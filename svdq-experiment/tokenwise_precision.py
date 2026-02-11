# tokenwise_precision.py
# -*- coding: utf-8 -*-
"""Tokenwise precision / forward-output error evaluation for Fake SVD-Quant.

This script implements three functions the user requested:

1) save_ori_XY(model, save_path, label)
   - Build representative inputs via get_veri_data(seed, cn=50, eu=80, len=128)
   - Run ONE forward pass over these inputs and capture X/Y of allowlisted *_proj linears
   - Save to {save_path}/{label}.safetensors (or .pt fallback)
     keys are "{mod_name}.X" and "{mod_name}.Y"

2) eval_tokenwise_fake(model, xy_path, out_json_path, ...)
   - Grid-search configs; for each config, apply fake_op(model, cfg) (no checkpointing)
   - For each allowlisted linear, use stored X/Y to compute Y~ = X @ W~^T and compare
   - Save results as JSON: {layer_idx: {layer_type: {"by_proj": {proj: {cfg: metrics}}, "agg": {...}}}}

3) eval_tokenwise_fake_smooth(smoothed_model, xy_path, out_json_path, ...)
   - Same as (2) but pre-process X with per-module smooth_factor (X' = X / smooth_factor)
   - smooth_factor is expected to be registered as a buffer on each target linear.

Notes
-----
* This file assumes your repo provides:
    - data_gen.get_veri_data
    - fake_svdq.fake_op
    - smooth_util.load_smoothed_model (optional; only for "eval_smooth" CLI)
* We do NOT run full end-to-end forward for every config. We only do X@W^T.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

os.environ['MODELSCOPE_CACHE'] = "/workspace/ms_cache"
os.environ['HF_HOME'] = "/workspace/hf_cache"
# ----------------------------
# Strict allowlist: ONLY these linears are considered.
# ----------------------------

_ALLOWED_PROJ = {
    ("self_attn", "q_proj"): "q",
    ("self_attn", "k_proj"): "k",
    ("self_attn", "v_proj"): "v",
    ("self_attn", "o_proj"): "o",
    ("mlp", "gate_proj"): "gate",
    ("mlp", "up_proj"): "up",
    ("mlp", "down_proj"): "down",
}

_TARGET_RE = re.compile(r"^model\.layers\.(\d+)\.(mlp|self_attn)\.([A-Za-z0-9_]+)$")


def _set_or_register_buffer(mod: nn.Module, name: str, tensor: torch.Tensor):
    if name in mod._buffers:
        mod._buffers[name] = tensor
    else:
        mod.register_buffer(name, tensor)


def _iter_allowed_linears(model: nn.Module) -> Dict[str, nn.Module]:
    """Return {module_path: module} for STRICT allowlisted linears.

    module_path example: "model.layers.0.mlp.gate_proj"
    """
    out: Dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        m = _TARGET_RE.match(name)
        if not m:
            continue
        proj = m.group(3)
        block = m.group(2)
        if (block, proj) not in _ALLOWED_PROJ:
            continue
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            out[name] = mod
    return out


def _name_to_layer_block_proj(name: str) -> Tuple[int, str, str]:
    m = _TARGET_RE.match(name)
    if not m:
        raise ValueError(f"Not a target linear name: {name}")
    layer = int(m.group(1))
    block = m.group(2)
    proj = m.group(3)
    return layer, block, proj


# ----------------------------
# I/O: safetensors (preferred) or torch.save fallback
# ----------------------------

def _try_import_safetensors():
    try:
        from safetensors.torch import load_file as st_load_file  # type: ignore
        from safetensors.torch import save_file as st_save_file  # type: ignore

        return st_load_file, st_save_file
    except Exception:
        return None, None


def save_tensor_dict(path: str, tensors: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None):
    st_load, st_save = _try_import_safetensors()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".safetensors") and st_save is not None:
        st_save(tensors, path, metadata=metadata or {})
        return
    # fallback
    torch.save({"tensors": tensors, "metadata": metadata or {}}, path)


def load_tensor_dict(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    st_load, _st_save = _try_import_safetensors()
    if path.endswith(".safetensors") and st_load is not None:
        td = st_load(path)
        # safetensors stores metadata separately; torch loader returns only tensors
        # we keep empty metadata here.
        return dict(td), {}
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "tensors" in obj:
        return dict(obj["tensors"]), dict(obj.get("metadata", {}))
    # tolerate raw dict
    if isinstance(obj, dict):
        return dict(obj), {}
    raise TypeError(f"Unsupported file content: {type(obj)}")


# ----------------------------
# Representative inputs
# ----------------------------

@dataclass
class VeriCfg:
    seed: int = 0
    cn: int = 50
    eu: int = 80
    length: int = 128
    device: str = "cpu"  # where input_ids live when generated (we may move later)


def build_veri_inputs(tokenizer: Any, cfg: VeriCfg) -> List[torch.Tensor]:
    """Use your repo's get_veri_data() to build representative input_ids tensors."""
    from data_gen import get_veri_data  # provided by your repo

    data = get_veri_data(
        tokenizer,
        seed=cfg.seed,
        cn=cfg.cn,
        cn_len=cfg.length,
        eu=cfg.eu,
        eu_len=cfg.length,
        device=cfg.device,
    )
    # Expected: {"cn": [tensor,...], "eu": [tensor,...]}
    inputs: List[torch.Tensor] = []
    for k in ("cn", "eu"):
        xs = data.get(k, [])
        if isinstance(xs, list):
            inputs.extend(xs)
    return inputs


def _infer_input_device(model: nn.Module) -> torch.device:
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    # common HF layout: model.model.embed_tokens
    try:
        mm = getattr(model, "model", None)
        if mm is not None:
            emb = getattr(mm, "embed_tokens", None)
            if emb is not None and hasattr(emb, "weight"):
                return emb.weight.device
    except Exception:
        pass
    # fallback
    return next(model.parameters()).device


def _build_attention_mask(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    if pad_id is None or pad_id < 0:
        return torch.ones_like(input_ids, dtype=torch.long)
    return (input_ids != pad_id).long()


# ----------------------------
# 1) Save original X/Y (hook capture)
# ----------------------------

@torch.no_grad()
def save_ori_XY(
    model: nn.Module,
    save_path: str,
    label: str,
    *,
    veri_cfg: VeriCfg = VeriCfg(),
    save_dtype: str = "bf16",  # "bf16" or "fp32"
    use_cache: bool = False,
) -> str:
    """Capture (X,Y) for each allowlisted linear across representative inputs and save.

    Returns the written file path.
    """

    # try to infer tokenizer if user attached it to model; else, user should use CLI.
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        raise RuntimeError(
            "Tokenizer not found on model. Use CLI (which loads tokenizer), or set model.tokenizer = tok."
        )

    device_in = _infer_input_device(model)
    model.eval()

    targets = _iter_allowed_linears(model)
    if not targets:
        raise RuntimeError("No target linears found. Check model structure or allowlist regex.")

    # storage lists per module
    xs: Dict[str, List[torch.Tensor]] = {name: [] for name in targets}
    ys: Dict[str, List[torch.Tensor]] = {name: [] for name in targets}

    if save_dtype.lower() == "fp32":
        out_dtype = torch.float32
    else:
        out_dtype = torch.bfloat16

    hooks = []

    def _make_hook(mod_name: str):
        def _hook(_mod: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            x = inp[0].detach()
            y = out.detach()
            # move to cpu for saving
            if isinstance(x, torch.Tensor) and x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if isinstance(y, torch.Tensor) and y.ndim == 3:
                y = y.reshape(-1, y.shape[-1])

            xs[mod_name].append(x.to("cpu", dtype=out_dtype, non_blocking=True))
            ys[mod_name].append(y.to("cpu", dtype=out_dtype, non_blocking=True))

        return _hook

    for name, mod in targets.items():
        hooks.append(mod.register_forward_hook(_make_hook(name)))

    # run forward over representative inputs (batch=1 each)
    pad_id = getattr(tok, "pad_token_id", None)
    for input_ids in build_veri_inputs(tok, veri_cfg):
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device_in)
        attn = _build_attention_mask(input_ids, pad_id if pad_id is not None else -1).to(device_in)
        _ = model(input_ids=input_ids, attention_mask=attn, use_cache=use_cache)

    for h in hooks:
        h.remove()

    # concat per module
    tensor_dict: Dict[str, torch.Tensor] = {}
    for name in targets.keys():
        if xs[name]:
            X = torch.cat(xs[name], dim=0)
            Y = torch.cat(ys[name], dim=0)
            tensor_dict[f"{name}.X"] = X
            tensor_dict[f"{name}.Y"] = Y

    meta = {
        "seed": str(veri_cfg.seed),
        "cn": str(veri_cfg.cn),
        "eu": str(veri_cfg.eu),
        "length": str(veri_cfg.length),
        "save_dtype": save_dtype,
    }

    out_file = os.path.join(save_path, f"{label}.safetensors")
    # If safetensors isn't available, save_tensor_dict will fall back to torch.save
    if _try_import_safetensors()[1] is None:
        out_file = os.path.join(save_path, f"{label}.pt")

    save_tensor_dict(out_file, tensor_dict, metadata=meta)
    return out_file


# ----------------------------
# Error metrics (tokenwise + NMSE)
# ----------------------------

def _apply_smooth_to_X(X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Apply SmoothQuant-style scaling: X' = X / s (s along input-channel)."""
    s = s.detach()
    if s.ndim == 0:
        return X / s
    if X.ndim == 3:
        return X / s.view(1, 1, -1)
    if X.ndim == 2:
        return X / s.view(1, -1)
    raise ValueError(f"Unsupported X shape: {tuple(X.shape)}")


@torch.no_grad()
def tokenwise_metrics(
    X: torch.Tensor,
    Y: torch.Tensor,
    W: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    *,
    device: Optional[torch.device] = None,
    chunk_tokens: int = 4096,
    eps: float = 1e-8,
    compute_percentiles: bool = True,
    percentiles: Tuple[float, float, float] = (0.95, 0.99, 0.999),
) -> Dict[str, float]:
    """Compute metrics between \u0176 = X @ W^T + b and Y.

    Metrics include:
      - nmse = ||Ŷ - Y||^2 / ||Y||^2
      - token_l2_mean / token_l2_max
      - token_rel_l2_mean / token_rel_l2_max, where token_rel_l2 = ||Ŷ_i - Y_i|| / (||Y_i|| + eps)
      - token_rel_l2_p95 / p99 / p999 (optional): percentiles of token_rel_l2 across tokens

    Shapes
    ------
    X: [B,T,in] or [N,in]
    Y: [B,T,out] or [N,out]
    W: [out,in]
    b: [out] (optional)
    """
    if X.ndim == 3:
        Xf = X.reshape(-1, X.shape[-1])
    else:
        Xf = X
    if Y.ndim == 3:
        Yf = Y.reshape(-1, Y.shape[-1])
    else:
        Yf = Y

    if Xf.shape[0] != Yf.shape[0]:
        raise ValueError(f"Token count mismatch: X {Xf.shape} vs Y {Yf.shape}")

    dev = device or (W.device if isinstance(W, torch.Tensor) else torch.device("cpu"))
    Wd = W.to(dev, dtype=torch.float32)

    sse = 0.0
    denom = 0.0
    sum_tok_l2 = 0.0
    max_tok_l2 = 0.0
    sum_tok_rel_l2 = 0.0
    max_tok_rel_l2 = 0.0
    n_tok = int(Xf.shape[0])

    # Optionally collect token_rel_l2 for percentile stats.
    tok_rel_all: Optional[torch.Tensor] = None
    if compute_percentiles:
        tok_rel_all = torch.empty(n_tok, dtype=torch.float32, device="cpu")

    # Preprocess bias once (avoid repeated device transfers).
    bd_dev: Optional[torch.Tensor] = None
    if b is not None:
        bd = b.detach()
        if bd.ndim != 1:
            bd = bd.view(-1)
        bd_dev = bd.to(dev, dtype=torch.float32)

    for i0 in range(0, n_tok, chunk_tokens):
        i1 = min(i0 + chunk_tokens, n_tok)
        Xc = Xf[i0:i1].to(dev, dtype=torch.float32)
        Yc = Yf[i0:i1].to(dev, dtype=torch.float32)

        Yh = Xc @ Wd.t()
        if bd_dev is not None:
            Yh = Yh + bd_dev.unsqueeze(0)

        D = Yh - Yc
        sse += float(D.pow(2).sum().item())
        denom += float(Yc.pow(2).sum().item())

        tok_l2 = torch.sqrt(torch.clamp_min(D.pow(2).sum(dim=1), 0.0))
        y_l2 = torch.sqrt(torch.clamp_min(Yc.pow(2).sum(dim=1), 0.0))
        tok_rel = tok_l2 / (y_l2 + eps)

        sum_tok_l2 += float(tok_l2.sum().item())
        max_tok_l2 = max(max_tok_l2, float(tok_l2.max().item()))

        sum_tok_rel_l2 += float(tok_rel.sum().item())
        max_tok_rel_l2 = max(max_tok_rel_l2, float(tok_rel.max().item()))

        if tok_rel_all is not None:
            tok_rel_all[i0:i1].copy_(tok_rel.detach().to("cpu", dtype=torch.float32))

    nmse = sse / max(denom, eps)
    out: Dict[str, float] = {
        "nmse": float(nmse),
        "sse": float(sse),
        "denom": float(denom),
        "token_l2_mean": float(sum_tok_l2 / max(n_tok, 1)),
        "token_l2_max": float(max_tok_l2),
        "token_rel_l2_mean": float(sum_tok_rel_l2 / max(n_tok, 1)),
        "token_rel_l2_max": float(max_tok_rel_l2),
        "n_tokens": float(n_tok),
    }

    if tok_rel_all is not None:
        # torch.quantile expects q in [0,1].
        qs = torch.tensor(list(percentiles), dtype=torch.float32)
        qv = torch.quantile(tok_rel_all, qs)
        out["token_rel_l2_p95"] = float(qv[0].item())
        out["token_rel_l2_p99"] = float(qv[1].item())
        out["token_rel_l2_p999"] = float(qv[2].item())

    return out


# ----------------------------
# Grid definition
# ----------------------------

def _parse_int_list(s: str) -> List[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def _parse_qspec_list(s: str) -> List[Tuple[str, Union[int, str]]]:
    """Parse "nvfp4@128,fp4@in" -> [("nvfp4",128),("fp4","in")]"""
    out: List[Tuple[str, Union[int, str]]] = []
    if not s.strip():
        return out
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if "@" not in item:
            raise ValueError(f"Bad qspec {item!r}, expected fmt@blk")
        fmt, blk = item.split("@", 1)
        fmt = fmt.strip()
        blk = blk.strip()
        if blk.isdigit():
            out.append((fmt, int(blk)))
        else:
            out.append((fmt, blk))
    return out


def _cfg_key(rank: Optional[int], qspec: Optional[Tuple[str, Union[int, str]]]) -> str:
    if rank is None:
        r = "none"
    else:
        r = str(int(rank))
    if qspec is None:
        return f"rank={r};quant=none"
    fmt, blk = qspec
    return f"rank={r};fmt={fmt};blk={blk}"


def build_svdq_cfg_all(rank: Optional[int], qspec: Optional[Tuple[str, Union[int, str]]]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"svd": {}, "quant": {}}
    if rank is not None:
        cfg["svd"] = {"all": int(rank)}
    if qspec is not None:
        fmt, blk = qspec
        cfg["quant"] = {"all": (fmt, blk)}
    return cfg


# ----------------------------
# 2) Evaluate forward-output error (no end-to-end forward)
# ----------------------------

@torch.no_grad()
def eval_tokenwise_fake(
    model: nn.Module,
    xy_path: str,
    out_json_path: str,
    *,
    ranks: List[int],
    qspecs: List[Tuple[str, Union[int, str]]],
    grid_mode: str = "combined",  # "combined" or "full"
    chunk_tokens: int = 4096,
    restore_from_cpu_backup: bool = True,
    backup_dtype: torch.dtype = torch.bfloat16,
    smooth: bool = False,
) -> Dict[str, Any]:
    """Grid-search configs and evaluate tokenwise output error for each allowlisted linear.

    If smooth=True, X will be preprocessed by each module.smooth_factor: X' = X / smooth_factor.
    """
    model.eval()
    targets = _iter_allowed_linears(model)
    if not targets:
        raise RuntimeError("No target linears found.")

    xy_tensors, xy_meta = load_tensor_dict(xy_path)

    # CPU backup for restoring weights between configs (optional but recommended for grid)
    cpu_backup: Dict[str, torch.Tensor] = {}
    if restore_from_cpu_backup:
        for name, mod in targets.items():
            cpu_backup[name] = mod.weight.detach().to("cpu", dtype=backup_dtype).clone()

    from fake_svdq import fake_op  # in-place op provided by your repo

    # Build grid
    cfg_items: List[Tuple[Optional[int], Optional[Tuple[str, Union[int, str]]]]] = []
    if grid_mode == "combined":
        for r in ranks:
            for q in qspecs:
                cfg_items.append((r, q))
    elif grid_mode == "full":
        # include quant-only and svd-only
        for q in qspecs:
            cfg_items.append((None, q))
        for r in ranks:
            cfg_items.append((r, None))
        for r in ranks:
            for q in qspecs:
                cfg_items.append((r, q))
    else:
        raise ValueError("grid_mode must be 'combined' or 'full'")

    # results
    results: Dict[str, Any] = {
        "meta": {
            "xy_path": xy_path,
            "xy_meta": xy_meta,
            "grid_mode": grid_mode,
            "ranks": ranks,
            "qspecs": [(a, str(b)) for a, b in qspecs],
            "smooth": bool(smooth),
            "chunk_tokens": int(chunk_tokens),
        },
        "by_layer": {},
    }

    # Helper to restore weights
    def _restore_weights():
        if not restore_from_cpu_backup:
            return
        for nm, mod in targets.items():
            w0 = cpu_backup[nm]
            mod.weight.data.copy_(w0.to(mod.weight.device, dtype=mod.weight.dtype))

    # Evaluate
    for rank, qspec in cfg_items:
        cfg_key = _cfg_key(rank, qspec)
        svdq_cfg = build_svdq_cfg_all(rank, qspec)
        _restore_weights()
        _ = fake_op(model, svdq_cfg)

        # per-module metrics
        per_layer_acc: Dict[Tuple[int, str], List[float]] = {}

        for name, mod in targets.items():
            Xk = f"{name}.X"
            Yk = f"{name}.Y"
            if Xk not in xy_tensors or Yk not in xy_tensors:
                continue
            X = xy_tensors[Xk]
            Y = xy_tensors[Yk]

            if smooth:
                if not hasattr(mod, "smooth_factor") or getattr(mod, "smooth_factor") is None:
                    raise RuntimeError(f"smooth=True but {name} has no smooth_factor buffer")
                X = _apply_smooth_to_X(X, getattr(mod, "smooth_factor"))

            W = mod.weight.detach()
            m = tokenwise_metrics(
                X,
                Y,
                W,
                b=mod.bias,
                device=W.device,
                chunk_tokens=chunk_tokens,
            )

            layer, block, proj = _name_to_layer_block_proj(name)
            layer_s = str(layer)
            if layer_s not in results["by_layer"]:
                results["by_layer"][layer_s] = {}
            if block not in results["by_layer"][layer_s]:
                results["by_layer"][layer_s][block] = {"by_proj": {}, "agg": {}}
            by_proj = results["by_layer"][layer_s][block]["by_proj"]
            if proj not in by_proj:
                by_proj[proj] = {}
            by_proj[proj][cfg_key] = m

            per_layer_acc.setdefault((layer, block), []).append(float(m["nmse"]))

        # fill aggregates for this cfg
        for (layer, block), vals in per_layer_acc.items():
            layer_s = str(layer)
            agg = results["by_layer"][layer_s][block]["agg"].setdefault(cfg_key, {})
            agg["nmse_mean_over_proj"] = float(sum(vals) / max(len(vals), 1))
            agg["nmse_max_over_proj"] = float(max(vals) if vals else 0.0)
            agg["n_proj"] = int(len(vals))

    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


# ----------------------------
# 3) Smooth version: wrapper
# ----------------------------

@torch.no_grad()
def eval_tokenwise_fake_smooth(
    model: nn.Module,
    xy_path: str,
    out_json_path: str,
    *,
    ranks: List[int],
    qspecs: List[Tuple[str, Union[int, str]]],
    grid_mode: str = "combined",
    chunk_tokens: int = 4096,
    restore_from_cpu_backup: bool = True,
    backup_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    return eval_tokenwise_fake(
        model,
        xy_path,
        out_json_path,
        ranks=ranks,
        qspecs=qspecs,
        grid_mode=grid_mode,
        chunk_tokens=chunk_tokens,
        restore_from_cpu_backup=restore_from_cpu_backup,
        backup_dtype=backup_dtype,
        smooth=True,
    )


# ----------------------------
# CLI
# ----------------------------

def _load_hf_model_tokenizer(model_dir: str, torch_dtype: str = "auto", device_map: Optional[str] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        # keep it deterministic; prefer eos as pad if needed
        tok.pad_token = tok.eos_token

    dtype = torch_dtype
    if torch_dtype not in ("auto", "fp16", "bf16", "fp32"):
        raise ValueError("torch_dtype must be auto|fp16|bf16|fp32")
    td = None
    if dtype == "fp16":
        td = torch.float16
    elif dtype == "bf16":
        td = torch.bfloat16
    elif dtype == "fp32":
        td = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=td,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    # attach tokenizer for save_ori_XY
    model.tokenizer = tok
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_save = sub.add_parser("save_xy")
    ap_save.add_argument("--model_dir", type=str, required=True)
    ap_save.add_argument("--save_path", type=str, required=True)
    ap_save.add_argument("--label", type=str, required=True)
    ap_save.add_argument("--seed", type=int, default=0)
    ap_save.add_argument("--cn", type=int, default=50)
    ap_save.add_argument("--eu", type=int, default=80)
    ap_save.add_argument("--length", type=int, default=128)
    ap_save.add_argument("--torch_dtype", type=str, default="auto")
    ap_save.add_argument("--device_map", type=str, default=None)
    ap_save.add_argument("--save_dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    ap_save.add_argument("--use_cache", action="store_true")

    ap_eval = sub.add_parser("eval_fake")
    ap_eval.add_argument("--model_dir", type=str, required=True)
    ap_eval.add_argument("--xy_path", type=str, required=True)
    ap_eval.add_argument("--out_json", type=str, required=True)
    ap_eval.add_argument("--ranks", type=str, default="16,32")
    ap_eval.add_argument("--qspecs", type=str, default="nvfp4@128")
    ap_eval.add_argument("--grid_mode", type=str, default="combined", choices=["combined", "full"])
    ap_eval.add_argument("--chunk_tokens", type=int, default=4096)
    ap_eval.add_argument("--torch_dtype", type=str, default="auto")
    ap_eval.add_argument("--device_map", type=str, default=None)
    ap_eval.add_argument("--no_restore", action="store_true")
    ap_eval.add_argument("--backup_dtype", type=str, default="bf16", choices=["bf16","fp16","fp32"])

    ap_smt = sub.add_parser("eval_smooth")
    ap_smt.add_argument("--smooth_model_dir", type=str, required=True)
    ap_smt.add_argument("--xy_path", type=str, required=True)
    ap_smt.add_argument("--out_json", type=str, required=True)
    ap_smt.add_argument("--ranks", type=str, default="16,32")
    ap_smt.add_argument("--qspecs", type=str, default="nvfp4@128")
    ap_smt.add_argument("--grid_mode", type=str, default="combined", choices=["combined", "full"])
    ap_smt.add_argument("--chunk_tokens", type=int, default=4096)
    ap_smt.add_argument("--torch_dtype", type=str, default="auto")
    ap_smt.add_argument("--device_map", type=str, default=None)
    ap_smt.add_argument("--no_restore", action="store_true")
    ap_smt.add_argument("--backup_dtype", type=str, default="bf16", choices=["bf16","fp16","fp32"])

    args = ap.parse_args()

    if args.cmd == "save_xy":
        model, _tok = _load_hf_model_tokenizer(args.model_dir, torch_dtype=args.torch_dtype, device_map=args.device_map)
        veri_cfg = VeriCfg(seed=args.seed, cn=args.cn, eu=args.eu, length=args.length, device="cpu")
        out_file = save_ori_XY(
            model,
            args.save_path,
            args.label,
            veri_cfg=veri_cfg,
            save_dtype=args.save_dtype,
            use_cache=bool(args.use_cache),
        )
        print(out_file)
        return

    if args.cmd == "eval_fake":
        model, _tok = _load_hf_model_tokenizer(args.model_dir, torch_dtype=args.torch_dtype, device_map=args.device_map)
        ranks = _parse_int_list(args.ranks)
        qspecs = _parse_qspec_list(args.qspecs)
        bd = torch.bfloat16 if args.backup_dtype == "bf16" else (torch.float16 if args.backup_dtype == "fp16" else torch.float32)
        _ = eval_tokenwise_fake(
            model,
            args.xy_path,
            args.out_json,
            ranks=ranks,
            qspecs=qspecs,
            grid_mode=args.grid_mode,
            chunk_tokens=int(args.chunk_tokens),
            backup_dtype=bd,
            restore_from_cpu_backup=(not bool(args.no_restore)),
        )
        print(args.out_json)
        return

    if args.cmd == "eval_smooth":
        # load smoothed model (your repo should provide it)
        try:
            from smooth_util import load_smoothed_model  # type: ignore

            model = load_smoothed_model(args.smooth_model_dir)
            # attach tokenizer for consistency (some impl already bundles it)
            if not hasattr(model, "tokenizer"):
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(args.smooth_model_dir, use_fast=True, trust_remote_code=True)
                if tok.pad_token_id is None:
                    tok.pad_token = tok.eos_token
                model.tokenizer = tok
        except Exception:
            # fallback: load as normal HF model
            model, _tok = _load_hf_model_tokenizer(
                args.smooth_model_dir,
                torch_dtype=args.torch_dtype,
                device_map=args.device_map,
            )

        ranks = _parse_int_list(args.ranks)
        qspecs = _parse_qspec_list(args.qspecs)
        bd = torch.bfloat16 if args.backup_dtype == "bf16" else (torch.float16 if args.backup_dtype == "fp16" else torch.float32)
        _ = eval_tokenwise_fake_smooth(
            model,
            args.xy_path,
            args.out_json,
            ranks=ranks,
            qspecs=qspecs,
            grid_mode=args.grid_mode,
            chunk_tokens=int(args.chunk_tokens),
            backup_dtype=bd,
            restore_from_cpu_backup=(not bool(args.no_restore)),
        )
        print(args.out_json)
        return


if __name__ == "__main__":
    main()