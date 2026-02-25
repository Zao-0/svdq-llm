#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
save_fake_quant.py (v2)

Save a *fake-quantized* HuggingFace Transformers checkpoint.

Key behaviors (per your spec)
-----------------------------
1) Support both:
   - raw (no SmoothQuant): normal HF checkpoint
   - smoothed (calibration): checkpoint that contains per-linear `smooth_factor` tensors
     (saved by smooth_util.py). We re-register smooth_factor buffers before saving.

2) Fake-quantization is done by **replacing Linear.weight in-place** for the allowed *_proj linears.
   - For raw models: the saved checkpoint can be loaded by Transformers as a normal model.
   - For smoothed models: load exactly like smooth_util.load_smoothed_model (i.e. re-register
     smooth_factor buffers from the saved weights).

3) Quant schemes:
   - uniform scheme applied to all allowlisted *_proj weights, OR
   - per-weight `strategy.json` mapping:
        "model.layers.0.mlp.down_proj.weight": "rank=128;fmt=fp4;blk=in"
     (each weight may have a different scheme)

4) All runtime parameters come from a YAML config; CLI only takes --config.

Notes
-----
- v2: avoid torch.linalg.svd_lowrank (not available on some torch builds); use compat SVD.
- This script ONLY touches:
    model.layers.{i}.{mlp|self_attn}.{q,k,v,o,gate,up,down}_proj.weight
  and does NOT touch embed_tokens or lm_head.
- The fake-quant kernels and SVD are taken from `fake_svdq.py` in your repo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn

# Optional cache defaults (safe to remove if you don't use ModelScope).
os.environ.setdefault("MODELSCOPE_CACHE", "/workspace/ms_cache")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")


# -------------------------
# YAML loader
# -------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to read the config. Install with: pip install pyyaml"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise TypeError(f"YAML root must be a dict, got {type(obj)}")
    return obj


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# -------------------------
# Allowlist: ONLY target these *_proj linears
# -------------------------
_ALLOWED_PROJ = {
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("mlp", "down_proj"),
}
_TARGET_RE = re.compile(r"^model\.layers\.(\d+)\.(mlp|self_attn)\.([A-Za-z0-9_]+)$")


def iter_allowed_linears(model: nn.Module) -> Dict[str, nn.Module]:
    """
    Return {module_path: module} for STRICT allowlisted linears.

    module_path example: "model.layers.0.mlp.gate_proj"
    """
    out: Dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        m = _TARGET_RE.match(name)
        if not m:
            continue
        block, proj = m.group(2), m.group(3)
        if (block, proj) not in _ALLOWED_PROJ:
            continue
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            out[name] = mod
    return out


# -------------------------
# Scheme parsing
# -------------------------
@dataclass(frozen=True)
class Scheme:
    rank: Optional[int] = None
    fmt: Optional[str] = None
    blk: Optional[Union[int, str]] = None

    @property
    def mode(self) -> str:
        if self.rank is None and self.fmt is None:
            return "noop"
        if self.rank is None and self.fmt is not None:
            return "quant_only"
        if self.rank is not None and self.fmt is None:
            return "svd_only"
        return "svd_plus_quant"

    def to_string(self) -> str:
        r = "none" if self.rank is None else str(int(self.rank))
        if self.fmt is None:
            return f"rank={r};quant=none"
        return f"rank={r};fmt={self.fmt};blk={self.blk}"


def _parse_blk(v: str) -> Union[int, str]:
    v = v.strip()
    if v.isdigit():
        return int(v)
    return v


def parse_scheme(obj: Union[str, Dict[str, Any], None]) -> Scheme:
    """
    Supported forms:
      - "rank=128;fmt=fp4;blk=in"
      - "rank=none;fmt=nvfp4;blk=na" (blk ignored by nvfp4/mxfp4/mxfp6 in fake_svdq)
      - "rank=32;quant=none"
      - {"rank": 128, "fmt": "fp4", "blk": "in"} or {"rank": None, "quant": "none"}
    """
    if obj is None:
        return Scheme()

    if isinstance(obj, dict):
        rank = obj.get("rank", None)
        fmt = obj.get("fmt", None)
        blk = obj.get("blk", None)

        # allow {"quant": "none"}
        q = obj.get("quant", None)
        if q is not None and str(q).lower() == "none":
            fmt = None
            blk = None

        if rank is None or str(rank).lower() in ("none", "null", ""):
            rank_i = None
        else:
            rank_i = int(rank)

        if fmt is None or str(fmt).lower() in ("none", "null", ""):
            fmt_s = None
            blk_v = None
        else:
            fmt_s = str(fmt).lower().strip()
            blk_v = _parse_blk(str(blk)) if blk is not None else None

        return Scheme(rank=rank_i, fmt=fmt_s, blk=blk_v)

    if not isinstance(obj, str):
        raise TypeError(f"scheme must be str|dict|None, got {type(obj)}")

    s = obj.strip()
    if not s:
        return Scheme()

    parts = [p.strip() for p in s.split(";") if p.strip()]
    kv: Dict[str, str] = {}
    for p in parts:
        if "=" not in p:
            # tolerate "nvfp4@na" as a shorthand
            if "@" in p:
                fmt, blk = p.split("@", 1)
                kv["fmt"] = fmt.strip()
                kv["blk"] = blk.strip()
                continue
            raise ValueError(f"Bad scheme token {p!r} in {obj!r}")
        k, v = p.split("=", 1)
        kv[k.strip().lower()] = v.strip()

    rank_v = kv.get("rank", None)
    if rank_v is None or rank_v.lower() in ("none", "null", ""):
        rank = None
    else:
        rank = int(rank_v)

    # allow "quant=none"
    qv = kv.get("quant", None)
    if qv is not None and qv.lower() == "none":
        return Scheme(rank=rank, fmt=None, blk=None)

    fmt_v = kv.get("fmt", None)
    if fmt_v is None or fmt_v.lower() in ("none", "null", ""):
        fmt = None
        blk = None
    else:
        fmt = fmt_v.lower().strip()
        blk = _parse_blk(kv.get("blk", "in"))

    return Scheme(rank=rank, fmt=fmt, blk=blk)


def load_strategy_json(path: str) -> Dict[str, Scheme]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"strategy json root must be dict, got {type(obj)}")
    out: Dict[str, Scheme] = {}
    for k, v in obj.items():
        out[str(k)] = parse_scheme(v)
    return out


# -------------------------
# Smooth-factor re-registration (for smoothed checkpoints)
# -------------------------
def set_or_register_buffer(mod: nn.Module, name: str, tensor: torch.Tensor):
    if name in mod._buffers:
        mod._buffers[name] = tensor
    else:
        mod.register_buffer(name, tensor)


def load_smooth_factors_from_checkpoint(model_dir: str) -> Dict[str, torch.Tensor]:
    """
    Reuse smooth_util's fast reader if available; otherwise do a minimal safe_open scan.
    Returns state-dict keys like "model.layers.0.mlp.gate_proj.smooth_factor".
    """
    try:
        import smooth_util  # your repo module
        if hasattr(smooth_util, "_load_smooth_factors_only"):
            return smooth_util._load_smooth_factors_only(model_dir)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Fallback: scan single-file safetensors only (no sharded index handling).
    out: Dict[str, torch.Tensor] = {}
    st_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(st_path):
        return out
    try:
        from safetensors import safe_open  # type: ignore
    except Exception:
        return out

    with safe_open(st_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith(".smooth_factor"):
                out[k] = f.get_tensor(k)
    return out


def reregister_smooth_factors(model: nn.Module, model_dir: str):
    """
    Ensure every allowlisted *_proj linear has a `smooth_factor` buffer registered.
    - If the checkpoint stores it, load it.
    - Otherwise fallback to ones.
    """
    targets = iter_allowed_linears(model)
    smooth_sd = load_smooth_factors_from_checkpoint(model_dir)

    for name, mod in targets.items():
        key = f"{name}.smooth_factor"
        if key in smooth_sd:
            t = smooth_sd[key].detach().to(dtype=torch.float32, device="cpu")
        else:
            in_dim = int(mod.weight.shape[1])
            t = torch.ones((in_dim,), dtype=torch.float32, device="cpu")
        set_or_register_buffer(mod, "smooth_factor", t)


# -------------------------
# Fake quant apply
# -------------------------
def _import_fake_svdq():
    """
    Prefer importing from your repo:
      - fake_svdq.py
    If not found, fall back to 'fake_svdq_v2' (in case you renamed).
    """
    try:
        import fake_svdq as m  # type: ignore
        return m
    except Exception:
        import importlib
        return importlib.import_module("fake_svdq_v2")


def _svd_lowrank_auto_compat(W: torch.Tensor, r: int, svd_cfg: Dict[str, Any]) -> torch.Tensor:
    """
    Torch-version-compatible low-rank reconstruction.

    Why this exists:
    - Some environments don't have `torch.linalg.svd_lowrank` (AttributeError).
    - We therefore avoid calling it entirely, and instead use:
        1) exact `torch.linalg.svd` for small matrices, or
        2) `torch.svd_lowrank` if available, else
        3) a lightweight randomized SVD (Halko-style) implemented with QR + SVD.

    Returns:
      L ≈ best rank-r approximation of W (in least-squares sense).
    """
    r = int(max(0, r))
    if r == 0:
        return torch.zeros_like(W)

    if W.ndim != 2:
        raise ValueError(f"SVD expects 2D matrix, got shape={tuple(W.shape)}")

    m, n = W.shape
    k = int(min(r, m, n))
    if k == 0:
        return torch.zeros_like(W)
    if k >= min(m, n):
        # full-rank request: just return W (already float32 in our pipeline)
        return W.clone()

    exact_threshold = int(svd_cfg.get("exact_threshold", 2048))
    niter = int(svd_cfg.get("niter", 2))
    oversample = int(svd_cfg.get("oversample", 8))

    # 1) Exact SVD for small dims (fast enough + stable)
    if min(m, n) <= exact_threshold:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]

    # 2) Prefer torch.svd_lowrank if present (works on many torch versions)
    if hasattr(torch, "svd_lowrank"):
        try:
            U, S, V = torch.svd_lowrank(W, q=k, niter=max(0, niter))
            return (U * S.unsqueeze(0)) @ V.transpose(0, 1)
        except Exception:
            # fall through to randomized SVD below
            pass

    # 3) Randomized SVD fallback (Halko)
    q = int(min(n, k + max(0, oversample)))
    # Random test matrix
    Omega = torch.randn((n, q), device=W.device, dtype=W.dtype)
    Y = W @ Omega
    # Power iteration to improve spectral decay separation
    for _ in range(max(0, niter)):
        Y = W @ (W.transpose(0, 1) @ Y)
    Q, _ = torch.linalg.qr(Y, mode="reduced")          # (m, q)
    B = Q.transpose(0, 1) @ W                          # (q, n)
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub[:, :k]
    return (U * S[:k].unsqueeze(0)) @ Vh[:k, :]


@torch.no_grad()
def apply_fake_quant_inplace(
    model: nn.Module,
    *,
    mode: str,
    uniform: Optional[Scheme] = None,
    strategy: Optional[Dict[str, Scheme]] = None,
    default: Optional[Scheme] = None,
    svd_cfg: Optional[Dict[str, Any]] = None,
    eps: float = 1e-6,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Mutate allowlisted linears' weights in-place and return an apply report.
    """
    svd_cfg = svd_cfg or {}
    fake_svdq_mod = _import_fake_svdq()

    if not hasattr(fake_svdq_mod, "fake_quantize"):
        raise RuntimeError("fake_svdq module must provide fake_quantize()")
    fake_quantize = fake_svdq_mod.fake_quantize  # type: ignore[attr-defined]

    targets = iter_allowed_linears(model)
    names = sorted(targets.keys())

    report: Dict[str, Any] = {
        "time_utc": now_utc_iso(),
        "mode": mode,
        "num_targets": len(names),
        "touched": 0,
        "skipped": 0,
        "per_weight": {},
    }

    for i, name in enumerate(names):
        mod = targets[name]
        w_key = f"{name}.weight"

        if mode == "uniform":
            scheme = uniform or Scheme()
        elif mode == "strategy":
            assert strategy is not None
            scheme = strategy.get(w_key, default or Scheme())
        else:
            raise ValueError("mode must be 'uniform' or 'strategy'")

        if scheme.mode == "noop":
            report["skipped"] += 1
            report["per_weight"][w_key] = {"scheme": scheme.to_string(), "status": "noop"}
            continue

        W0 = mod.weight.data
        dtype0 = W0.dtype
        dev0 = W0.device

        # compute in fp32 on the same device as W0
        W = W0.float()

        try:
            if scheme.mode == "quant_only":
                W_hat = fake_quantize(W, fmt=str(scheme.fmt), blk=scheme.blk, eps=float(eps))
            elif scheme.mode == "svd_only":
                L = _svd_lowrank_auto_compat(W, r=int(scheme.rank or 0), svd_cfg=svd_cfg)
                W_hat = L
            else:
                L = _svd_lowrank_auto_compat(W, r=int(scheme.rank or 0), svd_cfg=svd_cfg)
                R = W - L
                Rq = fake_quantize(R, fmt=str(scheme.fmt), blk=scheme.blk, eps=float(eps))
                W_hat = L + Rq

            mod.weight.data = W_hat.to(device=dev0, dtype=dtype0)
            report["touched"] += 1
            report["per_weight"][w_key] = {
                "scheme": scheme.to_string(),
                "status": "ok",
                "shape": list(W0.shape),
                "dtype": str(dtype0).replace("torch.", ""),
                "device": str(dev0),
            }
        except Exception as e:
            report["per_weight"][w_key] = {
                "scheme": scheme.to_string(),
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "shape": list(W0.shape),
                "dtype": str(dtype0).replace("torch.", ""),
                "device": str(dev0),
            }
            # keep going (best-effort)
            if verbose:
                print(f"[apply_fake_quant] ERROR @ {w_key}: {type(e).__name__}: {e}")

        if verbose and ((i + 1) % 20 == 0 or (i + 1) == len(names)):
            print(f"[apply_fake_quant] {i+1}/{len(names)} done")

        # free GPU cache between weights (best-effort)
        if torch.cuda.is_available() and dev0.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return report


# -------------------------
# Model loading & saving
# -------------------------
def _parse_torch_dtype(s: str):
    s = (s or "auto").lower().strip()
    if s == "auto":
        return "auto"
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {s!r}")


def load_model_tokenizer(cfg: Dict[str, Any]):
    from transformers import AutoTokenizer, AutoModelForCausalLM  # local import

    model_cfg = cfg.get("model", {}) or {}
    model_dir = str(model_cfg.get("model_dir", ""))
    if not model_dir:
        raise ValueError("config.model.model_dir is required")

    kind = str(model_cfg.get("kind", "raw")).lower().strip()
    if kind not in ("raw", "smoothed"):
        raise ValueError("config.model.kind must be 'raw' or 'smoothed'")

    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    low_cpu_mem_usage = bool(model_cfg.get("low_cpu_mem_usage", True))
    device_map = model_cfg.get("device_map", None)
    torch_dtype = _parse_torch_dtype(str(model_cfg.get("torch_dtype", "auto")))

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=None if torch_dtype == "auto" else torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )

    # For smoothed checkpoints, re-register smooth_factor buffers for allowlisted linears.
    if kind == "smoothed":
        reregister_smooth_factors(model, model_dir=model_dir)

    # If not sharded, optionally move to one device.
    if device_map is None:
        device = str(model_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        model = model.to(torch.device(device))

    model.eval()
    return model, tok, {"model_dir": model_dir, "kind": kind}


def save_checkpoint(
    model: nn.Module,
    tok: Any,
    out_dir: str,
    save_cfg: Dict[str, Any],
    extra_files: Optional[Dict[str, Any]] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    tok.save_pretrained(out_dir)

    safe_serialization = bool(save_cfg.get("safe_serialization", True))
    max_shard_size = str(save_cfg.get("max_shard_size", "10GB"))

    model.save_pretrained(
        out_dir,
        safe_serialization=safe_serialization,
        max_shard_size=max_shard_size,
    )

    if extra_files:
        for fn, obj in extra_files.items():
            path = os.path.join(out_dir, fn)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config path")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # 1) load model
    model, tok, model_meta = load_model_tokenizer(cfg)

    # 2) prepare quant spec
    qcfg = cfg.get("quant", {}) or {}
    qmode = str(qcfg.get("mode", "uniform")).lower().strip()
    if qmode not in ("uniform", "strategy"):
        raise ValueError("config.quant.mode must be 'uniform' or 'strategy'")

    uniform_scheme = None
    strategy_map = None

    if qmode == "uniform":
        uniform_scheme = parse_scheme(qcfg.get("uniform", ""))
        if uniform_scheme.mode == "noop":
            raise ValueError("uniform scheme is noop/empty; set config.quant.uniform properly")
    else:
        strat_path = str(qcfg.get("strategy_json", ""))
        if not strat_path:
            raise ValueError("config.quant.strategy_json is required when mode=strategy")
        strategy_map = load_strategy_json(strat_path)

    default_scheme = parse_scheme(qcfg.get("default", None)) if qcfg.get("default", None) is not None else None

    # 3) apply fake quant
    svd_cfg = cfg.get("svd", {}) or {}
    eps = float(qcfg.get("eps", 1e-6))

    report = apply_fake_quant_inplace(
        model,
        mode=qmode,
        uniform=uniform_scheme,
        strategy=strategy_map,
        default=default_scheme,
        svd_cfg=svd_cfg,
        eps=eps,
        verbose=bool(cfg.get("verbose", True)),
    )

    # 4) stamp config meta into model.config (optional but helpful)
    try:
        model.config.fake_quant = {
            "time_utc": report.get("time_utc"),
            "mode": qmode,
            "uniform": None if uniform_scheme is None else uniform_scheme.to_string(),
            "strategy_json": None if qmode != "strategy" else str(qcfg.get("strategy_json", "")),
            "default": None if default_scheme is None else default_scheme.to_string(),
            "eps": eps,
            "svd_cfg": svd_cfg,
        }
        model.config.fake_quant_timestamp_utc = report.get("time_utc")
    except Exception:
        pass

    # 5) save
    save_cfg = cfg.get("save", {}) or {}
    out_dir = str(save_cfg.get("out_dir", ""))
    if not out_dir:
        raise ValueError("config.save.out_dir is required")

    meta = {
        "time_utc": now_utc_iso(),
        "model": model_meta,
        "config_path": os.path.abspath(args.config),
        "quant": {
            "mode": qmode,
            "uniform": None if uniform_scheme is None else uniform_scheme.to_string(),
            "strategy_json": None if qmode != "strategy" else str(qcfg.get("strategy_json", "")),
            "default": None if default_scheme is None else default_scheme.to_string(),
            "eps": eps,
        },
        "svd": svd_cfg,
        "report_summary": {
            "num_targets": report.get("num_targets", 0),
            "touched": report.get("touched", 0),
            "skipped": report.get("skipped", 0),
        },
    }

    save_checkpoint(
        model,
        tok,
        out_dir=out_dir,
        save_cfg=save_cfg,
        extra_files={
            "fake_quant_meta.json": meta,
            "fake_quant_report.json": report,
        },
    )

    print(f"[save_fake_quant] saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
