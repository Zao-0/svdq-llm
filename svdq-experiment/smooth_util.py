# smooth_util.py
# -*- coding: utf-8 -*-
"""
Offline SmoothQuant-style smooth factor extraction & weight-only smoothing for *_proj linears.

STRICT CONSTRAINT (per your updated agreement):
- Only touch the 7 linear weights (torch.nn.Linear.weight) per layer:
    self_attn.{q,k,v,o}_proj.weight
    mlp.{gate,down,up}_proj.weight
- Do NOT modify LayerNorm/RMSNorm or any other parameters.
- Do NOT change model structure (only register per-linear buffer: smooth_factor).

Implication:
- We scale weights columns: W' = W * diag(smooth_factor)  (i.e., multiply each input-channel column by s_i)
- To preserve exact function, runtime should scale linear inputs by 1/s_i.
  (You said runtime absorption is for later engineering; here we just bake weights + store factors.)

Public APIs:
- smoothen_model(model_dir, smt_cfg, save_path, label) -> (model, tokenizer)
- load_smoothed_model(model_dir) -> (model, tokenizer)
"""

from __future__ import annotations

import os
import re
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Config
# -------------------------
@dataclass
class SmoothCfg:
    # calibration sampling
    seed: int = 0
    cn: int = 100
    eu: int = 160

    # devices
    device: str = "cuda:0"                 # where model runs if device_map is None
    device_for_inputs: Optional[str] = None  # where input_ids are placed; default inferred

    # smoothquant params
    alpha: float = 0.5
    eps: float = 1e-6
    act_stat: str = "absmax"               # "absmax" or "meanabs"

    # model loading
    torch_dtype: Any = "auto"
    device_map: Optional[Any] = None       # if not None, accelerate dispatch

    # saving
    safe_serialization: bool = True
    max_shard_size: str = "10GB"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_device(s: str) -> torch.device:
    return torch.device(s)


def _infer_input_device(cfg: SmoothCfg) -> str:
    # If model is sharded via device_map, CPU inputs are the safest default.
    if cfg.device_for_inputs is not None and len(cfg.device_for_inputs) > 0:
        return cfg.device_for_inputs
    return "cpu" if cfg.device_map is not None else cfg.device


# -------------------------
# Target linears: STRICT allowlist
# -------------------------
_ALLOWED_PROJ = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "down_proj", "up_proj",
}

# module path should match:
#   model.layers.{i}.(mlp|self_attn).{proj_name}
_TARGET_RE = re.compile(r"^model\.layers\.(\d+)\.(mlp|self_attn)\.([A-Za-z0-9_]+)$")


def _iter_allowed_linears(model: nn.Module) -> Dict[str, nn.Module]:
    """
    Return {module_path: module} for STRICT allowlisted linears.
    module_path example: "model.layers.0.mlp.gate_proj"
    """
    out: Dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        m = _TARGET_RE.match(name)
        if not m:
            continue
        proj = m.group(3)
        if proj not in _ALLOWED_PROJ:
            continue
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            out[name] = mod
    return out


# -------------------------
# Activation stats via linear input hooks
# -------------------------
class _LinearInputStats:
    """
    Collect per-linear input-channel stats on calibration data.
    Stores per-module vector [in_features] (fp32) as:
      - absmax: max(|x|) over all tokens seen
      - meanabs: mean(|x|) over all tokens seen (proper sum/count)
    """
    def __init__(self, act_stat: str = "absmax"):
        if act_stat not in ("absmax", "meanabs"):
            raise ValueError(f"act_stat must be 'absmax' or 'meanabs', got {act_stat!r}")
        self.act_stat = act_stat
        self.handles: List[Any] = []
        self.absmax: Dict[str, torch.Tensor] = {}          # name -> [H]
        self.sum_count: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}  # name -> (sum[H], count[1])

    def add(self, name: str, linear: nn.Module):
        # Use forward_pre_hook to capture input tensor to linear
        def _pre_hook(_mod: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            # x: [..., in_features]
            if x.ndim < 2:
                return
            h = x.shape[-1]
            x2 = x.reshape(-1, h).float().abs()  # [N, H]

            if self.act_stat == "absmax":
                v = x2.max(dim=0).values  # [H]
                if name not in self.absmax:
                    self.absmax[name] = v.detach()
                else:
                    self.absmax[name] = torch.maximum(self.absmax[name], v.detach())
            else:
                s = x2.sum(dim=0)  # [H]
                c = torch.tensor([x2.shape[0]], device=s.device, dtype=s.dtype)  # [1]
                if name not in self.sum_count:
                    self.sum_count[name] = (s.detach(), c.detach())
                else:
                    s0, c0 = self.sum_count[name]
                    self.sum_count[name] = (s0 + s.detach(), c0 + c.detach())

        self.handles.append(linear.register_forward_pre_hook(_pre_hook))

    def finalize(self) -> Dict[str, torch.Tensor]:
        if self.act_stat == "absmax":
            # ensure fp32
            return {k: v.float() for k, v in self.absmax.items()}

        out: Dict[str, torch.Tensor] = {}
        for k, (s, c) in self.sum_count.items():
            mean = s / torch.clamp_min(c, 1.0)
            out[k] = mean.float()
        return out

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []


def _weight_col_absmax(w: torch.Tensor) -> torch.Tensor:
    # w: [out, in] -> [in]
    return w.float().abs().max(dim=0).values


def _compute_smooth_factor(
    act: torch.Tensor,          # [in]
    w_absmax: torch.Tensor,     # [in]
    alpha: float,
    eps: float,
) -> torch.Tensor:
    act = torch.clamp_min(act, eps)
    w = torch.clamp_min(w_absmax, eps)
    a = float(alpha)
    s = (act.pow(a)) / (w.pow(1.0 - a))
    s = torch.clamp(s, min=eps, max=1.0 / eps)
    return s


def _set_or_register_buffer(mod: nn.Module, name: str, tensor: torch.Tensor):
    if name in mod._buffers:
        mod._buffers[name] = tensor
    else:
        mod.register_buffer(name, tensor)


def _run_calibration_forward(model: nn.Module, ids_list: List[torch.Tensor]):
    model.eval()
    with torch.inference_mode():
        for ids in ids_list:
            if ids.ndim != 1:
                ids = ids.view(-1)
            inp = ids.unsqueeze(0)  # [1, T]
            try:
                model(input_ids=inp, use_cache=False)
            except TypeError:
                model(input_ids=inp)


# -------------------------
# Saving / Loading
# -------------------------
def smoothen_model(model_dir: str, smt_cfg: Dict[str, Any], save_path: str, label: str):
    """
    Load model+tokenizer, compute smooth_factor for STRICT allowlisted *_proj linears,
    apply weight-only smoothing (scale columns), register per-linear smooth_factor buffers,
    set config.smoothed=True, then save to save_path/label.

    Returns: (model, tokenizer) with updated weights + buffers.
    """
    cfg = SmoothCfg(**smt_cfg) if not isinstance(smt_cfg, SmoothCfg) else smt_cfg
    input_device = _infer_input_device(cfg)

    # 1) Load tokenizer + model
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=cfg.torch_dtype,
        device_map=cfg.device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if cfg.device_map is None:
        model = model.to(_as_device(cfg.device))

    # 2) Load calibration data
    from data_gen import get_cali_data

    cali = get_cali_data(tok, seed=cfg.seed, cn=cfg.cn, eu=cfg.eu, device=input_device)
    ids_all: List[torch.Tensor] = list(cali.get("cn", [])) + list(cali.get("eu", []))
    if not ids_all:
        raise RuntimeError("Empty calibration data from get_cali_data().")

    # deterministic shuffle
    rng = np.random.RandomState(cfg.seed & 0xFFFFFFFF)
    perm = rng.permutation(len(ids_all))
    ids_all = [ids_all[int(i)] for i in perm]

    # 3) Find STRICT allowlisted linears
    target = _iter_allowed_linears(model)  # module_path -> module
    if not target:
        raise RuntimeError("No allowlisted *_proj linears found. Check model structure/naming.")

    # 4) Collect activation stats (linear inputs)
    stats = _LinearInputStats(act_stat=cfg.act_stat)
    for name, mod in target.items():
        stats.add(name, mod)

    _run_calibration_forward(model, ids_all)
    act_stats = stats.finalize()
    stats.remove()

    # 5) Compute smooth_factor + apply weight-only smoothing + register buffer
    #    IMPORTANT: only touch Linear.weight for the allowlisted modules.
    smooth_meta: Dict[str, Any] = {}
    with torch.no_grad():
        for name, mod in target.items():
            w = mod.weight.data
            in_dim = w.shape[1]

            act = act_stats.get(name, None)
            if act is None:
                # if this linear never executed (unlikely), fall back to ones
                s = torch.ones((in_dim,), device=w.device, dtype=torch.float32)
            else:
                act = act.to(device=w.device, dtype=torch.float32)
                w_abs = _weight_col_absmax(w).to(device=w.device, dtype=torch.float32)
                s = _compute_smooth_factor(act=act, w_absmax=w_abs, alpha=cfg.alpha, eps=cfg.eps)

            # register smooth_factor (buffer)
            _set_or_register_buffer(mod, "smooth_factor", s.detach())

            # smooth ONLY the weight (scale columns); do NOT touch bias or any other params
            mod.weight.data = (w.float() * s.view(1, -1)).to(w.dtype)

            smooth_meta[name] = {
                "in_dim": int(in_dim),
                "out_dim": int(w.shape[0]),
                "stat_found": bool(act_stats.get(name, None) is not None),
            }

    # 6) Update config and save
    model.config.smoothed = True
    model.config.smooth_timestamp_utc = _now_utc_iso()
    model.config.smooth_cfg = {
        "seed": cfg.seed,
        "cn": cfg.cn,
        "eu": cfg.eu,
        "alpha": cfg.alpha,
        "eps": cfg.eps,
        "act_stat": cfg.act_stat,
        "allowed_proj": sorted(list(_ALLOWED_PROJ)),
        "weight_only": True,
        "note": "Weights scaled by smooth_factor columns; runtime should scale linear inputs by 1/smooth_factor to preserve exact function.",
    }

    out_dir = os.path.join(save_path, label)
    os.makedirs(out_dir, exist_ok=True)

    # Save extra meta
    meta = {
        "time_utc": _now_utc_iso(),
        "model_dir": model_dir,
        "label": label,
        "num_cali": len(ids_all),
        "cn": cfg.cn,
        "eu": cfg.eu,
        "alpha": cfg.alpha,
        "eps": cfg.eps,
        "act_stat": cfg.act_stat,
        "num_target_linears": len(target),
        "targets": sorted(list(target.keys())),
        "per_linear": smooth_meta,
    }
    with open(os.path.join(out_dir, "smooth_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    tok.save_pretrained(out_dir)
    model.save_pretrained(
        out_dir,
        safe_serialization=cfg.safe_serialization,
        max_shard_size=cfg.max_shard_size,
    )

    return model, tok


# -------------------------
# Loader: restore smooth_factor buffers
# -------------------------
def _load_weight_map_index(model_dir: str) -> Tuple[Optional[Dict[str, str]], str]:
    st_index = os.path.join(model_dir, "model.safetensors.index.json")
    bin_index = os.path.join(model_dir, "pytorch_model.bin.index.json")

    if os.path.exists(st_index):
        with open(st_index, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj.get("weight_map", {}), "safetensors"

    if os.path.exists(bin_index):
        with open(bin_index, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj.get("weight_map", {}), "bin"

    if os.path.exists(os.path.join(model_dir, "model.safetensors")):
        return None, "safetensors"
    if os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        return None, "bin"

    return None, "none"


def _load_smooth_factors_only(model_dir: str) -> Dict[str, torch.Tensor]:
    weight_map, fmt = _load_weight_map_index(model_dir)
    out: Dict[str, torch.Tensor] = {}

    if fmt == "none":
        return out

    if fmt == "safetensors":
        from safetensors import safe_open  # type: ignore

        if weight_map is None:
            path = os.path.join(model_dir, "model.safetensors")
            with safe_open(path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.endswith(".smooth_factor"):
                        out[k] = f.get_tensor(k)
            return out

        shard_to_keys: Dict[str, List[str]] = {}
        for k, shard in weight_map.items():
            if k.endswith(".smooth_factor"):
                shard_to_keys.setdefault(shard, []).append(k)

        for shard, keys in shard_to_keys.items():
            path = os.path.join(model_dir, shard)
            with safe_open(path, framework="pt", device="cpu") as f:
                for k in keys:
                    out[k] = f.get_tensor(k)
        return out

    # fmt == "bin": need to load shards
    if weight_map is None:
        sd = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
        for k, v in sd.items():
            if k.endswith(".smooth_factor"):
                out[k] = v
        return out

    shards = sorted(set(weight_map.values()))
    for shard in shards:
        sd = torch.load(os.path.join(model_dir, shard), map_location="cpu")
        for k, v in sd.items():
            if k.endswith(".smooth_factor"):
                out[k] = v
    return out


def load_smoothed_model(model_dir: str):
    """
    Load model+tokenizer from smoothen_model output directory.
    Since base HF classes don't know about extra buffers on Linear, we re-register
    smooth_factor buffers from saved weights.

    Returns: (model, tokenizer)
    """
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Re-register smooth_factor buffers for allowlisted linears (and load tensors if present)
    target = _iter_allowed_linears(model)
    smooth_sd = _load_smooth_factors_only(model_dir)

    for name, mod in target.items():
        key = f"{name}.smooth_factor"
        if key in smooth_sd:
            t = smooth_sd[key].to(dtype=torch.float32)
        else:
            # fallback: ones
            in_dim = mod.weight.shape[1]
            t = torch.ones((in_dim,), dtype=torch.float32)
        _set_or_register_buffer(mod, "smooth_factor", t)

    return model, tok
