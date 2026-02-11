# fake_svdq.py
# -*- coding: utf-8 -*-
"""
Fake SVD-Quant (and MPO placeholder) for *_proj Linear weights.

Goal:
- For each allowed nn.Linear weight:
    self_attn.{q,k,v,o}_proj.weight
    mlp.{gate,down,up}_proj.weight
  apply (optionally) low-rank extraction + fake quant on residual, then reconstruct "restored weight".

Pipeline per linear:
1) (optional) low-rank: W ≈ L (rank=r)
2) residual: R = W - L
3) (optional) fake quant: Rq = Q(R)
4) reconstruct: What = L + Rq
5) overwrite linear.weight with What (cast back to original dtype)

Config:
svdq_config = {"svd": svd_dict, "quant": quant_dict}
- svd_dict supports:
    {"all": r}
    {layer_idx: r}
    {layer_idx: {"X": r}}  # X in {q,k,v,o,up,down,gate}
- quant_dict supports:
    {"all": (fmt, blk)}
    {layer_idx: (fmt, blk)}
    {layer_idx: {"X": (fmt, blk)}}
  fmt in {"fp4","fp4_e2m1","nvfp4","mxfp4","mxfp6","fp8"}
  blk in {64,128,256,512,"in","out"}

Rules:
- If neither rank nor quant specified for a given linear => no-op.
- If rank specified but quant missing => use low-rank only (drop residual): What = L
- If quant specified but rank missing => quantize full W: What = Q(W)

Decomposition switch:
- You can set method to "mpo" (NotImplemented for now) via:
    svdq_config.get("method") or svdq_config["svd"].get("method")
  Default: "svd".

Public API:
- fake_op(model, svdq_config) -> model   (in-place modification)
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


# ----------------------------
# Allowed targets & name parse
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

# module path: model.layers.{i}.{mlp|self_attn}.{proj}
_TARGET_RE = re.compile(r"^model\.layers\.(\d+)\.(mlp|self_attn)\.([A-Za-z0-9_]+)$")


def _iter_allowed_linears(model: nn.Module) -> Dict[str, nn.Module]:
    out: Dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        m = _TARGET_RE.match(name)
        if not m:
            continue
        block = m.group(2)
        proj = m.group(3)
        if (block, proj) not in _ALLOWED_PROJ:
            continue
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            out[name] = mod
    return out


def _name_to_layer_block_x(name: str) -> Tuple[int, str, str]:
    m = _TARGET_RE.match(name)
    if not m:
        raise ValueError(f"Not a target linear name: {name}")
    layer = int(m.group(1))
    block = m.group(2)
    proj = m.group(3)
    x = _ALLOWED_PROJ[(block, proj)]
    return layer, block, x


# ----------------------------
# Config resolver
# ----------------------------
def _key_to_int(k: Any) -> Optional[int]:
    if isinstance(k, int):
        return k
    if isinstance(k, str) and k.isdigit():
        return int(k)
    return None


def _resolve_svd_rank(svd_dict: Dict[str, Any], layer: int, x: str) -> Optional[int]:
    if not isinstance(svd_dict, dict):
        return None

    if "all" in svd_dict:
        v = svd_dict["all"]
        return int(v) if isinstance(v, (int, float)) else None

    # layer-specific
    if layer in svd_dict:
        v = svd_dict[layer]
    elif str(layer) in svd_dict:
        v = svd_dict[str(layer)]
    else:
        return None

    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, dict):
        if x in v and isinstance(v[x], (int, float)):
            return int(v[x])
    return None


def _resolve_quant_spec(
    quant_dict: Dict[str, Any], layer: int, x: str
) -> Optional[Tuple[str, Union[int, str]]]:
    if not isinstance(quant_dict, dict):
        return None

    if "all" in quant_dict:
        v = quant_dict["all"]
        if isinstance(v, (tuple, list)) and len(v) == 2:
            return str(v[0]), v[1]
        return None

    if layer in quant_dict:
        v = quant_dict[layer]
    elif str(layer) in quant_dict:
        v = quant_dict[str(layer)]
    else:
        return None

    if isinstance(v, (tuple, list)) and len(v) == 2:
        return str(v[0]), v[1]
    if isinstance(v, dict):
        if x in v:
            vv = v[x]
            if isinstance(vv, (tuple, list)) and len(vv) == 2:
                return str(vv[0]), vv[1]
    return None


def _resolve_method(svdq_config: Dict[str, Any]) -> str:
    # allow svdq_config["method"] or svdq_config["svd"]["method"]
    m = svdq_config.get("method", None)
    if isinstance(m, str) and m:
        return m.lower()
    svd = svdq_config.get("svd", {})
    if isinstance(svd, dict):
        m2 = svd.get("method", None)
        if isinstance(m2, str) and m2:
            return m2.lower()
    return "svd"


# ----------------------------
# Fake quantization kernels
# ----------------------------

def _em_for_fmt(fmt: str) -> Tuple[int, int]:
    fmt = fmt.lower()
    if fmt in ("fp4", "fp4_e2m1", "nvfp4", "mxfp4"):
        return 2, 1  # e2m1
    if fmt in ("mxfp6",):
        return 3, 2  # e3m2  (as you requested)
    if fmt in ("fp8",):
        return 4, 3  # e4m3  (as you requested)
    raise ValueError(f"Unknown quant fmt: {fmt!r}")


def _max_finite_minifloat(e: int, m: int) -> float:
    # IEEE-like: exponent all-ones reserved -> max exp_field = 2^e - 2
    bias = (1 << (e - 1)) - 1
    max_exp_field = (1 << e) - 2
    max_exp = max_exp_field - bias
    max_mant = 2.0 - (2.0 ** (-m))  # (2 - 2^{-m})
    return float(max_mant * (2.0 ** max_exp))


def _quantize_minifloat_unit(x: torch.Tensor, e: int, m: int) -> torch.Tensor:
    """
    Quantize x into an IEEE-like minifloat(e,m) domain with:
      - normals + subnormals
      - saturate to max finite
    x is assumed float32, and roughly scaled so values are in a reasonable range.
    Returns float32 (dequantized).
    """
    x = x.float()
    sign = torch.sign(x)
    ax = x.abs()

    # constants
    bias = (1 << (e - 1)) - 1
    min_exp = 1 - bias
    max_exp_field = (1 << e) - 2
    max_exp = max_exp_field - bias
    max_finite = _max_finite_minifloat(e, m)

    # masks
    is_zero = ax == 0

    # frexp: ax = mant * 2**exp, mant in [0.5, 1) for nonzero
    mant, exp = torch.frexp(ax)  # mant float, exp int
    mant2 = mant * 2.0          # [1,2)
    exp2 = exp - 1              # unbiased exponent as int tensor

    # classify
    is_over = exp2 > max_exp
    is_normal = (exp2 >= min_exp) & (exp2 <= max_exp) & (~is_zero)
    is_under = (exp2 < min_exp) & (~is_zero)

    # normals: quantize mantissa
    if m > 0:
        frac = mant2 - 1.0
        frac_q = torch.round(frac * (1 << m)) / float(1 << m)
        mant_q = 1.0 + frac_q
    else:
        mant_q = torch.ones_like(mant2)

    val_norm = torch.ldexp(mant_q, exp2)  # mant_q * 2**exp2

    # subnormals: exponent fixed to min_exp, mantissa has no implicit leading 1
    # value = (mant_bits / 2^m) * 2^min_exp
    scale_sub = 2.0 ** float(min_exp)
    mant_sub = ax / scale_sub
    mant_sub_q = torch.round(mant_sub * (1 << m)) / float(1 << m)
    mant_sub_q = torch.clamp(mant_sub_q, 0.0, (float((1 << m) - 1) / float(1 << m)) if m > 0 else 0.0)
    val_sub = mant_sub_q * scale_sub

    # overflow: saturate
    val_over = torch.full_like(ax, float(max_finite))

    # assemble
    val = torch.zeros_like(ax)
    val = torch.where(is_normal, val_norm, val)
    val = torch.where(is_under, val_sub, val)
    val = torch.where(is_over, val_over, val)
    val = torch.where(is_zero, torch.zeros_like(val), val)

    return sign * val


def _fake_quant_block_fp(x: torch.Tensor, e: int, m: int, eps: float) -> torch.Tensor:
    """
    Blockwise scaling + minifloat quantization:
      scale = amax / max_finite
      y = Q_minifloat(x/scale) * scale
    """
    x = x.float()
    a = x.abs().max()
    max_f = _max_finite_minifloat(e, m)

    # if all zeros, return zeros
    if a.item() == 0.0:
        return torch.zeros_like(x)

    scale = torch.clamp_min(a / max_f, eps)
    y = _quantize_minifloat_unit(x / scale, e=e, m=m) * scale
    return y


def _fake_quant_channel_in_fp(x: torch.Tensor, e: int, m: int, eps: float) -> torch.Tensor:
    x = x.float()
    a = x.abs().max(dim=0).values  # [in]
    max_f = _max_finite_minifloat(e, m)
    scale = torch.clamp_min(a / max_f, eps)  # [in]
    y = _quantize_minifloat_unit(x / scale.view(1, -1), e=e, m=m) * scale.view(1, -1)
    return y


def _fake_quant_channel_out_fp(x: torch.Tensor, e: int, m: int, eps: float) -> torch.Tensor:
    x = x.float()
    a = x.abs().max(dim=1).values  # [out]
    max_f = _max_finite_minifloat(e, m)
    scale = torch.clamp_min(a / max_f, eps)  # [out]
    y = _quantize_minifloat_unit(x / scale.view(-1, 1), e=e, m=m) * scale.view(-1, 1)
    return y


def _fake_quant_tile2d_fp(x: torch.Tensor, tile: int, e: int, m: int, eps: float) -> torch.Tensor:
    x = x.float()
    out, inp = x.shape
    y = torch.empty_like(x)
    for i0 in range(0, out, tile):
        i1 = min(i0 + tile, out)
        for j0 in range(0, inp, tile):
            j1 = min(j0 + tile, inp)
            blk = x[i0:i1, j0:j1]
            y[i0:i1, j0:j1] = _fake_quant_block_fp(blk, e=e, m=m, eps=eps)
    return y


def fake_quantize(x: torch.Tensor, fmt: str, blk: Union[int, str], eps: float = 1e-6) -> torch.Tensor:
    """
    Return dequantized float32 tensor after fake *minifloat* quant.

    fmt:
      - fp4/fp4_e2m1/nvfp4/mxfp4 -> e2m1
      - mxfp6 -> e3m2
      - fp8 -> e4m3

    blk:
      - int (64/128/256/512): tile2d scale per tile×tile
      - "in": per input-channel scale (per column)
      - "out": per output-channel scale (per row)
    """
    e, m = _em_for_fmt(fmt)

    if isinstance(blk, str):
        b = blk.lower()
        if b == "in":
            return _fake_quant_channel_in_fp(x, e=e, m=m, eps=eps)
        if b == "out":
            return _fake_quant_channel_out_fp(x, e=e, m=m, eps=eps)
        raise ValueError(f"Invalid blk string: {blk!r} (expected 'in' or 'out')")

    if isinstance(blk, int):
        if blk not in (64, 128, 256, 512):
            raise ValueError(f"Invalid blk int: {blk} (expected 64/128/256/512)")
        return _fake_quant_tile2d_fp(x, tile=int(blk), e=e, m=m, eps=eps)

    raise TypeError(f"Invalid blk type: {type(blk)}")


# ----------------------------
# SVD / MPO (placeholder)
# ----------------------------
def svd_lowrank(W: torch.Tensor, r: int) -> torch.Tensor:
    """
    Truncated SVD low-rank approximation L (float32).
    W: [out, in] float32
    """
    out, inp = W.shape
    r_eff = int(max(0, min(r, out, inp)))
    if r_eff == 0:
        return torch.zeros_like(W)
    if r_eff == min(out, inp):
        return W

    # full_matrices=False gives U:[out,k], S:[k], Vh:[k,inp], where k=min(out,inp)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :r_eff]
    S_r = S[:r_eff]
    Vh_r = Vh[:r_eff, :]
    # (U_r * S_r) @ Vh_r
    L = (U_r * S_r.unsqueeze(0)) @ Vh_r
    return L


def mpo_lowrank_not_implemented(*args, **kwargs):
    raise NotImplementedError("MPO decomposition is TODO (not implemented in fake_svdq.py).")


# ----------------------------
# Main public API
# ----------------------------
@torch.no_grad()
def fake_op(model: nn.Module, svdq_config: Dict[str, Any]) -> nn.Module:
    """
    Apply fake SVD+quant reconstruction to selected *_proj weights in-place.
    Returns the same model object (modified).
    """
    if not isinstance(svdq_config, dict):
        raise TypeError("svdq_config must be a dict")

    svd_dict = svdq_config.get("svd", {}) or {}
    quant_dict = svdq_config.get("quant", {}) or {}
    method = _resolve_method(svdq_config)

    if method == "mpo":
        mpo_lowrank_not_implemented()

    if method != "svd":
        raise ValueError(f"Unknown decomposition method: {method!r} (expected 'svd' or 'mpo')")

    targets = _iter_allowed_linears(model)

    for name, mod in targets.items():
        layer, block, x = _name_to_layer_block_x(name)

        rank = _resolve_svd_rank(svd_dict, layer, x)
        qspec = _resolve_quant_spec(quant_dict, layer, x)

        # no-op if neither specified
        if rank is None and qspec is None:
            continue

        W0 = mod.weight.data
        dtype0 = W0.dtype
        dev0 = W0.device

        W = W0.float()  # compute in fp32

        # Determine operation mode
        if rank is None and qspec is not None:
            # quant-only: quantize full weight
            fmt, blk = qspec
            W_hat = fake_quantize(W, fmt=fmt, blk=blk, eps=1e-6)

        elif rank is not None and qspec is None:
            # svd-only: keep low-rank part (drop residual)
            L = svd_lowrank(W, r=int(rank))
            W_hat = L

        else:
            # svd + quant residual
            fmt, blk = qspec  # type: ignore[misc]
            L = svd_lowrank(W, r=int(rank))  # type: ignore[arg-type]
            R = W - L
            Rq = fake_quantize(R, fmt=fmt, blk=blk, eps=1e-6)
            W_hat = L + Rq

        mod.weight.data = W_hat.to(device=dev0, dtype=dtype0)

    return model
