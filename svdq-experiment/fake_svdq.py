# -*- coding: utf-8 -*-
"""fake_svdq_v3.py

Fake SVD-Quant **and MPO-Quant** for *_proj Linear weights.

What's new vs v2
----------------
1) Support `method: mpo` as a *high-precision side path* alternative to SVD:

   - Compute a 2-core MPO (a.k.a. 2-site tensor-train) approximation `W_hat` where
     `rank` is the virtual bond dimension.
   - Quantize the residual `R = W - W_hat` (fake minifloat).
   - Reconstruct `W' = W_hat + fake_quant(R)` and write back (same logic as your LoRA
     "side path" idea, but using MPO).

2) Remove the dependency on `torch.linalg.svd_lowrank` (some builds don't have it).
   We use a version-compatible low-rank backend:
     - exact SVD for small matrices
     - torch.svd_lowrank when available
     - randomized SVD fallback (Halko-style)

Public API
----------
* `fake_op(model, svdq_config)`   : in-place weight replacement for allowlisted linears.
* `stable_scanner(model, ...)`    : NMSE scan (no mutation) across schemes.
* `fake_quantize(x, fmt, blk)`    : minifloat fake-quant kernel (shared with saver).

Config
------
`svdq_config` is a dict:
{
  "method": "svd" | "mpo",           # default: "svd"
  "svd":   { ... ranks ... },         # ranks by layer (same format as v2)
  "quant": { ... qspec ... },         # qspec by layer (same format as v2)
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

# Optional preferred 2-way factorizations for MPO reshape.
# If a dim isn't present, we auto-factor it into (a,b) with a*b=dim and a≈sqrt(dim).
_IN_DIM_HINTS: Dict[int, Tuple[int, int]] = {
    1024: (32, 32),
    4096: (64, 64),
    12288: (96, 128),
}
_OUT_DIM_HINTS: Dict[int, Tuple[int, int]] = {
    1024: (32, 32),
    4096: (64, 64),
    12288: (128, 96),
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
# Config resolver (kept for fake_op)
# ----------------------------
def _resolve_svd_rank(svd_dict: Dict[str, Any], layer: int, x: str) -> Optional[int]:
    if not isinstance(svd_dict, dict):
        return None

    if "all" in svd_dict:
        v = svd_dict["all"]
        return int(v) if isinstance(v, (int, float)) else None

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
    m = svdq_config.get("method", None)
    if isinstance(m, str) and m:
        return m.lower().strip()
    svd = svdq_config.get("svd", {})
    if isinstance(svd, dict):
        m2 = svd.get("method", None)
        if isinstance(m2, str) and m2:
            return m2.lower().strip()
    return "svd"


# ----------------------------
# Fake quantization kernels
# ----------------------------


def _em_for_fmt(fmt: str) -> Tuple[int, int]:
    fmt = fmt.lower()
    if fmt in ("fp4", "fp4_e2m1", "nvfp4", "mxfp4"):
        return 2, 1  # e2m1
    if fmt in ("mxfp6", "fp6"):
        return 3, 2  # e3m2
    if fmt in ("fp8",):
        return 4, 3  # e4m3
    raise ValueError(f"Unknown quant fmt: {fmt!r}")


def _max_finite_minifloat(e: int, m: int) -> float:
    # IEEE-like: exponent all-ones reserved -> max exp_field = 2^e - 2
    bias = (1 << (e - 1)) - 1
    max_exp_field = (1 << e) - 2
    max_exp = max_exp_field - bias
    max_mant = 2.0 - (2.0 ** (-m))
    return float(max_mant * (2.0 ** max_exp))


def _quantize_minifloat_unit(x: torch.Tensor, e: int, m: int) -> torch.Tensor:
    """Quantize x into an IEEE-like minifloat(e,m) domain and return dequantized float32."""
    x = x.float()
    sign = torch.sign(x)
    ax = x.abs()

    bias = (1 << (e - 1)) - 1
    min_exp = 1 - bias
    max_exp_field = (1 << e) - 2
    max_exp = max_exp_field - bias
    max_finite = _max_finite_minifloat(e, m)

    is_zero = ax == 0

    mant, exp = torch.frexp(ax)
    mant2 = mant * 2.0
    exp2 = exp - 1

    is_over = exp2 > max_exp
    is_normal = (exp2 >= min_exp) & (exp2 <= max_exp) & (~is_zero)
    is_under = (exp2 < min_exp) & (~is_zero)

    if m > 0:
        frac = mant2 - 1.0
        frac_q = torch.round(frac * (1 << m)) / float(1 << m)
        mant_q = 1.0 + frac_q
    else:
        mant_q = torch.ones_like(mant2)

    val_norm = torch.ldexp(mant_q, exp2)

    # subnormals
    scale_sub = 2.0 ** float(min_exp)
    mant_sub = ax / scale_sub
    mant_sub_q = torch.round(mant_sub * (1 << m)) / float(1 << m)
    if m > 0:
        mant_sub_q = torch.clamp(mant_sub_q, 0.0, float((1 << m) - 1) / float(1 << m))
    else:
        mant_sub_q = torch.zeros_like(mant_sub_q)
    val_sub = mant_sub_q * scale_sub

    val_over = torch.full_like(ax, float(max_finite))

    val = torch.zeros_like(ax)
    val = torch.where(is_normal, val_norm, val)
    val = torch.where(is_under, val_sub, val)
    val = torch.where(is_over, val_over, val)
    val = torch.where(is_zero, torch.zeros_like(val), val)

    return sign * val


def _fake_quant_block_fp(x: torch.Tensor, e: int, m: int, eps: float) -> torch.Tensor:
    x = x.float()
    a = x.abs().max()
    max_f = _max_finite_minifloat(e, m)
    if a.item() == 0.0:
        return torch.zeros_like(x)
    scale = torch.clamp_min(a / max_f, eps)
    return _quantize_minifloat_unit(x / scale, e=e, m=m) * scale


def _fake_quant_channel_in_fp(x: torch.Tensor, e: int, m: int, eps: float) -> torch.Tensor:
    x = x.float()
    a = x.abs().amax(dim=0)  # [in]
    max_f = _max_finite_minifloat(e, m)
    scale = torch.clamp_min(a / max_f, eps)  # [in]
    return _quantize_minifloat_unit(x / scale.view(1, -1), e=e, m=m) * scale.view(1, -1)


def _fake_quant_channel_out_fp(x: torch.Tensor, e: int, m: int, eps: float) -> torch.Tensor:
    x = x.float()
    a = x.abs().amax(dim=1)  # [out]
    max_f = _max_finite_minifloat(e, m)
    scale = torch.clamp_min(a / max_f, eps)  # [out]
    return _quantize_minifloat_unit(x / scale.view(-1, 1), e=e, m=m) * scale.view(-1, 1)


def _fake_quant_tile2d_fp(x: torch.Tensor, tile: int, e: int, m: int, eps: float) -> torch.Tensor:
    """Vectorized 2D tiling quant (avoids Python loops)."""
    x = x.float()
    out, inp = x.shape
    t = int(tile)
    out_pad = (t - (out % t)) % t
    inp_pad = (t - (inp % t)) % t

    if out_pad or inp_pad:
        x_pad = torch.nn.functional.pad(x, (0, inp_pad, 0, out_pad), mode="constant", value=0.0)
    else:
        x_pad = x

    O, I = x_pad.shape
    ob = O // t
    ib = I // t
    # [ob, t, ib, t] -> [ob, ib, t, t]
    xb = x_pad.view(ob, t, ib, t).permute(0, 2, 1, 3).contiguous()
    a = xb.abs().amax(dim=(-1, -2), keepdim=True)
    max_f = _max_finite_minifloat(e, m)
    scale = torch.clamp_min(a / max_f, eps)
    yb = _quantize_minifloat_unit(xb / scale, e=e, m=m) * scale
    # back to [O, I]
    y_pad = yb.permute(0, 2, 1, 3).contiguous().view(O, I)
    return y_pad[:out, :inp]


def fake_quantize(x: torch.Tensor, fmt: str, blk: Union[int, str], eps: float = 1e-6) -> torch.Tensor:
    """Return dequantized float32 tensor after fake *minifloat* quant."""
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
# Low-rank backends (SVD / MPO)
# ----------------------------


def _svd_lowrank_exact(W: torch.Tensor, r: int) -> torch.Tensor:
    """Exact truncated SVD low-rank approximation (float32)."""
    if W.ndim != 2:
        raise ValueError(f"SVD expects 2D matrix, got shape={tuple(W.shape)}")
    out, inp = W.shape
    r_eff = int(max(0, min(r, out, inp)))
    if r_eff == 0:
        return torch.zeros_like(W)
    if r_eff == min(out, inp):
        return W
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return (U[:, :r_eff] * S[:r_eff].unsqueeze(0)) @ Vh[:r_eff, :]


def _svd_lowrank_auto_compat(
    W: torch.Tensor,
    r: int,
    *,
    oversample: int = 8,
    niter: int = 2,
    exact_threshold: int = 2048,
) -> torch.Tensor:
    """Torch-version-compatible low-rank reconstruction (SVD).

    Strategy:
      1) exact `torch.linalg.svd` for small matrices
      2) `torch.svd_lowrank` if available
      3) randomized SVD (Halko) fallback
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
        return W.clone()

    if min(m, n) <= int(exact_threshold):
        return _svd_lowrank_exact(W, r=k)

    # Prefer torch.svd_lowrank if present.
    if hasattr(torch, "svd_lowrank"):
        try:
            U, S, V = torch.svd_lowrank(W, q=k, niter=int(max(0, niter)))
            return (U * S.unsqueeze(0)) @ V.transpose(0, 1)
        except Exception:
            pass

    # Randomized SVD fallback.
    q = int(min(n, k + max(0, int(oversample))))
    Omega = torch.randn((n, q), device=W.device, dtype=W.dtype)
    Y = W @ Omega
    for _ in range(int(max(0, niter))):
        Y = W @ (W.transpose(0, 1) @ Y)
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.transpose(0, 1) @ W
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub[:, :k]
    return (U * S[:k].unsqueeze(0)) @ Vh[:k, :]


def _factor2(n: int) -> Tuple[int, int]:
    """Find (a,b) with a*b=n, a<=b, and a as close to sqrt(n) as possible."""
    n = int(n)
    if n <= 0:
        raise ValueError(f"dim must be positive, got {n}")
    r = int(n**0.5)
    for a in range(r, 0, -1):
        if n % a == 0:
            b = n // a
            return int(a), int(b)
    return 1, n


def _split_dim(n: int, hints: Dict[int, Tuple[int, int]]) -> Tuple[int, int]:
    if int(n) in hints:
        a, b = hints[int(n)]
        if int(a) * int(b) == int(n):
            return int(a), int(b)
    return _factor2(int(n))


def mpo_lowrank(
    W: torch.Tensor,
    r: int,
    *,
    out_factors: Optional[Tuple[int, int]] = None,
    in_factors: Optional[Tuple[int, int]] = None,
    svd_oversample: int = 8,
    svd_niter: int = 2,
    svd_exact_threshold: int = 2048,
) -> torch.Tensor:
    """2-core MPO approximation of a matrix.

    We reshape W[out,in] -> W[o0,o1,i0,i1], then group (o0,i0) and (o1,i1) as a 2D matrix
    and take a rank-r truncated SVD there. This corresponds to a bond-dimension-r MPO.
    """
    if W.ndim != 2:
        raise ValueError(f"MPO expects 2D matrix, got shape={tuple(W.shape)}")
    out, inp = map(int, W.shape)
    r = int(max(0, min(r, out, inp)))
    if r == 0:
        return torch.zeros_like(W)
    if r >= min(out, inp):
        return W.clone()

    o0, o1 = out_factors if out_factors is not None else _split_dim(out, _OUT_DIM_HINTS)
    i0, i1 = in_factors if in_factors is not None else _split_dim(inp, _IN_DIM_HINTS)
    if o0 * o1 != out or i0 * i1 != inp:
        raise ValueError(
            f"Bad MPO factors: out={out} -> ({o0},{o1}), in={inp} -> ({i0},{i1})"
        )

    # W[o0,o1,i0,i1] -> permute to [o0,i0,o1,i1] -> view as [o0*i0, o1*i1]
    W4 = W.contiguous().view(o0, o1, i0, i1)
    W2 = W4.permute(0, 2, 1, 3).contiguous().view(o0 * i0, o1 * i1)

    W2_hat = _svd_lowrank_auto_compat(
        W2,
        r=r,
        oversample=int(svd_oversample),
        niter=int(svd_niter),
        exact_threshold=int(svd_exact_threshold),
    )

    W4_hat = W2_hat.view(o0, i0, o1, i1).permute(0, 2, 1, 3).contiguous()
    return W4_hat.view(out, inp)


def _lowrank(
    W: torch.Tensor,
    r: int,
    method: str,
    *,
    svd_oversample: int = 8,
    svd_niter: int = 2,
    svd_exact_threshold: int = 2048,
) -> torch.Tensor:
    method = (method or "svd").lower().strip()
    if method == "svd":
        return _svd_lowrank_auto_compat(
            W,
            r=r,
            oversample=svd_oversample,
            niter=svd_niter,
            exact_threshold=svd_exact_threshold,
        )
    if method == "mpo":
        return mpo_lowrank(
            W,
            r=r,
            svd_oversample=svd_oversample,
            svd_niter=svd_niter,
            svd_exact_threshold=svd_exact_threshold,
        )
    raise ValueError(f"Unknown method: {method!r} (expected 'svd' or 'mpo')")


# ----------------------------
# Main public API (in-place)
# ----------------------------


@torch.no_grad()
def fake_op(model: nn.Module, svdq_config: Dict[str, Any]) -> nn.Module:
    """Apply fake (low-rank + quant residual) reconstruction to selected *_proj weights in-place."""
    if not isinstance(svdq_config, dict):
        raise TypeError("svdq_config must be a dict")

    svd_dict = svdq_config.get("svd", {}) or {}
    quant_dict = svdq_config.get("quant", {}) or {}
    method = _resolve_method(svdq_config)

    # Optional low-rank backend knobs
    lr_cfg = svdq_config.get("lowrank", None)
    if lr_cfg is None:
        lr_cfg = svdq_config.get("svd_cfg", None)
    if lr_cfg is None:
        lr_cfg = svdq_config.get("svd", {}) if isinstance(svdq_config.get("svd", {}), dict) else {}
    lr_cfg = lr_cfg or {}

    oversample = int(lr_cfg.get("oversample", 8))
    niter = int(lr_cfg.get("niter", 2))
    exact_threshold = int(lr_cfg.get("exact_threshold", 2048))

    targets = _iter_allowed_linears(model)

    for name, mod in targets.items():
        layer, _block, x = _name_to_layer_block_x(name)
        rank = _resolve_svd_rank(svd_dict, layer, x)
        qspec = _resolve_quant_spec(quant_dict, layer, x)

        if rank is None and qspec is None:
            continue

        W0 = mod.weight.data
        dtype0 = W0.dtype
        dev0 = W0.device
        W = W0.float()

        if rank is None and qspec is not None:
            fmt, blk = qspec
            W_hat = fake_quantize(W, fmt=fmt, blk=blk, eps=1e-6)
        elif rank is not None and qspec is None:
            W_hat = _lowrank(
                W,
                r=int(rank),
                method=method,
                svd_oversample=oversample,
                svd_niter=niter,
                svd_exact_threshold=exact_threshold,
            )
        else:
            fmt, blk = qspec  # type: ignore[misc]
            W_side = _lowrank(
                W,
                r=int(rank),
                method=method,
                svd_oversample=oversample,
                svd_niter=niter,
                svd_exact_threshold=exact_threshold,
            )
            R = W - W_side
            Rq = fake_quantize(R, fmt=fmt, blk=blk, eps=1e-6)
            W_hat = W_side + Rq

        mod.weight.data = W_hat.to(device=dev0, dtype=dtype0)

    return model


# ----------------------------
# Stable scanner (no mutation)
# ----------------------------


@dataclass(frozen=True)
class ScanScheme:
    """One evaluation scheme."""

    rank: Optional[int] = None
    quant_fmt: Optional[str] = None
    quant_blk: Optional[Union[int, str]] = None
    method: str = "svd"  # "svd" | "mpo"

    @property
    def mode(self) -> str:
        if self.rank is None and self.quant_fmt is None:
            return "noop"
        if self.rank is None and self.quant_fmt is not None:
            return "quant_only"
        if self.rank is not None and self.quant_fmt is None:
            return "lowrank_only"
        return "lowrank_plus_quant"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "method": self.method,
            "rank": self.rank,
            "quant": None
            if self.quant_fmt is None
            else {"fmt": self.quant_fmt, "blk": self.quant_blk},
        }


def _nmse(W_hat: torch.Tensor, W: torch.Tensor) -> float:
    """NMSE = ||W_hat - W||_2^2 / ||W||_2^2 (flattened)."""
    diff = (W_hat - W).float()
    num = torch.sum(diff * diff, dtype=torch.float64)
    den = torch.sum(W.float() * W.float(), dtype=torch.float64)
    if den.item() == 0.0:
        return 0.0
    return float((num / den).item())


def _reconstruct_weight_for_scheme(
    W: torch.Tensor,
    scheme: ScanScheme,
    *,
    eps: float = 1e-6,
    svd_oversample: int = 8,
    svd_niter: int = 2,
    svd_exact_threshold: int = 2048,
) -> torch.Tensor:
    """Return reconstructed weight (float32) without touching the model."""
    W = W.float()

    if scheme.mode == "noop":
        return W

    if scheme.mode == "quant_only":
        return fake_quantize(W, fmt=str(scheme.quant_fmt), blk=scheme.quant_blk, eps=eps)

    if scheme.mode == "lowrank_only":
        return _lowrank(
            W,
            r=int(scheme.rank or 0),
            method=str(scheme.method),
            svd_oversample=svd_oversample,
            svd_niter=svd_niter,
            svd_exact_threshold=svd_exact_threshold,
        )

    # lowrank + quant residual
    L = _lowrank(
        W,
        r=int(scheme.rank or 0),
        method=str(scheme.method),
        svd_oversample=svd_oversample,
        svd_niter=svd_niter,
        svd_exact_threshold=svd_exact_threshold,
    )
    R = W - L
    Rq = fake_quantize(R, fmt=str(scheme.quant_fmt), blk=scheme.quant_blk, eps=eps)
    return L + Rq


def _default_scan_schemes(method: str = "svd") -> List[ScanScheme]:
    """Default grid: moderate size, focuses on practical candidates."""
    ranks = [8, 16, 32, 64]
    quant_specs: List[Tuple[str, Union[int, str]]] = [
        ("fp4", "in"),
        ("fp4", "out"),
        ("fp4", 128),
        ("fp4_e2m1", "in"),
        ("nvfp4", "in"),
        ("mxfp4", "in"),
        ("mxfp6", "in"),
        ("fp8", "in"),
        ("fp8", 128),
    ]

    method = (method or "svd").lower().strip()
    schemes: List[ScanScheme] = []

    # quant-only
    for fmt, blk in quant_specs:
        schemes.append(ScanScheme(rank=None, quant_fmt=fmt, quant_blk=blk, method=method))

    # lowrank-only
    for r in ranks:
        schemes.append(ScanScheme(rank=int(r), quant_fmt=None, quant_blk=None, method=method))

    # lowrank + quant
    for r in ranks:
        for fmt, blk in quant_specs:
            schemes.append(ScanScheme(rank=int(r), quant_fmt=fmt, quant_blk=blk, method=method))

    return schemes


@torch.no_grad()
def stable_scanner(model: nn.Module, save_path: str, save_label: str) -> Dict[str, Any]:
    """Scan all target linears and evaluate NMSE under a grid of schemes.

    Environment controls:
      - FAKE_SVDQ_SCAN_DEVICE=cpu|cuda
      - FAKE_SVDQ_SCAN_METHOD=svd|mpo
    """
    t0 = time.time()
    Path(save_path).mkdir(parents=True, exist_ok=True)
    out_file = Path(save_path) / f"{save_label}.json"

    targets = _iter_allowed_linears(model)

    env_method = os.environ.get("FAKE_SVDQ_SCAN_METHOD", "").strip().lower() or "svd"
    schemes = _default_scan_schemes(method=env_method)

    # Choose computation device heuristic (can override by env var)
    env_dev = os.environ.get("FAKE_SVDQ_SCAN_DEVICE", "").strip().lower()

    results: Dict[str, Any] = {
        "meta": {
            "save_label": save_label,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "torch": torch.__version__,
            "num_targets": len(targets),
            "num_schemes": len(schemes),
            "scheme_preview": [s.as_dict() for s in schemes[: min(8, len(schemes))]],
            "scan_device_env": env_dev or None,
            "scan_method_env": env_method,
        },
        "targets": {},
    }

    names = sorted(targets.keys())
    for idx, name in enumerate(names):
        mod = targets[name]
        layer, block, x = _name_to_layer_block_x(name)
        W0 = mod.weight.detach()
        dtype0 = str(W0.dtype).replace("torch.", "")
        dev0 = str(W0.device)
        shape = list(W0.shape)

        if env_dev in ("cpu", "cuda"):
            compute_device = torch.device(env_dev if env_dev == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            compute_device = W0.device

        entry: Dict[str, Any] = {
            "layer": layer,
            "block": block,
            "x": x,
            "shape": shape,
            "dtype": dtype0,
            "device": dev0,
            "compute_device": str(compute_device),
            "nmse": [],
        }

        try:
            W = W0.to(device=compute_device, dtype=torch.float32, non_blocking=True)
            for s in schemes:
                try:
                    W_hat = _reconstruct_weight_for_scheme(W, s)
                    entry["nmse"].append({**s.as_dict(), "value": _nmse(W_hat, W)})
                except Exception as e:
                    entry["nmse"].append({**s.as_dict(), "error": f"{type(e).__name__}: {e}"})

            ok = [r for r in entry["nmse"] if "value" in r]
            ok_sorted = sorted(ok, key=lambda d: d["value"])
            entry["best"] = ok_sorted[: min(10, len(ok_sorted))]
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"

        results["targets"][name] = entry

        if (idx + 1) % 10 == 0 or (idx + 1) == len(names):
            print(f"[stable_scanner] {idx+1}/{len(names)} done: {name}")

        if torch.cuda.is_available() and compute_device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    results["meta"]["elapsed_sec"] = float(time.time() - t0)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[stable_scanner] saved: {out_file}")
    return results


# ----------------------------
# CLI demo
# ----------------------------


def _load_hf_model(model_path: str, device: str, dtype: str) -> nn.Module:
    try:
        from transformers import AutoModelForCausalLM
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for --model_path loading in __main__. "
            "Install transformers or import and call stable_scanner(model, ...) directly."
        ) from e

    torch_dtype = None
    d = dtype.lower().strip()
    if d in ("fp16", "float16"):
        torch_dtype = torch.float16
    elif d in ("bf16", "bfloat16"):
        torch_dtype = torch.bfloat16
    elif d in ("fp32", "float32"):
        torch_dtype = torch.float32

    m = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    m.eval()
    m.to(device)
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake SVDQ/MPO stable scanner (NMSE, no calibration, no forward)")
    parser.add_argument("--model_path", type=str, default="", help="HF checkpoint path (AutoModelForCausalLM)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16/fp16/fp32/auto")
    parser.add_argument("--save_path", type=str, default="./svdq_scan")
    parser.add_argument("--save_label", type=str, default="nmse_scan")
    args = parser.parse_args()

    if not args.model_path:
        raise SystemExit("Please provide --model_path, or import this file and call stable_scanner(model, ...) directly.")

    model = _load_hf_model(args.model_path, device=args.device, dtype=args.dtype)
    stable_scanner(model, save_path=args.save_path, save_label=args.save_label)
