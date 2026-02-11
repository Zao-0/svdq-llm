# fake_svdq_v2.py
# -*- coding: utf-8 -*-
"""fake_svdq_v2.py

Fake SVD-Quant (and MPO placeholder) for *_proj Linear weights.

This file extends the original `fake_svdq.py` with an **(no-calibration, no-forward)**
weight reconstruction error scanner:

    stable_scanner(model, save_path, save_label)

It enumerates all legal target linears:

    model.layers.{i}.{mlp|self_attn}.{*_proj}.weight

and evaluates the **normalized mean-squared error (NMSE)** between reconstructed
weights \tilde{W} and original W under a grid of (rank, quant_fmt, quant_blk)
schemes **without mutating** the input model.

Notes
-----
* `fake_op()` remains an in-place operator (same as v1).
* `stable_scanner()` never writes back to `model`.
* For practical scalability on large matrices, `stable_scanner()` uses randomized
  truncated SVD (`torch.linalg.svd_lowrank`) by default; it will fall back to exact
  SVD for small matrices.
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
    if fmt in ("mxfp6","fp6"):
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
# SVD / MPO (placeholder)
# ----------------------------
def svd_lowrank(W: torch.Tensor, r: int) -> torch.Tensor:
    """Exact truncated SVD low-rank approximation (float32)."""
    out, inp = W.shape
    r_eff = int(max(0, min(r, out, inp)))
    if r_eff == 0:
        return torch.zeros_like(W)
    if r_eff == min(out, inp):
        return W
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :r_eff]
    S_r = S[:r_eff]
    Vh_r = Vh[:r_eff, :]
    return (U_r * S_r.unsqueeze(0)) @ Vh_r


def mpo_lowrank_not_implemented(*args, **kwargs):
    raise NotImplementedError("MPO decomposition is TODO (not implemented in fake_svdq_v2.py).")


def _svd_lowrank_auto(
    W: torch.Tensor,
    r: int,
    *,
    oversample: int = 8,
    niter: int = 2,
    exact_threshold: int = 2048,
) -> torch.Tensor:
    """Truncated low-rank approximation used by `stable_scanner()`.

    - For small matrices (min_dim <= exact_threshold): exact SVD.
    - Otherwise: randomized SVD (torch.linalg.svd_lowrank) with q=r+oversample.
    """
    out, inp = W.shape
    r_eff = int(max(0, min(r, out, inp)))
    if r_eff == 0:
        return torch.zeros_like(W)
    if r_eff == min(out, inp):
        return W

    k = min(out, inp)
    if k <= exact_threshold:
        return svd_lowrank(W, r=r_eff)

    # randomized
    q = int(min(k, r_eff + max(0, int(oversample))))
    # torch.linalg.svd_lowrank returns U:[out,q], S:[q], V:[inp,q]
    U, S, V = torch.linalg.svd_lowrank(W, q=q, niter=int(max(0, niter)))
    U_r = U[:, :r_eff]
    S_r = S[:r_eff]
    V_r = V[:, :r_eff]
    return (U_r * S_r.unsqueeze(0)) @ V_r.transpose(0, 1)


# ----------------------------
# Main public API (in-place)
# ----------------------------
@torch.no_grad()
def fake_op(model: nn.Module, svdq_config: Dict[str, Any]) -> nn.Module:
    """Apply fake SVD+quant reconstruction to selected *_proj weights in-place."""
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
            W_hat = svd_lowrank(W, r=int(rank))
        else:
            fmt, blk = qspec  # type: ignore[misc]
            L = svd_lowrank(W, r=int(rank))  # type: ignore[arg-type]
            R = W - L
            Rq = fake_quantize(R, fmt=fmt, blk=blk, eps=1e-6)
            W_hat = L + Rq

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

    @property
    def mode(self) -> str:
        if self.rank is None and self.quant_fmt is None:
            return "noop"
        if self.rank is None and self.quant_fmt is not None:
            return "quant_only"
        if self.rank is not None and self.quant_fmt is None:
            return "svd_only"
        return "svd_plus_quant"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
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

    if scheme.mode == "svd_only":
        return _svd_lowrank_auto(
            W,
            r=int(scheme.rank or 0),
            oversample=svd_oversample,
            niter=svd_niter,
            exact_threshold=svd_exact_threshold,
        )

    # svd + quant residual
    L = _svd_lowrank_auto(
        W,
        r=int(scheme.rank or 0),
        oversample=svd_oversample,
        niter=svd_niter,
        exact_threshold=svd_exact_threshold,
    )
    R = W - L
    Rq = fake_quantize(R, fmt=str(scheme.quant_fmt), blk=scheme.quant_blk, eps=eps)
    return L + Rq


def _default_scan_schemes() -> List[ScanScheme]:
    """Default grid: moderate size, focuses on practical candidates.

    You can override these by editing this function locally.
    """
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

    schemes: List[ScanScheme] = []

    # quant-only
    for fmt, blk in quant_specs:
        schemes.append(ScanScheme(rank=None, quant_fmt=fmt, quant_blk=blk))

    # svd-only
    for r in ranks:
        schemes.append(ScanScheme(rank=int(r), quant_fmt=None, quant_blk=None))

    # svd + quant
    for r in ranks:
        for fmt, blk in quant_specs:
            schemes.append(ScanScheme(rank=int(r), quant_fmt=fmt, quant_blk=blk))

    return schemes


@torch.no_grad()
def stable_scanner(model: nn.Module, save_path: str, save_label: str) -> Dict[str, Any]:
    """Scan all target linears and evaluate NMSE under a grid of schemes.

    Requirements from user:
    - find all legal linears to fake-quantize
    - compare errors before/after under different quant schemes
    - DO NOT register modifications back to the original model
    - save to {save_path}/{save_label}.json
    """
    t0 = time.time()
    Path(save_path).mkdir(parents=True, exist_ok=True)
    out_file = Path(save_path) / f"{save_label}.json"

    targets = _iter_allowed_linears(model)
    schemes = _default_scan_schemes()

    # Choose computation device heuristic (can override by env var)
    #   FAKE_SVDQ_SCAN_DEVICE=cpu|cuda
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
        },
        "targets": {},
    }

    # Iterate
    names = sorted(targets.keys())
    for idx, name in enumerate(names):
        mod = targets[name]
        layer, block, x = _name_to_layer_block_x(name)
        W0 = mod.weight.detach()
        dtype0 = str(W0.dtype).replace("torch.", "")
        dev0 = str(W0.device)
        shape = list(W0.shape)

        # Decide compute device
        if env_dev in ("cpu", "cuda"):
            compute_device = torch.device(env_dev if env_dev == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            # default: keep on original device
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
            # also store noop baseline (nmse=0)
            for s in schemes:
                try:
                    W_hat = _reconstruct_weight_for_scheme(W, s)
                    entry["nmse"].append({
                        **s.as_dict(),
                        "value": _nmse(W_hat, W),
                    })
                except Exception as e:
                    entry["nmse"].append({
                        **s.as_dict(),
                        "error": f"{type(e).__name__}: {e}",
                    })

            # a small summary: best configs (by nmse)
            ok = [r for r in entry["nmse"] if "value" in r]
            ok_sorted = sorted(ok, key=lambda d: d["value"])
            entry["best"] = ok_sorted[: min(10, len(ok_sorted))]

        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"

        results["targets"][name] = entry

        if (idx + 1) % 10 == 0 or (idx + 1) == len(names):
            print(f"[stable_scanner] {idx+1}/{len(names)} done: {name}")

        # free per-layer intermediates (best-effort)
        if torch.cuda.is_available() and compute_device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    results["meta"]["elapsed_sec"] = float(time.time() - t0)

    # write json
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
    parser = argparse.ArgumentParser(description="Fake SVDQ stable scanner (NMSE, no calibration, no forward)")
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
