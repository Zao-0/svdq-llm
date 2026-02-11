# data_gen.py
# -*- coding: utf-8 -*-
"""
Data preparation for SVDQuant / calibration & verification.

What this file does:
- CN dataset: load from ModelScope MsDataset, each sample has `messages` (list of {"role","content"}).
  Tokenization uses tokenizer.apply_chat_template(..., tokenize=True, add_generation_prompt=False, enable_thinking=True).
- EU dataset: load from HuggingFace datasets `SinclairSchneider/politico_eu`, tokenize raw text with tokenizer(text).
- Calibration: NO truncation (use full sequences).
- Verification: truncation only (CN default 1024, EU default 512).
- Return: dict {"cn": List[torch.Tensor], "eu": List[torch.Tensor]}, each tensor is 1D LongTensor on `device`.

Notes:
- `device` is a str like "cpu" or "cuda:0". We move tensors onto device right after tokenization.
- For verification, we truncate BEFORE moving (less transfer); calibration has no truncation by design.

Dependencies:
- pip install modelscope datasets
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

# -----------------------------
# Dataset identifiers
# -----------------------------
CN_DATASET_ID = "swift/Chinese-Qwen3-235B-Thinking-2507-Distill-data-110k-SFT"
EU_DATASET_ID = "SinclairSchneider/politico_eu"

# Global cache to avoid repeated downloads in one process
_CN_DS_CACHE = None
_EU_DS_CACHE = None


# -----------------------------
# Utilities
# -----------------------------
def _as_torch_device(device: str) -> torch.device:
    try:
        return torch.device(device)
    except Exception as e:
        raise ValueError(f"Invalid device string: {device!r}") from e


def _rng(seed: int, salt: int) -> np.random.RandomState:
    """Deterministic RNG with per-purpose salt to reduce unintended overlap."""
    return np.random.RandomState((seed ^ salt) & 0xFFFFFFFF)


def _ensure_1d_input_ids(enc: Any) -> torch.Tensor:
    """
    Normalize tokenizer outputs to a 1D torch.LongTensor [seq_len].
    Supports:
      - BatchEncoding-like dict with "input_ids": [1, L] or [L]
      - torch.Tensor [1, L] or [L]
      - list[int] (rare)
    """
    ids = enc["input_ids"]

    if isinstance(ids, list):
        ids = torch.tensor(ids, dtype=torch.long)
    if not isinstance(ids, torch.Tensor):
        raise TypeError(f"Unsupported input_ids type: {type(ids)}")
    
    ids = ids.to(dtype=torch.long)

    if ids.dim() == 2:
        # assume [B, L], take first sample
        ids = ids[0]
    elif ids.dim() != 1:
        raise ValueError(f"input_ids must be 1D or 2D, got shape={tuple(ids.shape)}")

    return ids.contiguous()


def _truncate_1d(ids: torch.Tensor, max_len: Optional[int]) -> torch.Tensor:
    if max_len is None:
        return ids
    if ids.numel() <= max_len:
        return ids
    return ids[:max_len].contiguous()


def _pick_first_available_split(ds_dict: Any) -> Any:
    """Pick a reasonable default split from a HF DatasetDict."""
    for name in ("train", "validation", "test"):
        if name in ds_dict:
            return ds_dict[name]
    keys = list(ds_dict.keys())
    if not keys:
        raise ValueError("Empty DatasetDict.")
    return ds_dict[keys[0]]


# -----------------------------
# Load datasets
# -----------------------------
def _load_cn_dataset():
    global _CN_DS_CACHE
    if _CN_DS_CACHE is not None:
        return _CN_DS_CACHE

    try:
        from modelscope.msdatasets import MsDataset  # type: ignore
    except Exception as e:
        raise ImportError(
            "Failed to import modelscope MsDataset. Please install modelscope.\n"
            "Example: pip install modelscope"
        ) from e

    ds = MsDataset.load(CN_DATASET_ID)
    _CN_DS_CACHE = ds
    return ds


def _load_eu_dataset():
    global _EU_DS_CACHE
    if _EU_DS_CACHE is not None:
        return _EU_DS_CACHE

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise ImportError(
            "Failed to import HuggingFace datasets. Please install datasets.\n"
            "Example: pip install datasets"
        ) from e

    ds_dict = load_dataset(EU_DATASET_ID)
    ds = _pick_first_available_split(ds_dict)
    _EU_DS_CACHE = ds
    return ds


# -----------------------------
# CN tokenization
# -----------------------------
def _apply_chat_template_safely(tok, messages: Sequence[Dict[str, str]]) -> torch.Tensor:
    """
    Call tok.apply_chat_template with best-effort support for:
      tokenize=True, add_generation_prompt=False, enable_thinking=True, return_tensors="pt", return_dict=True

    Different transformers/tokenizer versions may expose slightly different signatures.
    """
    fn = getattr(tok, "apply_chat_template", None)
    if fn is None:
        raise AttributeError("Tokenizer has no apply_chat_template().")

    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}

    # Required by your spec (when supported)
    if "tokenize" in sig.parameters:
        kwargs["tokenize"] = True
    if "add_generation_prompt" in sig.parameters:
        kwargs["add_generation_prompt"] = False
    if "return_tensors" in sig.parameters:
        kwargs["return_tensors"] = "pt"

    # Strongly recommended to get enc["input_ids"] consistently (when supported)
    if "return_dict" in sig.parameters:
        kwargs["return_dict"] = True

    # Your requested thinking flag (when supported)
    if "enable_thinking" in sig.parameters:
        kwargs["enable_thinking"] = True

    enc = fn(messages, **kwargs)
    # Some versions return tensor directly even with return_tensors
    ids = _ensure_1d_input_ids(enc)
    return ids


def _cn_sample_to_ids(tok, sample: Dict[str, Any]) -> torch.Tensor:
    if "messages" not in sample:
        raise KeyError("CN sample does not contain 'messages' field.")
    messages = sample["messages"]
    if not isinstance(messages, (list, tuple)):
        raise TypeError(f"'messages' must be list/tuple, got {type(messages)}")
    return _apply_chat_template_safely(tok, messages)


# -----------------------------
# EU text extraction & tokenization
# -----------------------------
_TEXT_KEY_CANDIDATES = (
    "text",
    "content",
    "article",
    "body",
    "document",
    "raw",
    "maintext",
    "sentence",
    "sentences",
    "paragraph",
    "paragraphs",
    "description",
    "summary",
)


def _extract_text_from_example(ex: Dict[str, Any]) -> Optional[str]:
    """
    Best-effort extraction of a text field from politico_eu examples.
    Returns None if no usable text found.
    """
    for k in _TEXT_KEY_CANDIDATES:
        if k in ex:
            v = ex[k]
            if isinstance(v, str) and v.strip():
                return v
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                joined = "\n".join([x for x in v if x.strip()])
                return joined if joined.strip() else None

    # Fallback: first string-like field
    for _, v in ex.items():
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            joined = "\n".join([x for x in v if x.strip()])
            return joined if joined.strip() else None
    return None


def _eu_text_to_ids(tok, text: str) -> torch.Tensor:
    enc = tok(text, return_tensors="pt")
    return _ensure_1d_input_ids(enc)


# -----------------------------
# Sampling helpers
# -----------------------------
def _sample_indices(n_total: int, n_need: int, rng: np.random.RandomState) -> np.ndarray:
    if n_need <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n_total < n_need:
        raise ValueError(f"Dataset too small: total={n_total} < need={n_need}")
    return rng.choice(n_total, size=n_need, replace=False).astype(np.int64)


def _collect_cn_ids(
    tok,
    n: int,
    seed: int,
    *,
    truncate_len: Optional[int],
    device: str,
) -> List[torch.Tensor]:
    ds = _load_cn_dataset()['train']['messages']
    r = _rng(seed, salt=0xC0C0_C0C0)
    dev = _as_torch_device(device)

    idxs = _sample_indices(len(ds), n, r)
    out: List[torch.Tensor] = []
    for i in idxs:
        sample = ds[int(i)]
        ids = _apply_chat_template_safely(tok, sample)
        # verification truncation happens here (before move, less transfer)
        ids = _truncate_1d(ids, truncate_len)
        ids = ids.to(dev, non_blocking=True)
        out.append(ids)
    return out


def _collect_eu_ids(
    tok,
    n: int,
    seed: int,
    *,
    truncate_len: Optional[int],
    device: str,
) -> List[torch.Tensor]:
    ds = _load_eu_dataset()
    r = _rng(seed, salt=0xE0E0_E0E0)
    dev = _as_torch_device(device)

    # We may need to skip samples without usable text, so we sample indices progressively.
    out: List[torch.Tensor] = []
    tried = 0
    max_tries = max(n * 20, 5000)  # enough cushion

    while len(out) < n and tried < max_tries:
        i = int(r.randint(0, len(ds)))
        tried += 1
        ex = ds[i]
        text = _extract_text_from_example(ex)
        if not text:
            continue
        ids = _eu_text_to_ids(tok, text)
        ids = _truncate_1d(ids, truncate_len)
        ids = ids.to(dev, non_blocking=True)
        out.append(ids)

    if len(out) < n:
        raise RuntimeError(
            f"Failed to collect enough EU samples: got={len(out)} need={n}. "
            f"Tried={tried}, dataset_len={len(ds)}. "
            "Maybe the dataset fields changed; please inspect one example and adjust _extract_text_from_example()."
        )

    return out


# -----------------------------
# Public APIs (updated with device: str)
# -----------------------------
def get_cali_data(
    tok,
    seed: int = 0,
    cn: int = 100,
    eu: int = 100,
    device: str = "cpu",
) -> Dict[str, List[torch.Tensor]]:
    """
    Calibration data:
      - NO truncation
      - return {"cn": [...], "eu": [...]}, tensors are on `device`
    """
    cn_ids = _collect_cn_ids(tok, cn, seed=seed, truncate_len=None, device=device)
    eu_ids = _collect_eu_ids(tok, eu, seed=seed, truncate_len=None, device=device)
    return {"cn": cn_ids, "eu": eu_ids}


def get_veri_data(
    tok,
    seed: int = 0,
    cn: int = 1000,
    cn_len: int = 1024,
    eu: int = 1600,
    eu_len: int = 512,
    device: str = "cpu",
) -> Dict[str, List[torch.Tensor]]:
    """
    Verification data:
      - truncation enabled (CN->cn_len, EU->eu_len)
      - return {"cn": [...], "eu": [...]}, tensors are on `device`
    """
    cn_ids = _collect_cn_ids(tok, cn, seed=seed, truncate_len=cn_len, device=device)
    eu_ids = _collect_eu_ids(tok, eu, seed=seed, truncate_len=eu_len, device=device)
    return {"cn": cn_ids, "eu": eu_ids}


# -----------------------------
# Quick test (runnable)
# -----------------------------
if __name__ == "__main__":
    import argparse

    from transformers import AutoTokenizer  # type: ignore

    ap = argparse.ArgumentParser(description="Quick test for data_gen.py")
    ap.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Tokenizer name or local path for transformers.AutoTokenizer.from_pretrained().",
    )
    ap.add_argument("--device", type=str, default="cpu", help='e.g. "cpu" or "cuda:0"')
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--cali_cn", type=int, default=2)
    ap.add_argument("--cali_eu", type=int, default=2)

    ap.add_argument("--veri_cn", type=int, default=2)
    ap.add_argument("--veri_eu", type=int, default=2)
    ap.add_argument("--cn_len", type=int, default=1024)
    ap.add_argument("--eu_len", type=int, default=512)

    args = ap.parse_args()

    dev = _as_torch_device(args.device)
    print(f"[QuickTest] tokenizer_path={args.tokenizer_path}")
    print(f"[QuickTest] device={dev} (cuda_available={torch.cuda.is_available()})")

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # Calibration (no truncation)
    cali = get_cali_data(
        tok,
        seed=args.seed,
        cn=args.cali_cn,
        eu=args.cali_eu,
        device=args.device,
    )
    print("\n[Calibration]")
    for k in ("cn", "eu"):
        lens = [int(x.numel()) for x in cali[k]]
        print(
            f"  {k}: n={len(lens)}  min={min(lens) if lens else 0}  "
            f"max={max(lens) if lens else 0}  mean={sum(lens)/max(len(lens),1):.1f}  "
            f"device={cali[k][0].device if lens else dev}"
        )

    # Verification (truncation)
    veri = get_veri_data(
        tok,
        seed=args.seed,
        cn=args.veri_cn,
        cn_len=args.cn_len,
        eu=args.veri_eu,
        eu_len=args.eu_len,
        device=args.device,
    )
    print("\n[Verification]")
    for k in ("cn", "eu"):
        lens = [int(x.numel()) for x in veri[k]]
        print(
            f"  {k}: n={len(lens)}  min={min(lens) if lens else 0}  "
            f"max={max(lens) if lens else 0}  mean={sum(lens)/max(len(lens),1):.1f}  "
            f"device={veri[k][0].device if lens else dev}"
        )

    # Show a tiny peek
    if veri["cn"]:
        print("\n[Peek] veri['cn'][0][:16] =", veri["cn"][0][:16].tolist())
    if veri["eu"]:
        print("[Peek] veri['eu'][0][:16] =", veri["eu"][0][:16].tolist())

    print("\nDone.")
