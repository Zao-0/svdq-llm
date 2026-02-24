#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Config-driven evaluator (multiple-choice + strict letter-order).

Supported datasets:
  - mmlu            (cais/mmlu)
  - mmlu-pro        (TIGER-Lab/MMLU-Pro)
  - gpqa-main       (Idavidrein/gpqa, config=gpqa_main)
  - gpqa-extended   (Idavidrein/gpqa, config=gpqa_extended)
  - gpqa-diamond    (Idavidrein/gpqa, config=gpqa_diamond)
  - arc-easy        (allenai/ai2_arc, subset=ARC-Easy)
  - arc-challenge   (allenai/ai2_arc, subset=ARC-Challenge)
  - casehold        (casehold/casehold)
  - econlogicqa     (yinzhu-quan/econ_logic_qa)

Key logging behaviors:
  - Always save model generation with thinking removed: `gen_no_think`
  - If a <think> tag appears but no closing </think> is found:
      - gen_no_think = "### Answer Extracted Failure ###"
      - pred_raw     = "### Answer Extracted Failure ###"
      - pred_norm    = None
  - For multiple-choice (mmlu/mmlu-pro/gpqa/casehold):
      - If an option letter can be extracted (highest priority), we score by strict letter match.
      - If no valid option letter can be extracted, we keep the cleaned generation in `pred_raw`,
        set `pred_norm=None`, and score as incorrect (acc=0).
  - For EconLogicQA:
      - Target is a permutation of A,B,C,D. We extract and normalize as "A,B,C,D" (no spaces).
      - If extraction fails, we keep generation in `pred_raw`, set `pred_norm=None`.

Run:
  python multi-choice-eval.py --config /path/to/eval.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
os.environ['HF_HOME'] = "/workspace/hf_cache"

# ==============================
#  Model dirs (tags)
# ==============================

DIR_DICT = {
    "base":"../models/Qwen3-8B/",
    "fp6":"../models/Qwen3-8B-fp6/",
    "nvfp4":"../models/Qwen3-8B-nvfp4/",
    "46agg":"../models/Qwen3-8B-nvfp4-fp6-aggressive/",
    "46con":"../models/Qwen3-8B-nvfp4-fp6-conservative/",
    "rk64nvfp4":"../models/Qwen3-8B-rk64-nvfp4/"
}


# ==============================
#  Config structures
# ==============================

@dataclass
class ModelConfig:
    tag: str
    device: str = "auto"            # transformers device_map
    torch_dtype: str = "bfloat16"   # "bfloat16" | "float16" | "float32"
    trust_remote_code: bool = True
    tokenizer_padding_side: str = "left"  # "left" recommended for decoder-only


@dataclass
class DatasetConfig:
    name: str
    split: str = "test"
    limit: Optional[float] = None
    seed: int = 42


@dataclass
class GenConfig:
    batch_size: int = 1
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_k: int = 0


@dataclass
class RunConfig:
    model: ModelConfig
    datasets: List[DatasetConfig]
    gen: GenConfig = dataclasses.field(default_factory=GenConfig)
    output_dir: str = "./outputs"
    thinking: bool = True
    max_input_length: Optional[int] = None


def _load_yaml_or_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required for .yaml config files. Install with: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _as_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def parse_config(path: str) -> RunConfig:
    raw = _load_yaml_or_json(path)

    model_raw = raw.get("model", {})
    model = ModelConfig(
        tag=str(model_raw.get("tag", "base")),
        device=str(model_raw.get("device", "auto")),
        torch_dtype=str(model_raw.get("torch_dtype", "bfloat16")),
        trust_remote_code=_as_bool(model_raw.get("trust_remote_code", True), True),
        tokenizer_padding_side=str(model_raw.get("tokenizer_padding_side", "left")),
    )

    gen_raw = raw.get("gen", {})
    gen = GenConfig(
        batch_size=int(gen_raw.get("batch_size", 1)),
        max_new_tokens=int(gen_raw.get("max_new_tokens", 1024)),
        temperature=float(gen_raw.get("temperature", 0.0)),
        top_k=int(gen_raw.get("top_k", 0)),
    )

    datasets_raw = raw.get("datasets")
    if not datasets_raw:
        one = raw.get("dataset")
        if one:
            datasets_raw = [one]
    if not datasets_raw:
        raise ValueError("Config must contain `datasets:` (list) or `dataset:` (single).")

    datasets: List[DatasetConfig] = []
    for d in datasets_raw:
        d = d or {}
        datasets.append(
            DatasetConfig(
                name=str(d.get("name")),
                split=str(d.get("split", "test")),
                limit=d.get("limit", None),
                seed=int(d.get("seed", raw.get("seed", 42))),
            )
        )

    output_dir = str(raw.get("output_dir", raw.get("output", "./outputs")))
    thinking = _as_bool(raw.get("thinking", True), True)
    max_input_length = raw.get("max_input_length", None)
    if max_input_length is not None:
        max_input_length = int(max_input_length)

    return RunConfig(
        model=model,
        datasets=datasets,
        gen=gen,
        output_dir=output_dir,
        thinking=thinking,
        max_input_length=max_input_length,
    )


# ==============================
#  Prompt templates
# ==============================

MC_SYSTEM_PROMPT = (
    "You are an expert multiple-choice question answering assistant.\n"
    "You MUST answer ONLY with:\n\n"
    "###Answer: X\n\n"
    "where X is a single capital letter (A, B, C, ...).\n"
    "Do not output anything else after that line."
)

MC_FEW_SHOT_BLOCK = r"""Below are examples of the required format.

###Question: What is 2 + 2?

###Options:
A. 3
B. 4
C. 5
D. 6

###Answer: B

###Question: The capital of France is?

###Options:
A. London
B. Berlin
C. Paris
D. Madrid

###Answer: C

Now answer the next question in the same format.
"""

ORDER_SYSTEM_PROMPT = (
    "You are an expert reasoning assistant.\n"
    "The task is to output the correct logical order of statements.\n"
    "You MUST answer ONLY with:\n\n"
    "###Answer: A,B,C,D\n\n"
    "where the answer is a permutation of A,B,C,D separated by commas.\n"
    "Do not output anything else after that line."
)

ORDER_FEW_SHOT_BLOCK = r"""Below are examples of the required format.

###Question: Arrange the statements in the correct order.

###Statements:
A. Wake up.
B. Brush teeth.
C. Eat breakfast.
D. Leave home.

###Answer: A,B,C,D

###Question: Arrange the steps to bake a cake.

###Statements:
A. Bake in the oven.
B. Mix ingredients.
C. Serve.
D. Preheat the oven.

###Answer: D,B,A,C

Now answer the next question in the same format.
"""


def system_block_for_task(task_type: str) -> str:
    if task_type == "mc":
        return MC_SYSTEM_PROMPT + "\n\n" + MC_FEW_SHOT_BLOCK
    if task_type == "order":
        return ORDER_SYSTEM_PROMPT + "\n\n" + ORDER_FEW_SHOT_BLOCK
    raise ValueError(f"Unknown task_type: {task_type}")


# ==============================
#  Thinking removal + answer extraction
# ==============================

FAIL_THINK = "### Answer Extracted Failure ###"

THINK_OPEN_TAG = re.compile(r"<think\b[^>]*>", flags=re.IGNORECASE)
THINK_CLOSE_TAG = re.compile(r"</think\s*>", flags=re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", flags=re.DOTALL | re.IGNORECASE)

# Accept both "###Answer:" and "### Answer:"
ANSWER_LINE_RE = re.compile(r"^#{3}\s*Answer\s*:\s*(.*)$", flags=re.IGNORECASE | re.MULTILINE)
ALT_ANSWER_LINE_RE = re.compile(r"^\s*Answer\s*:\s*(.*)$", flags=re.IGNORECASE | re.MULTILINE)


def strip_think_blocks_or_fail(text: str) -> Tuple[str, bool]:
    """Remove <think>...</think> blocks.

    Returns (cleaned_text, think_failed).

    If we encounter a '<think' without a later '</think>', we mark it as failure.
    """
    s = text or ""
    out_parts: List[str] = []
    pos = 0
    while True:
        m_open = THINK_OPEN_TAG.search(s, pos)
        if not m_open:
            out_parts.append(s[pos:])
            break
        m_close = THINK_CLOSE_TAG.search(s, m_open.end())
        if not m_close:
            return FAIL_THINK, True
        out_parts.append(s[pos : m_open.start()])
        pos = m_close.end()

    cleaned = "".join(out_parts)
    cleaned = THINK_BLOCK_RE.sub("", cleaned)
    return cleaned.strip(), False


def _first_nonempty_line(s: str) -> str:
    for line in (s or "").splitlines():
        t = line.strip()
        if t:
            return t
    return ""


def extract_answer_payload(cleaned_text: str) -> Optional[str]:
    """Try to extract the payload after an 'Answer:' line. Returns the payload string (single line)."""
    m = ANSWER_LINE_RE.search(cleaned_text or "")
    if not m:
        m = ALT_ANSWER_LINE_RE.search(cleaned_text or "")
    if not m:
        # if the whole output is short, maybe it's just the answer
        short = cleaned_text.strip()
        if 0 < len(short) <= 10:
            return _first_nonempty_line(short)
        return None
    payload = (m.group(1) or "").strip()
    payload = payload.splitlines()[0].strip()
    return payload or None


def extract_mc_letter(cleaned_text: str, valid_letters: str) -> Optional[str]:
    """Extract a single option letter with highest priority given to an Answer line."""
    valid = set(valid_letters)

    payload = extract_answer_payload(cleaned_text)
    if payload:
        m = re.search(r"\b([A-Z])\b", payload.upper())
        if m and m.group(1) in valid:
            return m.group(1)
        # sometimes "(C)" or "C." etc.
        m2 = re.search(r"([A-Z])", payload.upper())
        if m2 and m2.group(1) in valid:
            return m2.group(1)

    # Fallback: search near the end for a standalone valid letter token
    tail = (cleaned_text or "")[-600:]
    tokens = list(re.finditer(r"\b([A-Z])\b", tail.upper()))
    for m in reversed(tokens):
        c = m.group(1)
        if c in valid:
            return c

    # Fallback 2: any letter char (less strict)
    tokens2 = list(re.finditer(r"([A-Z])", tail.upper()))
    for m in reversed(tokens2):
        c = m.group(1)
        if c in valid:
            return c

    return None


def normalize_order_answer(s: str) -> Optional[str]:
    """Normalize a permutation of A-D into 'A,B,C,D'. Return None if invalid."""
    if not s:
        return None
    # Extract letters A-D in order of appearance
    letters = re.findall(r"[ABCD]", s.upper())
    if len(letters) < 4:
        return None
    # Keep first 4 (common when model repeats)
    letters = letters[:4]
    if len(set(letters)) != 4:
        return None
    return ",".join(letters)


def extract_order_abcd(cleaned_text: str) -> Optional[str]:
    payload = extract_answer_payload(cleaned_text)
    if payload:
        norm = normalize_order_answer(payload)
        if norm:
            return norm

    # Fallback: find last occurrence of 4 A-D letters in the whole text
    # e.g. "A, B, C, D" or "D B A C"
    matches = list(re.finditer(r"([ABCD])[^ABCD]*([ABCD])[^ABCD]*([ABCD])[^ABCD]*([ABCD])", cleaned_text.upper()))
    for m in reversed(matches):
        cand = ",".join([m.group(1), m.group(2), m.group(3), m.group(4)])
        if len(set(cand.split(","))) == 4:
            return cand
    return None


# ==============================
#  Dataset loading & formatting
# ==============================

GPQA_CONFIG_MAP = {
    "gpqa-main": "gpqa_main",
    "gpqa-extended": "gpqa_extended",
    "gpqa-diamond": "gpqa_diamond",
}


def _arc_unpack_question(q: Any) -> str:
    """ARC's `question` is usually a dict like {stem: ...}, but we accept a plain string too."""
    if isinstance(q, dict):
        for k in ("stem", "question", "text"):
            if k in q and q[k] is not None:
                return str(q[k]).strip()
        return str(q).strip()
    return str(q).strip()


def _arc_unpack_choices(choices: Any) -> Tuple[List[str], List[str]]:
    """Return (labels, texts) from ARC's `choices`.

    HF commonly provides:
      - choices = {"text": [...], "label": [...]} OR
      - choices = [{"text": ..., "label": ...}, ...]
    """
    if isinstance(choices, dict):
        labels = [str(x).strip() for x in (choices.get("label") or [])]
        texts = [str(x).strip() for x in (choices.get("text") or [])]
        return labels, texts
    if isinstance(choices, list):
        labels: List[str] = []
        texts: List[str] = []
        for item in choices:
            if isinstance(item, dict):
                labels.append(str(item.get("label", "")).strip())
                texts.append(str(item.get("text", "")).strip())
            else:
                texts.append(str(item).strip())
                labels.append("")
        return labels, texts
    return [], []


def _arc_answer_index(answer_key: str, labels: List[str], n: int) -> int:
    """Infer the correct choice index from ARC's answerKey and labels.

    ARC answerKey might be a letter (A/B/...) or a digit string ("1"/"2"/...)
    depending on the source/version. We try both conventions.
    """
    ak = (answer_key or "").strip()
    if not ak:
        raise ValueError("Empty answerKey")

    # 1) Exact / case-insensitive match within provided labels
    if ak in labels:
        return labels.index(ak)
    ak_u = ak.upper()
    labels_u = [x.upper() for x in labels]
    if ak_u in labels_u:
        return labels_u.index(ak_u)

    # 2) Numeric convention: "1" -> first option
    if ak.isdigit():
        idx = int(ak) - 1
        if 0 <= idx < n:
            return idx

    # 3) Letter convention: "A" -> first option
    if len(ak_u) == 1 and "A" <= ak_u <= "Z":
        idx = ord(ak_u) - ord("A")
        if 0 <= idx < n:
            return idx

    raise ValueError(f"Cannot map answerKey={answer_key!r} to an index (labels={labels!r}, n={n})")


def build_mc_question_block(question: str, options: List[str]) -> str:
    lines: List[str] = []
    lines.append(f"###Question: {question.strip()}")
    lines.append("")
    lines.append("###Options:")
    letters = [chr(ord("A") + i) for i in range(len(options))]
    for letter, opt in zip(letters, options):
        lines.append(f"{letter}. {str(opt).strip()}")
    lines.append("")
    return "\n".join(lines)


def build_order_question_block(question: str, statements: Dict[str, str]) -> str:
    lines: List[str] = []
    lines.append(f"###Question: {question.strip()}")
    lines.append("")
    lines.append("###Statements:")
    for k in ["A", "B", "C", "D"]:
        lines.append(f"{k}. {statements[k].strip()}")
    lines.append("")
    return "\n".join(lines)


def apply_limit(examples: List[Dict[str, Any]], limit: Optional[float]) -> List[Dict[str, Any]]:
    if limit is None:
        return examples
    n = len(examples)
    if limit <= 0:
        return []
    if limit < 1:
        k = int(n * limit)
        k = max(1, min(k, n))
        return examples[:k]
    k = int(limit)
    k = min(k, n)
    return examples[:k]


def _canonical_split(name: str, split: str) -> str:
    s = (split or "").strip().lower()
    if s in {"val", "valid"}:
        return "validation"
    return split


def load_and_build_examples(ds_cfg: DatasetConfig) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (task_type, examples).

    For task_type == 'mc':
      example includes: gold_raw, gold_norm (letter), meta.valid_letters

    For task_type == 'order':
      example includes: gold_raw, gold_norm (A,B,C,D), meta.valid_letters="ABCD"
    """
    name = ds_cfg.name.lower()
    random.seed(ds_cfg.seed)

    # -------- GPQA (4 choices) --------
    if name in GPQA_CONFIG_MAP:
        ds = load_dataset("Idavidrein/gpqa", GPQA_CONFIG_MAP[name], split="train")
        examples: List[Dict[str, Any]] = []
        for i, ex in tqdm(enumerate(ds), desc=f"[{name}] building examples", colour="cyan"):
            question = str(ex["Question"]).strip()
            correct = str(ex["Correct Answer"]).strip()
            incorrects = [
                str(ex["Incorrect Answer 1"]).strip(),
                str(ex["Incorrect Answer 2"]).strip(),
                str(ex["Incorrect Answer 3"]).strip(),
            ]
            choice_pairs = [(correct, True)] + [(inc, False) for inc in incorrects]
            random.shuffle(choice_pairs)
            letters = [chr(ord("A") + j) for j in range(len(choice_pairs))]
            options: List[str] = []
            gold_letter: Optional[str] = None
            for letter, (opt_text, is_correct) in zip(letters, choice_pairs):
                options.append(opt_text)
                if is_correct:
                    gold_letter = letter
            assert gold_letter is not None
            user_content = build_mc_question_block(question, options)
            examples.append(
                {
                    "index": i,
                    "user_content": user_content,
                    "gold_raw": gold_letter,
                    "gold_norm": gold_letter,
                    "meta": {
                        "gold_content": correct,
                        "valid_letters": "".join(letters),
                    },
                }
            )
        examples = apply_limit(examples, ds_cfg.limit)
        for j, ex in enumerate(examples):
            ex["index"] = j
        return "mc", examples

    # -------- MMLU (4 choices) --------
    if name == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        import numpy as np

        examples: List[Dict[str, Any]] = []
        for i, ex in tqdm(enumerate(ds), desc="[mmlu] building examples", colour="green"):
            question = str(ex["question"]).strip()
            choices = [str(c).strip() for c in ex["choices"]]
            ans = ex["answer"]
            if isinstance(ans, (int, np.integer)):
                idx = int(ans)
                gold_letter = chr(ord("A") + idx)
            elif isinstance(ans, str):
                m = re.search(r"[A-Z]", ans.upper())
                if m:
                    gold_letter = m.group(0)
                else:
                    gold_letter = chr(ord("A") + int(ans))
            else:
                raise ValueError(f"Unexpected mmlu answer type: {type(ans)}")
            user_content = build_mc_question_block(question, choices)
            letters = "".join([chr(ord("A") + k) for k in range(len(choices))])
            examples.append(
                {
                    "index": i,
                    "user_content": user_content,
                    "gold_raw": gold_letter,
                    "gold_norm": gold_letter,
                    "meta": {
                        "gold_content": choices[ord(gold_letter) - ord("A")],
                        "valid_letters": letters,
                    },
                }
            )
        examples = apply_limit(examples, ds_cfg.limit)
        for j, ex in enumerate(examples):
            ex["index"] = j
        return "mc", examples

    # -------- MMLU-Pro (usually 10 choices A-J) --------
    if name == "mmlu-pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        examples: List[Dict[str, Any]] = []
        for i, ex in tqdm(enumerate(ds), desc="[mmlu-pro] building examples", colour="magenta"):
            question = str(ex["question"]).strip()
            options = [str(o).strip() for o in ex["options"]]
            ans = ex.get("answer")

            gold_letter: str
            if isinstance(ans, str):
                m = re.search(r"[A-Z]", ans.upper())
                if not m:
                    raise ValueError(f"Cannot parse mmlu-pro answer: {ans!r}")
                gold_letter = m.group(0)
            else:
                # Some variants provide `answer_index`
                idx = ex.get("answer_index", ans)
                gold_letter = chr(ord("A") + int(idx))

            user_content = build_mc_question_block(question, options)
            letters = "".join([chr(ord("A") + k) for k in range(len(options))])
            examples.append(
                {
                    "index": int(ex.get("question_id", i)),
                    "user_content": user_content,
                    "gold_raw": gold_letter,
                    "gold_norm": gold_letter,
                    "meta": {
                        "gold_content": options[ord(gold_letter) - ord("A")] if (0 <= (ord(gold_letter) - ord("A")) < len(options)) else None,
                        "valid_letters": letters,
                    },
                }
            )
        examples = apply_limit(examples, ds_cfg.limit)
        for j, ex in enumerate(examples):
            ex["index"] = j
        return "mc", examples

    # -------- ARC (AI2) (typically 4 choices, but we handle variable N) --------
    if name in {"arc-easy", "arc_easy", "arc", "ai2_arc_easy", "ai2-arc-easy"}:
        split = _canonical_split(name, ds_cfg.split)
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
        examples: List[Dict[str, Any]] = []
        for i, ex in tqdm(enumerate(ds), desc=f"[arc-easy:{split}] building examples", colour="green"):
            q = _arc_unpack_question(ex.get("question"))
            labels, texts = _arc_unpack_choices(ex.get("choices"))
            if not texts:
                # Some dataset variants may store choices under different keys
                labels, texts = _arc_unpack_choices(ex.get("options"))
            n = len(texts)
            if n <= 0:
                continue
            idx = _arc_answer_index(str(ex.get("answerKey", "")), labels, n)
            gold_letter = chr(ord("A") + idx)
            user_content = build_mc_question_block(q, texts)
            valid_letters = "".join([chr(ord("A") + k) for k in range(n)])
            examples.append(
                {
                    "index": i,
                    "user_content": user_content,
                    "gold_raw": gold_letter,
                    "gold_norm": gold_letter,
                    "meta": {
                        "id": ex.get("id"),
                        "answerKey": ex.get("answerKey"),
                        "choice_labels": labels,
                        "gold_content": texts[idx] if 0 <= idx < n else None,
                        "valid_letters": valid_letters,
                    },
                }
            )
        examples = apply_limit(examples, ds_cfg.limit)
        for j, ex in enumerate(examples):
            ex["index"] = j
        return "mc", examples

    if name in {"arc-challenge", "arc_challenge", "ai2_arc_challenge", "ai2-arc-challenge"}:
        split = _canonical_split(name, ds_cfg.split)
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
        examples: List[Dict[str, Any]] = []
        for i, ex in tqdm(enumerate(ds), desc=f"[arc-challenge:{split}] building examples", colour="red"):
            q = _arc_unpack_question(ex.get("question"))
            labels, texts = _arc_unpack_choices(ex.get("choices"))
            if not texts:
                labels, texts = _arc_unpack_choices(ex.get("options"))
            n = len(texts)
            if n <= 0:
                continue
            idx = _arc_answer_index(str(ex.get("answerKey", "")), labels, n)
            gold_letter = chr(ord("A") + idx)
            user_content = build_mc_question_block(q, texts)
            valid_letters = "".join([chr(ord("A") + k) for k in range(n)])
            examples.append(
                {
                    "index": i,
                    "user_content": user_content,
                    "gold_raw": gold_letter,
                    "gold_norm": gold_letter,
                    "meta": {
                        "id": ex.get("id"),
                        "answerKey": ex.get("answerKey"),
                        "choice_labels": labels,
                        "gold_content": texts[idx] if 0 <= idx < n else None,
                        "valid_letters": valid_letters,
                    },
                }
            )
        examples = apply_limit(examples, ds_cfg.limit)
        for j, ex in enumerate(examples):
            ex["index"] = j
        return "mc", examples

    # -------- CaseHOLD (5 choices A-E) --------
    if name == "casehold":
        split = _canonical_split(name, ds_cfg.split)
        ds = load_dataset("casehold/casehold", split=split,trust_remote_code=True)
        examples: List[Dict[str, Any]] = []
        for i, ex in tqdm(enumerate(ds), desc=f"[casehold:{split}] building examples", colour="blue"):
            prompt = str(ex["citing_prompt"]).strip()
            options = [str(ex[f"holding_{k}"]).strip() for k in range(5)]
            label = int(ex["label"])  # 0..4
            gold_letter = chr(ord("A") + label)
            user_content = build_mc_question_block(prompt, options)
            examples.append(
                {
                    "index": i,
                    "user_content": user_content,
                    "gold_raw": gold_letter,
                    "gold_norm": gold_letter,
                    "meta": {"example_id": ex.get("example_id"), "label": label, "valid_letters": "ABCDE"},
                }
            )
        examples = apply_limit(examples, ds_cfg.limit)
        for j, ex in enumerate(examples):
            ex["index"] = j
        return "mc", examples

    # -------- EconLogicQA (order A-D) --------
    if name in {"econlogicqa", "econ_logic_qa", "econ-logic-qa"}:
        split = _canonical_split(name, ds_cfg.split)
        try:
            ds = load_dataset("yinzhu-quan/econ_logic_qa", split=split)
        except Exception:
            # Try common alternative name
            ds = load_dataset("yinzhu-quan/econ_logic_qa", split="validation" if split == "val" else split)

        examples: List[Dict[str, Any]] = []
        for i, ex in tqdm(enumerate(ds), desc=f"[econlogicqa:{split}] building examples", colour="white"):
            q = str(ex.get("Question", "")).strip()
            A = str(ex.get("A", "")).strip()
            B = str(ex.get("B", "")).strip()
            C = str(ex.get("C", "")).strip()
            D = str(ex.get("D", "")).strip()
            gold_raw = str(ex.get("Answer", "")).strip()
            gold_norm = normalize_order_answer(gold_raw) or gold_raw.replace(" ", "")
            user_content = build_order_question_block(q, {"A": A, "B": B, "C": C, "D": D})
            examples.append(
                {
                    "index": i,
                    "user_content": user_content,
                    "gold_raw": gold_raw,
                    "gold_norm": gold_norm,
                    "meta": {"valid_letters": "ABCD"},
                }
            )
        examples = apply_limit(examples, ds_cfg.limit)
        for j, ex in enumerate(examples):
            ex["index"] = j
        return "order", examples

    raise ValueError(f"Unknown dataset name: {ds_cfg.name}")


def build_chat_prompts(tokenizer, examples: List[Dict[str, Any]], system_block: str, thinking: bool) -> List[str]:
    prompts: List[str] = []
    for ex in tqdm(examples, desc="Building prompts"):
        messages = [
            {"role": "system", "content": system_block.strip()},
            {"role": "user", "content": str(ex["user_content"]).strip()},
        ]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        prompts.append(prompt_text)
    return prompts


# ==============================
#  Inference helpers
# ==============================

def _torch_dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown torch_dtype: {s}")


def load_model_and_tokenizer(cfg: ModelConfig):
    tag = cfg.tag
    if tag not in DIR_DICT:
        raise ValueError(f"Unknown model tag: {tag}. Add it to DIR_DICT or support list.")

    device_map = cfg.device
    torch_dtype = _torch_dtype_from_str(cfg.torch_dtype)

    model_dir = DIR_DICT[tag]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=cfg.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    

    if cfg.tokenizer_padding_side:
        tokenizer.padding_side = cfg.tokenizer_padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def get_inference_device(model, device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if hasattr(model, "hf_device_map"):
        for dev in model.hf_device_map.values():
            if isinstance(dev, (list, tuple)):
                dev = dev[0]
            if dev not in ("cpu", "disk", "meta"):
                return str(dev)
        return "cpu"
    if hasattr(model, "device"):
        return str(model.device)
    return "cpu"


# ==============================
#  Scoring + output
# ==============================

def compute_summary(
    dataset_name: str,
    split: str,
    limit: Optional[float],
    gold_norm: List[str],
    pred_norm: List[Optional[str]],
) -> Tuple[Dict[str, Any], List[float]]:
    assert len(gold_norm) == len(pred_norm)
    n = len(gold_norm)
    correct = 0
    per_item_acc: List[float] = []
    for g, p in zip(gold_norm, pred_norm):
        ok = (p is not None) and (p == g)
        correct += int(ok)
        per_item_acc.append(1.0 if ok else 0.0)
    if n == 0:
        acc = 0.0
        std_err = 0.0
    else:
        p_hat = correct / n
        acc = round(p_hat, 4)
        std_err = round(math.sqrt(p_hat * (1.0 - p_hat) / n), 4)
    summary = {
        "dataset": dataset_name,
        "split": split,
        "limit": limit,
        "correct": correct,
        "total": n,
        "accuracy": acc,
        "std_err": std_err,
    }
    return summary, per_item_acc


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ==============================
#  Main evaluation
# ==============================

def evaluate_one_dataset(
    model,
    tokenizer,
    run_cfg: RunConfig,
    ds_cfg: DatasetConfig,
) -> Dict[str, Any]:
    task_type, examples = load_and_build_examples(ds_cfg)
    if not examples:
        return {"summary": {"dataset": ds_cfg.name, "split": ds_cfg.split, "limit": ds_cfg.limit,
                            "correct": 0, "total": 0, "accuracy": 0.0, "std_err": 0.0}, "details": []}

    system_block = system_block_for_task(task_type)
    prompts = build_chat_prompts(tokenizer, examples, system_block, thinking=run_cfg.thinking)

    device_for_inputs = get_inference_device(model, run_cfg.model.device)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": run_cfg.gen.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
    }
    if run_cfg.gen.temperature and run_cfg.gen.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(run_cfg.gen.temperature)
        if run_cfg.gen.top_k and run_cfg.gen.top_k > 0:
            gen_kwargs["top_k"] = int(run_cfg.gen.top_k)
    else:
        gen_kwargs["do_sample"] = False

    gold_norm = [str(ex["gold_norm"]) for ex in examples]
    gold_raw = [str(ex["gold_raw"]) for ex in examples]

    pred_raw: List[str] = []
    pred_norm: List[Optional[str]] = []
    gen_no_think: List[str] = []

    bs = max(1, int(run_cfg.gen.batch_size))
    for start in tqdm(
        range(0, len(examples), bs),
        desc=f"[{ds_cfg.name}:{ds_cfg.split}] evaluating",
        colour="yellow",
    ):
        end = min(start + bs, len(examples))
        batch_prompts = prompts[start:end]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=run_cfg.max_input_length,
        )

        input_ids = enc["input_ids"].to(device_for_inputs)
        attention_mask = enc["attention_mask"].to(device_for_inputs)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        sequences = out.sequences if hasattr(out, "sequences") else out

        # IMPORTANT: generation starts after the *padded* prompt length (input_ids.shape[1]),
        # NOT after the unpadded length (attention_mask.sum()).
        prompt_len_padded = input_ids.shape[1]
        gen_ids_batch = sequences[:, prompt_len_padded:]
        decoded_texts = tokenizer.batch_decode(gen_ids_batch, skip_special_tokens=True)

        for i, text in enumerate(decoded_texts):

            cleaned_text, think_failed = strip_think_blocks_or_fail(text)
            gen_no_think.append(cleaned_text)

            if think_failed:
                pred_raw.append(FAIL_THINK)
                pred_norm.append(None)
                continue

            ex_meta = examples[start + i].get("meta", {}) or {}
            valid_letters = str(ex_meta.get("valid_letters", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

            if task_type == "mc":
                letter = extract_mc_letter(cleaned_text, valid_letters=valid_letters)
                if letter is not None:
                    pred_raw.append(letter)
                    pred_norm.append(letter)
                else:
                    # extraction failure: keep generation (no-think) for later analysis
                    pred_raw.append(cleaned_text)
                    pred_norm.append(None)

            elif task_type == "order":
                order = extract_order_abcd(cleaned_text)
                if order is not None:
                    pred_raw.append(order)
                    pred_norm.append(order)
                else:
                    pred_raw.append(cleaned_text)
                    pred_norm.append(None)
            else:
                raise ValueError(f"Unknown task_type: {task_type}")

    summary, per_item_acc = compute_summary(
        dataset_name=ds_cfg.name,
        split=ds_cfg.split,
        limit=ds_cfg.limit,
        gold_norm=gold_norm,
        pred_norm=pred_norm,
    )

    details: List[Dict[str, Any]] = []
    for ex, g_raw, g_norm, gen_txt, p_raw, p_norm, acc in zip(
        examples, gold_raw, gold_norm, gen_no_think, pred_raw, pred_norm, per_item_acc
    ):
        details.append(
            {
                "index": ex["index"],
                "gold_raw": g_raw,
                "gold_norm": g_norm,
                "gen_no_think": gen_txt,
                "pred_raw": p_raw,
                "pred_norm": p_norm,
                "acc": float(acc),
                "meta": ex.get("meta", {}),
            }
        )

    return {"summary": summary, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config file")
    args = parser.parse_args()

    run_cfg = parse_config(args.config)
    ensure_dir(run_cfg.output_dir)

    model, tokenizer = load_model_and_tokenizer(run_cfg.model)
    think_switch = "non-think"
    if run_cfg.thinking:
        think_switch = "thinking"
    all_summaries = []
    for ds_cfg in run_cfg.datasets:
        result = evaluate_one_dataset(model, tokenizer, run_cfg, ds_cfg)
        all_summaries.append(result["summary"])
        limit_setting = ""
        if ds_cfg.limit is not None:
            limit_setting = f"_{ds_cfg.limit}"
        out_path = os.path.join(run_cfg.output_dir, f"{run_cfg.model.tag}_{think_switch}{limit_setting}_{ds_cfg.name}_{ds_cfg.split}.json")
        save_json(result, out_path)

        s = result["summary"]
        print(
            f"Dataset={s['dataset']} split={s['split']} limit={s['limit']} "
            f"correct={s['correct']}/{s['total']} acc={s['accuracy']:.4f} se={s['std_err']:.4f} "
            f"-> {out_path}"
        )

    save_json({"summaries": all_summaries}, os.path.join(run_cfg.output_dir, f"{run_cfg.model.tag}_{think_switch}{limit_setting}_{ds_cfg.name}_{ds_cfg.split}_summary.json"))


if __name__ == "__main__":
    main()
