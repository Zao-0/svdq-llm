# ppl_evaluator.py
# -*- coding: utf-8 -*-
"""
A slim PPL evaluator aligned with your original evaluator.py saving style.

- Only evaluates: wikitext, lambada_openai
- PPL eval forces batch_size=1 (even if caller passes others)
- Saves per-task json as:  result_path + "_" + full_scenario_name + ".json"
  where full_scenario_name = (scenario_name + "_" + task) or task

Saved json content keeps the original structure (task_name -> metrics),
and adds a root "__meta__" field:
{
  "__meta__": {task_name, scenario_name, model_signature, time_utc, ...},
  "wikitext": {...metrics...}
}

Public API:
  evl(model, tokenizer, result_path, scenario_name=None, limit=None, batch_size="auto") -> Dict[str, float]
Returns:
  {"wikitext": ppl, "lambada_openai": ppl}  (ppl rounded to 4 decimals)
"""

from __future__ import annotations

from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM

from typing import Optional, Any, Dict, Union
from torch import nn
import torch

import os
import json
import math
import inspect
from datetime import datetime, timezone


NumberLike = Union[int, float]


# -----------------------------
# Helpers: signature / time / metrics
# -----------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _get_model_device(model: nn.Module) -> torch.device:
    dev = getattr(model, "device", None)
    if isinstance(dev, torch.device):
        return dev
    if isinstance(dev, str):
        return torch.device(dev)
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def _model_signature(model: nn.Module, tokenizer: Any) -> str:
    """
    Best-effort model signature string for distinguishing runs.
    You can still use scenario_name to encode your quant config; this is an extra safety net.
    """
    parts = []

    # model name/path
    name = None
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("_name_or_path", "name_or_path", "model_type"):
            if hasattr(cfg, attr):
                v = getattr(cfg, attr)
                if isinstance(v, str) and v:
                    name = v
                    break
    if not name and hasattr(model, "name_or_path"):
        v = getattr(model, "name_or_path")
        if isinstance(v, str) and v:
            name = v
    parts.append(f"model={name or model.__class__.__name__}")

    # dtype / device
    try:
        dtype = next(model.parameters()).dtype
    except Exception:
        dtype = None
    parts.append(f"dtype={str(dtype) if dtype is not None else 'unknown'}")
    parts.append(f"device={_get_model_device(model)}")

    # tokenizer name
    tok_name = None
    for attr in ("name_or_path",):
        if hasattr(tokenizer, attr):
            v = getattr(tokenizer, attr)
            if isinstance(v, str) and v:
                tok_name = v
                break
    parts.append(f"tokenizer={tok_name or tokenizer.__class__.__name__}")

    return " | ".join(parts)


def _make_hflm(model: nn.Module, tokenizer: Any, batch_size: int = 1) -> HFLM:
    """
    Create lm-eval HFLM wrapper, keeping compatibility with your evaluator.py:
    - pass enable_thinking=False when supported
    """
    kwargs = dict(
        pretrained=model,
        device=_get_model_device(model),
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    # keep best-effort compatibility
    try:
        sig = inspect.signature(HFLM.__init__)
        if "enable_thinking" in sig.parameters:
            kwargs["enable_thinking"] = False
    except Exception:
        pass

    try:
        return HFLM(**kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return HFLM(**kwargs)


def _extract_ppl(metrics: Dict[str, Any]) -> float:
    """
    Extract a PPL-like metric from lm-eval metrics dict.
    Priority: word_perplexity -> perplexity -> ppl -> byte_perplexity
    Fallback: exp(loss)
    """
    priority = ["word_perplexity", "perplexity", "ppl", "byte_perplexity"]
    keys = list(metrics.keys())

    for pat in priority:
        for k in keys:
            if pat in str(k).lower():
                v = metrics[k]
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    return float(v)

    for k in keys:
        if "loss" in str(k).lower():
            v = metrics[k]
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                return float(math.exp(float(v)))

    raise KeyError(
        "Cannot find PPL metric in task results. "
        f"Available metric keys: {keys}"
    )


# -----------------------------
# Core: run one task and save
# -----------------------------
def run_evaluation(
    model: nn.Module,
    tokenizer: Any,
    task_dict: Dict,
    result_path: str,
    scenario_name: str = None,
    limit: Optional[int | float] = None,
    batch_size: int | str = "auto",
    task_name: str = None,
) -> Dict:
    """
    Runs lm-eval evaluator.evaluate on a single task_dict (should contain one PPL task).
    Saves json in aligned naming scheme, without saving samples.

    NOTE: batch_size is forced to 1 for PPL tasks.
    """
    if scenario_name is not None:
        print(f"--- Starting PPL evaluation for scenario: {scenario_name} ---", flush=True)

    # PPL: force batch size = 1
    forced_bs = 1
    if batch_size != 1:
        # keep silent enough; but let you notice if you accidentally pass other value
        print(f"[ppl_evaluator] batch_size is forced to 1 (got {batch_size!r}).", flush=True)

    lm_eval_model = _make_hflm(model=model, tokenizer=tokenizer, batch_size=forced_bs)

    t0 = datetime.now(timezone.utc)
    print("Running evaluator.evaluate...", flush=True)
    results_all = evaluator.evaluate(
        lm=lm_eval_model,
        task_dict=task_dict,
        limit=limit,
        log_samples=False,
        apply_chat_template=False,  # PPL tasks are raw text
    )
    t1 = datetime.now(timezone.utc)

    results = results_all["results"]  # {task_name: metrics}

    # infer task_name if not provided
    if task_name is None:
        # usually only one key
        task_name = next(iter(results.keys())) if isinstance(results, dict) and results else "unknown_task"

    meta = {
        "task_name": task_name,
        "scenario_name": scenario_name,
        "model_signature": _model_signature(model, tokenizer),
        "time_utc": t1.replace(microsecond=0).isoformat(),
        "start_time_utc": t0.replace(microsecond=0).isoformat(),
        "duration_sec": float(f"{(t1 - t0).total_seconds():.4f}"),
        "limit": limit,
        "batch_size_forced": forced_bs,
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }

    # Save (aligned filename), but add "__meta__" at root.
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    save_obj = {"__meta__": meta}
    if isinstance(results, dict):
        save_obj.update(results)

    with open(result_path + "_" + scenario_name + ".json", "w", encoding="utf-8") as f:
        json.dump(save_obj, f, indent=4, ensure_ascii=False)

    if scenario_name is not None:
        print(f"--- PPL evaluation for scenario '{scenario_name}' finished and saved. ---", flush=True)

    return results


# -----------------------------
# Public API: evl()
# -----------------------------
def evl(
    model: nn.Module,
    tokenizer: Any,
    result_path: str,
    scenario_name: str = None,
    limit: Optional[int | float] = None,
    batch_size: int | str = "auto",
) -> Dict[str, float]:
    """
    Aligns with original evaluator.py signature & saving pattern.

    Evaluates tasks:
      - wikitext
      - lambada_openai

    Returns:
      {"wikitext": ppl, "lambada_openai": ppl} with 4-decimal floats
    """
    model.eval()

    task_manager = tasks.TaskManager()
    task_list = [
        {"task": "wikitext", "num_fewshot": 0},
        {"task": "lambada_openai", "num_fewshot": 0},
    ]

    # Kept for readability; actual extraction is robust via _extract_ppl()
    result_dict: Dict[str, float] = {}

    for tsk in task_list:
        tname = tsk["task"]
        full_scenario_name = (scenario_name + "_" + tname) if scenario_name is not None else tname
        task_dict = tasks.get_task_dict([tsk], task_manager=task_manager)

        res = run_evaluation(
            model=model,
            tokenizer=tokenizer,
            task_dict=task_dict,
            result_path=result_path,
            scenario_name=full_scenario_name,
            limit=limit,
            batch_size=batch_size,  # will be forced to 1 inside
            task_name=tname,
        )

        if tname not in res:
            raise KeyError(f"Task {tname!r} missing from results keys: {list(res.keys())}")

        ppl = _extract_ppl(res[tname])
        result_dict[tname] = float(f"{ppl:.4f}")

    return result_dict


# -----------------------------
# Quick manual test (optional)
# -----------------------------
if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser("ppl_evaluator quick test")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--result_path", type=str, default="./results/ppl")
    ap.add_argument("--scenario_name", type=str, default="baseline")
    ap.add_argument("--limit", type=float, default=None)  # e.g. 0.01 or 100
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=None,
        trust_remote_code=True,
    ).to(torch.device(args.device))

    out = evl(
        model=model,
        tokenizer=tok,
        result_path=args.result_path,
        scenario_name=args.scenario_name,
        limit=args.limit,
        batch_size=1,
    )
    print("[PPL]", out, flush=True)
