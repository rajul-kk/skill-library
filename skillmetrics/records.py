"""Record schema for attempts and training nights, and aggregation into per-seed cells."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

from .flops import ModelShape, attempt_flops, lora_train_flops

ATTEMPT_FIELDS = (
    "model", "seed", "arm", "milestone", "skill", "is_root", "success",
    "prompt_tokens", "gen_tokens", "lora_params",
)
TRAIN_FIELDS = (
    "model", "seed", "arm", "night", "library_size", "n_new", "n_replay",
    "train_tokens", "train_tokens_sq", "lora_params", "grad_checkpointing",
)


def attempt_record(**kw) -> Dict:
    """One evaluation attempt; raises if a required field is missing."""
    missing = [f for f in ATTEMPT_FIELDS if f not in kw]
    if missing:
        raise ValueError(f"attempt record missing {missing}")
    return {"kind": "attempt", **kw}


def train_record(**kw) -> Dict:
    """One consolidation night; raises if a required field is missing."""
    missing = [f for f in TRAIN_FIELDS if f not in kw]
    if missing:
        raise ValueError(f"train record missing {missing}")
    return {"kind": "train", **kw}


def read_jsonl(path: str | Path) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: str | Path, records: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


CellKey = Tuple[str, str, int, int]  # model, arm, milestone, seed


def aggregate(records: Iterable[Dict], shapes: Dict[str, ModelShape]) -> Dict[CellKey, Dict[str, float]]:
    """Per (model, arm, milestone, seed): success, tokens, inference FLOPs, cumulative training FLOPs."""
    attempts: Dict[CellKey, List[Dict]] = defaultdict(list)
    nights: Dict[Tuple[str, str, int], List[Dict]] = defaultdict(list)
    for r in records:
        if r.get("kind") == "attempt":
            attempts[(r["model"], r["arm"], int(r["milestone"]), int(r["seed"]))].append(r)
        elif r.get("kind") == "train":
            nights[(r["model"], r["arm"], int(r["seed"]))].append(r)

    out: Dict[CellKey, Dict[str, float]] = {}
    for key, rows in attempts.items():
        model, arm, milestone, seed = key
        shape = shapes[model]
        flops = [attempt_flops(shape, r["prompt_tokens"], r["gen_tokens"], r["lora_params"])["total"] for r in rows]
        roots = [r for r in rows if r["is_root"]]
        crafted = [r for r in rows if not r["is_root"]]
        train = [
            lora_train_flops(shape, n["train_tokens"], n["train_tokens_sq"], n["lora_params"], n["grad_checkpointing"])
            for n in nights.get((model, arm, seed), [])
            if n["library_size"] <= milestone
        ]
        train_tokens = sum(
            n["train_tokens"] for n in nights.get((model, arm, seed), []) if n["library_size"] <= milestone
        )
        out[key] = {
            "n": len(rows),
            "success": mean(float(r["success"]) for r in rows),
            "success_root": mean(float(r["success"]) for r in roots) if roots else float("nan"),
            "success_crafted": mean(float(r["success"]) for r in crafted) if crafted else float("nan"),
            "n_root": len(roots),
            "n_crafted": len(crafted),
            "prompt_tokens": mean(r["prompt_tokens"] for r in rows),
            "gen_tokens": mean(r["gen_tokens"] for r in rows),
            "inf_tokens": mean(r["prompt_tokens"] + r["gen_tokens"] for r in rows),
            "inf_flops": mean(flops),
            "train_flops_cum": float(sum(train)),
            "train_tokens_cum": float(train_tokens),
        }
    return out


def per_seed(cells: Dict[CellKey, Dict[str, float]], model: str, arm: str, milestone: int) -> Dict[int, Dict[str, float]]:
    """Seed -> cell for one (model, arm, milestone)."""
    return {s: v for (m, a, n, s), v in cells.items() if m == model and a == arm and n == milestone}


def grid(cells: Dict[CellKey, Dict[str, float]], model: Optional[str] = None) -> Dict[str, List]:
    """Models, arms and milestones present in the cells."""
    keys = [k for k in cells if model is None or k[0] == model]
    return {
        "models": sorted({k[0] for k in keys}),
        "arms": sorted({k[1] for k in keys}),
        "milestones": sorted({k[2] for k in keys}),
        "seeds": sorted({k[3] for k in keys}),
    }
