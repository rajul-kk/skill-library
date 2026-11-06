"""Write sweep cells in the Myrios shared result schema (myrios_eval.result 1.0) without importing myrios_eval.

One Myrios arm per (model, arm, milestone): conditions {"model", "library_size"}, per_seed rows carrying the
records.aggregate() metrics. Aggregates are left empty for the reader to compute (myrios_eval.schema.make_arm
does this when given per_seed only), so the bootstrap is done once, in one place.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

SCHEMA = "myrios_eval.result"
SCHEMA_VERSION = "1.0"
HARNESS_VERSION = "0.1.0"
METRICS = ("success", "success_root", "success_crafted", "prompt_tokens", "gen_tokens", "inf_tokens",
           "inf_flops", "train_flops_cum", "train_tokens_cum")
TOKEN_COUNTER = "model tokenizer (Qwen2.5), counted at generation time"
PROMPT_SCOPE = "full prompt per attempt, including retrieved or in-context skills"
FLOPS_METHOD = "skillmetrics.flops: 2*N*tokens matmul plus attention term; LoRA training 6*N_lora + 4*N_base per token"


def _num(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return float(v)


def to_myrios(cells: Dict, experiment: str, repo: str, commit: str, created: Optional[str] = None,
              extra: Optional[Dict] = None) -> Dict:
    """cells: records.aggregate() output keyed (model, arm, milestone, seed)."""
    groups: Dict = {}
    for (model, arm, milestone, seed), v in cells.items():
        groups.setdefault((model, arm, int(milestone)), []).append(
            {"seed": int(seed), "metrics": {m: _num(v.get(m)) for m in METRICS}, "n_attempts": int(v.get("n", 0))})
    arms = []
    for (model, arm, milestone), rows in sorted(groups.items()):
        arms.append({"arm": arm, "label": arm, "conditions": {"model": model, "library_size": milestone},
                     "per_seed": sorted(rows, key=lambda r: r["seed"]), "aggregate": {}})
    return {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "harness_version": HARNESS_VERSION,
        "experiment": experiment, "repo": repo, "commit": commit,
        "created": created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": sorted({c["conditions"]["model"] for c in arms})[0] if len({c["conditions"]["model"] for c in arms}) == 1 else None,
        "token_counter": TOKEN_COUNTER, "prompt_tokens_scope": PROMPT_SCOPE, "flops_method": FLOPS_METHOD,
        "arms": arms, "comparisons": [], "extra": dict(extra or {}),
    }


def write(path, doc: Dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
