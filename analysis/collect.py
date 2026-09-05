"""Assembling GPU run outputs into the payload the figures already understand."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from analysis.metrics import (
    SkillEval,
    Snapshot,
    curve_support,
    forgetting_curve,
    normalized_forgetting_curve,
)

LABELS = {
    "in_context": "Full in-context library (no consolidation)",
    "seq_no_replay": "Sequential LoRA, no replay",
    "isolated": "Isolated per-skill adapters (PEAM-style)",
    "single_shot": "One-shot text-to-LoRA per skill (LatentSkill-style)",
    "ours": "Cumulative LoRA + replay",
}


def snapshot_from_dict(d: dict) -> Snapshot:
    return Snapshot(
        night=d["night"],
        episode=d["episode"],
        library_size=d["library_size"],
        stored_params=d["stored_params"],
        seed=d.get("seed", 0),
        skills=[SkillEval(**s) for s in d["skills"]],
    )


def load_method(root: str | Path, method: str) -> Optional[List[Snapshot]]:
    path = Path(root) / method / "snapshots.json"
    if not path.exists():
        return None
    return [snapshot_from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def collect_run(
    root: str | Path, methods: Optional[List[str]] = None, run_id: str = "gpu"
) -> dict:
    """Build a results payload from whatever methods have finished so far."""
    root = Path(root)
    methods = methods or sorted(p.name for p in root.iterdir() if p.is_dir())
    payload: Dict[str, dict] = {}
    for method in methods:
        snapshots = load_method(root, method)
        if not snapshots:
            continue
        last_night = max(s.night for s in snapshots)
        finals = [s for s in snapshots if s.night == last_night]
        payload[method] = {
            "method": method,
            "label": LABELS.get(method, method),
            "final_success_rate": sum(s.success_rate for s in finals) / len(finals),
            "final_earliest_3": sum(s.earliest_k(3) for s in finals) / len(finals),
            "final_stored_params": finals[-1].stored_params,
            "forgetting_curve": {
                str(k): v for k, v in forgetting_curve(snapshots).items()
            },
            "forgetting_curve_by_skills": {
                str(k): v for k, v in forgetting_curve(snapshots, "skills").items()
            },
            "normalized_forgetting_curve": {
                str(k): v
                for k, v in normalized_forgetting_curve(snapshots, "skills").items()
            },
            "curve_support": {
                str(k): v for k, v in curve_support(snapshots, "skills").items()
            },
            "snapshots": [s.to_dict() for s in snapshots],
            "tokens": {"mean_prompt_tokens": finals[-1].mean_tokens},
            "transfer": {},
            "episodes": [],
            "nights": [],
        }
    return {"run_id": run_id, "backend": "lora", "methods": payload}


def write_collected(
    root: str | Path, methods: Optional[List[str]] = None, run_id: str = "gpu"
) -> Path:
    payload = collect_run(root, methods, run_id)
    path = Path(root) / "results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def summary_table(payload: dict) -> str:
    """A plain text table of the headline numbers, for pasting into the paper."""
    rows = [
        f"{'method':<34}{'final':>8}{'oldest3':>9}{'tokens':>9}{'params':>13}",
        "-" * 73,
    ]
    for method, m in payload["methods"].items():
        rows.append(
            f"{m['label'][:33]:<34}{m['final_success_rate']:>8.2f}"
            f"{m['final_earliest_3']:>9.2f}"
            f"{m['tokens'].get('mean_prompt_tokens', 0):>9.0f}"
            f"{m['final_stored_params']:>13,}"
        )
    return "\n".join(rows)
