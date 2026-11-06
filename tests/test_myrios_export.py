import json
import math
import sys
from pathlib import Path

import pytest

from skillmetrics.myrios_export import METRICS, to_myrios, write

MYRIOS_SRC = Path(__file__).resolve().parents[2] / "Myrios" / "packages" / "myrios-eval" / "src"


def _cells():
    base = {m: 0.5 for m in METRICS}
    return {
        ("Qwen/Qwen2.5-1.5B-Instruct", "ours", 10, 0): dict(base, n=40),
        ("Qwen/Qwen2.5-1.5B-Instruct", "ours", 10, 1): dict(base, n=40, success=0.6),
        ("Qwen/Qwen2.5-1.5B-Instruct", "ic_full", 10, 0): dict(base, n=40, success_crafted=float("nan")),
        ("Qwen/Qwen2.5-1.5B-Instruct", "ours", 25, 0): dict(base, n=40),
    }


def test_groups_by_arm_and_milestone_with_seeds_sorted():
    doc = to_myrios(_cells(), "sweep", "skill-library", "abc123", created="2026-11-06T00:00:00+00:00")
    keys = [(a["arm"], a["conditions"]["library_size"]) for a in doc["arms"]]
    assert keys == [("ic_full", 10), ("ours", 10), ("ours", 25)]
    ours10 = doc["arms"][1]
    assert [r["seed"] for r in ours10["per_seed"]] == [0, 1]
    assert ours10["per_seed"][1]["metrics"]["success"] == 0.6
    assert doc["model"] == "Qwen/Qwen2.5-1.5B-Instruct"


def test_nan_becomes_null_and_file_is_strict_json(tmp_path):
    doc = to_myrios(_cells(), "sweep", "skill-library", "abc123")
    p = write(tmp_path / "r.json", doc)
    back = json.loads(p.read_text())
    assert back["arms"][0]["per_seed"][0]["metrics"]["success_crafted"] is None


def test_validates_against_myrios_eval_when_available():
    if not MYRIOS_SRC.exists():
        pytest.skip("Myrios checkout not found next to this repo")
    sys.path.insert(0, str(MYRIOS_SRC))
    try:
        from myrios_eval.schema import make_arm, validate
    finally:
        sys.path.remove(str(MYRIOS_SRC))
    doc = to_myrios(_cells(), "sweep", "skill-library", "abc123")
    assert validate(doc)
    arm = doc["arms"][1]
    filled = make_arm(arm["arm"], arm["per_seed"], conditions=arm["conditions"])
    assert math.isclose(filled["aggregate"]["success"]["mean"], 0.55)
