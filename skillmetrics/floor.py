"""Floor-effect and headroom checks that decide which (model, size) cells enter the analysis."""
from __future__ import annotations

from statistics import mean
from typing import Dict, Iterable, Sequence

FLOOR_ROOT_SUCCESS = 0.30
HEADROOM_CRAFTED = 0.20


def _pooled(cells: Iterable[Dict[str, float]], field: str, count: str) -> float:
    """Attempt-weighted mean of a per-cell rate across seeds."""
    num = den = 0.0
    for c in cells:
        n = c.get(count, 0)
        v = c.get(field, float("nan"))
        if n and v == v:
            num += v * n
            den += n
    return num / den if den else float("nan")


def floor_check(no_skill_cells: Sequence[Dict[str, float]], threshold: float = FLOOR_ROOT_SUCCESS) -> Dict:
    """Pass if the no-skill arm solves root (skill-free) tasks at or above threshold, pooled over seeds."""
    rate = _pooled(no_skill_cells, "success_root", "n_root")
    return {
        "check": "floor",
        "no_skill_root_success": rate,
        "threshold": threshold,
        "n_root": sum(c.get("n_root", 0) for c in no_skill_cells),
        "passed": rate == rate and rate >= threshold,
    }


def headroom_check(
    in_context_cells: Dict[str, Sequence[Dict[str, float]]],
    no_skill_cells: Sequence[Dict[str, float]],
    margin: float = HEADROOM_CRAFTED,
) -> Dict:
    """Pass if the best in-context arm beats no-skill on crafted tasks by at least margin."""
    base = _pooled(no_skill_cells, "success_crafted", "n_crafted")
    arms = {a: _pooled(c, "success_crafted", "n_crafted") for a, c in in_context_cells.items()}
    best_arm = max(arms, key=lambda a: arms[a] if arms[a] == arms[a] else -1.0) if arms else None
    best = arms.get(best_arm, float("nan")) if best_arm else float("nan")
    return {
        "check": "headroom",
        "no_skill_crafted_success": base,
        "in_context_crafted_success": arms,
        "best_arm": best_arm,
        "margin": margin,
        "passed": best == best and base == base and best - base >= margin,
    }


def mean_or_nan(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x == x]
    return mean(xs) if xs else float("nan")
