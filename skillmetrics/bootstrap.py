"""Seed-level paired bootstrap with percentile intervals, stdlib only."""
from __future__ import annotations

import random
from statistics import mean
from typing import Callable, Dict, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def resample_indices(n: int, n_boot: int, seed: int = 0) -> List[List[int]]:
    """Index lists for n_boot resamples of n units with replacement."""
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(n_boot)]


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    xs = sorted(values)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def bootstrap(
    units: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Point estimate and percentile interval of statistic over resampled units."""
    units = list(units)
    point = statistic(units)
    stats = []
    for idx in resample_indices(len(units), n_boot, seed):
        value = statistic([units[i] for i in idx])
        if value == value:  # drop NaN resamples
            stats.append(value)
    return {
        "point": point,
        "lo": percentile(stats, 100 * alpha / 2),
        "hi": percentile(stats, 100 * (1 - alpha / 2)),
        "n_units": len(units),
        "n_boot": n_boot,
        "n_valid": len(stats),
    }


def mean_ci(values: Sequence[float], n_boot: int = 10_000, seed: int = 0) -> Dict[str, float]:
    """Bootstrap interval of a plain mean."""
    return bootstrap(list(values), lambda xs: mean(xs) if xs else float("nan"), n_boot, seed=seed)


def paired_units(table: Dict[str, Dict[int, T]], keys: Sequence[str]) -> Tuple[List[int], List[Tuple[T, ...]]]:
    """Seeds present for every key, and per-seed tuples, so resampling keeps arms paired."""
    seeds = sorted(set.intersection(*(set(table[k]) for k in keys))) if keys else []
    return seeds, [tuple(table[k][s] for k in keys) for s in seeds]
