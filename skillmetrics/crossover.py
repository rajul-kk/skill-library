"""Efficiency ratios between two arms across library sizes, crossover N* and break-even H*."""
from __future__ import annotations

import math
from statistics import mean
from typing import Dict, List, Optional, Sequence

from .bootstrap import percentile, resample_indices

INF = float("inf")


def _cost(cell: Dict[str, float], metric: str, horizon: float) -> float:
    if metric == "tokens":
        return cell["inf_tokens"]
    if metric == "flops":
        amortised = 0.0 if horizon == INF else cell["train_flops_cum"] / horizon
        return cell["inf_flops"] + amortised
    raise ValueError(f"unknown cost metric {metric!r}")


def efficiency_ratio(
    ours: Sequence[Dict[str, float]],
    comp: Sequence[Dict[str, float]],
    metric: str = "tokens",
    horizon: float = INF,
) -> float:
    """(success per unit cost of ours) / (success per unit cost of comp), ratio of seed means."""
    s_o, s_c = mean(c["success"] for c in ours), mean(c["success"] for c in comp)
    c_o = mean(_cost(c, metric, horizon) for c in ours)
    c_c = mean(_cost(c, metric, horizon) for c in comp)
    if s_c == 0:
        return INF if s_o > 0 else float("nan")
    return (s_o / c_o) / (s_c / c_c)


def break_even_queries(ours: Sequence[Dict[str, float]], comp: Sequence[Dict[str, float]]) -> float:
    """Queries H* after which ours' training pays off in success-adjusted FLOPs; inf if never."""
    s_o, s_c = mean(c["success"] for c in ours), mean(c["success"] for c in comp)
    i_o, i_c = mean(c["inf_flops"] for c in ours), mean(c["inf_flops"] for c in comp)
    t = mean(c["train_flops_cum"] for c in ours) - mean(c["train_flops_cum"] for c in comp)
    if s_c == 0 or s_o == 0:
        return INF if s_o == 0 else 0.0
    slack = s_o * i_c / s_c - i_o  # inference saving per query, success-adjusted
    if slack <= 0:
        return INF
    return max(0.0, t) / slack


def sustained_crossover(ratios: Dict[int, float]) -> Optional[int]:
    """Smallest grid size from which the ratio exceeds 1 at every larger size; None if never."""
    sizes = sorted(ratios)
    best = None
    for n in reversed(sizes):
        r = ratios[n]
        if r == r and r > 1.0:
            best = n
        else:
            break
    return best


def interpolated_crossover(ratios: Dict[int, float]) -> Optional[float]:
    """Crossover on a log-N scale between the last losing size and the first sustained winning size."""
    n_star = sustained_crossover(ratios)
    sizes = sorted(ratios)
    if n_star is None:
        return None
    i = sizes.index(n_star)
    if i == 0:
        return float(n_star)
    lo, hi = sizes[i - 1], n_star
    r_lo, r_hi = ratios[lo], ratios[hi]
    if not (r_lo > 0 and r_hi > 0) or r_lo == r_hi:
        return float(hi)
    x_lo, x_hi = math.log(lo), math.log(hi)
    t = (0.0 - math.log(r_lo)) / (math.log(r_hi) - math.log(r_lo))
    return float(math.exp(x_lo + t * (x_hi - x_lo)))


def crossover_analysis(
    ours: Dict[int, Dict[int, Dict[str, float]]],
    comp: Dict[int, Dict[int, Dict[str, float]]],
    metric: str = "tokens",
    horizon: float = INF,
    n_boot: int = 10_000,
    seed: int = 0,
) -> Dict:
    """ours/comp: milestone -> seed -> cell; seeds are resampled jointly across all milestones."""
    sizes = sorted(set(ours) & set(comp))
    seeds = sorted(set.intersection(*[set(ours[n]) & set(comp[n]) for n in sizes])) if sizes else []
    if not seeds:
        return {"sizes": sizes, "seeds": [], "error": "no paired seeds"}

    def ratios_for(sample: List[int]) -> Dict[int, float]:
        return {
            n: efficiency_ratio([ours[n][s] for s in sample], [comp[n][s] for s in sample], metric, horizon)
            for n in sizes
        }

    point = ratios_for(seeds)
    boot_ratios: Dict[int, List[float]] = {n: [] for n in sizes}
    boot_nstar: List[float] = []
    for idx in resample_indices(len(seeds), n_boot, seed):
        sample = [seeds[i] for i in idx]
        r = ratios_for(sample)
        for n in sizes:
            if r[n] == r[n]:
                boot_ratios[n].append(r[n])
        ns = sustained_crossover(r)
        boot_nstar.append(INF if ns is None else float(ns))

    dist = {}
    for v in boot_nstar:
        key = "not_reached" if v == INF else str(int(v))
        dist[key] = dist.get(key, 0) + 1
    return {
        "metric": metric,
        "horizon": horizon,
        "sizes": sizes,
        "seeds": seeds,
        "ratio": {n: {"point": point[n], "lo": percentile(boot_ratios[n], 2.5), "hi": percentile(boot_ratios[n], 97.5)} for n in sizes},
        "n_star": sustained_crossover(point),
        "n_star_interpolated": interpolated_crossover(point),
        "n_star_ci": [percentile(boot_nstar, 2.5), percentile(boot_nstar, 97.5)],
        "n_star_distribution": {k: v / len(boot_nstar) for k, v in sorted(dist.items())},
    }


def break_even_analysis(
    ours: Dict[int, Dict[int, Dict[str, float]]],
    comp: Dict[int, Dict[int, Dict[str, float]]],
    n_boot: int = 10_000,
    seed: int = 0,
) -> Dict[int, Dict[str, float]]:
    """Break-even query count H* per milestone with a percentile interval."""
    out = {}
    for n in sorted(set(ours) & set(comp)):
        seeds = sorted(set(ours[n]) & set(comp[n]))
        if not seeds:
            continue
        point = break_even_queries([ours[n][s] for s in seeds], [comp[n][s] for s in seeds])
        boot = []
        for idx in resample_indices(len(seeds), n_boot, seed):
            sample = [seeds[i] for i in idx]
            boot.append(break_even_queries([ours[n][s] for s in sample], [comp[n][s] for s in sample]))
        out[n] = {"point": point, "lo": percentile(boot, 2.5), "hi": percentile(boot, 97.5), "p_never": sum(b == INF for b in boot) / len(boot)}
    return out
