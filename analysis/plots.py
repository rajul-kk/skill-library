"""Figures."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE: Dict[str, dict] = {
    "in_context": dict(color="#444444", marker="o", ls="--", lw=1.6, ms=5),
    "seq_no_replay": dict(color="#c0392b", marker="s", ls="-", lw=1.8, ms=5),
    # Wide halo: our method lands exactly on top of this, and both must stay legible.
    "isolated": dict(color="#2980b9", marker="^", ls="-", lw=5.0, ms=9, alpha=0.35),
    "single_shot": dict(color="#8e44ad", marker="d", ls=":", lw=1.6, ms=5),
    "ours": dict(color="#27ae60", marker="*", ls="-", lw=1.8, ms=8),
}
DEFAULT_STYLE = dict(color="#7f8c8d", marker="o", ls="-", lw=1.6, ms=5)


def _style(key: str) -> dict:
    return dict(STYLE.get(key, DEFAULT_STYLE))


def _finish(ax, path: Optional[Path], legend_loc: str = "best"):
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc=legend_loc, frameon=False, fontsize=8)
    fig = ax.get_figure()
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
    return fig


def plot_forgetting_curve(
    results: dict,
    path: Optional[str | Path] = None,
    axis: str = "skills",
    order: Optional[Sequence[str]] = None,
    max_x: int = 12,
    normalized: bool = True,
):
    """Headline figure: retention vs elapsed time since a skill was learned."""
    if normalized:
        key = "normalized_forgetting_curve"
    else:
        key = "forgetting_curve_by_skills" if axis == "skills" else "forgetting_curve"
    methods = results["methods"]
    order = order or list(methods)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for m in order:
        curve = methods[m].get(key, {})
        xs = sorted(int(k) for k in curve)
        xs = [x for x in xs if x <= max_x]
        if not xs:
            continue
        ys = [curve[str(x)] for x in xs]
        ax.plot(xs, ys, label=methods[m]["label"], **_style(m))
    xlabel = "skills learned since this skill" if axis == "skills" else "nights since learned"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(
        "retention (share of fresh score)" if normalized else "retention (success rate)"
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Retention of earlier skills as the library grows", fontsize=10)
    return _finish(ax, Path(path) if path else None, "center right")


def plot_parameter_growth(
    results: dict,
    path: Optional[str | Path] = None,
    order: Optional[Sequence[str]] = None,
):
    """Secondary figure: stored adapter parameters as the library grows."""
    methods = results["methods"]
    order = order or list(methods)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for m in order:
        points: Dict[int, int] = {}
        for snap in methods[m]["snapshots"]:
            points[snap["library_size"]] = max(
                points.get(snap["library_size"], 0), snap["stored_params"]
            )
        if not points:
            continue
        xs = sorted(points)
        ys = [points[x] / 1e6 for x in xs]
        ax.plot(xs, ys, label=methods[m]["label"], **_style(m))
    ax.set_xlabel("skills in library")
    ax.set_ylabel("stored adapter parameters (M)")
    ax.set_title("Adapter storage: shared cumulative vs per-skill isolation", fontsize=10)
    return _finish(ax, Path(path) if path else None, "upper left")


def plot_token_cost(
    results: dict,
    path: Optional[str | Path] = None,
    order: Optional[Sequence[str]] = None,
):
    """In-context prompt tokens once the library is complete."""
    methods = results["methods"]
    order = [m for m in (order or list(methods)) if methods[m]["snapshots"]]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    labels, values = [], []
    for m in order:
        snaps = methods[m]["snapshots"]
        labels.append(m)
        values.append(snaps[-1]["mean_tokens"])
    bars = ax.bar(labels, values, color=[_style(m)["color"] for m in order], width=0.62)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}", ha="center",
                va="bottom", fontsize=8)
    ax.set_ylabel("mean prompt tokens per attempt")
    ax.set_title("In-context cost at full library size", fontsize=10)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.grid(alpha=0.25, axis="y", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
    return fig


def plot_ablation(
    results: dict,
    path: Optional[str | Path] = None,
    title: str = "Ablation",
    xlabel: str = "variant",
):
    """Final success and oldest-skill retention for one ablation grid."""
    methods = results["methods"]
    keys = list(methods)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    xs = range(len(keys))
    ax.plot(list(xs), [methods[k]["final_success_rate"] for k in keys],
            marker="o", color="#27ae60", label="all skills", linewidth=1.6)
    ax.plot(list(xs), [methods[k]["final_earliest_3"] for k in keys],
            marker="s", color="#c0392b", label="oldest 3 skills", linewidth=1.6)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([methods[k]["label"] if "label" in methods[k] else k for k in keys],
                       rotation=20, fontsize=8, ha="right")
    ax.set_ylabel("success rate")
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title, fontsize=10)
    return _finish(ax, Path(path) if path else None)


def make_all_figures(results: dict, outdir: str | Path, order=None) -> List[Path]:
    outdir = Path(outdir)
    paths = [
        outdir / "fig1_forgetting_curve.png",
        outdir / "fig2_parameter_growth.png",
        outdir / "fig3_token_cost.png",
    ]
    plot_forgetting_curve(results, paths[0], order=order)
    plot_parameter_growth(results, paths[1], order=order)
    plot_token_cost(results, paths[2], order=order)
    plt.close("all")
    return paths
