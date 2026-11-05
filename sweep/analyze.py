"""Pre-registered analysis of sweep records: cell table, floor checks, crossovers, break-even."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from skillmetrics import SHAPES, ModelShape, aggregate, read_jsonl
from skillmetrics.bootstrap import mean_ci
from skillmetrics.crossover import INF, break_even_analysis, crossover_analysis
from skillmetrics.floor import floor_check, headroom_check

PRIMARY_OURS = "ours_b24"
COMPARATORS = ("ic_full", "ic_topk")
PRIMARY_HORIZON = 1e4
PRIMARY_ALPHA = 0.05 / 4  # Bonferroni over 2 comparators x 2 cost metrics
HORIZONS = (1e2, 1e3, 1e4, 1e5, INF)
SECONDARY_OURS = ("ours_cover", "ours_b96", "isolated", "seq_no_replay")


def shape_for(model: str, shapes: Optional[Dict[str, ModelShape]] = None) -> ModelShape:
    """Shape for a model label; 'sim:<name>' runs are costed as <name>."""
    shapes = shapes or SHAPES
    base = model.split(":", 1)[1] if model.startswith("sim:") else model
    if base not in shapes:
        raise KeyError(f"no shape for {model!r}; add it to skillmetrics.flops.SHAPES")
    return shapes[base]


def cells_from(records: List[dict]) -> Dict:
    models = {r["model"] for r in records if "model" in r}
    return aggregate(records, {m: shape_for(m) for m in models})


def by_milestone(cells: Dict, model: str, arm: str) -> Dict[int, Dict[int, dict]]:
    out: Dict[int, Dict[int, dict]] = {}
    for (m, a, n, s), v in cells.items():
        if m == model and a == arm:
            out.setdefault(n, {})[s] = v
    return out


def checks(cells: Dict, model: str) -> Dict:
    """Floor check per milestone and the model-level headroom check at the smallest milestone."""
    no_skill = by_milestone(cells, model, "no_skill")
    out = {"floor": {}, "headroom": None}
    for n, per_seed in sorted(no_skill.items()):
        out["floor"][n] = floor_check(list(per_seed.values()))
    if no_skill:
        n0 = min(no_skill)
        ic = {a: list(by_milestone(cells, model, a).get(n0, {}).values()) for a in COMPARATORS}
        ic = {a: v for a, v in ic.items() if v}
        out["headroom"] = {"milestone": n0, **headroom_check(ic, list(no_skill[n0].values()))}
    return out


def table(cells: Dict, model: str, n_boot: int) -> List[dict]:
    """Per (arm, milestone): success with a seed-bootstrap interval, tokens, FLOPs."""
    rows = []
    arms = sorted({a for (m, a, _, _) in cells if m == model})
    for arm in arms:
        for n, per_seed in sorted(by_milestone(cells, model, arm).items()):
            vals = list(per_seed.values())
            ci = mean_ci([v["success"] for v in vals], n_boot=n_boot)
            rows.append({
                "arm": arm, "milestone": n, "seeds": len(vals),
                "success": ci["point"], "success_lo": ci["lo"], "success_hi": ci["hi"],
                "prompt_tokens": sum(v["prompt_tokens"] for v in vals) / len(vals),
                "inf_tokens": sum(v["inf_tokens"] for v in vals) / len(vals),
                "inf_gflops": sum(v["inf_flops"] for v in vals) / len(vals) / 1e9,
                "train_pflops_cum": sum(v["train_flops_cum"] for v in vals) / len(vals) / 1e15,
            })
    return rows


def contrasts(cells: Dict, model: str, ours_arms: Iterable[str], allowed: List[int], n_boot: int, alpha: float = 0.05) -> Dict:
    out = {}
    for ours in ours_arms:
        o = {n: v for n, v in by_milestone(cells, model, ours).items() if n in allowed}
        if not o:
            continue
        for comp in COMPARATORS:
            c = {n: v for n, v in by_milestone(cells, model, comp).items() if n in allowed}
            if not c:
                continue
            key = f"{ours}_vs_{comp}"
            out[key] = {
                "tokens": crossover_analysis(o, c, "tokens", n_boot=n_boot, alpha=alpha),
                "flops": {
                    ("inf" if h == INF else f"{h:g}"): crossover_analysis(o, c, "flops", horizon=h, n_boot=n_boot, alpha=alpha)
                    for h in HORIZONS
                },
                "break_even_queries": break_even_analysis(o, c, n_boot=n_boot),
            }
    return out


def analyze(records: List[dict], n_boot: int = 10_000) -> Dict:
    cells = cells_from(records)
    report = {"models": {}, "primary": {"ours": PRIMARY_OURS, "comparators": list(COMPARATORS), "horizon": PRIMARY_HORIZON, "alpha": PRIMARY_ALPHA}}
    for model in sorted({k[0] for k in cells}):
        chk = checks(cells, model)
        allowed = [n for n, f in chk["floor"].items() if f["passed"]]
        dropped_model = chk["headroom"] is not None and not chk["headroom"]["passed"]
        report["models"][model] = {
            "checks": chk,
            "milestones_in_analysis": [] if dropped_model else allowed,
            "model_dropped": dropped_model,
            "table": table(cells, model, n_boot=min(n_boot, 2000)),
            "primary": {} if dropped_model else contrasts(cells, model, [PRIMARY_OURS], allowed, n_boot, PRIMARY_ALPHA),
            "secondary": {} if dropped_model else contrasts(cells, model, SECONDARY_OURS, allowed, n_boot),
            "exploratory_all_milestones": contrasts(cells, model, [PRIMARY_OURS], sorted({k[2] for k in cells if k[0] == model}), min(n_boot, 2000)),
        }
    return report


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    if x == INF:
        return "inf"
    return f"{x:.3g}"


def summary_text(report: Dict) -> str:
    lines = []
    for model, m in report["models"].items():
        lines.append(f"== {model}")
        h = m["checks"]["headroom"]
        floors = ", ".join(f"N={n}:{'ok' if f['passed'] else 'FLOOR'}({_fmt(f['no_skill_root_success'])})" for n, f in m["checks"]["floor"].items())
        lines.append(f"floor (no_skill root success >= 0.30): {floors}")
        if h:
            lines.append(f"headroom at N={h['milestone']}: {'ok' if h['passed'] else 'FAILED'} best={h['best_arm']} {h['in_context_crafted_success']} vs no_skill {_fmt(h['no_skill_crafted_success'])}")
        lines.append(f"{'arm':<14}{'N':>5}{'seeds':>6}{'success':>9}{'95% CI':>16}{'tokens':>9}{'GFLOP/q':>10}{'train PF':>10}")
        for r in m["table"]:
            ci = f"[{r['success_lo']:.2f},{r['success_hi']:.2f}]"
            lines.append(f"{r['arm']:<14}{r['milestone']:>5}{r['seeds']:>6}{r['success']:>9.2f}{ci:>16}{r['inf_tokens']:>9.0f}{r['inf_gflops']:>10.1f}{r['train_pflops_cum']:>10.3f}")
        for group in ("primary", "secondary"):
            for key, c in m[group].items():
                t = c["tokens"]
                f = c["flops"][f"{PRIMARY_HORIZON:g}"]
                lines.append(f"[{group}] {key}: per-token N*={t['n_star']} (interp {_fmt(t['n_star_interpolated'])}, boot {t['n_star_distribution']}, CI-wins at {t['wins_at']}); "
                             f"per-FLOP(H=1e4) N*={f['n_star']} (boot {f['n_star_distribution']}, CI-wins at {f['wins_at']}); interval level {1 - t['alpha']:.4g}")
                for n, r in t["ratio"].items():
                    fr = f["ratio"][n]
                    be = c["break_even_queries"].get(n, {})
                    lines.append(f"    N={n:<4} token ratio {_fmt(r['point'])} [{_fmt(r['lo'])},{_fmt(r['hi'])}]  FLOP ratio {_fmt(fr['point'])} [{_fmt(fr['lo'])},{_fmt(fr['hi'])}]  break-even H* {_fmt(be.get('point', float('nan')))}")
        lines.append("")
    return "\n".join(lines)


def plot(report: Dict, out_dir: Path) -> List[Path]:
    """Success and efficiency-ratio curves against library size, one figure per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for model, m in report["models"].items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        arms = sorted({r["arm"] for r in m["table"]})
        for arm in arms:
            rows = [r for r in m["table"] if r["arm"] == arm]
            xs = [r["milestone"] for r in rows]
            axes[0].plot(xs, [r["success"] for r in rows], marker="o", label=arm)
            axes[0].fill_between(xs, [r["success_lo"] for r in rows], [r["success_hi"] for r in rows], alpha=0.15)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("library size N")
        axes[0].set_ylabel("success")
        axes[0].legend(fontsize=7)
        for key, c in m["exploratory_all_milestones"].items():
            for label, res in (("per token", c["tokens"]), ("per FLOP, H=1e4", c["flops"][f"{PRIMARY_HORIZON:g}"])):
                xs = list(res["ratio"])
                axes[1].plot(xs, [res["ratio"][n]["point"] for n in xs], marker="o", label=f"{key} {label}")
                axes[1].fill_between(xs, [res["ratio"][n]["lo"] for n in xs], [res["ratio"][n]["hi"] for n in xs], alpha=0.15)
        axes[1].axhline(1.0, color="grey", lw=0.8)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("library size N")
        axes[1].set_ylabel("efficiency ratio (ours / in-context)")
        axes[1].legend(fontsize=6)
        fig.suptitle(model)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"sweep_{model.replace('/', '_').replace(':', '_')}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return str(obj)
    return obj


def run_analysis(records_path: Path, out_dir: Path, n_boot: int = 10_000, figures: bool = True) -> Dict:
    report = analyze(read_jsonl(records_path), n_boot=n_boot)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    (out_dir / "summary.txt").write_text(summary_text(report) + "\n", encoding="utf-8")
    if figures:
        plot(report, out_dir)
    return report
