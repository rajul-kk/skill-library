"""Regenerate every figure from a results.json produced by the runner."""
import _paths  # noqa: F401

import argparse
from pathlib import Path

from analysis.plots import make_all_figures, plot_ablation
from baselines import MAIN_ORDER, load_results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/cpu_dev/results.json")
    ap.add_argument("--outdir", default="runs/cpu_dev/figures")
    ap.add_argument("--ablation", action="store_true", help="render as an ablation grid")
    ap.add_argument("--title", default="Ablation")
    args = ap.parse_args()

    results = load_results(args.results)
    outdir = Path(args.outdir)
    if args.ablation:
        path = outdir / "ablation.png"
        plot_ablation(results, path, title=args.title)
        print(path)
        return
    order = [m for m in MAIN_ORDER if m in results["methods"]]
    for p in make_all_figures(results, outdir, order=order):
        print(p)


if __name__ == "__main__":
    main()
