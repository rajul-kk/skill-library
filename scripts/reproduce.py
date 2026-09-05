"""Reproduce every CPU result and figure in one command."""
import _paths  # noqa: F401

import json
import shutil
from pathlib import Path

from analysis.metrics import forgetting_curve
from analysis.plots import make_all_figures, plot_ablation
from baselines import MAIN_ORDER, RunConfig, load_results, run_all, run_grid
from baselines.methods import frequency_ablation, replay_size_ablation, strategy_ablation

RESULTS = Path("results/cpu")
SEEDS = tuple(range(20))
GRIDS = {
    "buffer": (replay_size_ablation, "Replay buffer budget"),
    "frequency": (frequency_ablation, "Consolidation frequency K"),
    "strategy": (strategy_ablation, "Replay allocation under a scarce buffer"),
}


def main_experiment() -> dict:
    config = RunConfig(run_id="cpu_main", seeds=SEEDS, eval_repeats=5, checkpoint=True)
    run_all(MAIN_ORDER, config)
    results = load_results(Path("runs/cpu_main/results.json"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    make_all_figures(results, RESULTS / "figures", order=MAIN_ORDER)
    return results


def ablations() -> None:
    for name, (grid, title) in GRIDS.items():
        config = RunConfig(
            run_id=f"ablation_{name}", seeds=SEEDS, eval_repeats=5, checkpoint=False
        )
        run_grid(grid(), config)
        results = load_results(Path(f"runs/ablation_{name}/results.json"))
        out = RESULTS / "ablations"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        plot_ablation(results, out / f"{name}.png", title=title, xlabel=name)


def summary(results: dict) -> str:
    lines = [
        f"{'method':<36}{'final':>8}{'oldest3':>9}{'tokens':>9}{'params':>13}",
        "-" * 75,
    ]
    for key in MAIN_ORDER:
        m = results["methods"][key]
        lines.append(
            f"{m['label'][:35]:<36}{m['final_success_rate']:>8.2f}"
            f"{m['final_earliest_3']:>9.2f}"
            f"{m['tokens']['mean_prompt_tokens']:>9.0f}"
            f"{m['final_stored_params']:>13,}"
        )
    lines.append("")
    lines.append("normalised retention by skills learned since:")
    for key in MAIN_ORDER:
        curve = results["methods"][key]["normalized_forgetting_curve"]
        points = " ".join(f"{x}:{curve[x]:.2f}" for x in sorted(curve, key=int)[:10])
        lines.append(f"  {key:<16}{points}")
    return "\n".join(lines)


if __name__ == "__main__":
    results = main_experiment()
    ablations()
    text = summary(results)
    (RESULTS / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
