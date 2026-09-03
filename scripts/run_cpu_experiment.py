"""Run the four methods on CPU with the simulated backend and print a summary."""
import _paths  # noqa: F401

import argparse
import json

from analysis.metrics import forgetting_curve
from baselines import MAIN_ORDER, RunConfig, run_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="cpu_dev")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--eval-repeats", type=int, default=3)
    ap.add_argument("--methods", nargs="*", default=MAIN_ORDER)
    ap.add_argument("--no-checkpoint", action="store_true")
    args = ap.parse_args()

    cfg = RunConfig(
        run_id=args.run_id,
        passes=args.passes,
        seeds=tuple(args.seeds),
        eval_repeats=args.eval_repeats,
        checkpoint=not args.no_checkpoint,
    )
    results = run_all(args.methods, cfg)

    header = f"{'method':<34}{'final':>7}{'oldest3':>9}{'tokens':>9}{'params':>12}"
    print(header)
    print("-" * len(header))
    for key in args.methods:
        r = results[key]
        print(
            f"{r.label[:33]:<34}{r.final('success_rate'):>7.2f}"
            f"{r.final('earliest_3'):>9.2f}"
            f"{r.tokens['mean_prompt_tokens']:>9.0f}{int(r.final('stored_params')):>12,}"
        )
    print()
    for key in args.methods:
        curve = forgetting_curve(results[key].snapshots, "skills")
        pretty = " ".join(f"{k}:{v:.2f}" for k, v in list(curve.items())[:8])
        print(f"{key:<16} retention by skills-since-learned  {pretty}")


if __name__ == "__main__":
    main()
