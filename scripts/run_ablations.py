"""Replay-budget, consolidation-frequency and replay-strategy ablations."""
import _paths  # noqa: F401

import argparse

from analysis.metrics import forgetting_curve
from baselines import RunConfig, frequency_ablation, replay_size_ablation, run_grid
from baselines.methods import strategy_ablation

GRIDS = {
    "buffer": replay_size_ablation,
    "frequency": frequency_ablation,
    "strategy": strategy_ablation,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=sorted(GRIDS), default="buffer")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--retry-rounds", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    methods = GRIDS[args.grid]()
    cfg = RunConfig(
        run_id=args.run_id or f"ablation_{args.grid}",
        retry_rounds=args.retry_rounds,
        seeds=tuple(args.seeds),
        checkpoint=False,
    )
    results = run_grid(methods, cfg)

    header = f"{'variant':<26}{'final':>7}{'oldest3':>9}{'replay/night':>14}{'params':>12}"
    print(header)
    print("-" * len(header))
    for key, r in results.items():
        nights = r.nights or [{"stats": {"n_replay": 0}}]
        replay = sum(n["stats"]["n_replay"] for n in nights) / len(nights)
        print(
            f"{r.label[:25]:<26}{r.final('success_rate'):>7.2f}"
            f"{r.final('earliest_3'):>9.2f}{replay:>14.1f}"
            f"{int(r.final('stored_params')):>12,}"
        )
    print()
    for key, r in results.items():
        curve = forgetting_curve(r.snapshots, "skills")
        pretty = " ".join(f"{k}:{v:.2f}" for k, v in list(curve.items())[:8])
        print(f"{key:<16} {pretty}")


if __name__ == "__main__":
    main()
