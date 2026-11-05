"""Run a library-size sweep from a config; resumable, one records.jsonl per config."""
import _paths  # noqa: F401

import argparse
import time
from pathlib import Path

from sweep.backends import HFSweepBackend, SimulatedSweepBackend
from sweep.config import load_sweep_config
from sweep.runner import run_model_seed, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seeds", type=int, nargs="*", help="override the config's seeds")
    ap.add_argument("--models", nargs="*", help="override the config's models")
    ap.add_argument("--out-dir", help="override the config's out_dir")
    ap.add_argument("--max-hours", type=float, default=None, help="stop starting new seeds after this long")
    args = ap.parse_args()

    cfg = load_sweep_config(args.config)
    if args.seeds:
        cfg.seeds = args.seeds
        cfg.seeds_by_model = {}
    if args.models:
        cfg.models = args.models
    out_dir = Path(args.out_dir or cfg.out_dir)
    records = out_dir / "records.jsonl"
    write_manifest(cfg, out_dir, {"config_file": str(args.config)})

    started = time.time()
    for model in cfg.models:
        hf = None
        for seed in cfg.seeds_for(model):
            if args.max_hours is not None and (time.time() - started) / 3600 > args.max_hours:
                print("time budget reached; rerun the same command to resume")
                return
            if cfg.backend == "simulated":
                backend = SimulatedSweepBackend(cfg, seed)
            else:
                if hf is None:
                    hf = HFSweepBackend(cfg, model, seed)
                hf.seed = seed
                backend = hf
            run_model_seed(cfg, model, seed, backend, records)
        del hf
    print(f"records: {records}")


if __name__ == "__main__":
    main()
