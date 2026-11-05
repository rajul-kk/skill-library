"""Apply the pre-registered sweep analysis to a records.jsonl and write summary, json and figures."""
import _paths  # noqa: F401

import argparse
from pathlib import Path

from sweep.analyze import run_analysis, summary_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    records = Path(args.records)
    out_dir = Path(args.out_dir) if args.out_dir else records.parent / "analysis"
    report = run_analysis(records, out_dir, n_boot=args.n_boot, figures=not args.no_figures)
    print(summary_text(report))
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
