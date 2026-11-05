"""Print the GPU-hour table for the full, reduced and pilot sweep designs."""
import _paths  # noqa: F401

import argparse
from pathlib import Path

from library.retrieval import heuristic_token_count
from skillmetrics.flops import SHAPES
from sweep.budget import design_table, token_profile
from sweep.config import load_sweep_config

ROOT = Path(__file__).resolve().parents[1]


def counter_for(tokenizer_name: str):
    """Real tokenizer when cached locally (Qwen2.5 1.5B and 3B share one), else ~4 chars/token."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        return (lambda t: len(tok(t, add_special_tokens=False)["input_ids"])), tokenizer_name
    except Exception:  # noqa: BLE001 - any failure means fall back to the heuristic
        return heuristic_token_count, "heuristic (4 chars/token)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()

    designs = {}
    for name, file in (("full", "sweep_full.yaml"), ("reduced", "sweep_reduced.yaml"), ("pilot", "sweep_pilot.yaml")):
        cfg = load_sweep_config(ROOT / "configs" / file)
        designs[name] = {"cfg": cfg}

    counter, used = counter_for(args.tokenizer)
    shared = token_profile(designs["full"]["cfg"], counter)  # both Qwen2.5 sizes share one tokenizer
    profiles = {model: shared for d in designs.values() for model in d["cfg"].models}
    print(f"token counts from: {used}")
    prof = next(iter(profiles.values()))
    print("mean prompt tokens per attempt (seed-0 world):")
    for arm, by_n in prof["prompt"].items():
        print(f"  {arm:<14}" + "  ".join(f"N={n}:{v:,.0f}" for n, v in by_n.items()))
    print(f"  train example ~{prof['train_example']:.0f} tokens, program ~{prof['gen']:.0f} tokens")
    print()

    rows = design_table(designs, profiles, SHAPES)
    print("| design | model | arms | milestones | seeds | h/seed | train h | prefill h | decode h | total GPU-h |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    totals = {}
    for r in rows:
        print(f"| {r['design']} | {r['model'].split('/')[-1]} | {r['arms']} | {r['milestones']} | {r['seeds']} | "
              f"{r['hours_per_seed']:.2f} | {r['train_h']:.1f} | {r['prefill_h']:.1f} | {r['decode_h']:.1f} | {r['hours']:.1f} |")
        totals[r["design"]] = totals.get(r["design"], 0.0) + r["hours"]
    print()
    for name, h in totals.items():
        print(f"{name}: {h:.1f} GPU-hours total, {h / 30:.1f} Kaggle weeks at 30 h/week")


if __name__ == "__main__":
    main()
