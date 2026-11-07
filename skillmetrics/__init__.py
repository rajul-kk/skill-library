"""Shareable metrics for skill-library experiments: FLOPs, bootstrap, crossover, floor checks (stdlib only)."""
from .bootstrap import bootstrap, mean_ci, percentile, resample_indices
from .crossover import (
    break_even_analysis,
    break_even_queries,
    crossover_analysis,
    efficiency_ratio,
    interpolated_crossover,
    sustained_crossover,
)
from .floor import FLOOR_ROOT_SUCCESS, HEADROOM_CRAFTED, floor_check, headroom_check
from .flops import (
    SHAPES,
    ModelShape,
    attempt_flops,
    decode_flops,
    lora_train_flops,
    prefill_flops,
    shape_from_hf_config,
)
from .records import aggregate, append_jsonl, attempt_record, finished_runs_only, per_seed, read_jsonl, train_record

__all__ = [
    "SHAPES", "ModelShape", "FLOOR_ROOT_SUCCESS", "HEADROOM_CRAFTED",
    "aggregate", "append_jsonl", "attempt_flops", "attempt_record", "bootstrap",
    "break_even_analysis", "break_even_queries", "crossover_analysis", "decode_flops",
    "efficiency_ratio", "finished_runs_only", "floor_check", "headroom_check", "interpolated_crossover",
    "lora_train_flops", "mean_ci", "per_seed", "percentile", "prefill_flops",
    "read_jsonl", "resample_indices", "shape_from_hf_config", "sustained_crossover",
    "train_record",
]
