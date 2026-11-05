"""GPU-hour estimates for a sweep design from the FLOP model plus stated throughput assumptions."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from env.procgen import generate_world
from skillmetrics.flops import ModelShape, attempt_flops, lora_train_flops

from .config import ARMS, COVER, SweepConfig
from .prompts import build_prompt
from .runner import eval_set


@dataclass(frozen=True)
class Throughput:
    """Assumed T4 throughput; the pilot notebook measures the real numbers."""

    matmul_tflops: float = 10.0          # effective fp16 TFLOP/s for prefill and training
    decode_step_ms: float = 35.0         # one batched greedy step
    decode_steps_per_batch: int = 64     # a batch runs until its longest program stops
    load_hours: float = 0.15             # model load, warm-up, record writing per (model, seed)


THROUGHPUT: Dict[str, Throughput] = {
    "Qwen/Qwen2.5-1.5B-Instruct": Throughput(decode_step_ms=35.0),
    "Qwen/Qwen2.5-3B-Instruct": Throughput(matmul_tflops=10.0, decode_step_ms=50.0, load_hours=0.2),
}


def token_profile(cfg: SweepConfig, counter: Callable[[str], int], seed: int = 0) -> Dict:
    """Prompt tokens per arm and milestone, and the training tokens per example, from one world."""
    world = generate_world(cfg.n_max, seed)
    prompts: Dict[str, Dict[int, float]] = {}
    for arm in cfg.arm_list():
        prompts[arm.key] = {}
        for n in cfg.milestones:
            lib = [] if arm.prompt == "none" else world.prefix(n)
            targets = eval_set(world, n, seed, cfg)
            lengths = [counter(build_prompt(world, arm.prompt, s, lib, cfg.topk)) for s in targets]
            prompts[arm.key][n] = sum(lengths) / len(lengths)
    lib = world.prefix(cfg.n_max)
    example = [counter(build_prompt(world, "pointers", s, lib, cfg.topk) + s.program) for s in lib[:50]]
    gen = [counter(s.program) + 2 for s in lib]
    return {"prompt": prompts, "train_example": sum(example) / len(example), "gen": sum(gen) / len(gen)}


def train_examples_per_night(arm_key: str, cfg: SweepConfig) -> List[int]:
    """Examples (new + replay) at each night along the trajectory."""
    arm = ARMS[arm_key]
    out, consolidated, pending = [], 0, 0
    for k in range(1, cfg.n_max + 1):
        pending += 1
        if pending >= cfg.night_every or k in cfg.milestones:
            if arm.weights == "isolated":
                replay = 0
            else:
                cap = consolidated if arm.replay_cap == COVER else min(arm.replay_cap, consolidated)
                replay = cap
            out.append(pending + replay)
            consolidated += pending
            pending = 0
    return out


def eval_counts(cfg: SweepConfig) -> Dict[int, int]:
    return {n: (n if n <= cfg.eval_full_up_to else 5 * (cfg.eval_max // 5)) for n in cfg.milestones}


def estimate(cfg: SweepConfig, shape: ModelShape, profile: Dict, tp: Optional[Throughput] = None) -> Dict:
    """Hours for one (model, seed) trajectory over the configured arms, split by phase."""
    tp = tp or THROUGHPUT.get(shape.name, Throughput())
    lora = shape.lora_params(cfg.train.rank, cfg.train.target_modules)
    counts = eval_counts(cfg)
    per_arm = {}
    for arm in cfg.arm_list():
        flops_train = 0.0
        if arm.trains:
            t = profile["train_example"]
            n_ex = sum(train_examples_per_night(arm.key, cfg)) * cfg.train.epochs
            flops_train = lora_train_flops(shape, n_ex * t, n_ex * t * t, lora, cfg.train.grad_checkpointing)
        flops_prefill = 0.0
        decode_s = 0.0
        for n in cfg.milestones:
            L = profile["prompt"][arm.key][n]
            q = counts[n]
            a = attempt_flops(shape, int(L), int(profile["gen"]), lora if arm.trains else 0)
            flops_prefill += q * (a["prefill"] + a["lora"])
            per_batch = max(1, min(cfg.decode.batch_size, cfg.decode.max_batch_tokens // int(L + cfg.decode.max_new_tokens)))
            decode_s += math.ceil(q / per_batch) * tp.decode_steps_per_batch * tp.decode_step_ms / 1000.0
        hours = {
            "train": flops_train / (tp.matmul_tflops * 1e12) / 3600,
            "prefill": flops_prefill / (tp.matmul_tflops * 1e12) / 3600,
            "decode": decode_s / 3600,
        }
        hours["total"] = sum(hours.values())
        per_arm[arm.key] = hours
    total = sum(h["total"] for h in per_arm.values()) + tp.load_hours
    return {"per_arm": per_arm, "per_seed_hours": total}


def design_table(designs: Dict[str, Dict], profiles: Dict[str, Dict], shapes: Dict[str, ModelShape]) -> List[Dict]:
    """designs: name -> {"cfg": SweepConfig}; one row per (design, model)."""
    rows = []
    for name, d in designs.items():
        cfg = d["cfg"]
        for model in cfg.models:
            est = estimate(cfg, shapes[model], profiles[model])
            seeds = len(cfg.seeds_for(model))
            rows.append({
                "design": name, "model": model, "arms": len(cfg.arms), "milestones": list(cfg.milestones),
                "seeds": seeds, "hours_per_seed": est["per_seed_hours"],
                "hours": est["per_seed_hours"] * seeds,
                "train_h": sum(h["train"] for h in est["per_arm"].values()) * seeds,
                "prefill_h": sum(h["prefill"] for h in est["per_arm"].values()) * seeds,
                "decode_h": sum(h["decode"] for h in est["per_arm"].values()) * seeds,
            })
    return rows
