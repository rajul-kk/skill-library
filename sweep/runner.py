"""Run one (model, seed) trajectory to the largest milestone for every arm, writing records.jsonl."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from agent.lm_policy import extract_program
from env.dsl import DSLError
from env.procgen import GenSkill, WorldSpec, generate_world, run_program
from env.world import ExecutionError
from skillmetrics.records import append_jsonl, attempt_record, read_jsonl, train_record
from sleep.backend import Example
from sleep.consolidate import Consolidator
from sleep.replay import ReplayBuffer, ReplayConfig

from .backends import Query
from .config import COVER, Arm, SweepConfig
from .prompts import build_prompt, in_context_names, names

CUMULATIVE = "cumulative"


def eval_set(world: WorldSpec, n: int, seed: int, cfg: SweepConfig) -> List[GenSkill]:
    """All skills up to eval_full_up_to, else a seeded sample stratified by age quintile."""
    lib = world.prefix(n)
    if n <= cfg.eval_full_up_to:
        return list(lib)
    rng = random.Random(f"eval-{seed}-{n}")
    per = cfg.eval_max // 5
    picked: List[GenSkill] = []
    for q in range(5):
        chunk = lib[q * n // 5 : (q + 1) * n // 5]
        picked.extend(rng.sample(chunk, min(per, len(chunk))))
    return sorted(picked, key=lambda s: s.index)


def replay_config(arm: Arm, seed: int) -> ReplayConfig:
    cap = arm.replay_cap
    if cap == 0:
        return ReplayConfig(enabled=False, per_skill=0, max_total=0, seed=seed)
    return ReplayConfig(enabled=True, per_skill=1, max_total=10**9 if cap == COVER else cap, strategy="round_robin", seed=seed)


class ArmRun:
    """State of one arm along one trajectory."""

    def __init__(self, arm: Arm, cfg: SweepConfig, world: WorldSpec, backend, model: str, seed: int, out: Path) -> None:
        self.arm, self.cfg, self.world, self.backend = arm, cfg, world, backend
        self.model, self.seed, self.out = model, seed, out
        self.buffer = ReplayBuffer(replay_config(arm, seed))
        self.consolidated: List[GenSkill] = []
        self.pending: List[GenSkill] = []
        self.block_of: Dict[str, str] = {}
        self.night = 0
        self.seconds = {"train": 0.0, "eval": 0.0}

    def common(self) -> dict:
        return {"model": self.model, "seed": self.seed, "arm": self.arm.key}

    def library(self, k: int) -> List[GenSkill]:
        return [] if self.arm.prompt == "none" else self.world.prefix(k)

    def run_night(self, k: int) -> None:
        """Consolidate pending skills: cumulative arms add capped replay, isolated arms open a new block."""
        if not self.pending:
            return
        self.night += 1
        lib = self.world.prefix(k)
        new = [
            Example(s.name, build_prompt(self.world, self.arm.prompt, s, lib, self.cfg.topk), s.program, self.night)
            for s in self.pending
        ]
        if self.arm.weights == CUMULATIVE:
            adapter = CUMULATIVE
            replay = self.buffer.sample(exclude=[s.name for s in self.pending])
            batch = Consolidator._interleave(new, replay)
        else:
            adapter = f"blk_{self.night}"
            replay, batch = [], new
            for s in self.pending:
                self.block_of[s.name] = adapter
        started = time.time()
        stats = self.backend.train(batch, adapter, self.night)
        self.seconds["train"] += time.time() - started
        for s, e in zip(self.pending, new):
            self.buffer.add(e, tier=s.task.tier)
        self.consolidated.extend(self.pending)
        self.pending = []
        append_jsonl(self.out, [train_record(
            **self.common(), night=self.night, library_size=k, n_new=len(new), n_replay=len(replay),
            train_tokens=stats.train_tokens, train_tokens_sq=stats.train_tokens_sq,
            padded_tokens=stats.padded_tokens, steps=stats.steps, loss=stats.loss, seconds=stats.seconds,
            nan=stats.nan, lora_params=self.backend.lora_params, grad_checkpointing=self.cfg.train.grad_checkpointing,
        )])

    def adapter_for(self, skill: GenSkill) -> Optional[str]:
        if self.arm.weights == CUMULATIVE:
            return CUMULATIVE
        if self.arm.weights == "isolated":
            return self.block_of.get(skill.name)
        return None

    def evaluate(self, k: int) -> List[dict]:
        lib = self.library(k)
        known = names(lib)
        targets = eval_set(self.world, k, self.seed, self.cfg)
        queries = [
            Query(
                prompt=build_prompt(self.world, self.arm.prompt, s, lib, self.cfg.topk),
                skill=s,
                adapter=self.adapter_for(s),
                context_full=in_context_names(self.arm.prompt, s, lib, self.cfg.topk),
            )
            for s in targets
        ]
        started = time.time()
        gens = self.backend.generate(queries)
        self.seconds["eval"] += time.time() - started
        rows = []
        for q, g in zip(queries, gens):
            program = extract_program(g.text)
            ok, error = False, ""
            try:
                world = run_program(self.world, program or "goto(nowhere)", known, banned=(q.skill.name,))
                ok = q.skill.task.check(world)
                error = "" if ok else "goal not met"
            except (ExecutionError, DSLError) as exc:
                error = str(exc)[:120]
            rows.append(attempt_record(
                **self.common(), milestone=k, skill=q.skill.name, is_root=q.skill.is_root,
                age_rank=k - q.skill.index, success=int(ok), prompt_tokens=self.backend.count_tokens(q.prompt),
                gen_tokens=g.gen_tokens, lora_params=self.backend.lora_params if q.adapter else 0,
                in_context=q.skill.name in q.context_full, program=program, error=error,
            ))
        append_jsonl(self.out, rows)
        return rows


def finished_arms(out: Path, model: str, seed: int) -> set:
    return {
        r["arm"] for r in read_jsonl(out)
        if r.get("kind") == "arm_done" and r["model"] == model and int(r["seed"]) == seed
    }


def run_model_seed(cfg: SweepConfig, model: str, seed: int, backend, out: Path, log=print) -> Dict[str, dict]:
    """Every configured arm for one (model, seed); arms already marked done in out are skipped."""
    world = generate_world(cfg.n_max, seed)
    done = finished_arms(out, model, seed)
    summary: Dict[str, dict] = {}
    milestones = set(cfg.milestones)
    for arm in cfg.arm_list():
        if arm.key in done:
            log(f"[{model} seed={seed}] {arm.key}: already done, skipping")
            continue
        started = time.time()
        backend.start_arm(arm)
        run = ArmRun(arm, cfg, world, backend, model, seed, out)
        per_milestone = {}
        for k, skill in enumerate(world.prefix(cfg.n_max), start=1):
            if arm.trains:
                run.pending.append(skill)
                if len(run.pending) >= cfg.night_every or k in milestones:
                    run.run_night(k)
            if k in milestones:
                rows = run.evaluate(k)
                per_milestone[k] = sum(r["success"] for r in rows) / len(rows)
                log(f"[{model} seed={seed}] {arm.key} N={k}: success {per_milestone[k]:.2f} on {len(rows)}")
        stored = backend_stored(arm, run, backend)
        backend.end_arm()
        record = {
            "kind": "arm_done", "model": model, "seed": seed, "arm": arm.key, "backend": backend.name,
            "nights": run.night, "stored_lora_params": stored,
            "seconds_total": time.time() - started, "seconds": run.seconds, "success": per_milestone,
        }
        append_jsonl(out, [record])
        summary[arm.key] = record
    return summary


def backend_stored(arm: Arm, run: ArmRun, backend) -> int:
    """Adapter parameters that must be kept: one adapter, or one per block."""
    if arm.weights == CUMULATIVE:
        return backend.lora_params
    if arm.weights == "isolated":
        return backend.lora_params * len(set(run.block_of.values()))
    return 0


def write_manifest(cfg: SweepConfig, out_dir: Path, extra: Optional[dict] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps({"config": cfg.to_dict(), **(extra or {})}, indent=2), encoding="utf-8")
    return path
