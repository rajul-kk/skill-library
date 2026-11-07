"""Driving a GPU run from a config file, one resumable night at a time."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from agent.lm_policy import DecodeConfig, LMPolicy
from agent.loop import LoopConfig, VoyagerLoop
from analysis.metrics import evaluate_library
from env.tasks import curriculum_order
from library.retrieval import RetrievalConfig, hf_token_counter
from library.store import SkillLibrary
from sleep.checkpoint import Manifest, latest_night, load_night, save_night
from sleep.consolidate import Consolidator, SleepConfig
from sleep.lora import (
    IsolatedLoRABackend,
    LoRAAdapterBackend,
    LoRAConfig,
    load_base_model,
)
from sleep.replay import ReplayConfig


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_lora_config(cfg: dict) -> LoRAConfig:
    lora = cfg.get("lora", {})
    return LoRAConfig(
        rank=lora.get("rank", 16),
        alpha=lora.get("alpha", 32),
        dropout=lora.get("dropout", 0.05),
        target_modules=tuple(lora.get("target_modules", LoRAConfig.target_modules)),
        learning_rate=float(lora.get("learning_rate", 1e-4)),
        batch_size=lora.get("batch_size", 4),
        steps_per_night=lora.get("steps_per_night", 300),
        max_length=lora.get("max_length", 512),
    )


def build_sleep_config(cfg: dict, replay_override: Optional[dict] = None) -> SleepConfig:
    sleep_cfg = cfg.get("sleep", {})
    replay = dict(sleep_cfg.get("replay", {}))
    replay.update(replay_override or {})
    return SleepConfig(
        every_k_tasks=sleep_cfg.get("every_k_tasks", 3),
        drop_full_text_after_consolidation=sleep_cfg.get(
            "drop_full_text_after_consolidation", True
        ),
        replay=ReplayConfig(
            enabled=replay.get("enabled", True),
            per_skill=replay.get("per_skill", 2),
            max_total=replay.get("max_total", 24),
            strategy=replay.get("strategy", "round_robin"),
        ),
    )


METHOD_PRESETS: Dict[str, Dict[str, Any]] = {
    "ours": {"isolated": False, "replay": {"enabled": True}},
    "seq_no_replay": {
        "isolated": False,
        "replay": {"enabled": False, "per_skill": 0, "max_total": 0},
    },
    "isolated": {
        "isolated": True,
        "replay": {"enabled": False, "per_skill": 0, "max_total": 0},
    },
    "in_context": {"isolated": False, "replay": {"enabled": False}, "no_sleep": True},
}


class NightBudgetReached(Exception):
    """Internal signal that this session has done as many nights as asked."""


@dataclass
class GPURun:
    """A resumable GPU experiment for one method."""

    config: dict
    method: str = "ours"
    model: Any = None
    tokenizer: Any = None
    policy_factory: Any = None  # seam for tests and for scripted-policy runs

    def __post_init__(self) -> None:
        preset = METHOD_PRESETS[self.method]
        model_cfg = self.config.get("model", {})
        device = model_cfg.get("device", "cuda")
        if self.model is None:
            self.model, self.tokenizer = load_base_model(
                model_cfg.get("name", "Qwen/Qwen2.5-1.5B-Instruct"),
                device,
                model_cfg.get("dtype", "bfloat16"),
            )
        lora_config = build_lora_config(self.config)

        backend_cls = IsolatedLoRABackend if preset["isolated"] else LoRAAdapterBackend
        self.backend = backend_cls(self.model, self.tokenizer, lora_config, device)

        sleep_config = build_sleep_config(self.config, preset.get("replay"))
        if preset.get("no_sleep"):
            sleep_config.enabled = False
        self.consolidator = Consolidator(self.backend, sleep_config)

        loop_cfg = self.config.get("loop", {})
        self.loop_config = LoopConfig(
            max_attempts=loop_cfg.get("max_attempts", 4),
            retrieval=RetrievalConfig(
                max_skills=loop_cfg.get("max_skills_in_context", 12),
                serve_full_text_always=bool(preset.get("no_sleep")),
            ),
            token_counter=hf_token_counter(self.tokenizer),
        )
        decode_cfg = self.config.get("decode", {})
        self.decode_config = DecodeConfig(
            max_new_tokens=decode_cfg.get("max_new_tokens", 128),
            temperature=decode_cfg.get("temperature", 0.7),
            top_p=decode_cfg.get("top_p", 0.9),
            max_program_lines=decode_cfg.get("max_program_lines", 24),
        )
        run_cfg = self.config.get("run", {})
        self.root = (
            Path(run_cfg.get("root", "runs"))
            / run_cfg.get("run_id", "gpu")
            / self.method
        )
        self.eval_repeats = run_cfg.get("eval_repeats", 1)
        self.seed = (run_cfg.get("seeds") or [0])[0]
        self.snapshots: List[dict] = []
        self.library = SkillLibrary()
        self.next_episode = 0

    def resume(self) -> Optional[int]:
        """Reload the newest night, if there is one. Returns the night number."""
        night = latest_night(self.root)
        if night is None:
            return None
        self.library, manifest = load_night(self.root, night, self.consolidator)
        self.next_episode = manifest.episode + 1
        snapshot_file = self.root / "snapshots.json"
        if snapshot_file.exists():
            # snapshots are written before the night's checkpoint commits; drop any whose checkpoint never landed
            snapshots = json.loads(snapshot_file.read_text(encoding="utf-8"))
            self.snapshots = [s for s in snapshots if s["night"] <= night]
        return night

    def _policy(self, seed: int):
        if self.policy_factory is not None:
            return self.policy_factory(seed)
        return LMPolicy(self.backend, self.decode_config, seed=seed)

    def _on_night(self, record, library: SkillLibrary) -> None:
        snapshot = evaluate_library(
            library,
            self._policy,
            self.loop_config,
            night=record.night,
            episode=record.episode,
            stored_params=self.backend.stored_parameters(),
            repeats=self.eval_repeats,
            seed=self.seed,
        )
        self.snapshots.append(snapshot.to_dict())
        # snapshot first, then the checkpoint (whose manifest is the commit marker): a crash in between leaves an
        # extra snapshot that resume() drops, never a checkpointed night without its snapshot
        tmp = self.root / "snapshots.json.tmp"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(self.snapshots, indent=2), encoding="utf-8")
        tmp.replace(self.root / "snapshots.json")
        save_night(
            self.root,
            record.night,
            library,
            self.consolidator,
            Manifest(
                run_id=self.config.get("run", {}).get("run_id", "gpu"),
                method=self.method,
                night=record.night,
                episode=record.episode,
                library_size=len(library),
                stored_params=self.backend.stored_parameters(),
                backend="lora",
                seed=self.seed,
                extra={"loss": record.stats.loss, "seconds": record.stats.seconds},
            ),
        )

    def run(self, max_nights: Optional[int] = None) -> VoyagerLoop:
        """Run the curriculum, optionally stopping after `max_nights` nights."""
        start_night = self.consolidator.night_count
        loop = VoyagerLoop(
            self._policy(self.seed), self.consolidator, self.loop_config, self.library,
            first_episode=self.next_episode,
        )

        last_night_episode = [-1]

        def on_night(record, library):
            self._on_night(record, library)
            last_night_episode[0] = record.episode
            if max_nights is not None and record.night - start_night >= max_nights:
                raise NightBudgetReached()

        loop.on_night = on_night
        remaining = [t for t in curriculum_order() if t.name not in self.library]
        retry_rounds = self.config.get("loop", {}).get("retry_rounds", 2)
        try:
            loop.run_curriculum(remaining, retry_rounds=retry_rounds)
            loop.finish()
        except NightBudgetReached:
            pass
        self.library = loop.library
        # NightBudgetReached fires before the night's episode is appended to loop.episodes
        self.next_episode = max(loop.next_episode, last_night_episode[0] + 1)
        return loop
