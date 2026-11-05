"""Sweep configuration: arms, milestones, training and decoding settings, loaded from yaml."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

COVER = -1  # replay cap meaning "the whole library, every night"


@dataclass(frozen=True)
class Arm:
    """One arm of the sweep."""

    key: str
    prompt: str            # "none", "full", "topk", "pointers"
    weights: str           # "base", "cumulative", "isolated"
    replay_cap: int = 0    # cumulative arms only; COVER covers the library

    @property
    def trains(self) -> bool:
        return self.weights != "base"


ARMS: Dict[str, Arm] = {
    a.key: a
    for a in [
        Arm("no_skill", "none", "base"),
        Arm("ic_full", "full", "base"),
        Arm("ic_topk", "topk", "base"),
        Arm("ours_b24", "pointers", "cumulative", 24),
        Arm("ours_b96", "pointers", "cumulative", 96),
        Arm("ours_cover", "pointers", "cumulative", COVER),
        Arm("seq_no_replay", "pointers", "cumulative", 0),
        Arm("isolated", "pointers", "isolated"),
    ]
}


@dataclass
class TrainConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    learning_rate: float = 2e-4
    epochs: int = 4
    batch_size: int = 8
    max_length: int = 1024
    grad_clip: float = 1.0
    grad_checkpointing: bool = False


@dataclass
class DecodeConfig:
    max_new_tokens: int = 96
    batch_size: int = 16
    max_batch_tokens: int = 32_768


@dataclass
class SweepConfig:
    name: str = "sweep"
    backend: str = "hf"                       # "hf" or "simulated"
    models: List[str] = field(default_factory=lambda: ["Qwen/Qwen2.5-1.5B-Instruct"])
    seeds: List[int] = field(default_factory=lambda: list(range(20)))
    seeds_by_model: Dict[str, List[int]] = field(default_factory=dict)  # overrides seeds per model
    milestones: List[int] = field(default_factory=lambda: [10, 25, 50, 100, 200])
    arms: List[str] = field(default_factory=lambda: list(ARMS))
    night_every: int = 10
    topk: int = 8
    eval_max: int = 50
    eval_full_up_to: int = 50
    device: str = "cuda"
    dtype: str = "float16"
    out_dir: str = "runs/sweep"
    train: TrainConfig = field(default_factory=TrainConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)

    def seeds_for(self, model: str) -> List[int]:
        return list(self.seeds_by_model.get(model, self.seeds))

    @property
    def n_max(self) -> int:
        return max(self.milestones)

    def arm_list(self) -> List[Arm]:
        unknown = [a for a in self.arms if a not in ARMS]
        if unknown:
            raise KeyError(f"unknown arms {unknown}; known {sorted(ARMS)}")
        return [ARMS[a] for a in self.arms]

    def validate(self) -> None:
        self.arm_list()
        if sorted(self.milestones) != list(self.milestones) or self.milestones[0] < 1:
            raise ValueError("milestones must be increasing and positive")
        if self.backend not in ("hf", "simulated"):
            raise ValueError(f"unknown backend {self.backend!r}")
        unknown = set(self.seeds_by_model) - set(self.models)
        if unknown:
            raise ValueError(f"seeds_by_model names models not in models: {sorted(unknown)}")
        if self.night_every < 1:
            raise ValueError("night_every must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls, data: Optional[dict]):
    data = dict(data or {})
    names = {f.name for f in fields(cls)}
    extra = set(data) - names
    if extra:
        raise KeyError(f"unknown {cls.__name__} keys {sorted(extra)}")
    if "target_modules" in data:
        data["target_modules"] = tuple(data["target_modules"])
    return cls(**data)


def from_dict(data: dict) -> SweepConfig:
    data = dict(data)
    train = _build(TrainConfig, data.pop("train", None))
    decode = _build(DecodeConfig, data.pop("decode", None))
    cfg = _build(SweepConfig, data)
    cfg.train, cfg.decode = train, decode
    cfg.validate()
    return cfg


def load_sweep_config(path: str | Path) -> SweepConfig:
    return from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
