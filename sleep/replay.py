"""The capped replay buffer -- the core mechanism of the method."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .backend import Example

STRATEGIES = ("uniform", "recency", "round_robin", "tier_balanced")


@dataclass
class ReplayConfig:
    """Budget and allocation policy for one night's replay sample."""

    enabled: bool = True
    per_skill: int = 2
    max_total: int = 24
    strategy: str = "round_robin"
    seed: int = 0

    def validate(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown replay strategy {self.strategy!r}; use {STRATEGIES}")
        if self.per_skill < 0 or self.max_total < 0:
            raise ValueError("replay budgets must be non-negative")


class ReplayBuffer:
    """Holds one example pool per consolidated skill and samples within a cap."""

    def __init__(self, config: ReplayConfig | None = None) -> None:
        self.config = config or ReplayConfig()
        self.config.validate()
        self.pool: Dict[str, List[Example]] = {}
        self.tier: Dict[str, int] = {}
        self.order: List[str] = []
        self._cursor = 0
        self._rng = random.Random(self.config.seed)

    def __len__(self) -> int:
        return sum(len(v) for v in self.pool.values())

    @property
    def skills(self) -> List[str]:
        return list(self.order)

    def add(self, example: Example, tier: int = 0) -> None:
        """Register an example as replayable. Pools are themselves capped."""
        name = example.skill
        if name not in self.pool:
            self.pool[name] = []
            self.order.append(name)
            self.tier[name] = tier
        bucket = self.pool[name]
        if len(bucket) < max(1, self.config.per_skill * 2):
            bucket.append(example)

    def _ordered_skills(self, exclude: Sequence[str]) -> List[str]:
        cfg = self.config
        candidates = [s for s in self.order if s not in set(exclude)]
        if cfg.strategy == "recency":
            return list(reversed(candidates))
        if cfg.strategy == "uniform":
            shuffled = list(candidates)
            self._rng.shuffle(shuffled)
            return shuffled
        if cfg.strategy == "tier_balanced":
            by_tier: Dict[int, List[str]] = {}
            for s in candidates:
                by_tier.setdefault(self.tier.get(s, 0), []).append(s)
            out: List[str] = []
            while any(by_tier.values()):
                for t in sorted(by_tier):
                    if by_tier[t]:
                        out.append(by_tier[t].pop(0))
            return out
        # round_robin: start where the last night stopped so coverage rotates
        if not candidates:
            return []
        start = self._cursor % len(candidates)
        self._cursor += 1
        return candidates[start:] + candidates[:start]

    def sample(self, exclude: Sequence[str] = ()) -> List[Example]:
        """Draw this night's replay batch, honouring per_skill and max_total."""
        cfg = self.config
        if not cfg.enabled or cfg.max_total == 0 or cfg.per_skill == 0:
            return []
        skills = self._ordered_skills(exclude)
        picked: List[Example] = []
        for round_idx in range(cfg.per_skill):
            for name in skills:
                bucket = self.pool.get(name, [])
                if not bucket or len(picked) >= cfg.max_total:
                    continue
                chosen = bucket[round_idx % len(bucket)]
                picked.append(
                    Example(
                        skill=chosen.skill,
                        prompt=chosen.prompt,
                        target=chosen.target,
                        night_added=chosen.night_added,
                        is_replay=True,
                    )
                )
            if len(picked) >= cfg.max_total:
                break
        return picked[: cfg.max_total]

    def state_dict(self) -> dict:
        return {
            "config": self.config.__dict__,
            "order": self.order,
            "tier": self.tier,
            "cursor": self._cursor,
            "rng": _jsonable(self._rng.getstate()),
            "pool": {
                k: [e.__dict__ for e in v] for k, v in self.pool.items()
            },
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "ReplayBuffer":
        buf = cls(ReplayConfig(**d["config"]))
        buf.order = list(d["order"])
        buf.tier = {k: int(v) for k, v in d["tier"].items()}
        buf._cursor = int(d.get("cursor", 0))
        if "rng" in d:  # older checkpoints lack it and restart the uniform strategy from the seed
            buf._rng.setstate(_tupled(d["rng"]))
        buf.pool = {k: [Example(**e) for e in v] for k, v in d["pool"].items()}
        return buf


def _jsonable(state):
    return [_jsonable(x) for x in state] if isinstance(state, tuple) else state


def _tupled(state):
    return tuple(_tupled(x) for x in state) if isinstance(state, list) else state
