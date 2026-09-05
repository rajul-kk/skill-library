"""The adapter backend interface shared by every method in the study."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable


@dataclass
class Example:
    """One SFT example: prompt -> target DSL program."""

    skill: str
    prompt: str
    target: str
    night_added: int = 0
    is_replay: bool = False


@dataclass
class TrainStats:
    night: int
    n_new: int
    n_replay: int
    steps: int
    loss: float = 0.0
    seconds: float = 0.0
    trainable_params: int = 0
    stored_params: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class AdapterBackend(Protocol):
    """Weights-side memory for the agent."""

    def consolidate(self, examples: Sequence[Example], night: int) -> TrainStats: ...

    def knows(self, skill: str) -> float: ...

    def stored_parameters(self) -> int:
        """Adapter parameters that must be kept on disk."""

    def save(self, path: str | Path) -> None: ...

    def load(self, path: str | Path) -> None: ...


class JsonPersisted:
    """Shared save/load for the CPU backends."""

    def state_dict(self) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError

    def load_state_dict(self, d: dict) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        self.load_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class InterferenceModel:
    """Parameters of the CPU retention simulator."""

    capacity: float = 6.0      # skills a rank-r adapter holds before it strains
    decay: float = 0.55        # retention lost per interfering night at load 1.0
    floor: float = 0.02        # retention never quite reaches zero
    gain: float = 1.0          # retention granted to a skill trained this night
    replay_gain: float = 0.85  # retention restored by a replayed example


class SimulatedAdapter(JsonPersisted):
    """Retention dynamics for CPU runs of a single cumulative adapter."""

    def __init__(
        self,
        rank: int = 16,
        base_params: int = 4_718_592,
        interference: Optional[InterferenceModel] = None,
    ) -> None:
        self.rank = rank
        self.base_params = base_params
        self.interference = interference or InterferenceModel()
        self.retention: Dict[str, float] = {}
        self.history: List[TrainStats] = []

    def consolidate(self, examples: Sequence[Example], night: int) -> TrainStats:
        trained = {e.skill for e in examples}
        n_new = sum(1 for e in examples if not e.is_replay)
        im = self.interference

        load = max(1.0, len(set(self.retention) | trained) / im.capacity)
        for skill in list(self.retention):
            if skill in trained:
                continue
            self.retention[skill] = max(
                im.floor, self.retention[skill] * (1.0 - min(0.95, im.decay * load))
            )

        for e in examples:
            gain = im.replay_gain if e.is_replay else im.gain
            self.retention[e.skill] = max(self.retention.get(e.skill, 0.0), gain)

        stats = TrainStats(
            night=night,
            n_new=n_new,
            n_replay=len(examples) - n_new,
            steps=len(examples),
            trainable_params=self.base_params,
            stored_params=self.stored_parameters(),
        )
        self.history.append(stats)
        return stats

    def knows(self, skill: str) -> float:
        return self.retention.get(skill, 0.0)

    def stored_parameters(self) -> int:
        """One shared adapter: constant however many skills it holds."""
        return self.base_params

    def state_dict(self) -> dict:
        return {
            "kind": "simulated",
            "rank": self.rank,
            "base_params": self.base_params,
            "interference": asdict(self.interference),
            "retention": self.retention,
            "history": [h.to_dict() for h in self.history],
        }

    def load_state_dict(self, d: dict) -> None:
        self.rank = d["rank"]
        self.base_params = d["base_params"]
        self.interference = InterferenceModel(**d["interference"])
        self.retention = dict(d["retention"])
        self.history = [TrainStats(**h) for h in d.get("history", [])]


class IsolatedSimulatedAdapters(JsonPersisted):
    """PEAM-style isolation: one adapter per skill or category, never shared."""

    def __init__(
        self, rank: int = 16, base_params: int = 4_718_592, category_of=None
    ) -> None:
        self.rank = rank
        self.base_params = base_params
        self.category_of = category_of or (lambda name: name)
        self.categories: Dict[str, List[str]] = {}
        self.history: List[TrainStats] = []

    def consolidate(self, examples: Sequence[Example], night: int) -> TrainStats:
        for e in examples:
            if e.is_replay:
                continue  # isolation needs no replay, by construction
            self.categories.setdefault(self.category_of(e.skill), []).append(e.skill)
        n_new = sum(1 for e in examples if not e.is_replay)
        stats = TrainStats(
            night=night,
            n_new=n_new,
            n_replay=0,
            steps=n_new,
            trainable_params=self.base_params,
            stored_params=self.stored_parameters(),
        )
        self.history.append(stats)
        return stats

    def knows(self, skill: str) -> float:
        return 1.0 if self.category_of(skill) in self.categories else 0.0

    def stored_parameters(self) -> int:
        return self.base_params * max(1, len(self.categories))

    def state_dict(self) -> dict:
        return {
            "kind": "isolated_simulated",
            "rank": self.rank,
            "base_params": self.base_params,
            "categories": self.categories,
            "history": [h.to_dict() for h in self.history],
        }

    def load_state_dict(self, d: dict) -> None:
        self.rank = d["rank"]
        self.base_params = d["base_params"]
        self.categories = {k: list(v) for k, v in d["categories"].items()}
        self.history = [TrainStats(**h) for h in d.get("history", [])]


class NullAdapter(JsonPersisted):
    """No weight memory at all -- baseline (a), everything stays in context."""

    def __init__(self) -> None:
        self.history: List[TrainStats] = []

    def consolidate(self, examples: Sequence[Example], night: int) -> TrainStats:
        stats = TrainStats(night=night, n_new=0, n_replay=0, steps=0)
        self.history.append(stats)
        return stats

    def knows(self, skill: str) -> float:
        return 0.0

    def stored_parameters(self) -> int:
        return 0

    def state_dict(self) -> dict:
        return {"kind": "null", "history": [h.to_dict() for h in self.history]}

    def load_state_dict(self, d: dict) -> None:
        self.history = [TrainStats(**h) for h in d.get("history", [])]
