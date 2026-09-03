"""The sleep phase: every K completed tasks, distil new skills into weights.

One night is: collect the skills learned since the last night, build their SFT
examples, interleave a capped replay sample of everything learned earlier, run
one short LoRA pass over the mixture, then mark the new skills consolidated so
retrieval serves only their pointers. The adapter is never reset and never
forked -- the same cumulative adapter absorbs every night, which is the
difference from PEAM-style isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from library.schema import Skill
from library.store import SkillLibrary

from .backend import AdapterBackend, Example, TrainStats
from .dataset import example_from_skill
from .replay import ReplayBuffer, ReplayConfig


@dataclass
class SleepConfig:
    """When and how hard to consolidate.

    every_k_tasks is the night trigger studied in the frequency ablation;
    drop_full_text_after_consolidation is what converts weight memory into
    context savings and is disabled for the no-consolidation baseline.
    """

    enabled: bool = True
    every_k_tasks: int = 3
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    drop_full_text_after_consolidation: bool = True
    steps_per_example: int = 12
    max_steps: int = 400


@dataclass
class NightRecord:
    night: int
    episode: int
    new_skills: List[str]
    replayed_skills: List[str]
    stats: TrainStats
    stored_params: int
    library_size: int

    def to_dict(self) -> dict:
        return {
            "night": self.night,
            "episode": self.episode,
            "new_skills": self.new_skills,
            "replayed_skills": self.replayed_skills,
            "stats": self.stats.to_dict(),
            "stored_params": self.stored_params,
            "library_size": self.library_size,
        }


class Consolidator:
    """Drives the sleep phase over a growing library."""

    def __init__(
        self,
        backend: AdapterBackend,
        config: Optional[SleepConfig] = None,
    ) -> None:
        self.backend = backend
        self.config = config or SleepConfig()
        self.buffer = ReplayBuffer(self.config.replay)
        self.nights: List[NightRecord] = []
        self.pending: List[str] = []

    @property
    def night_count(self) -> int:
        return len(self.nights)

    def note_new_skill(self, name: str) -> None:
        """Queue a freshly-learned skill for the next night."""
        if name not in self.pending:
            self.pending.append(name)

    def should_sleep(self) -> bool:
        return (
            self.config.enabled
            and len(self.pending) >= max(1, self.config.every_k_tasks)
        )

    def run_night(self, library: SkillLibrary, episode: int) -> Optional[NightRecord]:
        """Consolidate everything pending. Returns None if there is nothing to do."""
        if not self.config.enabled or not self.pending:
            return None

        night = self.night_count + 1
        new_skills = [library[n] for n in self.pending if n in library]
        if not new_skills:
            self.pending.clear()
            return None

        new_examples = [example_from_skill(s, night) for s in new_skills]
        replay_examples = self.buffer.sample(exclude=[s.name for s in new_skills])
        batch: List[Example] = self._interleave(new_examples, replay_examples)

        stats = self.backend.consolidate(batch, night)
        stats.steps = min(
            self.config.max_steps, len(batch) * self.config.steps_per_example
        )

        for skill, example in zip(new_skills, new_examples):
            self.buffer.add(example, tier=skill.tier)
        if self.config.drop_full_text_after_consolidation:
            library.mark_consolidated([s.name for s in new_skills], night)

        record = NightRecord(
            night=night,
            episode=episode,
            new_skills=[s.name for s in new_skills],
            replayed_skills=[e.skill for e in replay_examples],
            stats=stats,
            stored_params=self.backend.stored_parameters(),
            library_size=len(library),
        )
        self.nights.append(record)
        self.pending.clear()
        return record

    @staticmethod
    def _interleave(new: Sequence[Example], replay: Sequence[Example]) -> List[Example]:
        """Alternate new and replayed examples rather than concatenating them.

        Interleaving keeps replay gradients spread across the pass instead of
        front-loading them, which is the whole point of training on the mixture
        together rather than sequentially.
        """
        out: List[Example] = []
        n, r = list(new), list(replay)
        if not n:
            return r
        per = max(1, len(r) // len(n)) if r else 0
        ri = 0
        for e in n:
            out.append(e)
            out.extend(r[ri : ri + per])
            ri += per
        out.extend(r[ri:])
        return out

    def state_dict(self) -> dict:
        return {
            "pending": self.pending,
            "buffer": self.buffer.state_dict(),
            "nights": [n.to_dict() for n in self.nights],
        }

    def load_state_dict(self, d: dict) -> None:
        self.pending = list(d.get("pending", []))
        self.buffer = ReplayBuffer.from_state_dict(d["buffer"])
        self.nights = [
            NightRecord(
                night=n["night"],
                episode=n["episode"],
                new_skills=n["new_skills"],
                replayed_skills=n["replayed_skills"],
                stats=TrainStats(**n["stats"]),
                stored_params=n["stored_params"],
                library_size=n["library_size"],
            )
            for n in d.get("nights", [])
        ]
