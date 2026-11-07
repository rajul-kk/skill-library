"""The growing skill library: append-only, JSON-checkpointable, cheap to resume."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .schema import Skill


class SkillLibrary:
    """Ordered store of learned skills, keyed by name."""

    def __init__(self, skills: Optional[Iterable[Skill]] = None) -> None:
        self._skills: Dict[str, Skill] = {}
        for s in skills or []:
            self._skills[s.name] = s

    # ---------------- basic container protocol ---------------- #

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def __iter__(self) -> Iterator[Skill]:
        return iter(self._skills.values())

    def __getitem__(self, name: str) -> Skill:
        return self._skills[name]

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    @property
    def names(self) -> List[str]:
        return list(self._skills)

    # ---------------- mutation ---------------- #

    def add(self, skill: Skill) -> Skill:
        """Insert or replace a skill, preserving its original learning time."""
        existing = self._skills.get(skill.name)
        if existing is not None:
            skill.learned_at_episode = existing.learned_at_episode
            skill.learned_at_night = existing.learned_at_night
            skill.learned_at_size = existing.learned_at_size
            skill.uses = existing.uses
            # consolidation is of a body: a replaced body is no longer what the adapter learned
            if skill.consolidated_night is None and skill.body == existing.body:
                skill.consolidated_night = existing.consolidated_night
        self._skills[skill.name] = skill
        return skill

    def mark_consolidated(self, names: Iterable[str], night: int) -> List[str]:
        """Record that these skills now live in the cumulative adapter."""
        marked = []
        for n in names:
            s = self._skills.get(n)
            if s is not None and s.consolidated_night is None:
                s.consolidated_night = night
                marked.append(n)
        return marked

    def unconsolidated(self) -> List[Skill]:
        return [s for s in self._skills.values() if not s.consolidated]

    def consolidated(self) -> List[Skill]:
        return [s for s in self._skills.values() if s.consolidated]

    # ---------------- execution support ---------------- #

    def resolver(self):
        """Callback for CraftingWorld.run(): skill name -> DSL body."""

        def _resolve(name: str) -> str:
            skill = self._skills.get(name)
            if skill is None:
                raise KeyError(name)
            skill.uses += 1
            return skill.body

        return _resolve

    # ---------------- persistence ---------------- #

    def to_dict(self) -> dict:
        return {"skills": [s.to_dict() for s in self._skills.values()]}

    @classmethod
    def from_dict(cls, d: dict) -> "SkillLibrary":
        return cls(Skill.from_dict(s) for s in d.get("skills", []))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SkillLibrary":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
