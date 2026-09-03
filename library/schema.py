"""Skill records stored in the growing library."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Skill:
    """One entry in the Voyager-style library.

    `consolidated_night` is the night on which this skill's full text was
    distilled into the cumulative adapter. Once set, retrieval serves only the
    pointer (name + one-line description) instead of the full entry, which is
    where the in-context token savings come from.
    """

    name: str
    description: str
    body: str
    task: str
    tier: int = 0
    learned_at_episode: int = 0
    learned_at_night: int = 0
    learned_at_size: int = 0
    consolidated_night: Optional[int] = None
    uses: int = 0
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def consolidated(self) -> bool:
        return self.consolidated_night is not None

    def fingerprint(self) -> str:
        return hashlib.sha1(f"{self.name}\n{self.body}".encode()).hexdigest()[:12]

    def full_text(self) -> str:
        """The full in-context form: header, pre/postcondition notes, DSL body."""
        parts = [f"# skill: {self.name}", f"# {self.description}"]
        if self.notes.strip():
            parts.append(self.notes.strip())
        parts.append(self.body.strip())
        return "\n".join(parts)

    def pointer_text(self) -> str:
        """The post-consolidation form: just enough to invoke it by name."""
        return f"# skill: {self.name} (in weights) -- {self.description}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Skill":
        return cls(**d)
