"""Policy interface and the CPU recall simulator.

A policy turns a prompt into a DSL program. Whether that program succeeds is
never decided here -- it is executed against the environment and judged by the
task's programmatic check. The simulator therefore only models one thing: can
the agent recall a skill it is supposed to reuse, given what is in context and
what is in the adapter. Everything downstream of that is the real system.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Set

from env.reference import REFERENCE_PROGRAMS
from env.tasks import Task


class Policy(Protocol):
    def propose(
        self,
        task: Task,
        prompt: str,
        context_skills: Set[str],
        attempt: int,
    ) -> str: ...


@dataclass
class SynthesisModel:
    """How hard it is to write a correct program for an unseen task.

    base is the chance of getting the non-call lines right on the first try;
    tier_penalty makes deeper tasks harder; prereq_bonus is the forward-transfer
    channel -- having the prerequisites reliably available makes composing the
    new skill easier, which is exactly the effect the transfer metric measures.
    """

    base: float = 0.62
    tier_penalty: float = 0.035
    prereq_bonus: float = 0.25
    retry_bonus: float = 0.12
    recall_threshold: float = 0.5


class SimulatedPolicy:
    """Recall-and-synthesis simulator used for CPU runs of the full pipeline.

    A skill is available if its full text is in the prompt, or if the adapter
    recalls it: retention is read as a probability, so a decayed skill fails
    intermittently, which is what produces a graded forgetting curve rather
    than a step function.
    """

    def __init__(
        self,
        backend,
        synthesis: Optional[SynthesisModel] = None,
        seed: int = 0,
    ) -> None:
        self.backend = backend
        self.synthesis = synthesis or SynthesisModel()
        self.seed = seed
        self._rng = random.Random(seed)
        self.recall_log: List[dict] = []

    # -- recall -- #

    def recalls(self, skill: str, context_skills: Set[str]) -> bool:
        if skill in context_skills:
            return True
        retention = self.backend.knows(skill)
        hit = self._rng.random() < retention
        self.recall_log.append({"skill": skill, "retention": retention, "hit": hit})
        return hit

    def available(self, task: Task, context_skills: Set[str]) -> bool:
        """True when every skill the reference program calls can be reached."""
        return all(
            self.recalls(name, context_skills) for name in called_skills(task.name)
        )

    # -- synthesis -- #

    def synthesis_probability(
        self, task: Task, context_skills: Set[str], attempt: int
    ) -> float:
        s = self.synthesis
        p = s.base - s.tier_penalty * task.tier + s.retry_bonus * attempt
        prereqs = set(task.prereqs)
        if prereqs and all(
            name in context_skills or self.backend.knows(name) >= s.recall_threshold
            for name in prereqs
        ):
            p += s.prereq_bonus
        return min(0.98, max(0.02, p))

    def propose(
        self, task: Task, prompt: str, context_skills: Set[str], attempt: int = 0
    ) -> str:
        """Emit a program. Degraded proposals are returned verbatim and fail in env."""
        program = REFERENCE_PROGRAMS[task.name].strip()
        if not self.available(task, context_skills):
            return _drop_calls(program)
        if self._rng.random() > self.synthesis_probability(task, context_skills, attempt):
            return _perturb(program, self._rng)
        return program


def called_skills(task_name: str) -> List[str]:
    """Skill names the reference program for a task invokes via call()."""
    out = []
    for line in REFERENCE_PROGRAMS[task_name].splitlines():
        line = line.strip()
        if line.startswith("call("):
            out.append(line[len("call(") : line.index(")")].strip())
    return out


def _drop_calls(program: str) -> str:
    """An unrecalled skill: the agent writes the program without the reuse step."""
    kept = [l for l in program.splitlines() if not l.strip().startswith("call(")]
    return "\n".join(kept) if kept else "goto(forest)"


def _perturb(program: str, rng: random.Random) -> str:
    """A plausible-but-wrong program: one quantity or destination is off."""
    lines = program.splitlines()
    idx = rng.randrange(len(lines))
    line = lines[idx].strip()
    if line.startswith("gather(") or line.startswith("craft("):
        lines[idx] = line.replace(")", ", 1)") if "," not in line else line
        head, _, rest = line.partition("(")
        args = [a.strip() for a in rest.rstrip(")").split(",")]
        args[-1] = "1" if len(args) > 1 else args[-1]
        lines[idx] = f"{head}({', '.join(args)})"
    elif line.startswith("goto("):
        lines[idx] = "goto(river)"
    else:
        lines.pop(idx)
    return "\n".join(lines) if lines else "goto(river)"
