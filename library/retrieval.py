"""Library retrieval, and the token accounting that makes its cost measurable."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from env.dsl import dsl_reference
from env.tasks import Task

from .schema import Skill
from .store import SkillLibrary

TokenCounter = Callable[[str], int]


def heuristic_token_count(text: str) -> int:
    """Tokenizer-free estimate (~4 chars/token) for CPU-only runs."""
    return max(1, (len(text) + 3) // 4)


def hf_token_counter(tokenizer) -> TokenCounter:
    """Exact counts when a HF tokenizer is available (used in the GPU notebooks)."""

    def _count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return _count


@dataclass
class RetrievalConfig:
    """How the library is rendered into the prompt."""

    max_skills: int = 12
    serve_full_text_always: bool = False
    include_pointers_for_consolidated: bool = True


def relevant_skills(
    library: SkillLibrary, task: Task, config: RetrievalConfig
) -> List[Skill]:
    """Rank skills for a task: declared prereqs first, then nearest tiers."""
    prereqs = set(task.prereqs)

    def rank(s: Skill) -> tuple:
        return (0 if s.name in prereqs else 1, abs(s.tier - task.tier), s.name)

    return sorted(library, key=rank)[: config.max_skills]


def render_skill(skill: Skill, config: RetrievalConfig) -> Optional[str]:
    if config.serve_full_text_always or not skill.consolidated:
        return skill.full_text()
    if config.include_pointers_for_consolidated:
        return skill.pointer_text()
    return None


def render_library(
    library: SkillLibrary, task: Task, config: RetrievalConfig
) -> str:
    chunks = [
        text
        for skill in relevant_skills(library, task, config)
        if (text := render_skill(skill, config)) is not None
    ]
    return "\n\n".join(chunks)


def build_prompt(
    library: SkillLibrary, task: Task, config: RetrievalConfig, observation: str
) -> str:
    """The full agent prompt: grammar, retrieved library, state, goal."""
    lib = render_library(library, task, config)
    return (
        f"{dsl_reference()}\n"
        f"Known skills:\n{lib if lib else '(none yet)'}\n\n"
        f"State: {observation}\n"
        f"Task: {task.description}\n"
        f"Goal: obtain {task.goal_text()}\n"
        f"Write a DSL program:\n"
    )


def context_token_cost(
    library: SkillLibrary,
    task: Task,
    config: RetrievalConfig,
    observation: str = "",
    counter: TokenCounter = heuristic_token_count,
) -> int:
    return counter(build_prompt(library, task, config, observation))
