"""Turning library entries into SFT examples for the sleep phase.

The training example teaches the model to emit a skill's program from the task
description *without* the skill text in context -- that is precisely the
capability that lets retrieval drop to a pointer afterwards. So the prompt is
built with an empty library view, and the target is the verified program.
"""
from __future__ import annotations

from typing import List, Sequence

from env.dsl import dsl_reference
from env.tasks import TASKS, Task
from library.schema import Skill
from library.store import SkillLibrary

from .backend import Example

WEIGHTS_ONLY_HEADER = "Known skills:\n(recall from memory)"


def consolidation_prompt(skill: Skill, task: Task | None = None) -> str:
    """The prompt used at both train and eval time for a consolidated skill.

    Nothing about the skill body appears here; if the model produces the right
    program it is doing so from weights.
    """
    task = task or TASKS.get(skill.task)
    goal = task.goal_text() if task else skill.description
    return (
        f"{dsl_reference()}\n"
        f"{WEIGHTS_ONLY_HEADER}\n\n"
        f"Task: {skill.description}\n"
        f"Goal: obtain {goal}\n"
        f"Write a DSL program:\n"
    )


def example_from_skill(skill: Skill, night: int, is_replay: bool = False) -> Example:
    return Example(
        skill=skill.name,
        prompt=consolidation_prompt(skill),
        target=skill.body.strip(),
        night_added=night,
        is_replay=is_replay,
    )


def examples_for_night(skills: Sequence[Skill], night: int) -> List[Example]:
    return [example_from_skill(s, night) for s in skills]


def evaluation_examples(library: SkillLibrary) -> List[Example]:
    """One weights-only probe per skill, used to measure retention on GPU."""
    return [example_from_skill(s, s.learned_at_night) for s in library]
