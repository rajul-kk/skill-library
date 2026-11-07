"""Prompt rendering per arm and the lexical retrieval used by the top-k arms."""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set

from env.procgen import GenSkill, WorldSpec

FORMAT_EXAMPLE = (
    "Example (a different world):\n"
    "Known skills:\n"
    "# skill: gather_oil -- Collect 2 oil from the barn.\n"
    "goto(barn)\n"
    "gather(oil, 2)\n"
    "Task: Make a lamp.\n"
    "Program:\n"
    "call(gather_oil)\n"
    "goto(shed)\n"
    "gather(wick, 1)\n"
    "craft(lamp)\n"
)

_STOP = {"a", "an", "the", "from", "make", "collect", "gather", "craft", "goto", "call", "of", "to"}
_WORD = re.compile(r"[a-z]+")


def full_text(skill: GenSkill) -> str:
    """In-context form of a skill: header plus its verified body."""
    return f"# skill: {skill.name} -- {skill.description}\n{skill.program}"


def pointer_text(skill: GenSkill) -> str:
    """Post-consolidation form: the name stays, the body lives in weights."""
    return f"# skill: {skill.name} -- {skill.description} (in weights)"


def _terms(text: str) -> Set[str]:
    return {w for w in _WORD.findall(text.lower().replace("_", " ")) if w not in _STOP}


def retrieve(task: GenSkill, library: Sequence[GenSkill], k: int) -> List[GenSkill]:
    """Top-k by term overlap (header terms count double, body terms once), ties to the newest."""
    query = _terms(task.description)

    def score(s: GenSkill) -> tuple:
        head = len(query & _terms(f"{s.name} {s.description}"))
        body = len(query & _terms(s.program))
        return (-(2 * head + body), -s.index)

    return sorted(library, key=score)[:k]


def skills_block(arm_prompt: str, task: GenSkill, library: Sequence[GenSkill], k: int) -> str:
    """The 'Known skills' section for one arm."""
    if arm_prompt == "none" or not library:
        return "(none)"
    if arm_prompt == "full":
        return "\n".join(full_text(s) for s in library)
    if arm_prompt == "topk":
        return "\n".join(full_text(s) for s in retrieve(task, library, k))
    if arm_prompt == "pointers":
        return "\n".join(pointer_text(s) for s in retrieve(task, library, k))
    raise ValueError(f"unknown prompt mode {arm_prompt!r}")


def build_prompt(world: WorldSpec, arm_prompt: str, task: GenSkill, library: Sequence[GenSkill], k: int) -> str:
    """Grammar, format example, the arm's skills block, then the task with its own skill banned.

    The rules line was revised after the first pilot (PROTOCOL_sweep section 12): the model called the banned skill it was
    asked to write, copying the example's call(), and wrote prose after the program."""
    return (
        f"{world.dsl_reference()}\n"
        f"{FORMAT_EXAMPLE}\n"
        f"Known skills:\n{skills_block(arm_prompt, task, library, k)}\n"
        f"Task: {task.description}\n"
        f"Goal: obtain {task.task.goal_text()}\n"
        f"Rules: call() only skills listed under Known skills, and never call {task.name}. "
        f"Answer with the program only: DSL instructions, one per line, no other text.\n"
        f"Program:\n"
    )


def wrap(prompt: str, wrapper: str, tokenizer=None) -> str:
    """Prompt wrapper (PROTOCOL_sweep section 6): raw text, or the model's chat template as one user turn."""
    if wrapper == "raw":
        return prompt
    if wrapper == "chat":
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    raise ValueError(f"unknown prompt wrapper {wrapper!r}")


def in_context_names(arm_prompt: str, task: GenSkill, library: Sequence[GenSkill], k: int) -> Set[str]:
    """Skills whose full body is visible in the prompt."""
    if arm_prompt == "full":
        return {s.name for s in library}
    if arm_prompt == "topk":
        return {s.name for s in retrieve(task, library, k)}
    return set()


def names(skills: Iterable[GenSkill]) -> List[str]:
    return [s.name for s in skills]
