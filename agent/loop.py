"""The Voyager-style outer loop, minus the LLM critic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from env.tasks import Task, curriculum_order
from env.world import CraftingWorld, ExecutionError
from env.dsl import DSLError
from library.authoring import skill_from_episode
from library.retrieval import (
    RetrievalConfig,
    TokenCounter,
    build_prompt,
    heuristic_token_count,
    relevant_skills,
    render_skill,
)
from library.store import SkillLibrary
from sleep.consolidate import Consolidator, NightRecord


@dataclass
class EpisodeRecord:
    episode: int
    task: str
    tier: int
    success: bool
    attempts: int
    prompt_tokens: int
    env_steps: int
    error: str = ""
    night_after: Optional[int] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class LoopConfig:
    max_attempts: int = 4
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    token_counter: TokenCounter = heuristic_token_count


def context_skill_names(
    library: SkillLibrary, task: Task, config: RetrievalConfig
) -> Set[str]:
    """Skills whose full text is actually in the prompt this episode."""
    names = set()
    for skill in relevant_skills(library, task, config):
        text = render_skill(skill, config)
        if text is not None and text == skill.full_text():
            names.add(skill.name)
    return names


def attempt_task(
    task: Task,
    library: SkillLibrary,
    policy,
    config: LoopConfig,
) -> tuple[bool, int, int, str, CraftingWorld, str]:
    """Run up to max_attempts proposals. Returns the first verified success."""
    ctx = context_skill_names(library, task, config.retrieval)
    last_error, program, world = "", "", CraftingWorld()
    tokens = 0
    for attempt in range(config.max_attempts):
        world = CraftingWorld()
        prompt = build_prompt(library, task, config.retrieval, world.observe())
        tokens += config.token_counter(prompt)
        program = policy.propose(task, prompt, ctx, attempt)
        try:
            world.run(program, library.resolver())
        except (ExecutionError, DSLError) as exc:
            last_error = str(exc)
            continue
        if task.check(world):
            return True, attempt + 1, tokens, "", world, program
        last_error = f"goal not met: needed {task.goal_text()}"
    return False, config.max_attempts, tokens, last_error, world, program


class VoyagerLoop:
    """Curriculum driver shared by every method in the study."""

    def __init__(
        self,
        policy,
        consolidator: Consolidator,
        config: Optional[LoopConfig] = None,
        library: Optional[SkillLibrary] = None,
        on_night: Optional[Callable[[NightRecord, SkillLibrary], None]] = None,
        first_episode: int = 0,
    ) -> None:
        self.policy = policy
        self.consolidator = consolidator
        self.config = config or LoopConfig()
        self.library = library if library is not None else SkillLibrary()
        self.episodes: List[EpisodeRecord] = []
        self.on_night = on_night
        self.first_episode = first_episode  # > 0 when a resumed run continues an earlier session's numbering

    @property
    def next_episode(self) -> int:
        return self.first_episode + len(self.episodes)

    @property
    def night(self) -> int:
        return self.consolidator.night_count

    def run_episode(self, task: Task, episode: int) -> EpisodeRecord:
        was_known = task.name in self.library
        success, attempts, tokens, error, world, program = attempt_task(
            task, self.library, self.policy, self.config
        )
        if success:
            skill = skill_from_episode(
                task,
                program,
                world,
                episode=episode,
                night=self.night,
                library_size=len(self.library),
            )
            self.library.add(skill)
            if not was_known:
                # Re-consolidating a known task would be replay by the back door.
                self.consolidator.note_new_skill(skill.name)

        record = EpisodeRecord(
            episode=episode,
            task=task.name,
            tier=task.tier,
            success=success,
            attempts=attempts,
            prompt_tokens=tokens,
            env_steps=world.steps,
            error=error,
        )

        if success and self.consolidator.should_sleep():
            night = self.consolidator.run_night(self.library, episode)
            if night is not None:
                record.night_after = night.night
                if self.on_night is not None:
                    self.on_night(night, self.library)

        self.episodes.append(record)
        return record

    def run_curriculum(
        self, tasks: Optional[List[Task]] = None, retry_rounds: int = 2
    ):
        """One pass in prerequisite order, then retries of what is still unlearned."""
        tasks = tasks or curriculum_order()
        episode = self.next_episode
        for task in tasks:
            self.run_episode(task, episode)
            episode += 1
        for _ in range(retry_rounds):
            pending = [t for t in tasks if t.name not in self.library]
            if not pending:
                break
            for task in pending:
                self.run_episode(task, episode)
                episode += 1
        return self.episodes

    def finish(self, episode: Optional[int] = None) -> Optional[NightRecord]:
        """Consolidate any stragglers so the run ends on a clean night."""
        if not self.consolidator.pending:
            return None
        night = self.consolidator.run_night(
            self.library, episode if episode is not None else self.next_episode
        )
        if night is not None and self.on_night is not None:
            self.on_night(night, self.library)
        return night
