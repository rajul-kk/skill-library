"""Evaluation and metrics."""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, Dict, List, Optional

from agent.loop import LoopConfig, attempt_task
from agent.policy import called_skills
from env.tasks import TASKS
from library.retrieval import context_token_cost
from library.store import SkillLibrary


@dataclass
class SkillEval:
    skill: str
    tier: int
    learned_at_night: int
    nights_since_learned: int
    skills_since_learned: int
    success_rate: float
    prompt_tokens: float


@dataclass
class Snapshot:
    """One post-night measurement of the whole library."""

    night: int
    episode: int
    library_size: int
    stored_params: int
    seed: int = 0
    skills: List[SkillEval] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return mean([s.success_rate for s in self.skills]) if self.skills else 0.0

    @property
    def mean_tokens(self) -> float:
        return mean([s.prompt_tokens for s in self.skills]) if self.skills else 0.0

    def earliest_k(self, k: int = 3) -> float:
        """Success on the earliest-learned skills -- the headline retention number."""
        oldest = sorted(self.skills, key=lambda s: (-s.skills_since_learned, s.tier))[:k]
        return mean([s.success_rate for s in oldest]) if oldest else 0.0

    def to_dict(self) -> dict:
        return {
            "night": self.night,
            "episode": self.episode,
            "library_size": self.library_size,
            "stored_params": self.stored_params,
            "seed": self.seed,
            "success_rate": self.success_rate,
            "mean_tokens": self.mean_tokens,
            "earliest_3": self.earliest_k(3),
            "skills": [s.__dict__ for s in self.skills],
        }


def eval_seed(night: int, skill: str, repeat: int) -> int:
    """Seed for one evaluation attempt. Stable across processes: hash() of a string is randomised per process
    (PYTHONHASHSEED), which made every evaluation differ between runs."""
    return zlib.crc32(f"{night}|{skill}|{repeat}".encode("utf-8")) % (2**31)


def evaluate_library(
    library: SkillLibrary,
    policy_factory: Callable[[int], object],
    config: LoopConfig,
    night: int,
    episode: int,
    stored_params: int,
    repeats: int = 3,
    eval_attempts: int = 1,
    seed: int = 0,
) -> Snapshot:
    """Re-attempt every learned skill under the deployed retrieval config."""
    eval_config = LoopConfig(
        max_attempts=eval_attempts,
        retrieval=config.retrieval,
        token_counter=config.token_counter,
    )
    results: List[SkillEval] = []
    for skill in library:
        task = TASKS[skill.task]
        successes, tokens = [], []
        for r in range(repeats):
            policy = policy_factory(eval_seed(night, skill.name, r))
            ok, _, tok, _, _, _ = attempt_task(task, library, policy, eval_config)
            successes.append(1.0 if ok else 0.0)
            tokens.append(tok)
        results.append(
            SkillEval(
                skill=skill.name,
                tier=skill.tier,
                learned_at_night=skill.learned_at_night,
                nights_since_learned=max(0, night - skill.learned_at_night),
                skills_since_learned=max(0, len(library) - skill.learned_at_size),
                success_rate=mean(successes),
                prompt_tokens=mean(tokens),
            )
        )
    return Snapshot(
        night=night,
        episode=episode,
        library_size=len(library),
        stored_params=stored_params,
        seed=seed,
        skills=results,
    )


def forgetting_curve(
    snapshots: List[Snapshot],
    axis: str = "nights",
    exclude_recall_free: bool = True,
    min_samples: int = 3,
) -> Dict[int, float]:
    """Retention as a function of elapsed time since a skill was learned."""
    if axis not in ("nights", "skills"):
        raise ValueError("axis must be 'nights' or 'skills'")
    key = "nights_since_learned" if axis == "nights" else "skills_since_learned"
    buckets: Dict[int, List[float]] = {}
    for snap in snapshots:
        for s in snap.skills:
            if exclude_recall_free and not called_skills(s.skill):
                continue
            buckets.setdefault(getattr(s, key), []).append(s.success_rate)
    return {
        k: mean(v) for k, v in sorted(buckets.items()) if len(v) >= min_samples
    }


def curve_support(
    snapshots: List[Snapshot], axis: str = "nights", exclude_recall_free: bool = True
) -> Dict[int, int]:
    """How many (skill, snapshot) pairs back each point of the curve."""
    key = "nights_since_learned" if axis == "nights" else "skills_since_learned"
    counts: Dict[int, int] = {}
    for snap in snapshots:
        for s in snap.skills:
            if exclude_recall_free and not called_skills(s.skill):
                continue
            counts[getattr(s, key)] = counts.get(getattr(s, key), 0) + 1
    return dict(sorted(counts.items()))


def forward_transfer(episodes, library: SkillLibrary) -> Dict[str, float]:
    """Attempts needed for the first success on each task."""
    first: Dict[str, int] = {}
    for e in episodes:
        if e.success and e.task not in first:
            first[e.task] = e.attempts
    tiers = {name: TASKS[name].tier for name in first}
    late = [v for k, v in first.items() if tiers[k] >= 5]
    early = [v for k, v in first.items() if tiers[k] < 5]
    return {
        "mean_attempts_all": mean(first.values()) if first else 0.0,
        "mean_attempts_early": mean(early) if early else 0.0,
        "mean_attempts_late": mean(late) if late else 0.0,
        "tasks_solved": float(len(first)),
    }


def token_cost_summary(episodes) -> Dict[str, float]:
    per_episode = [e.prompt_tokens for e in episodes]
    return {
        "mean_prompt_tokens": mean(per_episode) if per_episode else 0.0,
        "total_prompt_tokens": float(sum(per_episode)),
    }


def normalized_forgetting_curve(
    snapshots: List[Snapshot],
    axis: str = "skills",
    exclude_recall_free: bool = True,
    min_samples: int = 3,
) -> Dict[int, float]:
    """Retention as a fraction of each skill's own freshly-learned score."""
    key = "nights_since_learned" if axis == "nights" else "skills_since_learned"
    trajectories: Dict[tuple, List[tuple]] = {}
    for snap in snapshots:
        for s in snap.skills:
            if exclude_recall_free and not called_skills(s.skill):
                continue
            trajectories.setdefault((snap.seed, s.skill), []).append(
                (getattr(s, key), s.success_rate)
            )

    buckets: Dict[int, List[float]] = {}
    for points in trajectories.values():
        points.sort()
        baseline = points[0][1]
        if baseline <= 0:
            continue  # never demonstrated, so nothing to forget
        for x, value in points:
            buckets.setdefault(x, []).append(min(1.0, value / baseline))
    return {k: mean(v) for k, v in sorted(buckets.items()) if len(v) >= min_samples}
