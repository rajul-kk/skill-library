"""Turning a successful episode into a library entry."""
from __future__ import annotations

from typing import Optional

from env.tasks import Task
from env.world import CraftingWorld

from .schema import Skill

_MAX_LISTED = 6


def _inventory_line(world: CraftingWorld) -> str:
    items = sorted(world.inventory.items(), key=lambda kv: (-kv[1], kv[0]))[:_MAX_LISTED]
    return ", ".join(f"{k} x{v}" for k, v in items) or "nothing"


def write_notes(task: Task, world: CraftingWorld) -> str:
    prereqs = ", ".join(task.prereqs) or "none"
    stations = ", ".join(
        f"{s}@{loc}" for s, locs in world.stations.items() for loc in sorted(locs)
    ) or "none"
    return (
        f"# requires: {prereqs}\n"
        f"# guarantees: {task.goal_text()}\n"
        f"# ends at: {world.location}; leaves: {_inventory_line(world)}\n"
        f"# stations after run: {stations}\n"
        f"# verified programmatically in {world.steps} env steps"
    )


def skill_from_episode(
    task: Task,
    program: str,
    world: CraftingWorld,
    episode: int,
    night: int,
    library_size: int = 0,
    description: Optional[str] = None,
) -> Skill:
    """Build the library entry for a task the agent has just verifiably solved."""
    return Skill(
        name=task.name,
        description=description or task.description,
        body=program.strip(),
        notes=write_notes(task, world),
        task=task.name,
        tier=task.tier,
        learned_at_episode=episode,
        learned_at_night=night,
        learned_at_size=library_size,
        metadata={"env_steps": world.steps},
    )
