"""The curriculum: a 15-task skill tree with programmatic success checks.

Each task carries a goal predicate evaluated against `CraftingWorld` state.
There is no LLM critic anywhere in this project -- success is decided by
`check()` alone, which is what makes the loop cheap enough for a T4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from .world import CraftingWorld


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    goal_items: Tuple[Tuple[str, int], ...]
    prereqs: Tuple[str, ...] = ()
    tier: int = 0
    goal_stations: Tuple[str, ...] = ()

    def check(self, world: CraftingWorld) -> bool:
        """Programmatic success check -- the only verifier in the system."""
        if not all(world.has(item, n) for item, n in self.goal_items):
            return False
        return all(world.stations.get(s) for s in self.goal_stations)

    def goal_text(self) -> str:
        parts = [f"{n}x {item}" for item, n in self.goal_items]
        parts += [f"a placed {s}" for s in self.goal_stations]
        return " and ".join(parts)


def _t(name, desc, goals, prereqs=(), tier=0, stations=()) -> Task:
    return Task(name, desc, tuple(goals), tuple(prereqs), tier, tuple(stations))


CURRICULUM: List[Task] = [
    _t("gather_wood", "Collect wood from the forest.", [("wood", 4)], (), 0),
    _t("craft_planks", "Turn wood into planks.", [("plank", 6)], ("gather_wood",), 1),
    _t("craft_sticks", "Turn planks into sticks.", [("stick", 4)], ("craft_planks",), 1),
    _t("craft_table", "Build and place a crafting table.", [], ("craft_planks",), 2,
       stations=("table",)),
    _t("craft_wood_pickaxe", "Craft a wooden pickaxe at a table.",
       [("wood_pickaxe", 1)], ("craft_table", "craft_sticks"), 3),
    _t("mine_stone", "Mine stone in the cave with a wooden pickaxe.",
       [("stone", 6)], ("craft_wood_pickaxe",), 4),
    _t("mine_coal", "Mine coal in the cave.", [("coal", 2)], ("craft_wood_pickaxe",), 4),
    _t("craft_stone_pickaxe", "Craft a stone pickaxe.",
       [("stone_pickaxe", 1)], ("mine_stone",), 5),
    _t("build_furnace", "Craft and place a furnace.", [], ("mine_stone",), 5,
       stations=("furnace",)),
    _t("craft_torch", "Craft torches from coal and sticks.",
       [("torch", 4)], ("mine_coal", "craft_sticks"), 5),
    _t("mine_iron", "Mine iron ore with a stone pickaxe.",
       [("iron_ore", 3)], ("craft_stone_pickaxe",), 6),
    _t("smelt_iron", "Smelt iron ore into ingots in a furnace.",
       [("iron_ingot", 3)], ("mine_iron", "build_furnace"), 7),
    _t("craft_iron_pickaxe", "Craft an iron pickaxe.",
       [("iron_pickaxe", 1)], ("smelt_iron",), 8),
    _t("make_glass", "Smelt sand into glass.", [("glass", 1)], ("build_furnace",), 7),
    _t("craft_lantern", "Assemble a lantern from glass, iron and a torch.",
       [("lantern", 1)], ("make_glass", "smelt_iron", "craft_torch"), 9),
]

TASKS: Dict[str, Task] = {t.name: t for t in CURRICULUM}


def curriculum_order() -> List[Task]:
    """Prerequisite-respecting order (stable topological sort by tier then name)."""
    done: set = set()
    remaining = list(CURRICULUM)
    ordered: List[Task] = []
    while remaining:
        ready = [t for t in remaining if set(t.prereqs) <= done]
        if not ready:
            raise ValueError(f"cyclic prereqs among {[t.name for t in remaining]}")
        ready.sort(key=lambda t: (t.tier, t.name))
        for t in ready:
            ordered.append(t)
            done.add(t.name)
        remaining = [t for t in remaining if t.name not in done]
    return ordered


def validate_curriculum() -> None:
    names = {t.name for t in CURRICULUM}
    for t in CURRICULUM:
        missing = set(t.prereqs) - names
        if missing:
            raise ValueError(f"task {t.name} has unknown prereqs {sorted(missing)}")
    curriculum_order()
