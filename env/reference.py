"""Oracle solutions for every task in the curriculum."""
from __future__ import annotations

from typing import Dict

from .tasks import TASKS
from .world import CraftingWorld

REFERENCE_PROGRAMS: Dict[str, str] = {
    "gather_wood": """
goto(forest)
gather(wood, 2)
""",
    "craft_planks": """
call(gather_wood)
craft(plank, 3)
""",
    "craft_sticks": """
call(craft_planks)
craft(stick, 2)
""",
    "craft_table": """
call(craft_planks)
craft(table)
place(table)
""",
    "craft_wood_pickaxe": """
call(craft_table)
gather(wood, 3)
craft(plank, 3)
craft(stick, 2)
craft(wood_pickaxe)
""",
    "mine_stone": """
call(craft_wood_pickaxe)
goto(cave)
gather(stone, 3)
""",
    "mine_coal": """
call(craft_wood_pickaxe)
goto(cave)
gather(coal, 2)
""",
    "craft_stone_pickaxe": """
call(mine_stone)
goto(forest)
gather(wood, 2)
craft(plank, 2)
craft(stone_pickaxe)
""",
    "build_furnace": """
call(mine_stone)
gather(stone, 1)
goto(forest)
craft(furnace)
place(furnace)
""",
    "craft_torch": """
call(mine_coal)
craft(torch)
""",
    "mine_iron": """
call(craft_stone_pickaxe)
goto(cave)
gather(iron_ore, 3)
""",
    "make_glass": """
call(build_furnace)
goto(plains)
gather(sand, 1)
goto(cave)
gather(coal, 1)
goto(forest)
craft(glass)
""",
    "smelt_iron": """
call(mine_iron)
gather(coal, 3)
call(build_furnace)
craft(iron_ingot, 3)
""",
    "craft_iron_pickaxe": """
call(smelt_iron)
gather(wood, 2)
craft(plank, 2)
craft(stick, 2)
craft(iron_pickaxe)
""",
    "craft_lantern": """
call(smelt_iron)
goto(plains)
gather(sand, 1)
goto(cave)
gather(coal, 2)
goto(forest)
craft(glass)
gather(wood, 2)
craft(plank, 2)
craft(stick, 2)
craft(torch)
craft(lantern)
""",
}


def reference_resolver(name: str) -> str:
    try:
        return REFERENCE_PROGRAMS[name]
    except KeyError as exc:
        raise KeyError(name) from exc


def verify_reference(task_name: str) -> CraftingWorld:
    """Run a reference program from a fresh world; raise if it misses the goal."""
    world = CraftingWorld()
    world.run(REFERENCE_PROGRAMS[task_name], reference_resolver)
    if not TASKS[task_name].check(world):
        raise AssertionError(
            f"reference for {task_name} failed goal "
            f"({TASKS[task_name].goal_text()}); final {world.observe()}"
        )
    return world


def verify_all() -> Dict[str, int]:
    """Verify every reference program; returns steps used per task."""
    return {name: verify_reference(name).steps for name in REFERENCE_PROGRAMS}
