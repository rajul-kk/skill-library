import pytest

from env.dsl import DSLError, parse_program
from env.reference import REFERENCE_PROGRAMS, verify_reference
from env.tasks import CURRICULUM, TASKS, curriculum_order, validate_curriculum
from env.world import ExecutionError, CraftingWorld


def test_parser_accepts_comments_and_counts():
    prog = parse_program("goto(forest)  # go\n\ngather(wood, 3)")
    assert [(i.op, i.args) for i in prog] == [
        ("goto", ("forest",)),
        ("gather", ("wood", "3")),
    ]


@pytest.mark.parametrize(
    "bad",
    ["", "gather wood", "fly(forest)", "goto()", "goto(forest, plains)"],
)
def test_parser_rejects_malformed_programs(bad):
    with pytest.raises(DSLError):
        parse_program(bad)


def test_gathering_stone_requires_a_pickaxe():
    world = CraftingWorld()
    world.goto("cave")
    with pytest.raises(ExecutionError, match="wood_pickaxe"):
        world.gather("stone")


def test_resource_is_bound_to_its_locations():
    world = CraftingWorld()
    with pytest.raises(ExecutionError, match="not at forest"):
        world.gather("sand")


def test_crafting_requires_the_station_to_be_placed_here():
    world = CraftingWorld()
    world.inventory.update({"plank": 9, "stick": 4})
    with pytest.raises(ExecutionError, match="needs a table"):
        world.craft("wood_pickaxe")
    world.craft("table")
    world.place("table")
    world.craft("wood_pickaxe")
    assert world.has("wood_pickaxe")


def test_station_does_not_follow_the_agent():
    world = CraftingWorld()
    world.inventory.update({"plank": 4})
    world.craft("table")
    world.place("table")
    world.goto("cave")
    assert not world.station_here("table")


def test_call_without_a_library_is_an_error():
    world = CraftingWorld()
    with pytest.raises(ExecutionError, match="no skill library"):
        world.run("call(anything)")


def test_unknown_skill_call_is_reported():
    world = CraftingWorld()
    with pytest.raises(ExecutionError, match="unknown skill"):
        world.run("call(nope)", lambda name: (_ for _ in ()).throw(KeyError(name)))


def test_recursive_skills_hit_the_depth_limit_not_the_stack():
    world = CraftingWorld()
    with pytest.raises(ExecutionError, match="depth"):
        world.run("call(loop)", lambda name: "call(loop)")


def test_curriculum_is_acyclic_and_fully_defined():
    validate_curriculum()
    order = [t.name for t in curriculum_order()]
    assert len(order) == len(CURRICULUM) == 15
    seen = set()
    for task in curriculum_order():
        assert set(task.prereqs) <= seen
        seen.add(task.name)


@pytest.mark.parametrize("name", sorted(REFERENCE_PROGRAMS))
def test_every_reference_program_solves_its_task(name):
    world = verify_reference(name)
    assert TASKS[name].check(world)


def test_every_task_has_a_reference_program():
    assert set(REFERENCE_PROGRAMS) == set(TASKS)
