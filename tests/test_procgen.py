"""The procedural world used for the library-size sweep."""
import pytest

from env.dsl import ALL_ITEMS, LOCATIONS
from env.procgen import generate_world, run_program, verify_world
from env.world import ExecutionError


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_every_reference_program_reaches_its_goal(seed):
    world = generate_world(200, seed)
    steps = verify_world(world)
    assert len(steps) == 200
    assert max(steps.values()) <= 400


def test_smaller_worlds_are_prefixes_of_larger_ones():
    small, big = generate_world(50, 3), generate_world(200, 3)
    assert [s.program for s in small.skills] == [s.program for s in big.skills[:50]]
    assert [s.name for s in small.skills] == [s.name for s in big.prefix(50)]


def test_seeds_give_different_worlds_and_same_seed_is_stable():
    a, b, c = generate_world(30, 0), generate_world(30, 0), generate_world(30, 1)
    assert [s.program for s in a.skills] == [s.program for s in b.skills]
    assert [s.name for s in a.skills] != [s.name for s in c.skills]


def test_names_are_unique_and_disjoint_from_the_original_world():
    world = generate_world(200, 0)
    names = [s.name for s in world.skills]
    assert len(set(names)) == len(names)
    produced = {s.produces for s in world.skills}
    assert not produced & set(ALL_ITEMS)
    assert not set(world.locations) & set(LOCATIONS)


def test_every_prefix_has_root_and_composed_skills():
    world = generate_world(200, 0)
    for n in (10, 25, 50, 100, 200):
        kinds = {s.is_root for s in world.prefix(n)}
        assert kinds == {True, False}, n


def test_composed_skills_only_call_earlier_skills():
    world = generate_world(200, 5)
    seen = set()
    for s in world.skills:
        assert set(s.calls) <= seen
        seen.add(s.name)


def test_the_self_call_ban_blocks_the_trivial_shortcut():
    world = generate_world(20, 0)
    target = next(s for s in world.skills if not s.is_root)
    names = [s.name for s in world.skills]
    with pytest.raises(ExecutionError):
        run_program(world, f"call({target.name})", names, banned=(target.name,))
    done = run_program(world, target.program, names, banned=(target.name,))
    assert target.task.check(done)
