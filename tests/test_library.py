import pytest

from env.reference import REFERENCE_PROGRAMS, verify_reference
from env.tasks import TASKS
from library import (
    RetrievalConfig,
    Skill,
    SkillLibrary,
    context_token_cost,
    skill_from_episode,
)


@pytest.fixture
def library():
    lib = SkillLibrary()
    for name, body in REFERENCE_PROGRAMS.items():
        lib.add(
            skill_from_episode(
                TASKS[name], body, verify_reference(name), episode=0, night=0,
                library_size=len(lib),
            )
        )
    return lib


def test_notes_describe_what_the_run_actually_produced(library):
    notes = library["mine_iron"].notes
    assert "requires: craft_stone_pickaxe" in notes
    assert "guarantees: 3x iron_ore" in notes
    assert "ends at: cave" in notes


def test_resolver_executes_a_stored_skill(library):
    from env.world import CraftingWorld

    world = CraftingWorld()
    world.run("call(craft_table)", library.resolver())
    assert world.stations["table"]


def test_resolver_counts_uses(library):
    resolve = library.resolver()
    resolve("gather_wood")
    resolve("gather_wood")
    assert library["gather_wood"].uses == 2


def test_re_adding_a_skill_keeps_its_original_learning_time(library):
    original = library["mine_stone"]
    original.consolidated_night = 2
    replacement = Skill(
        name="mine_stone", description="d", body="goto(cave)", task="mine_stone"
    )
    library.add(replacement)
    stored = library["mine_stone"]
    assert stored.body == "goto(cave)"
    assert stored.learned_at_size == original.learned_at_size
    assert stored.consolidated_night == 2


def test_consolidation_swaps_full_text_for_a_pointer(library):
    task = TASKS["craft_lantern"]
    config = RetrievalConfig()
    before = context_token_cost(library, task, config)
    library.mark_consolidated([s.name for s in library], night=1)
    after = context_token_cost(library, task, config)
    assert after < before / 2


def test_no_consolidation_config_never_drops_full_text(library):
    task = TASKS["craft_lantern"]
    config = RetrievalConfig(serve_full_text_always=True)
    before = context_token_cost(library, task, config)
    library.mark_consolidated([s.name for s in library], night=1)
    assert context_token_cost(library, task, config) == before


def test_mark_consolidated_is_idempotent(library):
    assert library.mark_consolidated(["mine_stone"], night=1) == ["mine_stone"]
    assert library.mark_consolidated(["mine_stone"], night=5) == []
    assert library["mine_stone"].consolidated_night == 1


def test_retrieval_puts_prerequisites_first(library):
    from library.retrieval import relevant_skills

    task = TASKS["craft_lantern"]
    top = [s.name for s in relevant_skills(library, task, RetrievalConfig(max_skills=3))]
    assert set(top) == set(task.prereqs)


def test_library_survives_a_save_load_round_trip(library, tmp_path):
    library.mark_consolidated(["gather_wood"], night=3)
    path = library.save(tmp_path / "library.json")
    restored = SkillLibrary.load(path)
    assert restored.names == library.names
    assert restored["gather_wood"].consolidated_night == 3
    assert restored["mine_iron"].notes == library["mine_iron"].notes
