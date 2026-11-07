import json

import pytest

from env.reference import REFERENCE_PROGRAMS, verify_reference
from env.tasks import TASKS, curriculum_order
from library import SkillLibrary, skill_from_episode
from sleep import (
    Consolidator,
    Example,
    IsolatedSimulatedAdapters,
    NullAdapter,
    ReplayBuffer,
    ReplayConfig,
    SimulatedAdapter,
    SleepConfig,
    latest_night,
    load_night,
    save_night,
)
from sleep.checkpoint import Manifest


def build_library(n=9):
    lib = SkillLibrary()
    for task in curriculum_order()[:n]:
        lib.add(
            skill_from_episode(
                task, REFERENCE_PROGRAMS[task.name], verify_reference(task.name),
                episode=0, night=0, library_size=len(lib),
            )
        )
    return lib


# ---------------------------- replay buffer ---------------------------- #


def make_buffer(names, **kwargs):
    buf = ReplayBuffer(ReplayConfig(**kwargs))
    for i, n in enumerate(names):
        buf.add(Example(n, "p", "t", night_added=i), tier=i)
    return buf


def test_buffer_respects_the_total_cap():
    buf = make_buffer([f"s{i}" for i in range(10)], per_skill=3, max_total=7)
    assert len(buf.sample()) == 7


def test_buffer_respects_the_per_skill_cap():
    buf = make_buffer(["a", "b"], per_skill=1, max_total=99)
    assert len(buf.sample()) == 2


def test_disabled_buffer_yields_nothing():
    buf = make_buffer(["a", "b"], enabled=False)
    assert buf.sample() == []


def test_zero_budget_yields_nothing():
    buf = make_buffer(["a", "b"], max_total=0)
    assert buf.sample() == []


def test_sampled_examples_are_flagged_as_replay():
    buf = make_buffer(["a"], per_skill=1)
    assert all(e.is_replay for e in buf.sample())


def test_new_skills_are_excluded_from_their_own_replay():
    buf = make_buffer(["a", "b", "c"], per_skill=1)
    assert {e.skill for e in buf.sample(exclude=["b"])} == {"a", "c"}


def test_round_robin_rotates_which_skills_a_scarce_buffer_covers():
    buf = make_buffer(["a", "b", "c", "d"], per_skill=1, max_total=2)
    first = {e.skill for e in buf.sample()}
    second = {e.skill for e in buf.sample()}
    assert first != second


def test_recency_spends_a_scarce_buffer_on_the_newest_skills():
    buf = make_buffer(["old", "mid", "new"], per_skill=1, max_total=1, strategy="recency")
    assert [e.skill for e in buf.sample()] == ["new"]


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown replay strategy"):
        ReplayBuffer(ReplayConfig(strategy="vibes"))


def test_buffer_survives_a_state_round_trip():
    buf = make_buffer(["a", "b"], per_skill=1)
    restored = ReplayBuffer.from_state_dict(buf.state_dict())
    assert restored.skills == buf.skills
    assert len(restored) == len(buf)


def test_uniform_sampling_continues_identically_after_a_restore():
    buf = make_buffer([f"s{i}" for i in range(12)], per_skill=1, max_total=4, strategy="uniform")
    buf.sample()
    restored = ReplayBuffer.from_state_dict(json.loads(json.dumps(buf.state_dict())))
    for _ in range(3):
        assert [e.skill for e in restored.sample()] == [e.skill for e in buf.sample()]


# ---------------------------- consolidation ---------------------------- #


def test_a_night_fires_only_after_k_new_skills():
    c = Consolidator(SimulatedAdapter(), SleepConfig(every_k_tasks=3))
    c.note_new_skill("a")
    c.note_new_skill("b")
    assert not c.should_sleep()
    c.note_new_skill("c")
    assert c.should_sleep()


def test_duplicate_pending_skills_are_ignored():
    c = Consolidator(SimulatedAdapter(), SleepConfig(every_k_tasks=2))
    c.note_new_skill("a")
    c.note_new_skill("a")
    assert not c.should_sleep()


def test_a_night_consolidates_and_records_replay():
    lib = build_library(6)
    c = Consolidator(SimulatedAdapter(), SleepConfig(every_k_tasks=3))
    for name in lib.names[:3]:
        c.note_new_skill(name)
    c.run_night(lib, episode=3)
    for name in lib.names[3:6]:
        c.note_new_skill(name)
    second = c.run_night(lib, episode=6)
    assert second.night == 2
    assert set(second.replayed_skills) == set(lib.names[:3])
    assert all(lib[n].consolidated for n in lib.names[:6])


def test_replay_is_interleaved_rather_than_appended():
    c = Consolidator(SimulatedAdapter(), SleepConfig())
    new = [Example(f"n{i}", "p", "t") for i in range(2)]
    replay = [Example(f"r{i}", "p", "t", is_replay=True) for i in range(4)]
    order = [e.skill for e in c._interleave(new, replay)]
    assert order.index("n1") < order.index("r3")


def test_a_disabled_sleep_still_ticks_the_night_counter():
    lib = build_library(3)
    c = Consolidator(NullAdapter(), SleepConfig(enabled=False, every_k_tasks=3))
    for name in lib.names:
        c.note_new_skill(name)
    record = c.run_night(lib, episode=3)
    assert record.night == 1
    assert record.stats.steps == 0
    assert not any(s.consolidated for s in lib)


def test_shared_adapter_storage_stays_flat_while_isolated_grows():
    lib = build_library(9)
    shared = Consolidator(SimulatedAdapter(), SleepConfig(every_k_tasks=3))
    isolated = Consolidator(IsolatedSimulatedAdapters(), SleepConfig(every_k_tasks=3))
    for consolidator, library in ((shared, build_library(9)), (isolated, lib)):
        for i, name in enumerate(library.names, 1):
            consolidator.note_new_skill(name)
            if i % 3 == 0:
                consolidator.run_night(library, episode=i)
    assert shared.nights[0].stored_params == shared.nights[-1].stored_params
    assert isolated.nights[-1].stored_params == 9 * isolated.nights[0].stored_params // 3


def test_replay_keeps_the_oldest_skill_that_no_replay_forgets():
    order = [f"s{i}" for i in range(10)]
    without = SimulatedAdapter()
    with_replay = SimulatedAdapter()
    seen = []
    for night, name in enumerate(order, 1):
        without.consolidate([Example(name, "p", "t", night)], night)
        batch = [Example(name, "p", "t", night)]
        batch += [Example(s, "p", "t", night, True) for s in seen]
        with_replay.consolidate(batch, night)
        seen.append(name)
    assert without.knows("s0") < 0.1
    assert with_replay.knows("s0") > 0.5


def test_isolated_adapters_never_forget_but_cost_storage():
    adapter = IsolatedSimulatedAdapters()
    for night, name in enumerate([f"s{i}" for i in range(8)], 1):
        adapter.consolidate([Example(name, "p", "t", night)], night)
    assert adapter.knows("s0") == 1.0
    assert adapter.stored_parameters() == 8 * adapter.base_params


# ---------------------------- checkpoints ---------------------------- #


def test_a_run_resumes_from_its_last_night(tmp_path):
    lib = build_library(6)
    c = Consolidator(SimulatedAdapter(), SleepConfig(every_k_tasks=3))
    for name in lib.names[:3]:
        c.note_new_skill(name)
    c.run_night(lib, episode=3)
    manifest = Manifest(
        run_id="t", method="ours", night=1, episode=3, library_size=len(lib),
        stored_params=c.backend.stored_parameters(), backend="simulated",
    )
    save_night(tmp_path, 1, lib, c, manifest)

    assert latest_night(tmp_path) == 1
    fresh = Consolidator(SimulatedAdapter(), SleepConfig(every_k_tasks=3))
    restored_lib, restored_manifest = load_night(tmp_path, 1, fresh)

    assert restored_manifest.method == "ours"
    assert restored_lib.names == lib.names
    assert fresh.night_count == 1
    assert fresh.buffer.skills == c.buffer.skills
    assert fresh.backend.retention == c.backend.retention


def test_listing_nights_on_an_empty_root_is_not_an_error(tmp_path):
    assert latest_night(tmp_path / "nothing") is None
