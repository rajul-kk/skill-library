import pytest

from agent import LoopConfig, SimulatedPolicy, VoyagerLoop, called_skills
from agent.loop import context_skill_names
from analysis.metrics import (
    evaluate_library,
    forgetting_curve,
    forward_transfer,
    normalized_forgetting_curve,
)
from env.tasks import TASKS, curriculum_order
from library.retrieval import RetrievalConfig
from sleep import Consolidator, NullAdapter, SimulatedAdapter, SleepConfig
from sleep.replay import ReplayConfig


def build_loop(backend, sleep_config, retrieval, seed=0):
    consolidator = Consolidator(backend, sleep_config)
    policy = SimulatedPolicy(backend, seed=seed)
    return VoyagerLoop(policy, consolidator, LoopConfig(retrieval=retrieval))


def test_called_skills_reads_the_reference_program():
    assert called_skills("mine_stone") == ["craft_wood_pickaxe"]
    assert called_skills("gather_wood") == []


def test_root_skills_depend_on_no_recall():
    roots = [t.name for t in curriculum_order() if not called_skills(t.name)]
    assert roots == ["gather_wood"]


def test_in_context_baseline_learns_the_whole_tree():
    loop = build_loop(
        NullAdapter(), SleepConfig(enabled=False),
        RetrievalConfig(serve_full_text_always=True),
    )
    loop.run_curriculum()
    assert len(loop.library) == 15


def test_consolidated_skills_leave_the_prompt():
    backend = SimulatedAdapter()
    loop = build_loop(backend, SleepConfig(every_k_tasks=3), RetrievalConfig())
    loop.run_curriculum()
    loop.finish()
    task = TASKS["craft_lantern"]
    in_prompt = context_skill_names(loop.library, task, loop.config.retrieval)
    assert loop.library["gather_wood"].consolidated
    assert "gather_wood" not in in_prompt


def test_resolving_a_task_twice_does_not_requeue_it():
    backend = SimulatedAdapter()
    loop = build_loop(backend, SleepConfig(every_k_tasks=3), RetrievalConfig())
    loop.run_curriculum()
    loop.finish()
    before = loop.consolidator.night_count
    task = TASKS["gather_wood"]
    loop.run_episode(task, episode=999)
    assert loop.consolidator.pending == []
    assert loop.consolidator.night_count == before


def test_a_skill_keeps_the_library_size_it_was_learned_at():
    loop = build_loop(NullAdapter(), SleepConfig(enabled=False), RetrievalConfig())
    loop.run_curriculum()
    sizes = [s.learned_at_size for s in loop.library]
    assert sizes == sorted(sizes)


def run_for_snapshots(method_backend, sleep_config, seed=0):
    backend = method_backend()
    loop = build_loop(backend, sleep_config, RetrievalConfig(), seed=seed)
    snapshots = []

    def on_night(record, library):
        snapshots.append(
            evaluate_library(
                library, lambda s: SimulatedPolicy(backend, seed=s), loop.config,
                night=record.night, episode=record.episode,
                stored_params=backend.stored_parameters(), repeats=3, seed=seed,
            )
        )

    loop.on_night = on_night
    loop.run_curriculum()
    loop.finish()
    return snapshots


def test_no_replay_forgets_and_replay_does_not():
    """The result the whole study rests on, as a regression test."""
    no_replay = SleepConfig(
        every_k_tasks=3, replay=ReplayConfig(enabled=False, per_skill=0, max_total=0)
    )
    with_replay = SleepConfig(
        every_k_tasks=3, replay=ReplayConfig(per_skill=2, max_total=24)
    )

    forgetful, retentive = [], []
    for seed in range(4):
        forgetful += run_for_snapshots(SimulatedAdapter, no_replay, seed)
        retentive += run_for_snapshots(SimulatedAdapter, with_replay, seed)

    old_forgetful = forgetting_curve(forgetful, "skills")
    old_retentive = forgetting_curve(retentive, "skills")
    far = max(k for k in old_forgetful if k in old_retentive)

    assert old_forgetful[far] < 0.25, "no-replay baseline is not forgetting"
    assert old_retentive[far] > 0.5, "replay is not retaining"


def test_normalised_curve_starts_at_full_retention():
    snapshots = run_for_snapshots(
        SimulatedAdapter, SleepConfig(every_k_tasks=3, replay=ReplayConfig())
    )
    curve = normalized_forgetting_curve(snapshots, "skills", min_samples=1)
    assert curve[min(curve)] == pytest.approx(1.0)


def test_curve_ignores_skills_that_need_no_recall():
    snapshots = run_for_snapshots(SimulatedAdapter, SleepConfig(every_k_tasks=3))
    with_roots = forgetting_curve(snapshots, "skills", exclude_recall_free=False,
                                  min_samples=1)
    without = forgetting_curve(snapshots, "skills", min_samples=1)
    assert set(without) <= set(with_roots)


def test_forward_transfer_reports_attempts_to_first_success():
    loop = build_loop(NullAdapter(), SleepConfig(enabled=False),
                      RetrievalConfig(serve_full_text_always=True))
    loop.run_curriculum()
    transfer = forward_transfer(loop.episodes, loop.library)
    assert transfer["tasks_solved"] == 15
    assert 1.0 <= transfer["mean_attempts_all"] <= 4.0
