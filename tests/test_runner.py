"""End-to-end checks on the experiment driver and the figures it feeds."""
import json

from analysis.plots import make_all_figures
from baselines import MAIN_ORDER, RunConfig, get_method, run_method, run_all
from baselines.methods import frequency_ablation, replay_size_ablation, strategy_ablation


def small(tmp_path, **kwargs):
    return RunConfig(
        run_id="test", seeds=(0, 1), eval_repeats=2, root=str(tmp_path), **kwargs
    )


def test_every_named_method_runs(tmp_path):
    for key in MAIN_ORDER:
        result = run_method(get_method(key), small(tmp_path))
        assert result.snapshots, f"{key} produced no snapshots"
        assert result.transfer["tasks_solved"] > 0


def test_results_json_is_written_and_reloadable(tmp_path):
    config = small(tmp_path)
    run_all(["ours", "seq_no_replay"], config)
    payload = json.loads((tmp_path / "test" / "results.json").read_text())
    assert set(payload["methods"]) == {"ours", "seq_no_replay"}
    assert payload["methods"]["ours"]["normalized_forgetting_curve"]


def test_checkpoints_are_written_for_the_first_seed(tmp_path):
    run_method(get_method("ours"), small(tmp_path))
    nights = sorted((tmp_path / "test" / "ours").glob("night_*"))
    assert nights
    manifest = json.loads((nights[0] / "manifest.json").read_text())
    assert manifest["method"] == "ours"
    assert manifest["backend"] == "simulated"


def test_replay_beats_no_replay_on_the_oldest_skills(tmp_path):
    config = small(tmp_path)
    ours = run_method(get_method("ours"), config)
    naive = run_method(get_method("seq_no_replay"), config)
    assert ours.final("earliest_3") > naive.final("earliest_3")


def test_our_storage_is_flat_where_isolation_grows(tmp_path):
    config = small(tmp_path)
    ours = run_method(get_method("ours"), config)
    isolated = run_method(get_method("isolated"), config)
    assert ours.final("stored_params") < isolated.final("stored_params") / 5


def test_consolidation_cuts_prompt_tokens(tmp_path):
    config = small(tmp_path)
    ours = run_method(get_method("ours"), config)
    in_context = run_method(get_method("in_context"), config)
    assert ours.tokens["mean_prompt_tokens"] < in_context.tokens["mean_prompt_tokens"]


def test_ablation_grids_are_well_formed():
    assert len(replay_size_ablation()) == 5
    assert len(frequency_ablation()) == 3
    strategies = strategy_ablation()
    assert all(m.sleep.replay.max_total == 8 for m in strategies.values())


def test_figures_render_from_a_results_payload(tmp_path):
    config = small(tmp_path)
    run_all(["ours", "isolated", "seq_no_replay"], config)
    payload = json.loads((tmp_path / "test" / "results.json").read_text())
    paths = make_all_figures(payload, tmp_path / "figures")
    assert all(p.exists() and p.stat().st_size > 5000 for p in paths)
