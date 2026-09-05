"""Config plumbing and resume logic for the GPU runner."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from baselines.gpu_runner import (  # noqa: E402
    METHOD_PRESETS,
    GPURun,
    build_lora_config,
    build_sleep_config,
    load_config,
)
from tests.test_lora_backend import CharTokenizer  # noqa: E402


def tiny_model():
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "qwen2", vocab_size=260, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.config.use_cache = False
    return model


class ScriptedPolicy:
    """Always writes the reference program, so runner plumbing can be tested alone."""

    def __init__(self, seed=0):
        self.seed = seed

    def propose(self, task, prompt, context_skills, attempt=0):
        from env.reference import REFERENCE_PROGRAMS

        return REFERENCE_PROGRAMS[task.name]


def base_config(tmp_path):
    return {
        "model": {"device": "cpu"},
        "lora": {"rank": 4, "batch_size": 2, "steps_per_night": 2, "max_length": 256},
        "sleep": {"every_k_tasks": 3, "replay": {"per_skill": 1, "max_total": 4}},
        "decode": {"max_new_tokens": 4},
        "loop": {"max_attempts": 1, "retry_rounds": 0},
        "run": {"run_id": "t", "root": str(tmp_path), "eval_repeats": 1, "seeds": [0]},
    }


def test_shipped_configs_parse():
    for name in ("default", "cpu_pipeline", "ablations"):
        assert load_config(f"configs/{name}.yaml")


def test_default_config_stays_inside_the_compute_budget():
    cfg = load_config("configs/default.yaml")
    lora = build_lora_config(cfg)
    assert 8 <= lora.rank <= 16
    assert lora.steps_per_night <= 500


def test_replay_is_capped_in_the_shipped_config():
    sleep = build_sleep_config(load_config("configs/default.yaml"))
    assert sleep.replay.enabled
    assert 0 < sleep.replay.max_total <= 64


def test_presets_turn_replay_off_where_the_method_requires_it():
    cfg = load_config("configs/default.yaml")
    for method in ("seq_no_replay", "isolated"):
        sleep = build_sleep_config(cfg, METHOD_PRESETS[method]["replay"])
        assert not sleep.replay.enabled
        assert sleep.replay.max_total == 0


def test_in_context_preset_keeps_full_library_in_the_prompt(tmp_path):
    run = GPURun(base_config(tmp_path), "in_context", tiny_model(), CharTokenizer())
    assert run.loop_config.retrieval.serve_full_text_always
    assert not run.consolidator.config.enabled


def test_isolated_preset_selects_the_isolated_backend(tmp_path):
    from sleep.lora import IsolatedLoRABackend

    run = GPURun(base_config(tmp_path), "isolated", tiny_model(), CharTokenizer())
    assert isinstance(run.backend, IsolatedLoRABackend)


def test_a_night_budget_stops_the_session_and_checkpoints(tmp_path):
    run = GPURun(base_config(tmp_path), "ours", tiny_model(), CharTokenizer(),
                 policy_factory=ScriptedPolicy)
    run.run(max_nights=1)
    assert run.consolidator.night_count == 1
    assert (run.root / "night_001" / "manifest.json").exists()
    assert (run.root / "snapshots.json").exists()


def test_the_next_session_resumes_where_the_last_one_stopped(tmp_path):
    config = base_config(tmp_path)
    first = GPURun(config, "ours", tiny_model(), CharTokenizer(),
                   policy_factory=ScriptedPolicy)
    first.run(max_nights=1)
    learned = first.library.names

    second = GPURun(config, "ours", tiny_model(), CharTokenizer(),
                    policy_factory=ScriptedPolicy)
    assert second.resume() == 1
    assert second.library.names == learned
    assert second.consolidator.night_count == 1
    assert second.snapshots


def test_resuming_a_run_that_never_started_returns_nothing(tmp_path):
    run = GPURun(base_config(tmp_path), "ours", tiny_model(), CharTokenizer())
    assert run.resume() is None


def test_manifest_records_the_backend_and_method(tmp_path):
    run = GPURun(base_config(tmp_path), "ours", tiny_model(), CharTokenizer(),
                 policy_factory=ScriptedPolicy)
    run.run(max_nights=1)
    manifest = json.loads((run.root / "night_001" / "manifest.json").read_text())
    assert manifest["backend"] == "lora"
    assert manifest["method"] == "ours"
    assert manifest["library_size"] >= 3


def test_a_finished_run_collects_into_a_plottable_payload(tmp_path):
    """The path from checkpoints to figures, which only runs after a real run."""
    from analysis.collect import collect_run, summary_table, write_collected
    from analysis.plots import make_all_figures

    config = base_config(tmp_path)
    for method in ("ours", "seq_no_replay"):
        run = GPURun(config, method, tiny_model(), CharTokenizer(),
                     policy_factory=ScriptedPolicy)
        run.run()
        assert len(run.library) == 15

    root = tmp_path / "t"
    payload = collect_run(root, ["ours", "seq_no_replay"], run_id="t")
    assert set(payload["methods"]) == {"ours", "seq_no_replay"}
    assert payload["methods"]["ours"]["normalized_forgetting_curve"]
    assert payload["methods"]["ours"]["final_stored_params"] > 0

    write_collected(root, ["ours", "seq_no_replay"], run_id="t")
    assert (root / "results.json").exists()
    assert "method" in summary_table(payload)

    figures = make_all_figures(payload, tmp_path / "figures")
    assert all(p.exists() for p in figures)


def test_collecting_a_run_with_no_snapshots_is_empty(tmp_path):
    from analysis.collect import collect_run

    (tmp_path / "ours").mkdir(parents=True)
    assert collect_run(tmp_path, ["ours"])["methods"] == {}
