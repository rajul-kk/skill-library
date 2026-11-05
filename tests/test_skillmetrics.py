"""The shareable metrics module: FLOP accounting, bootstrap, crossover and floor checks."""
import ast
import math
from pathlib import Path

import pytest

from skillmetrics import (
    SHAPES,
    aggregate,
    attempt_record,
    bootstrap,
    break_even_queries,
    crossover_analysis,
    decode_flops,
    efficiency_ratio,
    floor_check,
    headroom_check,
    interpolated_crossover,
    lora_train_flops,
    mean_ci,
    prefill_flops,
    shape_from_hf_config,
    sustained_crossover,
    train_record,
)

QWEN15 = SHAPES["Qwen/Qwen2.5-1.5B-Instruct"]


def test_skillmetrics_imports_nothing_from_this_repo():
    repo_roots = {"env", "library", "sleep", "agent", "baselines", "analysis", "sweep", "scripts"}
    for path in Path("skillmetrics").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                assert node.module.split(".")[0] not in repo_roots, f"{path} imports {node.module}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in repo_roots


def test_qwen_15b_parameter_count_matches_the_published_size():
    total = QWEN15.non_embedding_params() + QWEN15.head_params()
    assert 1.50e9 < total < 1.56e9


def test_lora_params_match_peft_on_a_tiny_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = dict(vocab_size=260, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
               num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model("qwen2", **cfg))
    shape = shape_from_hf_config("tiny", {**cfg, "model_type": "qwen2"})
    blocks = sum(p.numel() for n, p in model.named_parameters() if ".layers." in n and p.dim() == 2)
    assert blocks == shape.non_embedding_params() - shape.n_layer * (shape.d_model + 2 * shape.d_kv)
    peft = get_peft_model(model, LoraConfig(r=4, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    trainable = sum(p.numel() for p in peft.parameters() if p.requires_grad)
    assert trainable == shape.lora_params(4)
    del torch


def test_prefill_is_about_2n_per_token_for_short_prompts_and_superlinear_for_long():
    n = QWEN15.non_embedding_params()
    short = prefill_flops(QWEN15, 100)
    assert 2 * n * 100 < short < 2.2 * n * 100 + 2 * QWEN15.head_params()
    assert prefill_flops(QWEN15, 20_000) / prefill_flops(QWEN15, 10_000) > 2.0


def test_decode_cost_grows_with_context():
    assert decode_flops(QWEN15, 10_000, 50) > decode_flops(QWEN15, 100, 50)


def test_training_costs_two_or_three_forwards():
    base = lora_train_flops(QWEN15, 1000, 1000 * 100, 0, False)
    ckpt = lora_train_flops(QWEN15, 1000, 1000 * 100, 0, True)
    assert ckpt / base == pytest.approx(1.5)


def test_bootstrap_interval_brackets_the_mean_and_is_deterministic():
    xs = [0.1, 0.4, 0.5, 0.6, 0.9, 0.3, 0.7]
    a, b = mean_ci(xs, n_boot=2000, seed=1), mean_ci(xs, n_boot=2000, seed=1)
    assert a == b
    assert a["lo"] <= a["point"] <= a["hi"]


def test_bootstrap_of_a_constant_has_zero_width():
    r = bootstrap([1.0] * 5, lambda xs: sum(xs) / len(xs), n_boot=200)
    assert r["lo"] == r["hi"] == 1.0


def test_sustained_crossover_requires_every_larger_size_to_win():
    assert sustained_crossover({10: 0.5, 25: 1.2, 50: 0.9, 100: 1.5, 200: 2.0}) == 100
    assert sustained_crossover({10: 1.1, 25: 1.2}) == 10
    assert sustained_crossover({10: 0.5, 25: 0.9}) is None


def test_interpolated_crossover_sits_between_the_bracketing_sizes():
    n = interpolated_crossover({10: 0.5, 25: 0.8, 50: 2.0})
    assert 25 < n < 50


def cell(success, tokens, flops, train=0.0):
    return {"success": success, "inf_tokens": tokens, "inf_flops": flops, "train_flops_cum": train}


def test_efficiency_ratio_trades_success_against_cost():
    ours, comp = [cell(0.7, 400, 1e12, train=1e15)], [cell(0.85, 900, 3e12)]
    assert efficiency_ratio(ours, comp, "tokens") == pytest.approx((0.7 / 400) / (0.85 / 900))
    assert efficiency_ratio(ours, comp, "flops", horizon=1.0) < efficiency_ratio(ours, comp, "flops")


def test_break_even_matches_the_amortisation_rule():
    ours, comp = [cell(1.0, 1, 1.0, train=100.0)], [cell(1.0, 1, 2.0)]
    assert break_even_queries(ours, comp) == pytest.approx(100.0)
    assert break_even_queries([cell(1.0, 1, 3.0, 5.0)], comp) == math.inf


def test_crossover_analysis_finds_a_planted_crossover():
    sizes = [10, 25, 50, 100, 200]
    ours, comp = {}, {}
    for n in sizes:
        ours[n] = {s: cell(0.7 + 0.01 * (s % 3), 300 + n, 1e12) for s in range(10)}
        comp[n] = {s: cell(0.9, 100 + 20 * n, 1e12) for s in range(10)}
    out = crossover_analysis(ours, comp, "tokens", n_boot=300)
    assert out["n_star"] == 25
    assert out["ratio"][200]["lo"] > 1


def test_floor_and_headroom_checks():
    ok = [{"success_root": 0.5, "n_root": 4, "success_crafted": 0.0, "n_crafted": 6}]
    bad = [{"success_root": 0.1, "n_root": 4, "success_crafted": 0.0, "n_crafted": 6}]
    assert floor_check(ok)["passed"] and not floor_check(bad)["passed"]
    ic = {"ic_full": [{"success_crafted": 0.6, "n_crafted": 6}]}
    assert headroom_check(ic, ok)["passed"]
    assert not headroom_check({"ic_full": [{"success_crafted": 0.1, "n_crafted": 6}]}, ok)["passed"]


def test_aggregate_charges_training_up_to_each_milestone():
    common = dict(model="Qwen/Qwen2.5-1.5B-Instruct", seed=0, arm="ours_b24", lora_params=4_358_144)
    recs = [
        attempt_record(**common, milestone=10, skill="a", is_root=True, success=1, prompt_tokens=200, gen_tokens=20),
        attempt_record(**common, milestone=10, skill="b", is_root=False, success=0, prompt_tokens=220, gen_tokens=30),
        attempt_record(**common, milestone=25, skill="a", is_root=True, success=1, prompt_tokens=200, gen_tokens=20),
        train_record(**common, night=1, library_size=10, n_new=10, n_replay=0, train_tokens=8000,
                     train_tokens_sq=8000 * 200, grad_checkpointing=False),
        train_record(**common, night=2, library_size=20, n_new=10, n_replay=10, train_tokens=16000,
                     train_tokens_sq=16000 * 200, grad_checkpointing=False),
    ]
    cells = aggregate(recs, SHAPES)
    c10 = cells[("Qwen/Qwen2.5-1.5B-Instruct", "ours_b24", 10, 0)]
    c25 = cells[("Qwen/Qwen2.5-1.5B-Instruct", "ours_b24", 25, 0)]
    assert c10["success"] == 0.5 and c10["success_root"] == 1.0 and c10["success_crafted"] == 0.0
    assert c25["train_flops_cum"] > c10["train_flops_cum"] > 0
    assert c10["train_tokens_cum"] == 8000 and c25["train_tokens_cum"] == 24000


def test_records_reject_missing_fields():
    with pytest.raises(ValueError):
        attempt_record(model="m", seed=0)
