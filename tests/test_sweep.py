"""The library-size sweep: configs, prompts, runner, resume and the pre-registered analysis."""
from pathlib import Path

import pytest

from env.procgen import generate_world
from skillmetrics import read_jsonl
from sweep.analyze import analyze, summary_text
from sweep.backends import SimulatedSweepBackend
from sweep.budget import train_examples_per_night
from sweep.config import ARMS, SweepConfig, from_dict, load_sweep_config
from sweep.prompts import build_prompt, in_context_names, retrieve
from sweep.runner import eval_set, run_model_seed

CONFIGS = sorted(Path("configs").glob("sweep_*.yaml"))
SIM = "sim:Qwen/Qwen2.5-1.5B-Instruct"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_sweep_config_loads(path):
    cfg = load_sweep_config(path)
    assert cfg.arm_list()


def test_full_design_matches_the_protocol():
    cfg = load_sweep_config("configs/sweep_full.yaml")
    assert cfg.milestones == [10, 25, 50, 100, 200]
    assert len(cfg.seeds) == 20
    assert set(cfg.arms) == set(ARMS)
    assert cfg.models == ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]


def test_pilot_seeds_are_disjoint_from_confirmatory_seeds():
    pilot = set(load_sweep_config("configs/sweep_pilot.yaml").seeds)
    full = set(load_sweep_config("configs/sweep_full.yaml").seeds)
    assert pilot and not pilot & full


def test_unknown_config_keys_are_rejected():
    with pytest.raises(KeyError):
        from_dict({"name": "x", "bogus": 1})
    with pytest.raises(KeyError):
        from_dict({"arms": ["not_an_arm"]})


@pytest.fixture(scope="module")
def world():
    return generate_world(60, 0)


def test_retrieval_puts_the_task_itself_first(world):
    lib = world.prefix(60)
    for s in lib[::7]:
        assert retrieve(s, lib, 8)[0].name == s.name


def test_prompts_carry_what_each_arm_is_allowed_to_see(world):
    lib = world.prefix(50)
    task = next(s for s in lib if not s.is_root and s.index > 30)
    full = build_prompt(world, "full", task, lib, 8)
    topk = build_prompt(world, "topk", task, lib, 8)
    ptr = build_prompt(world, "pointers", task, lib, 8)
    none = build_prompt(world, "none", task, [], 8)
    assert all(s.name in full for s in lib)
    assert task.program in topk and task.program in full
    assert task.program not in ptr and "(in weights)" in ptr
    assert "Known skills:\n(none)" in none
    assert len(none) < len(ptr) < len(topk) < len(full)
    assert "do not call the skill for this task itself" in ptr
    assert f"call {task.name}" not in none  # naming the banned skill primed the model to call it (pilot 1)
    assert in_context_names("pointers", task, lib, 8) == set()
    assert task.name in in_context_names("topk", task, lib, 8)


def test_eval_sets_are_stratified_and_fixed(world):
    cfg = SweepConfig(eval_max=10, eval_full_up_to=20, milestones=[10, 60])
    assert len(eval_set(world, 10, 0, cfg)) == 10
    picked = eval_set(world, 60, 0, cfg)
    assert len(picked) == 10
    assert [s.name for s in picked] == [s.name for s in eval_set(world, 60, 0, cfg)]
    quintiles = {s.index * 5 // 60 for s in picked}
    assert quintiles == {0, 1, 2, 3, 4}


def test_replay_caps_shape_the_nightly_training_set():
    cfg = SweepConfig(milestones=[10, 25, 50], night_every=10)
    assert train_examples_per_night("seq_no_replay", cfg) == [10, 10, 5, 10, 10, 5]
    b24 = train_examples_per_night("ours_b24", cfg)
    cover = train_examples_per_night("ours_cover", cfg)
    assert max(b24) <= 10 + 24
    assert cover[-1] == 10 + 40
    assert train_examples_per_night("isolated", cfg) == [10, 10, 5, 10, 10, 5]


def small_cfg(**kw):
    base = dict(backend="simulated", models=[SIM], seeds=[0, 1, 2], milestones=[10, 25],
                arms=["no_skill", "ic_full", "ic_topk", "ours_b24", "ours_cover", "seq_no_replay", "isolated"],
                device="cpu", dtype="float32")
    base.update(kw)
    return from_dict(base)


@pytest.fixture(scope="module")
def sim_records(tmp_path_factory):
    cfg = small_cfg()
    out = tmp_path_factory.mktemp("sweep") / "records.jsonl"
    for seed in cfg.seeds:
        run_model_seed(cfg, SIM, seed, SimulatedSweepBackend(cfg, seed), out, log=lambda *_: None)
    return out


def test_the_simulated_sweep_writes_every_record_kind(sim_records):
    recs = read_jsonl(sim_records)
    kinds = {r["kind"] for r in recs}
    assert kinds == {"attempt", "train", "arm_done"}
    done = [r for r in recs if r["kind"] == "arm_done"]
    assert len(done) == 3 * 7
    attempts = [r for r in recs if r["kind"] == "attempt"]
    assert {r["milestone"] for r in attempts} == {10, 25}
    assert all("text" in r for r in attempts)
    trains = [r for r in recs if r["kind"] == "train"]
    assert {r["arm"] for r in trains} == {"ours_b24", "ours_cover", "seq_no_replay", "isolated"}


def test_isolated_storage_grows_and_cumulative_does_not(sim_records):
    done = {(r["arm"], r["seed"]): r for r in read_jsonl(sim_records) if r["kind"] == "arm_done"}
    assert done[("isolated", 0)]["stored_lora_params"] == 3 * done[("ours_b24", 0)]["stored_lora_params"]
    assert done[("ic_full", 0)]["stored_lora_params"] == 0


def test_rerunning_skips_finished_arms(sim_records):
    cfg = small_cfg(seeds=[0])
    before = len(read_jsonl(sim_records))
    summary = run_model_seed(cfg, SIM, 0, SimulatedSweepBackend(cfg, 0), sim_records, log=lambda *_: None)
    assert summary == {}
    assert len(read_jsonl(sim_records)) == before


def test_no_skill_never_solves_a_composed_skill(sim_records):
    rows = [r for r in read_jsonl(sim_records) if r["kind"] == "attempt" and r["arm"] == "no_skill"]
    assert rows and not any(r["success"] for r in rows if not r["is_root"])


def test_analysis_runs_and_reports_the_primary_contrasts(sim_records):
    report = analyze(read_jsonl(sim_records), n_boot=200)
    m = report["models"][SIM]
    assert set(m["checks"]["floor"]) == {10, 25}
    assert {"ours_b24_vs_ic_full", "ours_b24_vs_ic_topk"} <= set(m["primary"]) or m["model_dropped"]
    full = m["exploratory_all_milestones"]["ours_b24_vs_ic_full"]
    assert set(full["tokens"]["ratio"]) == {10, 25}
    assert full["tokens"]["ratio"][25]["point"] > full["tokens"]["ratio"][10]["point"]
    text = summary_text(report)
    assert "floor" in text and "ours_b24" in text


def test_floor_failure_drops_the_cell(sim_records):
    recs = read_jsonl(sim_records)
    for r in recs:
        if r["kind"] == "attempt" and r["arm"] == "no_skill" and r["milestone"] == 25:
            r["success"] = 0
    m = analyze(recs, n_boot=100)["models"][SIM]
    assert not m["checks"]["floor"][25]["passed"]
    assert 25 not in m["milestones_in_analysis"]


def tiny_model_cached():
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache("HuggingFaceTB/SmolLM2-135M-Instruct", "config.json"), str)
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not tiny_model_cached(), reason="SmolLM2-135M-Instruct not in the local HF cache")
def test_real_tiny_model_runs_every_arm_kind(tmp_path):
    pytest.importorskip("peft")
    from sweep.backends import HFSweepBackend

    cfg = from_dict(dict(
        backend="hf", models=["HuggingFaceTB/SmolLM2-135M-Instruct"], seeds=[0], milestones=[10],
        arms=["no_skill", "ic_topk", "ours_b24", "isolated"], device="cpu", dtype="float32", prompt_wrapper="chat",
        train={"rank": 4, "alpha": 8, "epochs": 1, "batch_size": 4, "learning_rate": 1e-3},
        decode={"max_new_tokens": 24, "batch_size": 10},
    ))
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    backend = HFSweepBackend(cfg, cfg.models[0], 0)
    out = tmp_path / "records.jsonl"
    run_model_seed(cfg, cfg.models[0], 0, backend, out, log=lambda *_: None)
    recs = read_jsonl(out)
    trains = [r for r in recs if r["kind"] == "train"]
    assert trains and all(r["train_tokens"] > 0 and r["lora_params"] > 0 for r in trains)
    attempts = [r for r in recs if r["kind"] == "attempt"]
    assert len(attempts) == 4 * 10
    assert all(0 < r["gen_tokens"] <= 24 for r in attempts)
    assert all(isinstance(r["text"], str) for r in attempts)
    wrapped = backend.wrap("Program:\n")
    assert "Program:\n" in wrapped and len(wrapped) > len("Program:\n")  # the chat template surrounds the prompt
    assert all(r["lora_params"] > 0 for r in attempts if r["arm"] in ("ours_b24", "isolated"))
    # the base weights must come back untouched after LoRA arms unload
    assert not any("lora_" in n for n, _ in backend.base.named_parameters())


@pytest.mark.skipif(not tiny_model_cached(), reason="SmolLM2-135M-Instruct not in the local HF cache")
def test_micro_batches_give_the_full_batch_gradient():
    torch = pytest.importorskip("torch")
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sleep.backend import Example
    from sleep.lora import encode
    from sweep.backends import accumulate_batch_loss

    name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    texts = [("make a plank", "craft plank"), ("make two sticks from wood", "craft stick; craft stick"),
             ("a much longer request that needs more tokens to describe", "gather wood; craft plank; craft table"),
             ("x", "y")]
    items = [encode(tok, Example(skill=f"s{i}", prompt=p, target=t), 128) for i, (p, t) in enumerate(texts)]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    grads = []
    for micro in (0, 1, 3):
        model.zero_grad(set_to_none=True)
        loss = accumulate_batch_loss(model, items, pad, micro, "cpu")
        grads.append((loss, model.lm_head.weight.grad.clone()))
    for loss, g in grads[1:]:
        assert abs(loss - grads[0][0]) < 1e-4
        assert torch.allclose(g, grads[0][1], atol=1e-6, rtol=1e-4)
