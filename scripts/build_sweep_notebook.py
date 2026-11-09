"""Generate the Kaggle sweep notebooks: 05_sweep_pilot.ipynb (pilot) and 06_sweep_confirmatory.ipynb (reduced design)."""
import _paths  # noqa: F401

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip().splitlines(True)}


def write(name, cells):
    """Same notebook shape as scripts/build_notebooks.py, which runs on import so is not imported."""
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = OUT / name
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return path

INTRO = """
# 05 - Library-size sweep: 1B-class pilot (Qwen2.5-1.5B, N = 10 and 50)

Runs the pilot from `PROTOCOL_sweep.md` section 6: seeds 1000 and 1001, library sizes 10 and 50,
arms `no_skill`, `ic_full`, `ic_topk`, `ours_b24`, `ours_cover`, `seq_no_replay`, `isolated`.
Pilot seeds never enter the confirmatory analysis.

**Before running:** Settings -> Accelerator -> GPU T4 x1 (or P100); Internet on (model download).
Attach this repository as a Kaggle dataset (upload a zip of the repo), or set `REPO_URL` below to
a git URL you control. Estimated cost: about 0.6 GPU-hours plus model download.

**Outputs** (download from /kaggle/working): `runs/sweep_pilot/records.jsonl`,
`runs/sweep_pilot/throughput.json`, `runs/sweep_pilot/analysis/`, and `sweep_pilot_outputs.zip`.
The notebook is resumable: rerunning skips arms already marked done in `records.jsonl`.
"""

SETUP = '''
import glob, os, shutil, subprocess, sys

REPO_URL = ""  # optional: a git URL for this repo; leave empty to use an attached dataset
WORK = "/kaggle/working/skill-library"

if not os.path.exists(WORK):
    if REPO_URL:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, WORK], check=True)
    else:
        found = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/PROTOCOL_sweep.md", recursive=True)]
        if not found:
            raise SystemExit("attach the repo as a dataset or set REPO_URL")
        shutil.copytree(found[0], WORK)
sys.path.insert(0, WORK)
os.chdir(WORK)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft>=0.11", "transformers>=4.40", "accelerate", "pyyaml"], check=False)
# Kaggle images ship an old torchao that recent peft rejects at LoRA injection (ImportError); nothing here uses torchao.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "this notebook needs a GPU accelerator"
GPU = torch.cuda.get_device_name(0)
print(GPU, f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", "bf16:", torch.cuda.is_bf16_supported())
'''

CONFIG = '''
from pathlib import Path
from sweep.config import load_sweep_config

cfg = load_sweep_config("configs/sweep_pilot.yaml")
OUT = Path(cfg.out_dir)
RECORDS = OUT / "records.jsonl"
print(cfg.models, cfg.seeds, cfg.milestones, cfg.arms)
'''

ENV_CHECK = '''
from env.procgen import generate_world, verify_world

for seed in cfg.seeds:
    world = generate_world(cfg.n_max, seed)
    steps = verify_world(world)
    roots = sum(s.is_root for s in world.skills)
    print(f"seed {seed}: {len(steps)} skills verified, {roots} root, max {max(steps.values())} env steps")
'''

LOAD = '''
from sweep.backends import HFSweepBackend
from sweep.runner import write_manifest

backend = HFSweepBackend(cfg, cfg.models[0], seed=cfg.seeds[0])
write_manifest(cfg, OUT, {"gpu": GPU, "notebook": "05_sweep_pilot"})
print("loaded", cfg.models[0])
'''

THROUGHPUT = '''
# Measure the throughputs the budget table assumed (PROTOCOL_sweep.md section 13).
import json, time
from env.procgen import generate_world
from skillmetrics import SHAPES, lora_train_flops, prefill_flops
from sleep.backend import Example
from sweep.backends import Query
from sweep.config import ARMS
from sweep.prompts import build_prompt

shape = SHAPES[cfg.models[0]]
world = generate_world(50, 999)
lib = world.prefix(50)
probe = {}

prompts = [build_prompt(world, "full", s, lib, cfg.topk) for s in lib[:4]]
lengths = [backend.count_tokens(p) for p in prompts]
enc = backend.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
torch.cuda.synchronize(); t = time.time()
with torch.no_grad():
    backend.model(**enc)
torch.cuda.synchronize()
probe["prefill_tflops"] = sum(prefill_flops(shape, L) for L in lengths) / (time.time() - t) / 1e12

queries = [Query(build_prompt(world, "pointers", s, lib, cfg.topk), s, None, set()) for s in lib[:16]]
torch.cuda.synchronize(); t = time.time()
gens = backend.generate(queries)
torch.cuda.synchronize()
probe["decode_batch16_seconds"] = time.time() - t
probe["decode_mean_gen_tokens"] = sum(g.gen_tokens for g in gens) / len(gens)

backend.start_arm(ARMS["ours_b24"])
examples = [Example(s.name, build_prompt(world, "pointers", s, lib, cfg.topk), s.program) for s in lib[:16]]
stats = backend.train(examples, "probe", night=0)
lora = backend.lora_params
backend.end_arm()
probe["train_tflops"] = lora_train_flops(shape, stats.train_tokens, stats.train_tokens_sq, lora, cfg.train.grad_checkpointing) / stats.seconds / 1e12
probe["train_steps"], probe["train_seconds"], probe["train_loss"] = stats.steps, stats.seconds, stats.loss
probe["gpu"] = GPU
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "throughput.json").write_text(json.dumps(probe, indent=2))
print(json.dumps(probe, indent=2))
print("budget assumed 10 TFLOP/s for prefill and training; rescale the protocol table by the measured values")
'''

RUN = '''
# The pilot itself. Resumable: arms already marked done in records.jsonl are skipped.
import time
from sweep.runner import run_model_seed

MAX_HOURS = 8.0
started = time.time()
for seed in cfg.seeds:
    if (time.time() - started) / 3600 > MAX_HOURS:
        print("time budget reached; rerun this cell in a new session to resume")
        break
    backend.seed = seed
    run_model_seed(cfg, cfg.models[0], seed, backend, RECORDS)
    torch.cuda.empty_cache()
print("records:", RECORDS)
'''

ANALYSE = '''
# Pilot analysis. Exploratory by definition: pilot seeds never enter the confirmatory analysis.
from sweep.analyze import run_analysis, summary_text

report = run_analysis(RECORDS, OUT / "analysis", n_boot=2000)
print(summary_text(report))
'''

SAMPLES = '''
# Eyeball what the model actually wrote: first failures per arm at the largest pilot size.
from skillmetrics import finished_runs_only, read_jsonl

rows = [r for r in finished_runs_only(read_jsonl(RECORDS)) if r["kind"] == "attempt" and r["milestone"] == max(cfg.milestones)]
for arm in cfg.arms:
    bad = [r for r in rows if r["arm"] == arm and not r["success"]][:2]
    for r in bad:
        print(f"--- {arm} {r['skill']} ({r['error']})")
        print(r["program"] or "(empty)")
'''

PACK = '''
import shutil
shutil.make_archive("/kaggle/working/sweep_pilot_outputs", "zip", str(OUT))
print("download /kaggle/working/sweep_pilot_outputs.zip")
'''

CONF_INTRO = """
# 06 - Library-size sweep: confirmatory runs (reduced design, PROTOCOL_sweep.md section 5)

Runs `configs/sweep_reduced.yaml` in the registered order: Qwen2.5-1.5B seeds 0-9, then Qwen2.5-3B seeds 0-7, every arm.
One Kaggle session cannot hold the whole design, so the notebook is re-run until nothing is left:

- it starts from `results/sweep_reduced/records.jsonl` in the attached repo (every earlier session's rows, merged), so arms
  already marked done are skipped;
- it starts no new (model, seed) after `MAX_HOURS`; a seed cut off by the session limit loses only its unfinished arm,
  whose rows are dropped by the analysis (`finished_runs_only`);
- its output `runs/sweep_reduced/records.jsonl` is merged back into the repo by `scripts/kaggle_sweep.py`.
"""

CONF_CONFIG = '''
import shutil
from pathlib import Path
from sweep.config import load_sweep_config
from sweep.runner import finished_arms

cfg = load_sweep_config("configs/sweep_reduced.yaml")
OUT = Path(cfg.out_dir)
RECORDS = OUT / "records.jsonl"
OUT.mkdir(parents=True, exist_ok=True)
prior = Path("results/sweep_reduced/records.jsonl")
if prior.exists() and not RECORDS.exists():
    shutil.copyfile(prior, RECORDS)
arm_keys = {a.key for a in cfg.arm_list()}
TODO = [(m, s) for m in cfg.models for s in cfg.seeds_for(m) if not arm_keys <= finished_arms(RECORDS, m, s)]
print(len(TODO), "(model, seed) pairs left:", TODO)
'''

CONF_RUN = '''
import gc, time
from sweep.backends import HFSweepBackend
from sweep.runner import run_model_seed, write_manifest

MAX_HOURS = 8.5          # no new (model, seed) after this; Kaggle stops GPU sessions at 12 h
started, backend, loaded = time.time(), None, None
for model, seed in TODO:
    if (time.time() - started) / 3600 > MAX_HOURS:
        print("time budget reached; the next session continues from here")
        break
    if model != loaded:
        backend = None
        gc.collect(); torch.cuda.empty_cache()
        backend, loaded = HFSweepBackend(cfg, model, seed=seed), model
        write_manifest(cfg, OUT, {"gpu": GPU, "notebook": "06_sweep_confirmatory"})
    backend.seed = seed
    run_model_seed(cfg, model, seed, backend, RECORDS)
    torch.cuda.empty_cache()
left = [(m, s) for m in cfg.models for s in cfg.seeds_for(m) if not arm_keys <= finished_arms(RECORDS, m, s)]
print("left after this session:", left)
'''


if __name__ == "__main__":
    path = write("05_sweep_pilot.ipynb", [
        md(INTRO),
        md("## Setup"), code(SETUP), code(CONFIG),
        md("## Environment check (CPU, seconds)"), code(ENV_CHECK),
        md("## Load the model"), code(LOAD),
        md("## Throughput probe (calibrates the compute table)"), code(THROUGHPUT),
        md("## Run the pilot"), code(RUN),
        md("## Analyse (exploratory)"), code(ANALYSE), code(SAMPLES),
        md("## Package outputs"), code(PACK),
    ])
    print(path)
    path = write("06_sweep_confirmatory.ipynb", [
        md(CONF_INTRO),
        md("## Setup"), code(SETUP), code(CONF_CONFIG),
        md("## Run (resumes; stops starting seeds after MAX_HOURS)"), code(CONF_RUN),
    ])
    print(path)
