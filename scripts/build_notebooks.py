"""Generate the Kaggle notebooks from cell sources kept here as plain text."""
import _paths  # noqa: F401

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name, cells):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    return path


SETUP = '''
import os, sys, subprocess

REPO = "/kaggle/working/skill-library"
if not os.path.exists(REPO):
    # On Kaggle, add the repo as a dataset or clone it here.
    REPO = os.getcwd()
sys.path.insert(0, REPO)
os.chdir(REPO)

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "peft>=0.11", "transformers>=4.40", "accelerate"], check=False)

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"{name}  {total:.1f} GB  bf16={torch.cuda.is_bf16_supported()}")
'''


ENV_CHECK = '''
from env import curriculum_order, validate_curriculum
from env.reference import verify_all

validate_curriculum()
steps = verify_all()
print(f"{len(steps)} tasks, all reference programs verified")
for task in curriculum_order():
    print(f"  tier {task.tier}  {task.name:<22} -> {task.goal_text()}")
'''

LOAD_MODEL = '''
from baselines.gpu_runner import load_config
from sleep.lora import load_base_model

config = load_config("configs/default.yaml")
model, tokenizer = load_base_model(
    config["model"]["name"], config["model"]["device"], config["model"]["dtype"]
)
print(f"loaded {config['model']['name']}")
print(f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")
'''

BASE_PROMPT = '''
# What does the base model do with no library and no adapter?
from env.tasks import TASKS
from library.retrieval import RetrievalConfig, build_prompt
from library.store import SkillLibrary

task = TASKS["craft_wood_pickaxe"]
prompt = build_prompt(SkillLibrary(), task, RetrievalConfig(), "at=forest | inventory: empty")
print(prompt)
print("-" * 60)

from sleep.lora import generate_with
print(generate_with(model, tokenizer, prompt, config["model"]["device"], 96))
'''

ONE_NIGHT = '''
# One night end to end: learn three skills, consolidate, checkpoint.
from baselines.gpu_runner import GPURun

config["run"]["run_id"] = "gpu_smoke"
run = GPURun(config, method="ours", model=model, tokenizer=tokenizer)
loop = run.run(max_nights=1)

print(f"library: {len(run.library)} skills -> {run.library.names}")
for night in run.consolidator.nights:
    s = night.stats
    print(f"night {night.night}: {s.n_new} new + {s.n_replay} replayed, "
          f"{s.steps} steps, loss {s.loss:.3f}, {s.seconds:.0f}s")
print(f"adapter: {run.backend.stored_parameters():,} parameters")
'''

CHECK_RECALL = '''
# Did the night put anything in the weights? Decode with no library in context.
for skill in run.library:
    score = run.backend.knows(skill.name)
    print(f"{skill.name:<22} recall {score:.2f}")
'''

CHECK_RESUME = '''
# The resume path, which every later session depends on.
resumed = GPURun(config, method="ours", model=model, tokenizer=tokenizer)
night = resumed.resume()
print(f"resumed at night {night} with {len(resumed.library)} skills")
assert resumed.library.names == run.library.names
print("checkpoint round-trips")
'''

write("01_setup_and_smoke.ipynb", [
    md("""
# 01 - Setup and smoke test

Confirms the environment, loads the base model, and runs a single night end to
end so that a longer run does not fail on something trivial three hours in.

Expect roughly 10-15 minutes on a T4.
"""),
    md("## Environment"),
    code(SETUP),
    md("## The crafting environment\n\nCPU-only, and independent of the model."),
    code(ENV_CHECK),
    md("## Base model"),
    code(LOAD_MODEL),
    md("""
## What the model does before any consolidation

The base model has neither the library nor an adapter here, so weak output is
the expected result. This is the starting point the rest of the project moves
away from.
"""),
    code(BASE_PROMPT),
    md("## One night"),
    code(ONE_NIGHT),
    md("## Recall from weights"),
    code(CHECK_RECALL),
    md("## Resume"),
    code(CHECK_RESUME),
])
print("wrote 01")


MAIN_SETUP = '''
from baselines.gpu_runner import GPURun, load_config
from sleep.lora import load_base_model

config = load_config("configs/default.yaml")
config["run"]["run_id"] = "gpu_main"

# One night budget per session. Raise it if the session has time left;
# whatever finishes is checkpointed, so nothing is lost either way.
NIGHTS_THIS_SESSION = 2
METHODS = ["ours", "seq_no_replay", "isolated", "in_context"]

model, tokenizer = load_base_model(
    config["model"]["name"], config["model"]["device"], config["model"]["dtype"]
)
'''

MAIN_LOOP = '''
import time, torch

status = {}
for method in METHODS:
    run = GPURun(config, method=method, model=model, tokenizer=tokenizer)
    resumed = run.resume()
    if resumed:
        print(f"[{method}] resuming from night {resumed}, {len(run.library)} skills")
    if len(run.library) == 15 and not run.consolidator.pending:
        print(f"[{method}] already complete, skipping")
        status[method] = "complete"
        continue

    started = time.time()
    run.run(max_nights=NIGHTS_THIS_SESSION)
    status[method] = (
        f"night {run.consolidator.night_count}, {len(run.library)}/15 skills, "
        f"{(time.time()-started)/60:.1f} min"
    )
    print(f"[{method}] {status[method]}")

    del run
    torch.cuda.empty_cache()

status
'''

MAIN_PROGRESS = '''
# Where each method stands. Re-run this notebook until every method reaches 15.
from pathlib import Path
from sleep.checkpoint import latest_night

root = Path(config["run"]["root"]) / config["run"]["run_id"]
for method in METHODS:
    night = latest_night(root / method)
    print(f"{method:<16} last night: {night}")
'''

MAIN_COLLECT = '''
from analysis.collect import summary_table, write_collected

path = write_collected(root, METHODS, run_id=config["run"]["run_id"])
print(path)

import json
print(summary_table(json.loads(path.read_text())))
'''

write("02_main_run.ipynb", [
    md("""
# 02 - Main run

Runs the four methods through the curriculum on a real LoRA. Written to be run
repeatedly: each session does as many nights as it has time for, checkpoints
them, and picks up from the last one next time.

Set `NIGHTS_THIS_SESSION` to whatever the session budget allows.
"""),
    md("## Environment"),
    code(SETUP),
    md("## Configuration"),
    code(MAIN_SETUP),
    md("""
## Run

Each method gets its own checkpoint directory, so they can be run in any order
and across any number of sessions. A method that has already finished is
skipped rather than re-run.
"""),
    code(MAIN_LOOP),
    md("## Progress"),
    code(MAIN_PROGRESS),
    md("## Collect results\n\nOnly meaningful once every method has reached 15 skills."),
    code(MAIN_COLLECT),
])
print("wrote 02")


ABL_SETUP = '''
from baselines.gpu_runner import GPURun, load_config
from sleep.lora import load_base_model

config = load_config("configs/default.yaml")
model, tokenizer = load_base_model(
    config["model"]["name"], config["model"]["device"], config["model"]["dtype"]
)

# Ablations are the expensive part of the project. Each variant is a full run,
# so pick one grid per session rather than trying to do all three.
GRID = "buffer"     # "buffer" | "frequency"
NIGHTS_THIS_SESSION = 3
'''

ABL_RUN = '''
import copy, torch
from pathlib import Path

GRIDS = {
    "buffer": [("buffer_0", {"max_total": 0, "per_skill": 0, "enabled": False}),
               ("buffer_6", {"max_total": 6, "per_skill": 1}),
               ("buffer_12", {"max_total": 12, "per_skill": 1}),
               ("buffer_24", {"max_total": 24, "per_skill": 2})],
    "frequency": [("K_1", {}), ("K_3", {}), ("K_5", {})],
}
K_FOR = {"K_1": 1, "K_3": 3, "K_5": 5}

for variant, replay in GRIDS[GRID]:
    variant_config = copy.deepcopy(config)
    variant_config["run"]["run_id"] = f"gpu_ablation_{GRID}"
    variant_config["sleep"]["replay"].update(replay)
    if variant in K_FOR:
        variant_config["sleep"]["every_k_tasks"] = K_FOR[variant]

    run = GPURun(variant_config, method="ours", model=model, tokenizer=tokenizer)
    run.root = Path(variant_config["run"]["root"]) / variant_config["run"]["run_id"] / variant
    resumed = run.resume()
    print(f"[{variant}] resumed at {resumed}, {len(run.library)} skills")
    run.run(max_nights=NIGHTS_THIS_SESSION)
    print(f"[{variant}] night {run.consolidator.night_count}, {len(run.library)}/15")

    del run
    torch.cuda.empty_cache()
'''

ABL_COLLECT = '''
from pathlib import Path
from analysis.collect import summary_table, write_collected
import json

root = Path(config["run"]["root"]) / f"gpu_ablation_{GRID}"
variants = [v for v, _ in GRIDS[GRID]]
path = write_collected(root, variants, run_id=f"gpu_ablation_{GRID}")
print(summary_table(json.loads(path.read_text())))
'''

write("03_ablations.ipynb", [
    md("""
# 03 - Ablations

Two knobs: how large the replay buffer is, and how often consolidation happens.
Both are full runs per variant, so this is the most expensive notebook here.
Run one grid per session and let the checkpoints accumulate.

The CPU pipeline runs the same grids far more cheaply and is the right place to
sanity-check a grid before spending GPU time on it:

    python scripts/run_ablations.py --grid buffer
"""),
    md("## Environment"),
    code(SETUP),
    md("## Configuration"),
    code(ABL_SETUP),
    md("## Run the grid"),
    code(ABL_RUN),
    md("## Collect"),
    code(ABL_COLLECT),
])

FIG_LOAD = '''
import json
from pathlib import Path

from analysis.collect import summary_table, write_collected

RUN_ID = "gpu_main"
METHODS = ["in_context", "seq_no_replay", "isolated", "ours"]

root = Path("runs") / RUN_ID
results_path = root / "results.json"
if not results_path.exists():
    results_path = write_collected(root, METHODS, run_id=RUN_ID)

results = json.loads(results_path.read_text())
print(summary_table(results))
'''

FIG_MAIN = '''
from analysis.plots import plot_forgetting_curve

order = [m for m in METHODS if m in results["methods"]]
fig = plot_forgetting_curve(results, root / "figures" / "fig1_forgetting_curve.png",
                            order=order)
fig
'''

FIG_PARAMS = '''
from analysis.plots import plot_parameter_growth

fig = plot_parameter_growth(results, root / "figures" / "fig2_parameter_growth.png",
                            order=order)
fig
'''

FIG_TOKENS = '''
from analysis.plots import plot_token_cost

fig = plot_token_cost(results, root / "figures" / "fig3_token_cost.png", order=order)
fig
'''

FIG_TABLE = '''
# The numbers as they go into the paper.
rows = []
for key in order:
    m = results["methods"][key]
    rows.append({
        "method": m["label"],
        "final success": round(m["final_success_rate"], 3),
        "oldest 3": round(m["final_earliest_3"], 3),
        "prompt tokens": round(m["tokens"].get("mean_prompt_tokens", 0)),
        "stored params": m["final_stored_params"],
    })

try:
    import pandas as pd
    display(pd.DataFrame(rows))
except ImportError:
    for r in rows:
        print(r)
'''

write("04_figures.ipynb", [
    md("""
# 04 - Figures

Reads whatever the runs have produced and renders the figures. Works on a CPU
kernel: it only reads JSON.

Figure 1 is the headline claim (retention as the library grows) and figure 2 is
the cost it is bought at (adapter storage against skill count).
"""),
    md("## Load results"),
    code(FIG_LOAD),
    md("""
## Figure 1 - forgetting curve

Retention is normalised per skill against its own score when freshly learned,
so the curve is not just tracking which tasks happen to be old. Skills whose
program calls nothing are excluded: they need no recall, so they cannot show
forgetting.
"""),
    code(FIG_MAIN),
    md("""
## Figure 2 - adapter storage

The honest half of the comparison with PEAM-style isolation: isolation does not
forget, and it pays for that linearly in stored parameters.
"""),
    code(FIG_PARAMS),
    md("## Figure 3 - in-context cost"),
    code(FIG_TOKENS),
    md("## Table"),
    code(FIG_TABLE),
])
print("wrote 03 and 04")
