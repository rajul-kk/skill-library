# Library-size sweep: design

Question: **at what library size does cumulative LoRA + replay beat keeping skills in context,
per token and per FLOP?**

This document fixes the architecture and the reasons for each choice. The confirmatory rules
(endpoints, thresholds, decision rules) are frozen separately in `PROTOCOL_sweep.md`; where the two
disagree, the protocol wins.

## 1. What the existing repo gives and what it lacks

The repo has a 15-skill crafting environment, a Voyager-style loop, a capped replay buffer, real
LoRA backends (`sleep/lora.py`) and a CPU interference simulator. Four things block a size sweep:

1. **Only 15 skills.** Sizes up to 200 need a skill generator.
2. **Discovery confounds retention.** In the main loop each arm discovers skills itself, so arms end
   up with different libraries, and a 1-3B model will not discover 200 skills.
3. **A self-call shortcut.** `SkillLibrary.resolver()` executes any stored body, consolidated or
   not, and consolidated skills stay listed by name. On GPU a model asked to redo task X can write
   `call(X)` and succeed without recalling anything. The CPU simulator never does this, so the
   README numbers are unaffected, but a GPU sweep would be.
4. **Cost is measured in prompt tokens only.** Adapter training is not counted, so "per FLOP" is
   not answerable.

## 2. Architecture

```
skillmetrics/        shareable metrics, no imports from this repo (stdlib only)
  flops.py           model shape -> prefill / decode / LoRA-train FLOPs
  bootstrap.py       seed-level paired bootstrap, percentile intervals
  crossover.py       efficiency ratios, sustained crossover N*, break-even query count H*
  floor.py           floor-effect and headroom checks
  records.py         the attempt / training record schema and aggregation
env/procgen.py       procedural crafting world: nonce items, recipe DAG, reference programs
sweep/
  config.py          SweepConfig (yaml) and the arm table
  prompts.py         prompt rendering and lexical top-k retrieval
  backends.py        SimulatedSweepBackend (CPU) and HFSweepBackend (real model + LoRA)
  runner.py          one (model, seed) trajectory through all milestones, all arms
  analyze.py         records -> per-cell table, crossovers, figures
  budget.py          GPU-hour estimate from the same FLOP model
configs/sweep*.yaml  full, reduced, pilot, CPU smoke
scripts/             run_sweep.py, analyze_sweep.py, sweep_budget.py, build_sweep_notebook.py
notebooks/05_sweep_pilot.ipynb
```

Data flow, per (model, seed):

```
generate_world(n_max=200, seed) -> ordered skills s_1..s_200 (verified reference programs)
for arm in arms:
    state = arm.fresh_state()
    for k in 1..200:
        add s_k to the library (oracle acquisition)
        if k % K == 0: arm.night(state)          # LoRA arms train; others no-op
        if k in milestones {10,25,50,100,200}:
            eval_set = fixed stratified sample of s_1..s_k (seeded)
            for skill in eval_set: attempt(skill) -> AttemptRecord
    TrainRecords per night -> records.jsonl
skillmetrics: records -> success, tokens, FLOPs per (arm, N, seed) -> bootstrap -> N*, H*
```

`skillmetrics` takes plain dicts and returns plain dicts, so another project (the proposed shared
harness in Myrios) can import it without this repo's environment or models.

## 3. Decisions, with alternatives rejected

### D1. Skills come from a procedural second environment
`env/procgen.py` generates a world per seed: nonce resource and item names, nonce locations, and a
recipe DAG grown one skill at a time. Skill k either gathers a new resource (`goto` + `gather`) or
crafts a new item from 1-3 inputs among earlier items and raw resources, so its reference program
mixes `call(...)` of earlier skills with inline `goto`/`gather`/`craft`. Everything runs on the
existing DSL and a subclass of the existing executor.

- *Rejected: quantity/location variants of the 15 existing tasks.* Variants share programs, so
  "200 skills" would be ~15 skills plus noise and interference would be underestimated.
- *Rejected: Crafter text mode.* Not built, and it does not reach 200 distinct skills.
- *Why nonce names:* recipes cannot be known from pretraining, so any success on a crafted skill
  must come from the library or the adapter. This is what makes the no-skill arm a clean floor.
- *Cost:* the environment is synthetic and written by the evaluator (an existing limitation that
  this does not fix). The 15-skill environment stays as the anchor for the README numbers.

### D2. Nested milestones, one trajectory per (model, seed, arm)
The world is grown incrementally, so the library at N=25 is a prefix of the library at N=200. One
trajectory to 200 is evaluated at N in {10, 25, 50, 100, 200}.

- *Rejected: independent runs per size.* Five times the training compute, and a lifelong agent's
  library at size N is exactly the prefix state anyway.
- *Cost:* milestones within a seed are correlated. The analysis bootstraps over seeds, never over
  milestones, so this is accounted for.

### D3. Oracle acquisition
Skills enter the library with their verified reference programs, identically for every arm.

- *Rejected: agent discovery.* It confounds retention with acquisition, gives each arm a different
  library, and nonce recipes are not discoverable without being told.
- *Cost:* the sweep measures the cost of **keeping and using** a library, not of building one.

### D4. Self-call ban during evaluation
When skill X is evaluated, `call(X)` raises. Every other skill stays callable, as in deployment.
The in-context arms may still copy X's body from the prompt; that is what an in-context library is
for.

### D5. Arms
| arm | prompt carries | weights | callable |
|---|---|---|---|
| `no_skill` | grammar + format example + task | base | nothing |
| `ic_full` | every skill's full text, learning order | base | all but X |
| `ic_topk` | top-8 skills by lexical retrieval, full text | base | all but X |
| `ours_b24` | top-8 pointers (name + description) + unconsolidated full text | one cumulative LoRA, replay cap 24 | all but X |
| `ours_b96` | same | cap 96 | all but X |
| `ours_cover` | same | cap = library size (covers it every night) | all but X |
| `seq_no_replay` | same | cap 0 | all but X |
| `isolated` | same | one adapter per night block, oracle-routed | all but X |

- *Two in-context arms.* `ic_full` is the literal "keep the library in context" and its cost grows
  linearly. `ic_topk` is what a Voyager-style agent actually does and its cost saturates. A
  crossover against `ic_full` alone would be an easy win; reporting both stops that.
- *Replay caps.* 24 is the paper default. `cover` tests the README's "buffer must cover the
  library once" claim at scale, at a cost that grows with N. 96 sits between. The reduced design
  keeps 24 and `cover`.
- *Isolated per night block, not per skill.* 200 per-skill adapters do not fit a T4 session. One
  adapter per night block (K skills) is PEAM-style per-category isolation with the block as the
  category; storage still grows linearly in N. Routing is oracle (the block of X is given), so
  this arm is an upper bound on isolation.
- *Rejected: one-shot text-to-LoRA (arm d).* Needs a trained hypernetwork; out of budget.

### D6. Training on the deployment prompt
Weight arms are trained on the same prompt template they are evaluated with (pointers rendered at
training time). The existing pipeline trains on a "(recall from memory)" prompt and evaluates on
another; the gap would be charged to consolidation unfairly.

### D7. Nights every K=10 skills, fixed epochs
K=10 gives 20 nights to N=200. Each night runs 4 epochs over (new + replay) examples, batch 8,
lr 2e-4, rank 16 on q/k/v/o, LoRA weights in fp32 over an fp16 base on T4.

- *Rejected: K=3 (paper default).* 67 nights per trajectory; the README ablation also found larger
  K no worse (K=5 best).
- *Rejected: a fixed 300 steps per night.* Over-trains small nights and makes training cost
  independent of replay cap, which hides the cost the sweep is meant to measure.
- Hyperparameters may be changed **only** on pilot seeds (1000+), which are excluded from the
  confirmatory analysis (protocol section 6).

### D8. Evaluation
Greedy decoding, one attempt, `max_new_tokens=96`, generation stops at a blank line. Success means
the program parses, runs in a fresh world with X uncallable, and meets X's goal. Eval set at
milestone N: all skills if N <= 50, otherwise a seeded sample of 50 stratified by age quintile.
Greedy decoding makes repeats pointless; the variance comes from seeds (world, init, data order).

### D9. Cost accounting
Counted per attempt from the real token counts: prompt tokens L and generated tokens G.

- Forward FLOPs per token = 2 x (non-embedding params + LM head) + attention term
  2 x n_layer x ctx x d_model. Prefill of L tokens sums ctx over 1..L; decode sums L+1..L+G.
- Unmerged LoRA adds 2 x (LoRA params) per token.
- LoRA training per real (unpadded) token = 4 x forward-matmul FLOPs (forward + activation
  backward; the frozen base needs no weight gradients) + 6 x LoRA params, x 1.33 if gradient
  checkpointing is on.
- Per-query FLOPs at size N = inference FLOPs + (cumulative training FLOPs to N) / H.
  H is reported as a sweep, and the break-even H* (training cost / per-query saving, the
  amortisation rule of 2610.29357) is reported per N.

- *Rejected: wall-clock as the cost.* T4 vs P100 and session noise make it incomparable; it is
  logged as a secondary field.
- *Rejected: tokens only.* Ignores training cost, which is the whole price of consolidation.

### D10. Models
Qwen2.5-1.5B-Instruct ("1B class") and Qwen2.5-3B-Instruct. Same family and tokenizer, so model
size is not confounded with family; both ungated (no HF token); the repo already targets 1.5B.

- *Rejected: Llama-3.2-1B/3B.* Exact sizes, but gated behind an HF licence acceptance. Swappable
  in the config if the user has access.

### D11. Statistics
Unit of analysis = seed. Paired seed bootstrap (10,000 resamples, percentile intervals); all arms
share each seed's world, so differences are paired. The crossover N* is the smallest grid size
from which the efficiency ratio stays above 1 at every larger size; its interval comes from
re-running that rule on each bootstrap resample.

### D12. Floor and headroom checks
Each (model, N) cell is dropped if the `no_skill` arm's success on **root** (gather) skills is
below 0.30. Those tasks name their resource and location, so failing them means the model cannot
write the DSL. A model is dropped if at N=10 no in-context arm beats `no_skill` on crafted skills
by 0.20 (no headroom). Thresholds are frozen in the protocol.

## 4. Compute budget

`scripts/sweep_budget.py` turns the FLOP model plus throughput assumptions into GPU-hours; the
table it prints is copied into the protocol. Throughputs are assumptions until the pilot measures
them (the pilot notebook logs measured seconds per phase), and the plan is re-sized from the pilot
before confirmatory seeds start.

## 5. Risks

| risk | effect | mitigation |
|---|---|---|
| 1.5B cannot write the DSL | floor check drops cells | 1-shot format example in every arm's prompt; pilot measures it first |
| LoRA never learns nonce recipes | weight arms at 0, no crossover | pilot on seeds 1000+ may tune lr/epochs before confirmatory seeds |
| `ic_full` at N=200 (~12k tokens) on T4 | slow prefill, memory | batch size 1 for long prompts; sdpa attention; budget counts it |
| fp16 LoRA training instability on T4 | NaN losses | fp32 LoRA weights, grad clip 1.0; NaN logged and the night flagged |
| Kaggle session limits (12 h, 30 h/week) | runs cut | resumable per (arm, milestone) via records.jsonl; reruns skip finished cells |
| Synthetic env written by the evaluator | external validity | stated as a limitation; the 15-skill env stays as anchor |
| Bootstrap with 10 seeds is narrow | over-confident intervals | stated; full design uses 20 seeds |

## 6. Discrepancies found in the existing results (for the user to decide; nothing changed)

Reported by the Myrios shared-harness cross-check and confirmed against the code here. The README,
paper and `results/cpu` numbers are left exactly as published.

1. **Final success drops seeds.** `baselines/runner.py:RunResult.final` takes the maximum night
   over all seeds' snapshots and averages only snapshots at that night. Seeds whose run ends on an
   earlier night are silently excluded. Reported effect: `seq_no_replay` 0.098 from 15 of 20 seeds
   vs 0.107 when each seed's own last score is averaged; ours/isolated 0.733 vs 0.740. The sweep
   avoids this by construction (fixed milestones, a forced night at each) and the protocol defines
   cell success as the mean of every seed's own score.
2. **Adapter size is hard-coded.** `baselines/methods.py:LORA_PARAMS_R16 = 4,718,592`; the real
   rank-16 q/k/v/o adapter on Qwen2.5-1.5B has 4,358,144 parameters (k/v projections are
   1536 -> 256 under grouped-query attention). The README's 4.7 M and 70.8 M (15 x 4.7 M) are
   therefore about 8% high; the 15x storage ratio is unaffected. `tests/test_skillmetrics.py`
   checks the shape formula against peft on a tiny model.
3. **Identical rows.** In `results/cpu/summary.txt` the `ours` and `isolated` rows are identical
   to every printed digit, including the retention curve. With b=24 covering a 15-skill library,
   both retentions saturate in the simulator, but exact equality of every point deserves a look
   before the table is cited.
4. **Self-call shortcut on GPU.** See section 1 item 3; it does not affect CPU numbers.

Options for the user: re-run `scripts/reproduce.py` with a fixed `final()` and the computed adapter
size and update README/paper; or keep the published numbers and add an erratum note.

## 7. Interoperability

`skillmetrics` is stdlib-only and imports nothing from this repo (a test enforces it). The Myrios
`myrios_eval` package (schema 1.0, branch `feat/shared-eval-harness`) reportedly agrees with it on
bootstrap intervals, LoRA parameter counts and LoRA training FLOPs once the attention term is
removed; `skillmetrics` keeps the attention term because `ic_full` prompts reach ~8k tokens, where
it is no longer negligible. Writing sweep records in the `myrios_eval` schema is left as a follow-up
adapter over `records.jsonl`, so neither repo depends on the other.
