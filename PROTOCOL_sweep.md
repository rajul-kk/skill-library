# Pre-registration: library-size sweep

**Status: FROZEN.** This file was committed before any GPU run of the sweep and before any
confirmatory seed was started. CPU smoke runs (simulated backend, and SmolLM2-135M on CPU) test the
plumbing only and are not results. Any analysis, threshold, arm or rule added or changed after the
freeze commit is labelled **POST HOC** wherever it is reported, and the change is logged in
section 12 with its date and reason.

Design rationale: `docs/sweep_design.md`. Scoop check: `docs/scoop_check_sweep.md`.

## 1. Question

At what skill-library size N does a single cumulative LoRA adapter trained with capped replay
beat keeping the skills in the prompt, **per token** and **per FLOP**?

## 2. Predictions (stated so they can be wrong; they are not decision rules)

- P1. Per token, `ours_b24` beats `ic_full` from the smallest size tested (N* <= 25), because
  `ic_full` prompt length grows linearly with N.
- P2. Per token, `ours_b24` does **not** sustainably beat `ic_topk`: a capped buffer stops covering
  the library somewhere in 50-200 and success falls, while `ic_topk` cost is flat.
- P3. `ours_cover` holds success closer to `ic_topk` than `ours_b24` at N=200, at a training cost
  growing roughly quadratically in N.
- P4. Per FLOP at H = 10^4 queries, crossovers move to larger N than per token, and the break-even
  query count H* grows with N.
- P5. `seq_no_replay` is below `ours_b24` at every N >= 25 (the forgetting control works).

## 3. Environment and library

- Skills come from `env/procgen.py` (`generate_world(200, seed)`), as frozen at the freeze commit.
  The world for seed s is identical across arms and models. Libraries at N in {10, 25, 50, 100, 200}
  are prefixes of the seed's 200-skill sequence.
- Skills enter every arm's library in the same order with their verified reference programs
  (oracle acquisition). The sweep measures the cost of keeping and using a library, not of
  building one.
- Root skills (gather a named resource at a named location) need no recall; composed skills call
  earlier skills and craft a nonce item whose recipe is not in the prompt.

## 4. Arms

| arm | prompt | weights |
|---|---|---|
| `no_skill` | grammar, format example, task; nothing callable | base |
| `ic_full` | full text of all N skills | base |
| `ic_topk` | full text of top-8 skills by lexical retrieval | base |
| `ours_b24` | top-8 pointers (name + description) | one cumulative LoRA, replay cap 24 |
| `ours_b96` | same | cap 96 (full design only) |
| `ours_cover` | same | cap = all consolidated skills |
| `seq_no_replay` | same | cap 0 |
| `isolated` | same | one fresh adapter per night block, oracle routing |

Every prompt is built by `sweep/prompts.py:build_prompt` and contains the same grammar, format
example and the line "Do not call X itself". During evaluation of skill X, `call(X)` raises; all
other skills in the library are callable (none for `no_skill`).

## 5. Design

| | models | seeds | milestones | arms |
|---|---|---|---|---|
| full | Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct | 0-19 | 10, 25, 50, 100, 200 | all 8 |
| **reduced (confirmatory plan)** | 1.5B: seeds 0-9; 3B: seeds 0-7 | | 10, 25, 50, 100, 200 | all but `ours_b96` |
| pilot | 1.5B | 1000, 1001 | 10, 50 | as reduced |

- Configs: `configs/sweep_full.yaml`, `configs/sweep_reduced.yaml`, `configs/sweep_pilot.yaml`.
- Run order: pilot; then reduced 1.5B seeds 0-9 in order; then reduced 3B seeds 0-7 in order; then,
  budget permitting, the remaining full-design seeds and `ours_b96`, in seed order. If the budget
  runs out, the calendar extends; the design does not shrink. Results are reported for whatever
  seeds have completed **all** arms for a model, and the seed count is stated.
- Nights every 10 skills, plus a night at every milestone so that all learned skills are
  consolidated before evaluation. Each night trains 4 epochs over the interleaved new + replay
  examples, batch 8, lr 2e-4, LoRA rank 16 / alpha 32 / dropout 0.05 on q,k,v,o, fp32 adapter
  weights, gradient clipping 1.0. Replay uses the existing round-robin `ReplayBuffer`, one example
  per skill.
- Evaluation at each milestone: all skills if N <= 50, else 50 skills sampled 10 per age quintile
  with RNG seed `eval-{seed}-{N}` (the same set for every arm). Greedy decoding, one attempt,
  max 96 new tokens, stop at the first blank line. Success = the extracted program parses, runs
  from a fresh world with X uncallable, and meets X's goal.

## 6. Pilot rules

The pilot uses seeds 1000 and 1001, which never enter the confirmatory analysis. After the pilot
and before confirmatory seed 0 starts, only these may change, by a committed config edit that
records the reason in section 12: learning rate, epochs per night, `max_new_tokens`, decode batch
sizes, gradient checkpointing, and the prompt wrapper (raw text vs the model's chat template,
applied identically to every arm). Arms, milestones, endpoints, thresholds and the analysis code
may not change on the basis of pilot results.

## 7. Cost accounting

Counted per attempt from real tokenizer counts (prompt tokens L, generated tokens G up to and
including the stop) with `skillmetrics/flops.py`:

- prefill = 2 P L + 4 n_layer d L(L+1)/2 + 2 V d; decode = 2 (P + V d) G + 4 n_layer d (L G + G(G+1)/2);
  unmerged LoRA adds 2 P_lora (L + G). P = non-embedding matmul parameters, V d = LM head.
- training per night = (2 or 3 with gradient checkpointing) x [2 (P + V d) sum T + 4 n_layer d sum T(T+1)/2]
  + 6 P_lora sum T, over the real (unpadded) token counts T of every sequence processed.
- **Per-token cost** = mean (L + G) per attempt.
- **Per-FLOP cost** at horizon H = mean inference FLOPs per attempt + cumulative training FLOPs up
  to N / H. Primary H = 10^4; also reported for H in {10^2, 10^3, 10^5, infinity}.
- **Break-even H*** = training FLOPs / (success-adjusted inference saving per query); infinity if
  there is no saving.

## 8. Endpoints

Efficiency of an arm in a (model, N) cell = mean success / mean cost, means over seeds.

**Success definition.** A seed's success at milestone N is the mean over that seed's own evaluation
set at N. Every seed is evaluated at every milestone (a night is forced at each milestone), so there
is no "last night" that only some seeds reach. Cell success is the unweighted mean of every
completed seed's own score; no seed is dropped for ending on a different night. This differs from
the existing CPU pipeline's `RunResult.final`, which averages only snapshots at the globally last
night (see `docs/sweep_design.md` section 6).

**Adapter size.** LoRA parameter counts are taken from the actual model: the trainer counts the
adapter's parameters after attaching it, and the FLOP model uses `ModelShape.lora_params` from the
real config (4,358,144 for Qwen2.5-1.5B at rank 16 on q,k,v,o). The repo's hard-coded
`LORA_PARAMS_R16 = 4,718,592` is not used anywhere in the sweep.

**Primary** (per model): the ratio R(N) = efficiency(`ours_b24`) / efficiency(comparator), for
comparator in {`ic_full`, `ic_topk`} and cost in {tokens, FLOPs at H = 10^4}. Four contrasts per
model.

- N* = the smallest milestone from which the point estimate R(N) > 1 at every larger milestone in
  the analysis; "not reached" if R(200) <= 1. Reported with its bootstrap distribution.
- A claim "consolidation beats in-context at N" requires the lower bound of the
  Bonferroni-adjusted interval (level 1 - 0.05/4 = 98.75%) of R(N) to exceed 1.
- An interpolated N* on a log-N scale is reported as descriptive.

**Secondary**: the same contrasts for `ours_cover`, `ours_b96`, `isolated`, `seq_no_replay` (95%
intervals); H* per N; success differences vs each comparator with 95% intervals and a
non-inferiority margin of 0.05; stored adapter parameters; success by age quintile.

**Exploratory**: anything else, including the primary contrasts on milestones that failed the floor
check (always labelled).

## 9. Floor and headroom checks

- **Floor** (per model, per N): the `no_skill` arm's success on root skills in the evaluation sets,
  pooled over seeds, must be >= 0.30. Otherwise that (model, N) cell is dropped from primary and
  secondary analyses and reported as "floor".
- **Headroom** (per model, at N = 10): the better of `ic_full` and `ic_topk` must exceed `no_skill` on
  composed skills by >= 0.20 (pooled over seeds). Otherwise the model is dropped from primary and
  secondary analyses and reported as "no headroom".

## 10. Statistics

- Unit of analysis: the seed. Arms are paired within a seed (same world, same evaluation sets).
- Paired percentile bootstrap over seeds, 10,000 resamples, RNG seed 0; seeds are resampled jointly
  across milestones when computing N*.
- No other multiplicity correction. Secondary and exploratory intervals are 95% and are not used
  for claims.

## 11. Exclusions and failures

- A night whose loss is non-finite is skipped by the trainer and flagged in its record. An
  (arm, seed) with more than 20% flagged nights is excluded and reported.
- A crashed or time-limited session resumes from `records.jsonl`; an arm not marked done is rerun
  from scratch for that seed. Seeds are never replaced.
- Nothing else is excluded.

## 12. Deviations log

| date | change | reason | POST HOC? |
|---|---|---|---|
| 2026-11-07 | Pilot config: `grad_checkpointing: true` (allowed by section 6) and `micro_batch_size: 2`, a new memory-only setting that splits each batch of 8 into micro-batches with token-weighted gradient accumulation, so the optimizer update equals the batch-8 update (`tests/test_sweep.py::test_micro_batches_give_the_full_batch_gradient`). Batch size, lr, epochs unchanged. Notebook also removes Kaggle's torchao 0.10, which recent peft rejects. | First pilot run on a T4: CUDA out of memory in the training loss (float32 logits of 8 x 1024 x 151,936); before that, peft ImportError on torchao. No pilot result existed. | no |
| 2026-11-07 | (a) Pilot config `prompt_wrapper: chat` (allowed by section 6: the model's chat template, one user turn, identical for every arm and for training examples). (b) **Prompt text amendment, not on the section 6 list:** the line "Do not call X itself; write its steps (other skills may be called)." is replaced by "Rules: call() only skills listed under Known skills, and never call X. Answer with the program only: DSL instructions, one per line, no other text." Same for every arm; grammar, example, skills block, task and goal lines unchanged. (c) Attempt records also store the decoded text (`text`); the analysis code does not read it. Arms, milestones, endpoints, thresholds and analysis code unchanged. | First pilot (`results/sweep_pilot/`) failed the floor: 25 of 35 no_skill root attempts gave empty programs. A CPU reproduction on 4 root tasks of pilot seed 1000 showed that with raw text the model writes `gather(r, n)` followed by prose on the same line (nothing extractable), and with the chat template alone it writes `call(X)`, the banned skill, copying the example's `call()`; 0 of 4 either way. The pilot seeds never enter the confirmatory analysis; the amended prompt is re-piloted on the same seeds before any confirmatory run. | **yes** (b) |
| 2026-11-07 | Supersedes (b) of the previous row before any pilot run used it. Prompt: a second format example with `Known skills: (none)` (goto/gather only) follows the first, and the rules line no longer names the task's own skill: "Rules: call() only skills listed under Known skills, and do not call the skill for this task itself. Answer with the program only: DSL instructions, one per line, no other text." The executor still bans the skill. Wrapper stays `chat`. | Pre-flight on Kaggle T4, base weights, pilot seeds 1000-1001 (`results/sweep_prompt_check/`): six prompt variants on 9 root no_skill tasks and 16 + 16 composed tasks with top-k / full library in context; choice rule fixed before the run (most no_skill root successes, ties by in-context successes). Original prompt raw 1/9, 1/16, 1/16; previous row's amendment 0/9, 2/16, 1/16 (the model called the named banned skill); chosen variant 9/9, 5/16, 1/16. Pilot seeds only. | **yes** |
| 2026-11-08 | **Training amendment, not on the section 6 list:** every training target ends with the tokenizer's EOS token (`sleep/lora.py::encode`; `<|im_end|>` for Qwen2.5-Instruct, the token generation already stops on). Before, targets ended with a newline only. Applies identically to every arm that trains (cumulative, isolated, no-replay) and to the GPU runner. Arms, milestones, endpoints, thresholds, data, hyperparameters and analysis code unchanged. | Pilot 2 (`results/sweep_pilot2/`) passed floor and headroom, but every weight arm scored below no_skill (0.10 vs 0.45 at N = 10). The stored texts show why: at N = 10, 0.50 of the attempts in every weight arm are the exact reference program followed by further lines (mean 9-14 lines against 3), i.e. the adapter learned the programs but not where they end, because generation stops only at EOS or a blank line and no training target contained either. Reproduced on CPU with SmolLM2-135M (one cumulative night on 10 chat-wrapped examples): without EOS 6 of 10 generations run past the reference program. Pilot seeds only. | **yes** |
| 2026-11-08 | `configs/sweep_full.yaml` and `configs/sweep_reduced.yaml` set `prompt_wrapper: chat` (allowed by section 6), with the prompt of the 2026-11-07 rows; confirmatory runs use the same wrapper and prompt as the pilot. A third pilot on seeds 1000-1001 with the EOS amendment runs before any confirmatory seed; like pilots 1-2 it is exploratory and its seeds never enter the confirmatory analysis. | The raw wrapper failed the floor in pilot 1 (format failure); the chat wrapper with the amended prompt passed floor and headroom in pilot 2. | no |

## 13. Compute estimate (at freeze)

From `python scripts/sweep_budget.py` with real Qwen2.5 token counts and **assumed** T4
throughput (10 effective TFLOP/s for prefill and training, 35 ms / 50 ms per batched decode step
for 1.5B / 3B, 64 steps per batch, batch 16). The pilot measures the real numbers.

| design | model | arms | seeds | h/seed | train h | prefill h | decode h | total GPU-h |
|---|---|---|---|---|---|---|---|---|
| full | 1.5B | 8 | 20 | 1.63 | 26.0 | 1.9 | 1.6 | 32.6 |
| full | 3B | 8 | 20 | 3.11 | 51.9 | 4.0 | 2.3 | 62.3 |
| reduced | 1.5B | 7 | 10 | 1.17 | 8.6 | 0.9 | 0.7 | 11.7 |
| reduced | 3B | 7 | 8 | 2.21 | 13.7 | 1.5 | 0.8 | 17.7 |
| pilot | 1.5B | 7 | 2 (N in {10, 50}) | 0.29 | 0.2 | 0.0 | 0.0 | 0.6 |

Full: ~95 GPU-hours (about 3.2 Kaggle weeks). Reduced: ~29 GPU-hours (one week). Training of
`ours_cover` and `ours_b96` dominates. If real training throughput is half the assumption, the
reduced design takes about two weeks; per section 5 the calendar extends rather than the design.

Mean prompt tokens per attempt (seed-0 world, Qwen2.5 tokenizer): `no_skill` ~206; `ic_full` 562 /
1,098 / 2,008 / 3,954 / 7,703 at N = 10 / 25 / 50 / 100 / 200; `ic_topk` ~500; pointer arms ~365.
