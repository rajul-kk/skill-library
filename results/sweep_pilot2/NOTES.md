# Sweep pilot 2, 2026-11-07 (Kaggle T4, notebook 05 version 4; seeds 1000-1001, N = 10 and 50; exploratory)

Re-run of the pilot after the prompt amendments in PROTOCOL_sweep section 12 (chat-template wrapper, second no-skill
format example, unnamed ban; dataset at commit 50aa505). Same seeds and arms as pilot 1 (`results/sweep_pilot/`).
Raw generations are stored this time (`text` in every attempt record).

Arms ran for 24 min in total on one T4. Measured throughput: prefill 6.7 TFLOP/s, training 13.4 TFLOP/s, decode 5.3 s
per batch of 16 (mean 54 generated tokens).

Checks (PROTOCOL_sweep section 9): **floor passed** at both N (no_skill root success 1.0 at N = 10 and N = 50);
**headroom passed** at N = 10 (ic_full and ic_topk 0.27 on composed skills vs no_skill 0). The prompt fix removed the
format failure of pilot 1.

Success (mean of 2 seeds):

| arm | N = 10 | N = 50 |
|---|---|---|
| no_skill | 0.45 | 0.26 |
| ic_full | 0.40 | 0.09 |
| ic_topk | 0.40 | 0.24 |
| ours_b24 | 0.10 | 0.00 |
| ours_cover | 0.10 | 0.00 |
| seq_no_replay | 0.10 | 0.00 |
| isolated | 0.10 | 0.08 |

**New problem: every arm that trains weights is below no_skill.** In pilot 1 (raw prompt) the weight arms led at N = 50
(ours_cover 0.29, ours_b24 0.21). With the chat wrapper they produce full-length programs (no empty programs), but the
programs call skills that do not exist or the wrong ones (most common errors at N = 50 for ours_b24: unknown skill,
resource at the wrong location, unparsable line). Training loss per night ranged 0.34-1.12 with no NaN. The cause is not
established; it has to be diagnosed before the confirmatory sweep, since the primary contrast is ours_b24 against the
in-context arms. Also seen: ic_topk sometimes calls `gather_oil` from the format example.

Not decided here: whether `configs/sweep_full.yaml` switches to `prompt_wrapper: chat` (a deviation row is needed either
way).
