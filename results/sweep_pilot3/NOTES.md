# Sweep pilot 3, 2026-11-08 (Kaggle T4, notebook 05 version 5; seeds 1000-1001, N = 10 and 50; exploratory)

Same settings and seeds as pilot 2 (`results/sweep_pilot2/`) plus the EOS amendment of PROTOCOL_sweep section 12
(2026-11-08): every training target ends with the tokenizer's EOS token. Dataset at commit 8a0a882. Arms ran for 24 min
on one T4; measured prefill 6.6 TFLOP/s, training 12.4 TFLOP/s, decode 5.5 s per batch of 16.

Checks (PROTOCOL_sweep section 9): **floor passed** at both N (no_skill root success 1.0); **headroom passed** at N = 10
(ic_full and ic_topk 0.27 on composed skills vs no_skill 0).

Success (mean of 2 seeds; pilot 2 in brackets):

| arm | N = 10 | N = 50 |
|---|---|---|
| no_skill | 0.45 (0.45) | 0.26 (0.26) |
| ic_full | 0.40 (0.40) | 0.09 (0.09) |
| ic_topk | 0.40 (0.40) | 0.24 (0.24) |
| ours_b24 | 0.45 (0.10) | 0.28 (0.00) |
| ours_cover | 0.50 (0.10) | 0.31 (0.00) |
| seq_no_replay | 0.45 (0.10) | 0.22 (0.00) |
| isolated | 0.45 (0.10) | 0.25 (0.08) |

**The EOS amendment fixed the collapse.** In pilot 2, 0.50 of the N = 10 attempts in every weight arm were the exact
reference program followed by further lines; here that share is 0.00 in every arm, and 0.45 of the weight arms' N = 10
attempts are exactly the reference program. Arms that do not train are unchanged from pilot 2, as expected (same prompts,
seeds and greedy decoding).

Signal worth noting (2 seeds, not evidence): at N = 50 the replay arms (ours_cover 0.31, ours_b24 0.28) are above
seq_no_replay (0.22), isolated (0.25), no_skill (0.26) and both in-context arms (ic_topk 0.24, ic_full 0.09), with about
465 prompt tokens per attempt against 2,144 for ic_full. ic_full collapses at N = 50 in all three pilots.

With floor and headroom passed and the weight arms working, the pilot condition of the 2026-11-08 deviation row is met;
the confirmatory seeds can run with these settings.
