# Sweep pilot, 2026-11-07 (Kaggle T4, notebook 05 version 3; seeds 1000-1001, N = 10 and 50; exploratory)

Wall time 23 min on one T4 (training 800 s of it). Measured throughput: prefill 4.9 TFLOP/s and training 13.4 TFLOP/s
against the 10 TFLOP/s the budget table assumed; decode 4.9 s per batch of 16 (mean 37 generated tokens).

Checks (PROTOCOL_sweep section 9): **floor failed** at both N (no_skill root success 0.11 and 0.08, needs >= 0.30);
**headroom failed** at N = 10 (best in-context arm +0.18 on composed skills vs >= 0.20). Under the protocol as frozen,
these cells would leave the primary analysis, so the pilot settings cannot go to confirmatory seeds unchanged.

Diagnosis of the floor: 25 of the 35 no_skill root attempts produced ~30 tokens from which no DSL line was extracted
(`program` empty), although the prompt names the location and the answer is two lines. The other arms never produced an
empty program. So the floor failure is a format failure of the bare prompt, not inability. Raw generations are not
stored, so the exact text cannot be read from these records.

Signal worth noting (2 seeds, not evidence): at N = 50, ours_cover 0.29 and ours_b24 0.21 vs isolated 0.19,
seq_no_replay 0.07, ic_full 0.18, ic_topk 0.07; weight arms used ~400 prompt tokens vs 2,102 for ic_full.
