# Scoop check: library-size sweep (in-context vs cumulative LoRA + replay)

Date: 2026-11-05. Question checked: does any paper sweep **skill-library size** and compare an
**in-context skill library** against **weight-based consolidation** at **matched tokens or FLOPs**
(including the training cost of the adapters)?

## How the sources were read

All reads went through a web fetch tool that converts the page to text and answers questions about
it with a small model. That is weaker than reading the PDF yourself. Tags:

- **abstract**: the arXiv abstract page only.
- **full HTML (summarised)**: the arXiv HTML full text was fetched and queried section by section.
  Where the tool reported truncation, the tag says which part was covered.
- **README**: the GitHub README.
- **snippet**: a search-engine result only.

No PDF was opened directly. Any claim below of the form "the paper does not do X" means "X was not
found in what was read", not "X is absent".

## Per source

| ID | Title (short) | Read | Library-size sweep? | In-context baseline | Token / FLOP accounting | Training cost matched? | Code |
|---|---|---|---|---|---|---|---|
| 2606.16769 | Skill-to-LoRA (Zhang, Qi) | abstract + full HTML (summarised) | No. Fixed 21-skill subset of SWE-Skills-Bench; Shared-LoRA control on 6 skills | Yes, Full Skill Text | Tokens per step, inference only; no FLOPs | No ("inference-time token usage" only) | "on acceptance" |
| 2606.06087 | LatentSkill (Yu et al.) | abstract + full HTML (summarised) | No. 8 in-domain skills; composition of 2 skills | Yes, In-Context Skill | Prefill/decode tokens; no FLOPs | No; hypernetwork training cost not reported | none found |
| 2606.30015 | Parametric Skills (Zhao et al.) | abstract + full HTML (summarised) | No ablation over library size; 45.8k-skill library used for training; 31-task online accumulation via EMA merge, no replay | Yes, ICL | None reported | No | none found |
| 2605.27762 | PEAM (Guo et al.) | abstract + full HTML v2 (summarised) | No. 3 categories (craft, gather, combat) | Retrieval agents (Voyager, Optimus-1) | Tokens per task (31.2K vs 4.6K); no FLOPs | No | none found |
| 2606.04703 | Rethinking Continual Experience Internalization (Chen et al.) | abstract + full HTML (summarised) | No; pool size named as future work in Limitations | In-context results in Appendix D | No tokens/FLOPs; only turns per trajectory | No | github.com/RUCBM/ExpInternalization |
| 2606.03979 | Language Models Need Sleep (Behrouz et al.) | abstract + full HTML (summarised, both halves) | No; varies number of consolidation stages, not amount consolidated | ICL among baselines | Wall-clock only (App. B.5) | No | none found |
| 2608.03874 | ContinualSkillBench (Guan et al.) | abstract + full HTML first 100k chars (summarised; appendices F-I not read) | No; reports final skill-pool size, does not vary it | Every arm is in-context | None in text read | n/a | github.com/gtynnn060110-hash/continual-skill-bench-final |
| 2604.20087 | SkillLearnBench (Zhong et al.) | abstract + README | No | No-skill only | None | n/a | github.com/cxcscmu/SkillLearnBench |
| 2610.29357 | Memory-Centric Continual Learning (Li et al.) | abstract + full HTML first 200k of 353k chars (summarised) | Own experiments vary capacity of **external** memory only (Mem0, Raw RAG) | n/a | n/a | Gives a conceptual amortisation rule (crossover reuse count = extra construction cost / per-query saving); parametric evidence cited from other papers; no computed crossover | none found |
| 2605.16508 | Scaling Laws of Skills in LLM Agent Systems | abstract + search snippet | Yes, but in-context routing only: routing accuracy decays logarithmically with library size | In-context only | No | n/a | none found |

2605.16508 was not on the original list; it turned up in a search and is the only paper found that
varies library size at all. It has no weight-based arm.

## Verdict

**Not scooped, as of the sources read.** None of the ten papers sweeps library size against an
in-context baseline with a weight-based arm, and none counts adapter training compute against
inference savings. The pieces exist separately:

- token savings of per-skill adapters at one library size (Skill-to-LoRA, LatentSkill, PEAM);
- evidence that a single shared adapter without replay forgets (PEAM B4: 32.4% loss on early craft
  skills; Skill-to-LoRA Shared-LoRA below per-skill adapters);
- in-context degradation with library size (2605.16508, routing accuracy);
- a conceptual amortisation crossover (2610.29357) that nobody has measured for skills.

The sweep in this repo combines them: library size x {in-context, cumulative LoRA + replay,
isolated, sequential, no-skill}, with training FLOPs amortised into per-query cost.

## Closest threats, in order

1. **2610.29357** names the crossover as a scaling-framework prediction. A follow-up that runs it on
   parametric memory would pre-empt the framing. Cite it and position this work as the measurement
   of its amortisation rule for skills.
2. **Skill-to-LoRA** has the in-context vs adapter comparison and a shared-adapter control; adding a
   skill-count sweep would be a small step for them once code is released.
3. **PEAM** already shows shared-without-replay forgetting and token savings vs retrieval agents.

## Caveats

- Reads were summarised by a tool, not by a person reading the PDF. Before citing any number above,
  open the PDF.
- Google Scholar "cited by" sweeps for each paper were not run. 2026 arXiv IDs have shown
  month/prefix mismatches before (see the consolidated research note), so check dates on arxiv.org.
- The repo's existing related-work section cites PEAM's isolation as per-category MoE-LoRA; the
  full text confirms 3 categories and a router (DistilBERT), plus a "single shared LoRA" baseline
  without replay that the related-work section does not mention yet.
