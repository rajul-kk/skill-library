# Related work

## Two-timescale distillation

The setting this paper works in -- an explicit skill library that periodically
writes its own contents back into model weights -- is named as an open direction
by the *Dynamic Agent Skills* survey (arXiv:2607.10113), which lists
"two-timescale distillation" among the emerging and unresolved categories. The
survey is evidence that the question is recognised, not that it is answered; the
papers below each take one part of it and stop short of the combination studied
here.

## PEAM (arXiv:2605.27762)

PEAM is the closest prior work and the baseline this paper is measured against.
It internalises episodic trajectory pairs into LoRA in an embodied Minecraft
setting, and it decides *when* to consolidate with a self-triggered timing
mechanism.

Two differences matter.

**What is consolidated.** PEAM consolidates episodic trajectory pairs. There is
no explicit, growing, Voyager-style library of textual or code skills that the
agent retrieves from, reuses compositionally, and then stops paying context for.
Here the library is the primary object: skills call other skills by name, and
consolidation is what lets a skill's full text leave the prompt while remaining
callable.

**How forgetting is prevented.** PEAM prevents interference by *construction*:
physically isolated per-category MoE-LoRA adapters, so no two categories share
parameters. This paper prevents it by *replay* into one shared cumulative
adapter. The two strategies have opposite cost profiles -- isolation cannot
forget but its storage grows with every category, while a shared adapter is
flat but forgets unless something actively protects earlier skills.

Nobody has run that comparison directly, which is what this paper does. The
result (Section 4) is that replay matches isolation's retention at 15x less adapter
storage, and the paper reports the axis where isolation genuinely wins
(robustness by construction, no replay budget to tune) rather than arguing
around it.

## Skill-to-LoRA (arXiv:2606.16769)

Skill-to-LoRA distils each skill into its own independent LoRA. There is no
growing library, no shared weights, and no replay across skills, so the question
of interference between skills does not arise. It is the pure form of the
tradeoff PEAM makes: perfect retention, linear storage. It appears here as the
motivation for reporting adapter parameter count as a first-class metric rather
than a footnote.

## LatentSkill (arXiv:2606.06087)

LatentSkill converts a skill into LoRA parameters with a single hypernetwork
forward pass -- no gradient-based training at consolidation time at all. It is
fast and it is one-shot, but there is no growing library and no offline
consolidation phase, so nothing accumulates and nothing has to be protected.
Baseline (d) is the LatentSkill-shaped reference point: one adapter per skill
the moment it is learned, no library loop, no sleep phase.

## TMEM (arXiv:2606.04536)

TMEM writes distilled experience into a fast LoRA *within* an episode and relies
on an RL outer loop. Two things separate it from this work. The timescale is
intra-episode rather than a periodic offline phase over a cross-episode library,
and the learning signal is RL rather than SFT. This project is deliberately
SFT-only: the success signal comes from a programmatic check against environment
state, so there is no reward model, no critic, and no policy-gradient machinery
to fit on a single T4.

## Voyager

The outer loop is Voyager's -- curriculum, skill library, retrieval, compositional
reuse -- with one deliberate removal. Voyager's GPT-4 self-verification critic is
replaced by a programmatic predicate over world state. This is a simplification
made for compute, but it also makes the success signal exact rather than
model-mediated, which matters when the thing being measured is whether a skill
still works.

## Continual learning

Replay as a defence against catastrophic forgetting is old, and none of the
mechanism here is novel as continual learning. The contribution is not the
replay buffer; it is applying it at the seam between an agent's explicit skill
library and its weights, and measuring it against the isolation strategy the
agent literature has converged on instead.

## Summary

| | growing skill library | shared weights | replay | offline consolidation |
|---|---|---|---|---|
| PEAM | no | no (isolated) | no | yes |
| Skill-to-LoRA | no | no (per-skill) | no | yes |
| LatentSkill | no | no (per-skill) | no | no (hypernetwork) |
| TMEM | no | yes | no | no (intra-episode) |
| **this work** | **yes** | **yes** | **yes** | **yes** |
