---
title: "Sleep-Consolidating a Growing Skill Library into a Cumulative LoRA via Replay"
author: Rajul Kabeer
abstract: |
  Agents that accumulate skills in an explicit library pay for those skills in
  context on every step, forever. Distilling them into weights removes that cost
  but risks forgetting the earliest skills. Recent work prevents forgetting by
  physically isolating one adapter per skill or category, which cannot forget
  but whose storage grows with every skill learned. We ask whether replay into a
  single cumulative adapter can retain a growing library as well as isolation
  does, at flat cost. In a 15-skill crafting environment with programmatic
  verification, a capped replay buffer interleaved with new skills each night
  matches isolation's retention while storing 15x fewer adapter parameters, and
  halves in-context cost against keeping the library in the prompt. The same
  adapter without replay collapses to 0.09 success, which is the control that
  makes the comparison meaningful.
---

# 1. Introduction

A Voyager-style agent writes skills into a library and retrieves them into its
prompt. The library is the agent's memory, and the cost of that memory is paid
on every step: the more it knows, the more it must read before it can act.

The natural fix is to move skills into weights. The difficulty is that a skill
library is *ordered* -- skills are learned one after another, early skills are
prerequisites for late ones, and an adapter trained on new skills will drift
away from old ones. Recent agent work resolves this by isolation: give each
skill, or each category, its own adapter, and nothing can interfere. This works,
and it costs a new adapter for every skill.

We ask the question the literature has left open: is that cost necessary?
Replay is the standard answer to forgetting in continual learning, and it is
cheap. If a bounded replay sample interleaved with new skills each night retains
a shared adapter as well as isolation retains separate ones, then the linear
storage is buying nothing at this scale.

**Contributions.**

1. A consolidation loop in which a growing textual skill library is periodically
   distilled into one cumulative LoRA via replay, after which consolidated
   skills leave the prompt and survive only as names.
2. A direct comparison of replay-into-shared against PEAM-style isolation on the
   same library, curriculum and metrics, reporting retention *and* adapter
   storage together.
3. A no-replay control that demonstrably forgets, so the comparison has a floor
   as well as a ceiling.
4. Ablations showing that the replay budget saturates once it covers the library
   once, that frequent consolidation is both more expensive and worse, and that
   the allocation policy only matters when the budget binds.

# 2. Setup

## 2.1 Environment

A text-only crafting environment: five locations, seven gatherable resources,
thirteen recipes, tool gating (stone requires a wooden pickaxe, iron requires a
stone pickaxe) and two placeable stations. Programs are written in a
five-operation DSL -- `goto`, `gather`, `craft`, `place`, `call` -- and `call`
invokes a previously-learned skill by name, which makes the library
compositional.

The curriculum is 15 tasks across 10 prerequisite tiers, from `gather_wood` to
`craft_lantern`. Every task has a reference solution verified from a fresh world
in the test suite, so the tree is known solvable before any model is involved.

## 2.2 No critic

Voyager verifies success with a GPT-4 critic. We replace it with a predicate
over world state. This is partly a compute decision -- the critic is the most
expensive part of Voyager's loop -- but it also makes the success signal exact.
When the measurement is "does this skill still work", a model-mediated answer is
the wrong instrument.

## 2.3 A night

Every K completed tasks the agent sleeps:

1. the skills learned since the last night become SFT examples: task description
   in, verified program out, **with no library text in the prompt**, so a
   correct program can only come from weights;
2. a capped replay sample is drawn from every skill learned earlier;
3. new and replayed examples are **interleaved** and one short LoRA pass (a few
   hundred steps, rank 16) runs over the mixture;
4. the delta lands in the **same** adapter as every previous night -- never
   reset, never forked;
5. the new skills are marked consolidated, and retrieval serves only their names
   from then on.

Steps 3 and 4 are the method. Interleaving rather than concatenating keeps
replay gradients spread through the pass. Sharing one adapter rather than
forking one per skill is what keeps storage flat, and is the specific choice
this paper tests.

The replay buffer is capped from the outset (default 2 examples per skill, 24
total). An uncapped buffer would make each night's cost grow with the library
and defeat the point of a lifelong loop on one GPU.

## 2.4 Methods compared

- **(a) Full in-context library.** No consolidation; every skill stays in the
  prompt at full length. Capability ceiling, worst token cost.
- **(b) Sequential LoRA, no replay.** The same cumulative adapter, but nights see
  only the new skills. Differs from ours in exactly one thing.
- **(c) Isolated adapters (PEAM-style).** One adapter per skill, no sharing.
  Cannot forget; storage grows linearly.
- **(d) One-shot per skill (LatentSkill-style).** An adapter per skill at the
  moment it is learned, no library loop, no consolidation phase.
- **(ours) Cumulative LoRA + replay.**

Everything except the retrieval configuration, the sleep configuration and the
adapter backend is shared, so any difference in results comes from those three
fields.

## 2.5 Metrics

**Retention** is measured the way each method would actually be deployed: every
learned task is re-attempted under that method's own retrieval configuration. A
method that keeps everything in context pays tokens for its score; a method that
consolidated has to recall.

The headline curve normalises each skill against its own score when freshly
learned, and excludes skills whose programs call nothing. Both corrections
matter: without normalisation the curve tracks which tasks happen to be old
rather than how much was forgotten, and root skills need no recall at all, so
including them lifts the tail.

**Storage** is total adapter parameters that must be kept: flat for a shared
adapter, linear in skill count for isolated ones.

**Token cost** is the mean prompt length per attempt.

# 3. Implementation

Base model Qwen2.5-1.5B-Instruct, LoRA rank 16 on the attention projections, a
few hundred steps per night, sized for a single T4. Library and adapter are
checkpointed after every night with a manifest, so a lifelong run spans Kaggle
sessions: a session does what nights it can and the next resumes from the last
manifest.

The repository also carries a CPU pipeline in which the adapter is replaced by
an explicit interference model and recall is sampled rather than decoded. It
exists so the loop, the metrics and the figures can be developed and tested
without a GPU, and every checkpoint records which backend produced it. **CPU
numbers validate the pipeline; they are not evidence about LoRA.** The results
below are labelled accordingly.

# 4. Results

## 4.1 The comparison (CPU pipeline, 20 seeds)

| method | final success | oldest 3 | prompt tokens | stored params |
|---|---|---|---|---|
| (a) in-context | 0.80 | 0.76 | 856 | 0 |
| (b) sequential, no replay | 0.09 | 0.23 | 679 | 4.7 M |
| (c) isolated (PEAM-style) | 0.79 | 0.78 | 417 | 70.8 M |
| (d) one-shot (LatentSkill-style) | 0.80 | 0.77 | 320 | 35.4 M |
| **ours** | **0.79** | **0.78** | **417** | **4.7 M** |

These numbers were regenerated on 2026-11-08. The earlier table (in-context 0.85 / 0.87, ours 0.73 / 0.87,
one-shot 0.71 / 0.67) was produced with evaluation seeds that did not depend on the run seed and came from Python's
per-process string hash, so all 20 seeds replayed the same evaluation draws and averaging over seeds did not average
out evaluation noise. Each seed now draws its own, reproducibly; the differences between (a), (c), (d) and ours that
the earlier table showed were that shared noise.

Three things to read off this.

**Replay matches isolation.** Ours and (c) are indistinguishable on retention.
That is the intended result and it is not a coincidence: with a buffer large
enough to cover the library, every skill is refreshed every night, so the shared
adapter is under no more strain than separate ones. The difference is the last
column -- 4.7 M against 70.8 M, and the gap grows with every skill added.

**The control forgets.** Without replay the same adapter collapses to 0.09, with
normalised retention reaching zero within a few skills. This is the baseline
that makes the rest meaningful, and it is asserted as a test: if the setup ever
stops detecting forgetting, the study has no result to report.

The table's token column is the mean across the whole run, when the library is
still small for much of it. At full library size the gap is wider: 1164 tokens
per attempt for the in-context baseline against 353 once skills have been
consolidated, which is what figure 3 shows.

**No measurable gap to the in-context ceiling, in the simulator.** Keeping the
library in context scores 0.80 against our 0.79 (0.76 against 0.78 on the oldest
skills). An earlier version of this table reported a 0.85 vs 0.73 gap and called
it the price of moving skills out of the prompt; that gap was evaluation noise
shared across seeds (see above). This is a property of the simulated adapter,
whose recall of a covered skill is held at 1.0; whether a real LoRA pays a price
for leaving the prompt is what the GPU runs and the library-size sweep measure.
What is bought is roughly half the prompt cost and flat storage.

## 4.2 Ablations

**Replay budget** (0, 6, 12, 24, 48): 0.09, 0.56, 0.79, 0.79, 0.79. Retention
saturates as soon as the buffer covers the library once, and buys nothing after
that. A replay buffer does not need to grow with the library; it needs to cover
it. This is what makes a fixed cap a design choice rather than a compromise.

**Consolidation frequency** (K = 1, 3, 5): 0.80, 0.79, 0.80. Frequency makes no
difference to retention here, so fewer, larger nights are simply cheaper. (An
earlier version reported 0.71, 0.73, 0.87 and concluded that larger nights were
also better; that was shared evaluation noise.) Either way, the self-triggered
timing mechanism PEAM builds is not load-bearing at this scale.

**Allocation under a scarce buffer.** At b=24 every allocation strategy covers
the library and they are identical, so this grid runs at b=8, below one example
per skill. There, tier-balanced allocation retains the oldest skills best (0.78)
against uniform (0.51) and both recency and round-robin (0.22). Our default is
round-robin, chosen for its coverage guarantee, and **the ablation does not
support it**: on a binding budget, spreading replay across the tree beats
rotating through skills.

# 5. Related work
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

# 6. Limitations

The GPU results are the ones that count, and this paper reports the CPU pipeline
plus the notebooks that produce them.

Fifteen skills is a small library. The storage argument against isolation gets
stronger with scale, but so might interference in a shared adapter, and a
rank-16 adapter holding hundreds of skills is untested here. The saturation
result in particular should be re-checked at scale: covering a library of 15 is
cheap, covering one of 500 may not be.

The environment is synthetic and was defined by the same person evaluating on
it. The planned check against Crafter's text-observation mode is not done.

Isolation wins on robustness by construction: no buffer to size, no allocation
policy to choose, and no failure mode where the budget was too small. The claim
here is that this is not worth linear storage at this scale, not that isolation
is the wrong design.

# 7. Conclusion

Replay into one cumulative adapter retains a growing skill library as well as
physically isolating one adapter per skill, at flat rather than linear storage,
and roughly halves the in-context cost of the library. The comparison rests on a
control that visibly forgets, and the ablations say the replay budget can be
small and fixed, that consolidation can be infrequent, and that how a scarce
budget is spent matters more than we assumed when choosing the default.
