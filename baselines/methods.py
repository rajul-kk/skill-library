"""The four methods under comparison, as data.

Each is a retrieval configuration plus a sleep configuration plus an adapter
backend factory. Everything else -- environment, curriculum, policy, metrics --
is shared, so any difference in the results comes from these three fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Optional

from library.retrieval import RetrievalConfig
from sleep.backend import IsolatedSimulatedAdapters, NullAdapter, SimulatedAdapter
from sleep.consolidate import SleepConfig
from sleep.replay import ReplayConfig

# LoRA r=16 on Qwen2.5-1.5B q/k/v/o projections; used for the parameter figure
# and kept identical across methods so the storage comparison is apples to apples.
LORA_PARAMS_R16 = 4_718_592


@dataclass
class Method:
    key: str
    label: str
    retrieval: RetrievalConfig
    sleep: SleepConfig
    backend_factory: Callable[[], object]
    note: str = ""

    def build_backend(self):
        return self.backend_factory()


def _cumulative_replay(
    per_skill: int = 2, max_total: int = 24, k: int = 3, strategy: str = "round_robin"
) -> Method:
    return Method(
        key="ours",
        label=f"Cumulative LoRA + replay (b={max_total}, K={k})",
        retrieval=RetrievalConfig(),
        sleep=SleepConfig(
            every_k_tasks=k,
            replay=ReplayConfig(
                enabled=True, per_skill=per_skill, max_total=max_total, strategy=strategy
            ),
        ),
        backend_factory=lambda: SimulatedAdapter(rank=16, base_params=LORA_PARAMS_R16),
        note="one shared adapter, new skills interleaved with capped replay",
    )


def no_consolidation() -> Method:
    """(a) Voyager-style: the whole library stays in context forever."""
    return Method(
        key="in_context",
        label="Full in-context library (no consolidation)",
        retrieval=RetrievalConfig(serve_full_text_always=True),
        sleep=SleepConfig(enabled=False),
        backend_factory=NullAdapter,
        note="capability upper bound, worst token cost",
    )


def sequential_no_replay() -> Method:
    """(b) The same cumulative adapter, but nights see only the new skills."""
    return Method(
        key="seq_no_replay",
        label="Sequential LoRA, no replay",
        retrieval=RetrievalConfig(),
        sleep=SleepConfig(
            every_k_tasks=3,
            replay=ReplayConfig(enabled=False, per_skill=0, max_total=0),
        ),
        backend_factory=lambda: SimulatedAdapter(rank=16, base_params=LORA_PARAMS_R16),
        note="isolates replay as the only difference from ours",
    )


def isolated_adapters() -> Method:
    """(c) PEAM-style: physically isolated adapters, no sharing, no replay."""
    return Method(
        key="isolated",
        label="Isolated per-skill adapters (PEAM-style)",
        retrieval=RetrievalConfig(),
        sleep=SleepConfig(
            every_k_tasks=3,
            replay=ReplayConfig(enabled=False, per_skill=0, max_total=0),
        ),
        backend_factory=lambda: IsolatedSimulatedAdapters(
            rank=16, base_params=LORA_PARAMS_R16
        ),
        note="no forgetting by construction; storage grows linearly",
    )


def single_shot() -> Method:
    """(d) LatentSkill-style: one adapter per skill the moment it is learned."""
    return Method(
        key="single_shot",
        label="One-shot text-to-LoRA per skill (LatentSkill-style)",
        retrieval=RetrievalConfig(),
        sleep=SleepConfig(
            every_k_tasks=1,
            replay=ReplayConfig(enabled=False, per_skill=0, max_total=0),
        ),
        backend_factory=lambda: IsolatedSimulatedAdapters(
            rank=8, base_params=LORA_PARAMS_R16 // 2
        ),
        note="no growing-library loop, no offline consolidation",
    )


def ours() -> Method:
    return _cumulative_replay()


METHODS: Dict[str, Callable[[], Method]] = {
    "in_context": no_consolidation,
    "seq_no_replay": sequential_no_replay,
    "isolated": isolated_adapters,
    "single_shot": single_shot,
    "ours": ours,
}

MAIN_ORDER = ["in_context", "seq_no_replay", "isolated", "single_shot", "ours"]


def get_method(key: str) -> Method:
    if key not in METHODS:
        raise KeyError(f"unknown method {key!r}; known: {sorted(METHODS)}")
    return METHODS[key]()


def replay_size_ablation(sizes=(0, 6, 12, 24, 48)) -> Dict[str, Method]:
    """Replay budget vs retention: the buffer-size ablation."""
    out = {}
    for b in sizes:
        m = _cumulative_replay(per_skill=max(1, b // 12) or 1, max_total=b)
        out[f"buffer_{b}"] = replace(m, key=f"buffer_{b}", label=f"replay buffer = {b}")
    return out


def frequency_ablation(ks=(1, 3, 5)) -> Dict[str, Method]:
    """Consolidation frequency K vs retention and nightly compute."""
    out = {}
    for k in ks:
        m = _cumulative_replay(k=k)
        out[f"K_{k}"] = replace(m, key=f"K_{k}", label=f"consolidate every K = {k}")
    return out


def strategy_ablation(
    strategies=("round_robin", "uniform", "recency", "tier_balanced"),
    max_total: int = 8,
):
    """How to spend a *scarce* buffer.

    Run at a budget below one example per skill on purpose: once the buffer can
    cover the whole library every allocation policy is identical, so the
    question only has content when the cap binds.
    """
    return {
        s: replace(
            _cumulative_replay(per_skill=1, max_total=max_total, strategy=s),
            key=f"strat_{s}",
            label=f"replay: {s} (b={max_total})",
        )
        for s in strategies
    }
