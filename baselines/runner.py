"""One driver for every method.

The runner owns the loop, the per-night evaluation and the checkpointing, so
methods differ only by their Method record. Results are written as JSON under
runs/<run_id>/<method>/ alongside the night checkpoints, which is what makes a
run resumable across Kaggle sessions.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from agent.loop import LoopConfig, VoyagerLoop
from agent.policy import SimulatedPolicy, SynthesisModel
from analysis.metrics import (
    Snapshot,
    evaluate_library,
    forgetting_curve,
    forward_transfer,
    token_cost_summary,
)
from env.tasks import curriculum_order
from library.retrieval import heuristic_token_count
from library.store import SkillLibrary
from sleep.checkpoint import Manifest, save_night
from sleep.consolidate import Consolidator

from .methods import Method, get_method


@dataclass
class RunConfig:
    run_id: str = "cpu_dev"
    seed: int = 0
    seeds: tuple = (0, 1, 2)
    passes: int = 3
    eval_repeats: int = 3
    max_attempts: int = 4
    checkpoint: bool = True
    root: str = "runs"
    backend_name: str = "simulated"


@dataclass
class RunResult:
    method: str
    label: str
    snapshots: List[Snapshot]
    episodes: List[dict]
    nights: List[dict]
    transfer: Dict[str, float]
    tokens: Dict[str, float]
    seconds: float = 0.0
    per_seed_final: List[float] = field(default_factory=list)

    def final(self, field_name: str) -> float:
        """Mean over the last snapshot of each seed's run."""
        if not self.snapshots:
            return 0.0
        last_night = max(s.night for s in self.snapshots)
        finals = [s for s in self.snapshots if s.night == last_night]
        if field_name == "earliest_3":
            values = [s.earliest_k(3) for s in finals]
        else:
            values = [getattr(s, field_name) for s in finals]
        return sum(values) / len(values)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "label": self.label,
            "seconds": self.seconds,
            "per_seed_final": self.per_seed_final,
            "transfer": self.transfer,
            "tokens": self.tokens,
            "final_success_rate": self.final("success_rate"),
            "final_earliest_3": self.final("earliest_3"),
            "final_stored_params": int(self.final("stored_params")),
            "forgetting_curve": {str(k): v for k, v in forgetting_curve(self.snapshots).items()},
            "forgetting_curve_by_skills": {
                str(k): v for k, v in forgetting_curve(self.snapshots, "skills").items()
            },
            "snapshots": [s.to_dict() for s in self.snapshots],
            "episodes": self.episodes,
            "nights": self.nights,
        }


def run_method(
    method: Method,
    config: Optional[RunConfig] = None,
    policy_factory: Optional[Callable] = None,
) -> RunResult:
    """Run one method through the full curriculum, evaluating after each night."""
    config = config or RunConfig()
    started = time.time()

    backend = method.build_backend()
    consolidator = Consolidator(backend, method.sleep)
    loop_config = LoopConfig(
        max_attempts=config.max_attempts,
        retrieval=method.retrieval,
        token_counter=heuristic_token_count,
    )

    def default_factory(seed: int):
        return SimulatedPolicy(backend, SynthesisModel(), seed=seed)

    make_policy = policy_factory or default_factory
    policy = make_policy(config.seed)

    snapshots: List[Snapshot] = []
    run_root = Path(config.root) / config.run_id / method.key

    def on_night(record, library: SkillLibrary) -> None:
        snap = evaluate_library(
            library,
            make_policy,
            loop_config,
            night=record.night,
            episode=record.episode,
            stored_params=backend.stored_parameters(),
            repeats=config.eval_repeats,
        )
        snapshots.append(snap)
        if config.checkpoint:
            save_night(
                run_root,
                record.night,
                library,
                consolidator,
                Manifest(
                    run_id=config.run_id,
                    method=method.key,
                    night=record.night,
                    episode=record.episode,
                    library_size=len(library),
                    stored_params=backend.stored_parameters(),
                    backend=config.backend_name,
                    seed=config.seed,
                ),
            )

    loop = VoyagerLoop(policy, consolidator, loop_config, on_night=on_night)
    loop.run_curriculum(curriculum_order(), passes=config.passes)
    loop.finish()

    if not snapshots:
        # No-consolidation baseline never sleeps; measure it once at the end.
        snapshots.append(
            evaluate_library(
                loop.library,
                make_policy,
                loop_config,
                night=0,
                episode=len(loop.episodes),
                stored_params=backend.stored_parameters(),
                repeats=config.eval_repeats,
            )
        )

    return RunResult(
        method=method.key,
        label=method.label,
        snapshots=snapshots,
        episodes=[e.to_dict() for e in loop.episodes],
        nights=[n.to_dict() for n in consolidator.nights],
        transfer=forward_transfer(loop.episodes, loop.library),
        tokens=token_cost_summary(loop.episodes),
        seconds=time.time() - started,
    )


def run_method_seeds(method: Method, config: RunConfig) -> RunResult:
    """Run one method once per seed and pool the results.

    Pooling matters: a single 15-skill run is noisy enough that method
    differences smaller than a couple of skills are not readable off one seed.
    """
    from dataclasses import replace as _replace

    runs = [
        run_method(method, _replace(config, seed=s, checkpoint=config.checkpoint and i == 0))
        for i, s in enumerate(config.seeds)
    ]
    pooled = RunResult(
        method=method.key,
        label=method.label,
        snapshots=[snap for r in runs for snap in r.snapshots],
        episodes=[e for r in runs for e in r.episodes],
        nights=[n for r in runs for n in r.nights],
        transfer={
            k: sum(r.transfer[k] for r in runs) / len(runs) for k in runs[0].transfer
        },
        tokens={k: sum(r.tokens[k] for r in runs) / len(runs) for k in runs[0].tokens},
        seconds=sum(r.seconds for r in runs),
    )
    pooled.per_seed_final = [
        r.snapshots[-1].success_rate for r in runs if r.snapshots
    ]
    return pooled


def run_all(
    method_keys: List[str],
    config: Optional[RunConfig] = None,
    write: bool = True,
) -> Dict[str, RunResult]:
    config = config or RunConfig()
    results: Dict[str, RunResult] = {}
    for key in method_keys:
        results[key] = run_method_seeds(get_method(key), config)
    if write:
        write_results(results, config)
    return results


def run_grid(
    methods: Dict[str, Method], config: Optional[RunConfig] = None, write: bool = True
) -> Dict[str, RunResult]:
    """Run an ablation grid of pre-built Method records."""
    config = config or RunConfig()
    results = {k: run_method_seeds(m, config) for k, m in methods.items()}
    if write:
        write_results(results, config)
    return results


def write_results(results: Dict[str, RunResult], config: RunConfig) -> Path:
    out = Path(config.root) / config.run_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    payload = {
        "run_id": config.run_id,
        "seed": config.seed,
        "passes": config.passes,
        "backend": config.backend_name,
        "methods": {k: v.to_dict() for k, v in results.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_results(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
