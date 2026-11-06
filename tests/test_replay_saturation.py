"""Why `ours` and `isolated` print identical rows in results/cpu/summary.txt.

Round one of ReplayBuffer.sample takes one example of every known skill (up to max_total). The CPU curriculum
never exceeds 15 skills, so with b=24 every skill is replayed every night. SimulatedAdapter skips decay for any
trained skill and sets retention with max(old, gain), so a learned skill stays at exactly 1.0, which is what
IsolatedSimulatedAdapters returns. The two methods then see the same retention, the same recall draws and the
same snapshots: the CPU table's "replay matches isolation" holds by construction, not as a finding.
These tests pin that mechanism and show the comparison becomes informative once the library outgrows the buffer.
"""
from dataclasses import asdict, replace

from baselines.methods import get_method, replay_size_ablation
from baselines.runner import RunConfig, run_method
from sleep import backend as B


def _snapshots(result):
    return [{k: v for k, v in asdict(s).items() if k not in ("stored_params", "method")} for s in result.snapshots]


def _cfg(seed=0):
    return replace(RunConfig(), seed=seed, checkpoint=False)


def test_ours_retention_never_drops_below_one_when_buffer_covers_library(monkeypatch):
    seen = []
    orig = B.SimulatedAdapter.consolidate

    def record(self, examples, night):
        stats = orig(self, examples, night)
        seen.append((len(self.retention), min(self.retention.values(), default=1.0)))
        return stats

    monkeypatch.setattr(B.SimulatedAdapter, "consolidate", record)
    run_method(get_method("ours"), _cfg())
    assert max(n for n, _ in seen) <= get_method("ours").sleep.replay.max_total
    assert all(m == 1.0 for _, m in seen)


def test_ours_equals_isolated_when_buffer_covers_library():
    assert _snapshots(run_method(get_method("ours"), _cfg())) == _snapshots(run_method(get_method("isolated"), _cfg()))


def test_ours_differs_from_isolated_when_library_outgrows_buffer():
    small = replay_size_ablation(sizes=(6,))["buffer_6"]
    assert _snapshots(run_method(small, _cfg())) != _snapshots(run_method(get_method("isolated"), _cfg()))
