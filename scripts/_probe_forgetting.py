import _paths  # noqa: F401
from sleep.backend import Example, SimulatedAdapter

skills = [f"s{i}" for i in range(1, 9)]

no_replay = SimulatedAdapter()
for n, sk in enumerate(skills, 1):
    no_replay.consolidate([Example(sk, "p", "t", n)], n)

with_replay, seen = SimulatedAdapter(), []
for n, sk in enumerate(skills, 1):
    batch = [Example(sk, "p", "t", n)]
    batch += [Example(s, "p", "t", n, True) for s in seen[-4:]]
    with_replay.consolidate(batch, n)
    seen.append(sk)

print("no replay :", {k: round(v, 2) for k, v in no_replay.retention.items()})
print("replay(4) :", {k: round(v, 2) for k, v in with_replay.retention.items()})
