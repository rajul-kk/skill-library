import _paths  # noqa: F401

from agent import LoopConfig, SimulatedPolicy, VoyagerLoop
from library.retrieval import RetrievalConfig
from sleep.backend import NullAdapter
from sleep.consolidate import Consolidator, SleepConfig

backend = NullAdapter()
consolidator = Consolidator(backend, SleepConfig(enabled=False))
policy = SimulatedPolicy(backend, seed=0)
loop = VoyagerLoop(
    policy,
    consolidator,
    LoopConfig(retrieval=RetrievalConfig(serve_full_text_always=True)),
)
loop.run_curriculum(passes=3)

learned = sorted(s.name for s in loop.library)
print(f"library size: {len(learned)}/15")
last_pass = loop.episodes[-15:]
print(f"final-pass success: {sum(e.success for e in last_pass)}/15")
print(f"mean prompt tokens/episode: {sum(e.prompt_tokens for e in last_pass)/15:.0f}")
missing = {t for t in __import__("env").TASKS} - set(learned)
print("never learned:", sorted(missing) or "none")
