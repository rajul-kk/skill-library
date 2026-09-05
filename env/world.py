"""CPU-only executor for the crafting DSL."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .dsl import (
    LOCATIONS,
    RECIPES,
    RESOURCES,
    STATIONS,
    TOOLS,
    DSLError,
    Instruction,
    parse_program,
)

MAX_CALL_DEPTH = 12
MAX_STEPS = 2000


class ExecutionError(Exception):
    """A program ran but hit an illegal action."""


@dataclass
class CraftingWorld:
    location: str = "forest"
    inventory: Dict[str, int] = field(default_factory=dict)
    stations: Dict[str, set] = field(default_factory=lambda: {s: set() for s in STATIONS})
    steps: int = 0

    # ---------------- state helpers ---------------- #

    def count(self, item: str) -> int:
        return self.inventory.get(item, 0)

    def has(self, item: str, n: int = 1) -> bool:
        return self.count(item) >= n

    def _add(self, item: str, n: int) -> None:
        self.inventory[item] = self.count(item) + n

    def _take(self, item: str, n: int) -> None:
        if not self.has(item, n):
            raise ExecutionError(f"need {n} {item}, have {self.count(item)}")
        self.inventory[item] -= n
        if self.inventory[item] == 0:
            del self.inventory[item]

    def station_here(self, station: str) -> bool:
        return self.location in self.stations.get(station, set())

    def snapshot(self) -> dict:
        return {
            "location": self.location,
            "inventory": dict(sorted(self.inventory.items())),
            "stations": {k: sorted(v) for k, v in self.stations.items()},
            "steps": self.steps,
        }

    def observe(self) -> str:
        """Compact text observation given to the agent."""
        inv = ", ".join(f"{k} x{v}" for k, v in sorted(self.inventory.items())) or "empty"
        placed = ", ".join(
            f"{s}@{loc}" for s, locs in self.stations.items() for loc in sorted(locs)
        ) or "none"
        return f"at={self.location} | inventory: {inv} | stations: {placed}"

    # ---------------- primitive actions ---------------- #

    def goto(self, loc: str) -> None:
        if loc not in LOCATIONS:
            raise ExecutionError(f"no such location {loc!r}")
        self.location = loc

    def gather(self, resource: str, n: int = 1) -> None:
        res = RESOURCES.get(resource)
        if res is None:
            raise ExecutionError(f"{resource!r} is not gatherable")
        if self.location not in res.locations:
            raise ExecutionError(
                f"{resource} is not at {self.location} (try {'/'.join(res.locations)})"
            )
        if res.requires_tool and not self.has(res.requires_tool):
            raise ExecutionError(f"gathering {resource} requires a {res.requires_tool}")
        self._add(resource, res.yield_per_gather * n)

    def craft(self, item: str, n: int = 1) -> None:
        rec = RECIPES.get(item)
        if rec is None:
            raise ExecutionError(f"{item!r} has no recipe")
        if rec.station and not self.station_here(rec.station):
            raise ExecutionError(f"crafting {item} needs a {rec.station} placed here")
        for _ in range(n):
            for name, qty in rec.inputs:
                self._take(name, qty)
            self._add(item, rec.output_count)

    def place(self, station: str) -> None:
        if station not in STATIONS:
            raise ExecutionError(f"{station!r} is not a placeable station")
        self._take(station, 1)
        self.stations[station].add(self.location)

    # ---------------- program execution ---------------- #

    def run(
        self,
        program: str | List[Instruction],
        skill_resolver: Optional[Callable[[str], str]] = None,
        _depth: int = 0,
    ) -> None:
        if _depth > MAX_CALL_DEPTH:
            raise ExecutionError(f"skill call depth exceeded {MAX_CALL_DEPTH}")
        instructions = parse_program(program) if isinstance(program, str) else program
        for ins in instructions:
            self.steps += 1
            if self.steps > MAX_STEPS:
                raise ExecutionError(f"step budget of {MAX_STEPS} exhausted")
            self._exec(ins, skill_resolver, _depth)

    def _exec(
        self,
        ins: Instruction,
        skill_resolver: Optional[Callable[[str], str]],
        depth: int,
    ) -> None:
        def as_int(value: str, what: str) -> int:
            try:
                k = int(value)
            except ValueError as exc:
                raise ExecutionError(f"{what} count must be an integer, got {value!r}") from exc
            if k < 1:
                raise ExecutionError(f"{what} count must be >= 1, got {k}")
            return k

        if ins.op == "goto":
            self.goto(ins.args[0])
        elif ins.op == "gather":
            n = as_int(ins.args[1], "gather") if len(ins.args) > 1 else 1
            self.gather(ins.args[0], n)
        elif ins.op == "craft":
            n = as_int(ins.args[1], "craft") if len(ins.args) > 1 else 1
            self.craft(ins.args[0], n)
        elif ins.op == "place":
            self.place(ins.args[0])
        elif ins.op == "call":
            if skill_resolver is None:
                raise ExecutionError("call() used but no skill library is available")
            name = ins.args[0]
            try:
                body = skill_resolver(name)
            except KeyError as exc:
                raise ExecutionError(f"unknown skill {name!r}") from exc
            self.run(body, skill_resolver, _depth=depth + 1)
        else:  # pragma: no cover - parse_program rejects unknown ops
            raise ExecutionError(f"unhandled op {ins.op!r}")


def fresh_world() -> CraftingWorld:
    return CraftingWorld()
