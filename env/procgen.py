"""Procedural crafting worlds with nonce names, grown one skill at a time for size sweeps."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .tasks import Task
from .world import CraftingWorld, ExecutionError

_ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "gl", "kr", "pl", "st", "tr", "sk")
_VOWELS = ("a", "e", "i", "o", "u", "ai", "ou")
_CODAS = ("", "", "n", "r", "l", "x", "m", "sk", "th")

MAX_TIER = 7
MAX_EXPANDED_STEPS = 400


@dataclass(frozen=True)
class GenResource:
    """A raw resource in a generated world: one location, fixed yield of 1."""

    name: str
    location: str


@dataclass(frozen=True)
class GenRecipe:
    """A generated recipe: consumes inputs, outputs one item, needs no station."""

    item: str
    inputs: Tuple[Tuple[str, int], ...]


@dataclass
class GenSkill:
    """One generated skill: its task, verified reference program and provenance."""

    index: int
    task: Task
    program: str
    kind: str
    produces: str
    produces_qty: int
    calls: Tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.task.name

    @property
    def description(self) -> str:
        return self.task.description

    @property
    def is_root(self) -> bool:
        return not self.calls


@dataclass
class WorldSpec:
    """Everything that defines a generated world."""

    seed: int
    locations: Tuple[str, ...]
    resources: Dict[str, GenResource] = field(default_factory=dict)
    recipes: Dict[str, GenRecipe] = field(default_factory=dict)
    skills: List[GenSkill] = field(default_factory=list)

    def skill(self, name: str) -> GenSkill:
        for s in self.skills:
            if s.name == name:
                return s
        raise KeyError(name)

    def prefix(self, n: int) -> List[GenSkill]:
        """The library a lifelong agent holds after learning its first n skills."""
        if n > len(self.skills):
            raise ValueError(f"world has {len(self.skills)} skills, asked for {n}")
        return self.skills[:n]

    def dsl_reference(self) -> str:
        return (
            "DSL (one instruction per line):\n"
            "  goto(<location>)          move to a location\n"
            "  gather(<resource>[, n])   gather n of a resource at the current location\n"
            "  craft(<item>[, n])        craft n of an item from its inputs\n"
            "  call(<skill_name>)        run a known skill by name\n"
            f"Locations: {', '.join(self.locations)}\n"
        )


class SpecWorld(CraftingWorld):
    """The existing executor, re-pointed at a generated world's resources and recipes."""

    def __init__(self, spec: WorldSpec) -> None:
        super().__init__(location=spec.locations[0], stations={})
        self.spec = spec

    def goto(self, loc: str) -> None:
        if loc not in self.spec.locations:
            raise ExecutionError(f"no such location {loc!r}")
        self.location = loc

    def gather(self, resource: str, n: int = 1) -> None:
        res = self.spec.resources.get(resource)
        if res is None:
            raise ExecutionError(f"{resource!r} is not gatherable")
        if self.location != res.location:
            raise ExecutionError(f"{resource} is not at {self.location}")
        self._add(resource, n)

    def craft(self, item: str, n: int = 1) -> None:
        rec = self.spec.recipes.get(item)
        if rec is None:
            raise ExecutionError(f"{item!r} has no recipe")
        for _ in range(n):
            for name, qty in rec.inputs:
                self._take(name, qty)
            self._add(item, 1)

    def place(self, station: str) -> None:
        raise ExecutionError("generated worlds have no stations")


class _Names:
    """Unique pronounceable nonce words from a seeded RNG."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.used: set = set()

    def word(self, syllables: Tuple[int, int] = (2, 3)) -> str:
        for _ in range(1000):
            n = self.rng.randint(*syllables)
            w = "".join(self.rng.choice(_ONSETS) + self.rng.choice(_VOWELS) for _ in range(n))
            w += self.rng.choice(_CODAS)
            if w not in self.used and len(w) <= 12:
                self.used.add(w)
                return w
        raise RuntimeError("name space exhausted")


def _expanded_steps(spec: WorldSpec, program_lines: List[str], cache: Dict[str, int]) -> int:
    total = 0
    for line in program_lines:
        if line.startswith("call("):
            total += 1 + cache[line[5:-1]]
        else:
            total += 1
    return total


def generate_world(n_skills: int = 200, seed: int = 0, n_locations: int = 6, p_gather: float = 0.2) -> WorldSpec:
    """Grow a world skill by skill; any prefix is itself a coherent library."""
    rng = random.Random(f"procgen-{seed}")
    names = _Names(rng)
    spec = WorldSpec(seed=seed, locations=tuple(names.word((2, 2)) for _ in range(n_locations)))
    steps: Dict[str, int] = {}
    tier_of: Dict[str, int] = {}

    def add_gather(index: int) -> GenSkill:
        res = GenResource(names.word(), rng.choice(spec.locations))
        spec.resources[res.name] = res
        qty = rng.randint(2, 3)
        program = [f"goto({res.location})", f"gather({res.name}, {qty})"]
        task = Task(
            name=f"gather_{res.name}",
            description=f"Collect {qty} {res.name} from the {res.location}.",
            goal_items=((res.name, qty),),
            prereqs=(),
            tier=0,
        )
        return GenSkill(index, task, "\n".join(program), "gather", res.name, qty)

    def add_craft(index: int) -> Optional[GenSkill]:
        skill_items = [s for s in spec.skills]
        raw = list(spec.resources.values())
        n_inputs = rng.choices((1, 2, 3), weights=(0.3, 0.5, 0.2))[0]
        chosen_skills: List[GenSkill] = []
        chosen_raw: List[Tuple[GenResource, int]] = []
        for _ in range(n_inputs):
            if rng.random() < 0.6 and skill_items:
                pool = skill_items[-30:] if rng.random() < 0.7 else skill_items
                pick = rng.choice(pool)
                if pick in chosen_skills or tier_of[pick.name] + 1 > MAX_TIER:
                    continue
                chosen_skills.append(pick)
            else:
                res = rng.choice(raw)
                if any(r.name == res.name for r, _ in chosen_raw):
                    continue
                chosen_raw.append((res, rng.randint(1, 2)))
        if not chosen_skills:
            return None  # a craft skill must reuse at least one earlier skill
        item = names.word()
        lines: List[str] = []
        parts: List[List[str]] = [[f"call({s.name})"] for s in chosen_skills]
        parts += [[f"goto({r.location})", f"gather({r.name}, {q})"] for r, q in chosen_raw]
        rng.shuffle(parts)
        for p in parts:
            lines.extend(p)
        lines.append(f"craft({item})")
        if _expanded_steps(spec, lines, steps) > MAX_EXPANDED_STEPS:
            return None
        inputs = tuple((s.produces, 1) for s in chosen_skills) + tuple((r.name, q) for r, q in chosen_raw)
        spec.recipes[item] = GenRecipe(item, inputs)
        task = Task(
            name=f"make_{item}",
            description=f"Make a {item}.",
            goal_items=((item, 1),),
            prereqs=tuple(s.name for s in chosen_skills),
            tier=1 + max(tier_of[s.name] for s in chosen_skills),
        )
        return GenSkill(index, task, "\n".join(lines), "craft", item, 1, tuple(s.name for s in chosen_skills))

    while len(spec.skills) < n_skills:
        index = len(spec.skills)
        skill = None
        if index < 4 or rng.random() < p_gather:
            skill = add_gather(index)
        else:
            for _ in range(20):
                skill = add_craft(index)
                if skill is not None:
                    break
            if skill is None:
                skill = add_gather(index)
        spec.skills.append(skill)
        tier_of[skill.name] = skill.task.tier
        steps[skill.name] = _expanded_steps(spec, skill.program.splitlines(), steps)
    return spec


def resolver_for(spec: WorldSpec, known: List[str], banned: Tuple[str, ...] = ()):
    """Skill-name -> body for the skills an agent holds, minus any banned names."""
    bodies = {s.name: s.program for s in spec.skills if s.name in set(known)}

    def _resolve(name: str) -> str:
        if name in banned or name not in bodies:
            raise KeyError(name)
        return bodies[name]

    return _resolve


def run_program(spec: WorldSpec, program: str, known: List[str], banned: Tuple[str, ...] = ()) -> SpecWorld:
    world = SpecWorld(spec)
    world.run(program, resolver_for(spec, known, banned))
    return world


def verify_world(spec: WorldSpec) -> Dict[str, int]:
    """Run every reference program from a fresh world with only earlier skills callable."""
    out = {}
    for i, s in enumerate(spec.skills):
        known = [k.name for k in spec.skills[:i]]
        world = run_program(spec, s.program, known)
        if not s.task.check(world):
            raise AssertionError(f"reference for {s.name} misses its goal: {world.observe()}")
        out[s.name] = world.steps
    return out
