"""Crafting DSL: items, recipes, and the action grammar the agent writes programs in."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ------------------------------- World definition -------------------------- #

LOCATIONS: Tuple[str, ...] = ("forest", "plains", "cave", "mountain", "river")


@dataclass(frozen=True)
class Resource:
    """A gatherable resource: where it lives and what tool it needs."""

    name: str
    locations: Tuple[str, ...]
    requires_tool: Optional[str] = None
    yield_per_gather: int = 1


RESOURCES: Dict[str, Resource] = {
    r.name: r
    for r in [
        Resource("wood", ("forest",), None, 2),
        Resource("fiber", ("forest", "plains"), None, 2),
        Resource("sand", ("plains", "river"), None, 2),
        Resource("stone", ("cave", "mountain"), "wood_pickaxe", 2),
        Resource("coal", ("cave",), "wood_pickaxe", 1),
        Resource("iron_ore", ("cave", "mountain"), "stone_pickaxe", 1),
        Resource("gem_ore", ("mountain",), "iron_pickaxe", 1),
    ]
}


@dataclass(frozen=True)
class Recipe:
    """A crafting recipe. `station` is None, 'table' or 'furnace'."""

    name: str
    inputs: Tuple[Tuple[str, int], ...]
    output_count: int = 1
    station: Optional[str] = None


RECIPES: Dict[str, Recipe] = {
    r.name: r
    for r in [
        Recipe("plank", (("wood", 1),), 2),
        Recipe("stick", (("plank", 1),), 2),
        Recipe("rope", (("fiber", 2),), 1),
        Recipe("table", (("plank", 4),), 1),
        Recipe("wood_pickaxe", (("plank", 3), ("stick", 2)), 1, "table"),
        Recipe("stone_pickaxe", (("stone", 3), ("stick", 2)), 1, "table"),
        Recipe("furnace", (("stone", 8),), 1, "table"),
        Recipe("torch", (("coal", 1), ("stick", 1)), 4),
        Recipe("iron_ingot", (("iron_ore", 1), ("coal", 1)), 1, "furnace"),
        Recipe("glass", (("sand", 1), ("coal", 1)), 1, "furnace"),
        Recipe("iron_pickaxe", (("iron_ingot", 3), ("stick", 2)), 1, "table"),
        Recipe("gem", (("gem_ore", 1), ("coal", 2)), 1, "furnace"),
        Recipe("lantern", (("glass", 1), ("iron_ingot", 1), ("torch", 1)), 1, "table"),
    ]
}

TOOLS: Tuple[str, ...] = ("wood_pickaxe", "stone_pickaxe", "iron_pickaxe")
STATIONS: Tuple[str, ...] = ("table", "furnace")

ALL_ITEMS: Tuple[str, ...] = tuple(sorted(set(RESOURCES) | set(RECIPES)))


# ---------------------------- Program representation ----------------------- #


class DSLError(Exception):
    """Raised when a program cannot be parsed."""


@dataclass(frozen=True)
class Instruction:
    op: str
    args: Tuple[str, ...]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.op}({', '.join(self.args)})"


OPS: Dict[str, Tuple[int, int]] = {
    # op -> (min_args, max_args)
    "goto": (1, 1),
    "gather": (1, 2),
    "craft": (1, 2),
    "place": (1, 1),
    "call": (1, 1),
}

_LINE = re.compile(r"^\s*([a-z_]+)\s*\(([^)]*)\)\s*$")


def parse_program(text: str) -> List[Instruction]:
    """Parse a newline-separated DSL program."""
    instructions: List[Instruction] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _LINE.match(line)
        if not m:
            raise DSLError(f"line {lineno}: cannot parse {raw.strip()!r}")
        op, argstr = m.group(1), m.group(2).strip()
        args = tuple(a.strip() for a in argstr.split(",") if a.strip()) if argstr else ()
        if op not in OPS:
            raise DSLError(f"line {lineno}: unknown op {op!r} (known: {sorted(OPS)})")
        lo, hi = OPS[op]
        if not lo <= len(args) <= hi:
            raise DSLError(
                f"line {lineno}: {op} takes {lo}..{hi} args, got {len(args)}"
            )
        instructions.append(Instruction(op, args))
    if not instructions:
        raise DSLError("program is empty")
    return instructions


def format_program(instructions: List[Instruction]) -> str:
    return "\n".join(str(i) for i in instructions)


def dsl_reference() -> str:
    """The static grammar block shown to the agent in every prompt."""
    return (
        "DSL:\n"
        "  goto(<location>)          move to a location\n"
        "  gather(<resource>[, n])   gather n of a resource at the current location\n"
        "  craft(<item>[, n])        craft n of an item (needs its station placed here)\n"
        "  place(<station>)          place a crafting station at the current location\n"
        "  call(<skill_name>)        run a previously-learned skill from the library\n"
        f"Locations: {', '.join(LOCATIONS)}\n"
    )
