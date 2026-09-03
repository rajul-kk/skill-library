"""Resumable night-N checkpoints.

A lifelong run has to survive being chopped into Kaggle sessions, so after
every night we write a self-describing directory: the library JSON (cheap), the
adapter state, the consolidator state, and a manifest naming the run. Resuming
means reading the highest night directory and continuing -- no other state is
needed to reproduce the rest of the run.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from library.store import SkillLibrary

from .consolidate import Consolidator

MANIFEST = "manifest.json"
LIBRARY = "library.json"
ADAPTER = "adapter.json"
CONSOLIDATOR = "consolidator.json"
ADAPTER_DIR = "adapter"


def night_dir(root: str | Path, night: int) -> Path:
    return Path(root) / f"night_{night:03d}"


@dataclass
class Manifest:
    run_id: str
    method: str
    night: int
    episode: int
    library_size: int
    stored_params: int
    backend: str
    seed: int = 0
    created: float = 0.0
    extra: Dict[str, Any] = None

    def to_dict(self) -> dict:
        d = {
            "run_id": self.run_id,
            "method": self.method,
            "night": self.night,
            "episode": self.episode,
            "library_size": self.library_size,
            "stored_params": self.stored_params,
            "backend": self.backend,
            "seed": self.seed,
            "created": self.created or time.time(),
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        return cls(
            run_id=d["run_id"],
            method=d["method"],
            night=d["night"],
            episode=d["episode"],
            library_size=d["library_size"],
            stored_params=d["stored_params"],
            backend=d["backend"],
            seed=d.get("seed", 0),
            created=d.get("created", 0.0),
            extra=d.get("extra"),
        )


def save_night(
    root: str | Path,
    night: int,
    library: SkillLibrary,
    consolidator: Consolidator,
    manifest: Manifest,
) -> Path:
    """Write one night's checkpoint. Adapter backends choose their own format."""
    out = night_dir(root, night)
    out.mkdir(parents=True, exist_ok=True)
    library.save(out / LIBRARY)
    (out / CONSOLIDATOR).write_text(
        json.dumps(consolidator.state_dict(), indent=2), encoding="utf-8"
    )
    backend = consolidator.backend
    target = out / ADAPTER if _is_json_backend(backend) else out / ADAPTER_DIR
    backend.save(target)
    (out / MANIFEST).write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return out


def _is_json_backend(backend) -> bool:
    return hasattr(backend, "state_dict") and not hasattr(backend, "peft_model")


def list_nights(root: str | Path) -> List[int]:
    root = Path(root)
    if not root.exists():
        return []
    nights = []
    for p in root.glob("night_*"):
        if (p / MANIFEST).exists():
            nights.append(int(p.name.split("_")[1]))
    return sorted(nights)


def latest_night(root: str | Path) -> Optional[int]:
    nights = list_nights(root)
    return nights[-1] if nights else None


def load_night(
    root: str | Path, night: int, consolidator: Consolidator
) -> tuple[SkillLibrary, Manifest]:
    """Restore a run in place: library returned, consolidator/adapter mutated."""
    d = night_dir(root, night)
    if not (d / MANIFEST).exists():
        raise FileNotFoundError(f"no checkpoint at {d}")
    library = SkillLibrary.load(d / LIBRARY)
    consolidator.load_state_dict(json.loads((d / CONSOLIDATOR).read_text(encoding="utf-8")))
    backend = consolidator.backend
    target = d / ADAPTER if _is_json_backend(backend) else d / ADAPTER_DIR
    backend.load(target)
    manifest = Manifest.from_dict(json.loads((d / MANIFEST).read_text(encoding="utf-8")))
    return library, manifest


def resume_or_start(
    root: str | Path, consolidator: Consolidator
) -> tuple[Optional[SkillLibrary], Optional[Manifest]]:
    """Convenience for notebooks: pick up the newest night, or start fresh."""
    night = latest_night(root)
    if night is None:
        return None, None
    return load_night(root, night, consolidator)
