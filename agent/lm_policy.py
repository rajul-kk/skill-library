"""Policy backed by a real language model.

Same interface as the CPU simulator, so `VoyagerLoop`, the metrics and the
figures are all unchanged between a CPU pipeline run and a GPU one. The model
sees the retrieved library and writes a program; whether that program works is
still decided by executing it, not by anything the model says about itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set

from env.dsl import OPS
from env.tasks import Task

_CALL = re.compile(r"^\s*(" + "|".join(OPS) + r")\s*\([^)]*\)\s*$")
_FENCE = re.compile(r"```[a-zA-Z]*")


@dataclass
class DecodeConfig:
    """Greedy on the first attempt, then sample, so retries actually differ."""

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    retry_temperature_step: float = 0.15
    max_program_lines: int = 24


def extract_program(text: str, max_lines: int = 24) -> str:
    """Pull the DSL out of whatever the model wrote around it.

    Small instruct models pad answers with prose and code fences; keeping only
    lines that parse as DSL is more robust than demanding clean output, and any
    genuinely malformed program still fails in the environment.
    """
    lines: List[str] = []
    for raw in _FENCE.sub("", text).splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            if lines:
                break  # stop at the first blank line after the program starts
            continue
        if _CALL.match(line):
            lines.append(line)
        elif lines:
            break  # prose after the program means the program is over
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


class LMPolicy:
    """Generates DSL programs with a (LoRA-adapted) causal LM."""

    def __init__(self, backend, decode: Optional[DecodeConfig] = None, seed: int = 0):
        self.backend = backend
        self.decode = decode or DecodeConfig()
        self.seed = seed
        self.transcript: List[dict] = []

    def _generate(self, prompt: str, attempt: int) -> str:
        import torch

        model, tokenizer = self.backend.model, self.backend.tokenizer
        device = self.backend.device
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        ids = {k: v.to(device) for k, v in ids.items()}
        torch.manual_seed(self.seed + attempt)
        kwargs = dict(
            max_new_tokens=self.decode.max_new_tokens,
            pad_token_id=getattr(tokenizer, "eos_token_id", 0) or 0,
        )
        if attempt == 0:
            kwargs["do_sample"] = False
        else:
            kwargs.update(
                do_sample=True,
                temperature=self.decode.temperature
                + self.decode.retry_temperature_step * (attempt - 1),
                top_p=self.decode.top_p,
            )
        with torch.no_grad():
            out = model.generate(**ids, **kwargs)
        return tokenizer.decode(out[0][ids["input_ids"].shape[1] :])

    def propose(
        self, task: Task, prompt: str, context_skills: Set[str], attempt: int = 0
    ) -> str:
        raw = self._generate(prompt, attempt)
        program = extract_program(raw, self.decode.max_program_lines)
        self.transcript.append(
            {
                "task": task.name,
                "attempt": attempt,
                "in_context": sorted(context_skills),
                "raw": raw,
                "program": program,
            }
        )
        return program or "goto(forest)"
