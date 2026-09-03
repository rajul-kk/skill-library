"""Real LoRA backends, used by the GPU notebooks.

The cumulative backend keeps exactly one adapter for the whole run: every night
continues training the same LoRA parameters on new skills interleaved with
replay. Nothing is re-initialised and nothing is forked, so storage is flat in
the number of skills. The isolated backend is the PEAM-style contrast -- one
named adapter per skill, so nothing can interfere and storage grows linearly.

Model and tokenizer are injected rather than loaded here so the whole path can
be exercised on CPU with a tiny model in the tests.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .backend import Example, TrainStats

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass
class LoRAConfig:
    """Kept inside the constraints the project is built for: a single T4."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES
    learning_rate: float = 1e-4
    batch_size: int = 4
    steps_per_night: int = 300
    max_length: int = 512
    grad_clip: float = 1.0
    seed: int = 0


def build_peft_model(model, config: LoRAConfig, adapter_name: str = "default"):
    """Attach a fresh LoRA adapter to a base model."""
    from peft import LoraConfig, TaskType, get_peft_model

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias="none",
    )
    return get_peft_model(model, lora, adapter_name=adapter_name)


def encode(tokenizer, example: Example, max_length: int):
    """Tokenise prompt+target, masking the prompt out of the loss.

    Only the program is trained on: the prompt is the same boilerplate for every
    skill, and letting it into the loss would drown the signal we care about.
    """
    import torch

    prompt_ids = tokenizer(example.prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(example.target + "\n", add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + target_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate(batch, pad_id: int):
    import torch

    width = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        pad = width - len(b["input_ids"])
        input_ids.append(torch.cat([b["input_ids"], torch.full((pad,), pad_id)]))
        labels.append(torch.cat([b["labels"], torch.full((pad,), -100)]))
        mask.append(torch.cat([torch.ones(len(b["input_ids"])), torch.zeros(pad)]))
    return {
        "input_ids": torch.stack(input_ids).long(),
        "labels": torch.stack(labels).long(),
        "attention_mask": torch.stack(mask).long(),
    }


def program_agreement(produced: str, target: str) -> float:
    """Share of the target program's lines the model reproduced, in order."""

    def clean(text: str) -> List[str]:
        lines = []
        for raw in text.strip().splitlines():
            line = raw.split("#", 1)[0].strip().replace(" ", "")
            if line:
                lines.append(line)
        return lines

    want, got = clean(target), clean(produced)
    if not want:
        return 0.0
    matched, cursor = 0, 0
    for line in want:
        if line in got[cursor:]:
            cursor = got.index(line, cursor) + 1
            matched += 1
    return matched / len(want)


def train_lora(model, tokenizer, examples, config, device, trainable=None, night=0):
    """Run one short SFT pass over `examples` and report what it cost.

    Deliberately a few hundred steps at most: a night has to fit alongside
    rollout collection inside one Kaggle session, not take the whole session.
    """
    import torch

    if not examples:
        return {"steps": 0, "loss": 0.0, "seconds": 0.0}

    torch.manual_seed(config.seed + night)
    encoded = [encode(tokenizer, e, config.max_length) for e in examples]
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0) or 0

    if trainable is None:
        trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    model.train()

    started = time.time()
    losses = []
    epochs = math.ceil(len(encoded) / config.batch_size) * 8
    steps = min(config.steps_per_night, max(1, epochs))
    cursor = 0
    for _ in range(steps):
        items = []
        for _ in range(min(config.batch_size, len(encoded))):
            items.append(encoded[cursor % len(encoded)])
            cursor += 1
        batch = {k: v.to(device) for k, v in collate(items, pad_id).items()}
        out = model(**batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(out.loss.detach()))

    model.eval()
    return {
        "steps": steps,
        "loss": sum(losses) / len(losses),
        "seconds": time.time() - started,
    }


def generate_with(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 96) -> str:
    """Greedy decode, returning only the continuation."""
    import torch

    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = {k: v.to(device) for k, v in ids.items()}
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "eos_token_id", 0) or 0,
        )
    return tokenizer.decode(out[0][ids["input_ids"].shape[1] :])


class LoRAAdapterBackend:
    """One cumulative LoRA adapter, trained further every night."""

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[LoRAConfig] = None,
        device: str = "cpu",
        peft_model=None,
    ) -> None:
        self.config = config or LoRAConfig()
        self.tokenizer = tokenizer
        self.device = device
        self.model = (
            peft_model if peft_model is not None else build_peft_model(model, self.config)
        )
        self.model.to(device)
        self.history: List[TrainStats] = []
        self.retention_cache: Dict[str, float] = {}
        self.targets: Dict[str, str] = {}
        self.prompts: Dict[str, str] = {}

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def stored_parameters(self) -> int:
        """One shared adapter, so this does not move as the library grows."""
        return self.trainable_parameters()

    def consolidate(self, examples: Sequence[Example], night: int) -> TrainStats:
        if not examples:
            return TrainStats(night=night, n_new=0, n_replay=0, steps=0)

        for e in examples:
            self.targets[e.skill] = e.target
            self.prompts[e.skill] = e.prompt

        result = train_lora(
            self.model, self.tokenizer, examples, self.config, self.device, night=night
        )
        self.retention_cache.clear()
        stats = TrainStats(
            night=night,
            n_new=sum(1 for e in examples if not e.is_replay),
            n_replay=sum(1 for e in examples if e.is_replay),
            steps=result["steps"],
            loss=result["loss"],
            seconds=result["seconds"],
            trainable_params=self.trainable_parameters(),
            stored_params=self.stored_parameters(),
        )
        self.history.append(stats)
        return stats

    def generate(self, prompt: str, max_new_tokens: int = 96) -> str:
        return generate_with(
            self.model, self.tokenizer, prompt, self.device, max_new_tokens
        )

    def knows(self, skill: str) -> float:
        """Measured, not modelled: decode the skill and compare with what it was
        trained to produce.

        Line-level agreement rather than exact match, so a program that is one
        quantity off does not score the same as a program that is unrelated.
        """
        if skill in self.retention_cache:
            return self.retention_cache[skill]
        prompt, target = self.prompts.get(skill), self.targets.get(skill)
        if prompt is None or target is None:
            return 0.0
        score = program_agreement(self.generate(prompt), target)
        self.retention_cache[skill] = score
        return score

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        state = {
            "kind": "lora_cumulative",
            "config": {
                **self.config.__dict__,
                "target_modules": list(self.config.target_modules),
            },
            "targets": self.targets,
            "prompts": self.prompts,
            "history": [h.to_dict() for h in self.history],
        }
        (path / "backend_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        state = json.loads((path / "backend_state.json").read_text(encoding="utf-8"))
        self.targets = dict(state["targets"])
        self.prompts = dict(state["prompts"])
        self.history = [TrainStats(**h) for h in state["history"]]
        self.retention_cache.clear()
        self.model.load_adapter(str(path), adapter_name="default", is_trainable=True)


class IsolatedLoRABackend:
    """PEAM-style isolation over real adapters: one named adapter per category.

    All adapters live on a single base model and only the active one is applied,
    which is how peft keeps them genuinely separate -- wrapping the base model
    again per skill would nest adapters instead of isolating them. Retention
    cannot degrade because no two skills share parameters, and the cost of that
    is a linear pile of adapters: the tradeoff this project reports rather than
    argues away.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[LoRAConfig] = None,
        device: str = "cpu",
        category_of=None,
    ) -> None:
        self.config = config or LoRAConfig()
        self.tokenizer = tokenizer
        self.device = device
        self.base_model = model
        self.category_of = category_of or (lambda name: name)
        self.model = None
        self.categories: List[str] = []
        self.history: List[TrainStats] = []
        self.targets: Dict[str, str] = {}
        self.prompts: Dict[str, str] = {}
        self.retention_cache: Dict[str, float] = {}

    def _lora_config(self):
        from peft import LoraConfig, TaskType

        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.rank,
            lora_alpha=self.config.alpha,
            lora_dropout=self.config.dropout,
            target_modules=list(self.config.target_modules),
            bias="none",
        )

    def _ensure_adapter(self, category: str) -> None:
        if self.model is None:
            self.model = build_peft_model(self.base_model, self.config, category)
            self.model.to(self.device)
        elif category not in self.categories:
            self.model.add_adapter(category, self._lora_config())
            self.model.to(self.device)
        if category not in self.categories:
            self.categories.append(category)
        self.model.set_adapter(category)

    def _activate(self, category: str):
        """Select one adapter and make only its parameters trainable."""
        self._ensure_adapter(category)
        trainable = []
        for name, param in self.model.named_parameters():
            param.requires_grad = "lora_" in name and f".{category}." in name
            if param.requires_grad:
                trainable.append(param)
        return trainable

    def adapter_size(self) -> int:
        """Parameters in a single adapter."""
        if self.model is None or not self.categories:
            return 0
        first = self.categories[0]
        return sum(
            p.numel()
            for n, p in self.model.named_parameters()
            if "lora_" in n and f".{first}." in n
        )

    def stored_parameters(self) -> int:
        return self.adapter_size() * len(self.categories)

    def consolidate(self, examples: Sequence[Example], night: int) -> TrainStats:
        fresh = [e for e in examples if not e.is_replay]  # isolation needs no replay
        by_category: Dict[str, List[Example]] = {}
        for e in fresh:
            self.targets[e.skill] = e.target
            self.prompts[e.skill] = e.prompt
            by_category.setdefault(self.category_of(e.skill), []).append(e)

        seconds, losses, steps = 0.0, [], 0
        for category, batch in by_category.items():
            trainable = self._activate(category)
            result = train_lora(
                self.model, self.tokenizer, batch, self.config, self.device,
                trainable=trainable, night=night,
            )
            steps += result["steps"]
            seconds += result["seconds"]
            losses.append(result["loss"])

        self.retention_cache.clear()
        stats = TrainStats(
            night=night,
            n_new=len(fresh),
            n_replay=0,
            steps=steps,
            loss=sum(losses) / len(losses) if losses else 0.0,
            seconds=seconds,
            trainable_params=self.adapter_size(),
            stored_params=self.stored_parameters(),
        )
        self.history.append(stats)
        return stats

    def knows(self, skill: str) -> float:
        category = self.category_of(skill)
        if category not in self.categories or skill not in self.prompts:
            return 0.0
        if skill in self.retention_cache:
            return self.retention_cache[skill]
        self.model.set_adapter(category)
        produced = generate_with(self.model, self.tokenizer, self.prompts[skill], self.device)
        score = program_agreement(produced, self.targets[skill])
        self.retention_cache[skill] = score
        return score

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save_pretrained(str(path))
        index = {
            "kind": "lora_isolated",
            "categories": self.categories,
            "targets": self.targets,
            "prompts": self.prompts,
            "history": [h.to_dict() for h in self.history],
        }
        (path / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        index = json.loads((path / "index.json").read_text(encoding="utf-8"))
        self.history = [TrainStats(**h) for h in index["history"]]
        self.targets = dict(index["targets"])
        self.prompts = dict(index["prompts"])
        self.retention_cache.clear()
        for category in index["categories"]:
            self._ensure_adapter(category)
            self.model.load_adapter(str(path / category), adapter_name=category)


def load_base_model(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda",
    dtype: str = "bfloat16",
):
    """Load the base model and tokenizer for a GPU run.

    Falls back to float16 on cards without bfloat16 support, which is what a T4
    needs; a P100 takes float16 as well.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda" and not torch.cuda.is_bf16_supported():
        dtype = "float16"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, dtype)
    )
    model.config.use_cache = False
    return model.to(device), tokenizer
