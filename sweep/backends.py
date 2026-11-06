"""Sweep backends: a CPU simulator for plumbing and a real HF model with LoRA for GPU runs."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

from env.procgen import GenSkill
from library.retrieval import heuristic_token_count
from sleep.backend import Example, InterferenceModel, SimulatedAdapter

from .config import SweepConfig, TrainConfig


@dataclass
class Query:
    """One evaluation attempt to decode."""

    prompt: str
    skill: GenSkill
    adapter: Optional[str]          # None = base weights, else adapter name
    context_full: Set[str]          # skills whose body is in the prompt


@dataclass
class Generation:
    text: str
    gen_tokens: int


@dataclass
class NightStats:
    train_tokens: int
    train_tokens_sq: int
    padded_tokens: int
    steps: int
    loss: float
    seconds: float
    nan: bool = False


def _degrade(program: str, rng: random.Random) -> str:
    """A plausible wrong program: drop one line."""
    lines = program.splitlines()
    if len(lines) > 1:
        lines.pop(rng.randrange(len(lines)))
    else:
        lines = ["goto(nowhere)"]
    return "\n".join(lines)


class SimulatedSweepBackend:
    """Pipeline-only simulator: retention from the interference model, never evidence about LoRA."""

    name = "simulated"

    def __init__(self, cfg: SweepConfig, seed: int = 0, interference: Optional[InterferenceModel] = None) -> None:
        self.cfg = cfg
        self.seed = seed
        self.interference = interference or InterferenceModel()
        self.adapters: Dict[str, SimulatedAdapter] = {}
        self.rng = random.Random(seed)
        self.lora_params = 1_000_000

    def count_tokens(self, text: str) -> int:
        return heuristic_token_count(text)

    def start_arm(self, arm) -> None:
        self.adapters = {}
        self.rng = random.Random(f"{self.seed}-{arm.key}")

    def end_arm(self) -> None:
        self.adapters = {}

    def train(self, examples: Sequence[Example], adapter: str, night: int) -> NightStats:
        ad = self.adapters.setdefault(adapter, SimulatedAdapter(interference=self.interference))
        ad.consolidate(examples, night)
        tokens = [self.count_tokens(e.prompt + e.target) for e in examples] * self.cfg.train.epochs
        return NightStats(sum(tokens), sum(t * t for t in tokens), sum(tokens), len(tokens), 0.0, 0.0)

    def generate(self, queries: Sequence[Query]) -> List[Generation]:
        out = []
        for q in queries:
            s = q.skill
            if s.name in q.context_full:
                length = self.count_tokens(q.prompt)
                p = 0.97 - 0.06 * math.log10(1 + length / 1000)  # mild long-context penalty, a stand-in
            elif q.adapter is not None:
                p = self.adapters[q.adapter].knows(s.name) if q.adapter in self.adapters else 0.0
            elif s.is_root:
                p = 0.8  # root tasks name resource and location
            else:
                p = 0.0
            text = s.program if self.rng.random() < p else _degrade(s.program, self.rng)
            out.append(Generation(text, self.count_tokens(text) + 1))
        return out


class HFSweepBackend:
    """Real model: batched greedy decoding, cumulative or per-block LoRA adapters, token-counted training."""

    name = "hf"

    def __init__(self, cfg: SweepConfig, model_name: str, seed: int = 0, model=None, tokenizer=None) -> None:
        import torch

        self.torch = torch
        self.cfg = cfg
        self.seed = seed
        self.model_name = model_name
        self.device = cfg.device
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dtype = getattr(torch, cfg.dtype)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
            model.to(self.device)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.base = model
        self.base.config.use_cache = True
        self.tokenizer = tokenizer
        self.model = model
        self.adapters: List[str] = []
        self.lora_params = 0

    # ---------------- accounting ---------------- #

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    # ---------------- adapters ---------------- #

    def _lora_config(self, tc: TrainConfig):
        from peft import LoraConfig, TaskType

        return LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=tc.rank, lora_alpha=tc.alpha, lora_dropout=tc.dropout,
            target_modules=list(tc.target_modules), bias="none",
        )

    def start_arm(self, arm) -> None:
        self.model = self.base
        self.adapters = []
        self.lora_params = 0

    def _ensure_adapter(self, name: str) -> None:
        from peft import get_peft_model

        torch = self.torch
        if not self.adapters:
            torch.manual_seed(self.seed)
            self.model = get_peft_model(self.base, self._lora_config(self.cfg.train), adapter_name=name)
        elif name not in self.adapters:
            torch.manual_seed(self.seed + len(self.adapters))
            self.model.add_adapter(name, self._lora_config(self.cfg.train))
        if name not in self.adapters:
            self.adapters.append(name)
            for n, p in self.model.named_parameters():
                if "lora_" in n and f".{name}." in n:
                    p.data = p.data.float()  # fp32 adapter weights over an fp16 base
            self.lora_params = sum(
                p.numel() for n, p in self.model.named_parameters() if "lora_" in n and f".{name}." in n
            )
        self.model.set_adapter(name)
        self.model.to(self.device)

    def end_arm(self) -> None:
        if self.adapters:
            self.base = self.model.unload()
        self.model = self.base
        self.adapters = []
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    # ---------------- training ---------------- #

    def train(self, examples: Sequence[Example], adapter: str, night: int) -> NightStats:
        from sleep.lora import collate, encode

        torch = self.torch
        tc = self.cfg.train
        self._ensure_adapter(adapter)
        trainable = []
        for n, p in self.model.named_parameters():
            p.requires_grad = "lora_" in n and f".{adapter}." in n
            if p.requires_grad:
                trainable.append(p)
        if not examples:
            return NightStats(0, 0, 0, 0, 0.0, 0.0)
        encoded = [encode(self.tokenizer, e, tc.max_length) for e in examples]
        order = list(range(len(encoded))) * tc.epochs
        optimizer = torch.optim.AdamW(trainable, lr=tc.learning_rate)
        use_scaler = self.device == "cuda" and self.cfg.dtype == "float16"
        scaler = torch.amp.GradScaler("cuda") if use_scaler else None
        if tc.grad_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        self.base.config.use_cache = False
        self.model.train()
        pad_id = self.tokenizer.pad_token_id
        started = time.time()
        tokens = tokens_sq = padded = steps = 0
        losses, nan = [], False
        torch.manual_seed(self.seed * 1000 + night)
        for i in range(0, len(order), tc.batch_size):
            items = [encoded[j] for j in order[i : i + tc.batch_size]]
            lengths = [len(it["input_ids"]) for it in items]
            tokens += sum(lengths)
            tokens_sq += sum(x * x for x in lengths)
            padded += max(lengths) * len(lengths)
            loss = accumulate_batch_loss(self.model, items, pad_id, tc.micro_batch_size, self.device, scaler)
            if loss is None:
                nan = True
                optimizer.zero_grad(set_to_none=True)
                continue
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, tc.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable, tc.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss)
            steps += 1
        if tc.grad_checkpointing:
            self.model.gradient_checkpointing_disable()
        self.model.eval()
        self.base.config.use_cache = True
        if self.device == "cuda":
            torch.cuda.synchronize()
        return NightStats(
            tokens, tokens_sq, padded, steps, sum(losses) / len(losses) if losses else float("nan"),
            time.time() - started, nan,
        )

    # ---------------- decoding ---------------- #

    def _batches(self, queries: Sequence[Query]) -> List[List[int]]:
        """Group query indices by adapter, then by length under the batch and token budgets."""
        dc = self.cfg.decode
        lengths = [self.count_tokens(q.prompt) for q in queries]
        groups: Dict[Optional[str], List[int]] = {}
        for i, q in enumerate(queries):
            groups.setdefault(q.adapter, []).append(i)
        batches = []
        for idxs in groups.values():
            idxs.sort(key=lambda i: lengths[i])
            cur: List[int] = []
            for i in idxs:
                width = max([lengths[j] for j in cur] + [lengths[i]]) + dc.max_new_tokens
                if cur and (len(cur) >= dc.batch_size or width * (len(cur) + 1) > dc.max_batch_tokens):
                    batches.append(cur)
                    cur = []
                cur.append(i)
            if cur:
                batches.append(cur)
        return batches

    def generate(self, queries: Sequence[Query]) -> List[Generation]:
        torch = self.torch
        out: List[Optional[Generation]] = [None] * len(queries)
        for idxs in self._batches(queries):
            adapter = queries[idxs[0]].adapter
            if adapter is not None:
                self._ensure_adapter(adapter)
            prompts = [queries[i].prompt for i in idxs]
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            stop = _BlankLineStop(self.tokenizer, enc["input_ids"].shape[1])
            with torch.no_grad():
                if adapter is None and self.adapters:
                    with self.model.disable_adapter():
                        gen = self.model.generate(**enc, **self._gen_kwargs(stop))
                else:
                    gen = self.model.generate(**enc, **self._gen_kwargs(stop))
            for row, i in enumerate(idxs):
                ids = gen[row, enc["input_ids"].shape[1] :].tolist()
                text, used = _cut(self.tokenizer, ids)
                out[i] = Generation(text, used)
        return out  # type: ignore[return-value]

    def _gen_kwargs(self, stop) -> dict:
        from transformers import StoppingCriteriaList

        return dict(
            max_new_tokens=self.cfg.decode.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([stop]),
        )


def _cut(tokenizer, ids: List[int]):
    """Text up to the first blank line or EOS, and how many generated tokens that took."""
    eos = tokenizer.eos_token_id
    text = ""
    for n in range(1, len(ids) + 1):
        if ids[n - 1] == eos:
            return text, n
        text = tokenizer.decode(ids[:n], skip_special_tokens=True)
        if "\n\n" in text.lstrip("\n"):
            return text.lstrip("\n").split("\n\n", 1)[0], n
    return text, len(ids)


class _BlankLineStop:
    """Marks each row finished once it has produced a blank line or EOS after its program."""

    def __init__(self, tokenizer, prompt_len: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        eos = self.tokenizer.eos_token_id
        done = []
        for row in input_ids:
            new = row[self.prompt_len :].tolist()
            text = self.tokenizer.decode(new, skip_special_tokens=True).lstrip("\n")
            done.append(eos in new or "\n\n" in text)
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


def accumulate_batch_loss(model, items, pad_id, micro: int, device, scaler=None):
    """Backward pass for one optimizer batch, optionally in micro-batches. Each micro-batch loss (a mean over its label
    tokens) is weighted by its share of the batch's label tokens, so the accumulated gradient equals the full-batch one.
    Returns the batch loss as a float, or None if any micro-batch loss is non-finite (gradients are then discarded by the
    caller)."""
    import torch

    from sleep.lora import collate

    chunks = [items] if not micro or micro >= len(items) else [items[i:i + micro] for i in range(0, len(items), micro)]
    batches = [collate(c, pad_id) for c in chunks]
    counts = [int((b["labels"][:, 1:] != -100).sum()) for b in batches]
    total = max(sum(counts), 1)
    out = 0.0
    for b, n in zip(batches, counts):
        if n == 0:
            continue
        loss = model(**{k: v.to(device) for k, v in b.items()}).loss
        if not torch.isfinite(loss):
            return None
        w = loss * (n / total)
        (scaler.scale(w) if scaler is not None else w).backward()
        out += float(w.detach())
    return out
