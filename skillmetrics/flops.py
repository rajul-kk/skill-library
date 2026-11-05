"""Analytic FLOP accounting for decoder-only transformers, inference and LoRA training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence


@dataclass(frozen=True)
class ModelShape:
    """The shape parameters FLOP counts depend on."""

    name: str
    n_layer: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab: int
    gated_mlp: bool = True
    qkv_bias: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_kv(self) -> int:
        return self.n_kv_heads * self.head_dim

    def attention_params(self) -> int:
        d, kv = self.d_model, self.d_kv
        bias = (d + 2 * kv) if self.qkv_bias else 0
        return d * d + 2 * d * kv + d * d + bias

    def mlp_params(self) -> int:
        return (3 if self.gated_mlp else 2) * self.d_model * self.d_ff

    def non_embedding_params(self) -> int:
        """Matmul parameters inside the blocks; norms are ignored as negligible."""
        return self.n_layer * (self.attention_params() + self.mlp_params())

    def head_params(self) -> int:
        """The LM head matmul, counted whether or not it is tied to the embedding."""
        return self.vocab * self.d_model

    def lora_params(self, rank: int, targets: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj")) -> int:
        """Trainable parameters of one LoRA adapter (A: r x d_in, B: d_out x r) on the named projections."""
        d, kv, ff = self.d_model, self.d_kv, self.d_ff
        dims = {
            "q_proj": (d, d), "k_proj": (d, kv), "v_proj": (d, kv), "o_proj": (d, d),
            "gate_proj": (d, ff), "up_proj": (d, ff), "down_proj": (ff, d),
        }
        return self.n_layer * sum(rank * (dims[t][0] + dims[t][1]) for t in targets)

    def to_dict(self) -> Dict:
        return asdict(self)


SHAPES: Dict[str, ModelShape] = {
    "Qwen/Qwen2.5-0.5B-Instruct": ModelShape("Qwen/Qwen2.5-0.5B-Instruct", 24, 896, 14, 2, 4864, 151936),
    "Qwen/Qwen2.5-1.5B-Instruct": ModelShape("Qwen/Qwen2.5-1.5B-Instruct", 28, 1536, 12, 2, 8960, 151936),
    "Qwen/Qwen2.5-3B-Instruct": ModelShape("Qwen/Qwen2.5-3B-Instruct", 36, 2048, 16, 2, 11008, 151936),
    "HuggingFaceTB/SmolLM2-135M-Instruct": ModelShape("HuggingFaceTB/SmolLM2-135M-Instruct", 30, 576, 9, 3, 1536, 49152, qkv_bias=False),
}


def shape_from_hf_config(name: str, cfg: Dict) -> ModelShape:
    """Build a shape from a HF config dict (Llama/Qwen-style keys)."""
    return ModelShape(
        name=name,
        n_layer=cfg["num_hidden_layers"],
        d_model=cfg["hidden_size"],
        n_heads=cfg["num_attention_heads"],
        n_kv_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        d_ff=cfg["intermediate_size"],
        vocab=cfg["vocab_size"],
        qkv_bias=bool(cfg.get("attention_bias", cfg.get("model_type") == "qwen2")),
    )


def _attn(shape: ModelShape, ctx_sum: float) -> float:
    """QK^T and AV cost: 2 matmuls x d_model x context, 2 FLOPs per MAC, per layer."""
    return 4.0 * shape.n_layer * shape.d_model * ctx_sum


def prefill_flops(shape: ModelShape, prompt_tokens: int) -> float:
    """Forward over the prompt; the LM head is charged once (for the first generated token)."""
    L = prompt_tokens
    return 2.0 * shape.non_embedding_params() * L + _attn(shape, L * (L + 1) / 2.0) + 2.0 * shape.head_params()


def decode_flops(shape: ModelShape, prompt_tokens: int, gen_tokens: int) -> float:
    """Incremental decoding of gen_tokens after the prompt, KV cache assumed."""
    L, G = prompt_tokens, gen_tokens
    per_token = 2.0 * (shape.non_embedding_params() + shape.head_params())
    ctx_sum = L * G + G * (G + 1) / 2.0
    return per_token * G + _attn(shape, ctx_sum)


def lora_inference_flops(lora_params: int, tokens: int) -> float:
    """Extra cost of an unmerged adapter: one MAC per LoRA weight per token."""
    return 2.0 * lora_params * tokens


def attempt_flops(shape: ModelShape, prompt_tokens: int, gen_tokens: int, lora_params: int = 0) -> Dict[str, float]:
    """All inference FLOPs of one attempt, split by phase."""
    out = {
        "prefill": prefill_flops(shape, prompt_tokens),
        "decode": decode_flops(shape, prompt_tokens, gen_tokens),
        "lora": lora_inference_flops(lora_params, prompt_tokens + gen_tokens),
    }
    out["total"] = out["prefill"] + out["decode"] + out["lora"]
    return out


def lora_train_flops(
    shape: ModelShape,
    tokens: float,
    tokens_sq: float,
    lora_params: int,
    grad_checkpointing: bool = False,
) -> float:
    """LoRA SFT cost from sum(T) and sum(T^2) over real (unpadded) training sequences."""
    # frozen base: forward + activation backward (2x), +1 forward if checkpointing; head charged per token
    forward = 2.0 * (shape.non_embedding_params() + shape.head_params()) * tokens
    forward += _attn(shape, (tokens_sq + tokens) / 2.0)
    passes = 3.0 if grad_checkpointing else 2.0
    return passes * forward + 6.0 * lora_params * tokens
