"""Exercises the real LoRA training path on CPU with a tiny model."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from sleep.backend import Example  # noqa: E402
from sleep.lora import (  # noqa: E402
    IsolatedLoRABackend,
    LoRAAdapterBackend,
    LoRAConfig,
    encode,
    program_agreement,
)


class CharTokenizer:
    """Byte-level tokenizer: enough for the backend, no download required."""

    pad_token_id = 0
    eos_token_id = 1
    offset = 2

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [b + self.offset for b in text.encode("utf-8")]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        raw = [int(i) - self.offset for i in ids if int(i) >= self.offset]
        return bytes(b for b in raw if 0 <= b < 256).decode("utf-8", errors="ignore")


@pytest.fixture
def tiny():
    config = AutoConfig.for_model(
        "qwen2",
        vocab_size=260,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.config.use_cache = False
    return model, CharTokenizer()


def examples(n=3, night=1):
    return [
        Example(f"skill_{i}", f"Task: skill {i}\nWrite a DSL program:\n", f"goto(forest)\ngather(wood, {i})", night)
        for i in range(n)
    ]


def test_prompt_tokens_are_masked_out_of_the_loss(tiny):
    _, tokenizer = tiny
    encoded = encode(tokenizer, examples(1)[0], max_length=512)
    labels = encoded["labels"].tolist()
    assert labels.count(-100) == len(tokenizer("Task: skill 0\nWrite a DSL program:\n")["input_ids"])
    assert labels[-1] != -100


def test_a_night_trains_only_the_adapter(tiny):
    model, tokenizer = tiny
    backend = LoRAAdapterBackend(model, tokenizer, LoRAConfig(rank=4, batch_size=2))
    trainable = backend.trainable_parameters()
    total = sum(p.numel() for p in backend.model.parameters())
    assert 0 < trainable < total / 10


def test_consolidation_reduces_loss_on_what_it_trained_on(tiny):
    model, tokenizer = tiny
    backend = LoRAAdapterBackend(
        model, tokenizer, LoRAConfig(rank=4, batch_size=2, learning_rate=5e-3)
    )
    first = backend.consolidate(examples(2), night=1)
    second = backend.consolidate(examples(2), night=2)
    assert second.loss < first.loss
    assert first.steps > 0


def test_storage_is_flat_across_nights(tiny):
    model, tokenizer = tiny
    backend = LoRAAdapterBackend(model, tokenizer, LoRAConfig(rank=4, batch_size=2))
    backend.consolidate(examples(2, night=1), night=1)
    before = backend.stored_parameters()
    backend.consolidate(examples(4, night=2), night=2)
    assert backend.stored_parameters() == before


def test_replay_examples_are_counted_separately(tiny):
    model, tokenizer = tiny
    backend = LoRAAdapterBackend(model, tokenizer, LoRAConfig(rank=4, batch_size=2))
    batch = examples(2) + [Example("old", "p\n", "goto(cave)", 0, is_replay=True)]
    stats = backend.consolidate(batch, night=1)
    assert (stats.n_new, stats.n_replay) == (2, 1)


def test_isolated_storage_grows_with_each_new_skill(tiny):
    model, tokenizer = tiny
    backend = IsolatedLoRABackend(model, tokenizer, LoRAConfig(rank=4, batch_size=2))
    backend.consolidate(examples(1, night=1), night=1)
    one = backend.stored_parameters()
    backend.consolidate([examples(3)[2]], night=2)
    assert backend.stored_parameters() == pytest.approx(2 * one)


def test_training_one_isolated_adapter_leaves_the_others_alone(tiny):
    """The property isolation is bought with: no shared parameters, no interference."""
    model, tokenizer = tiny
    backend = IsolatedLoRABackend(model, tokenizer, LoRAConfig(rank=4, batch_size=2))
    backend.consolidate([Example("a", "p\n", "goto(cave)", 1)], night=1)

    before = {
        name: param.detach().clone()
        for name, param in backend.model.named_parameters()
        if "lora_" in name and ".a." in name
    }
    assert before, "expected named lora parameters for the first adapter"

    backend.consolidate([Example("b", "p\n", "goto(river)", 2)], night=2)
    for name, param in backend.model.named_parameters():
        if name in before:
            assert torch.equal(param, before[name]), f"{name} moved while training b"


def test_isolated_adapters_are_registered_once_each(tiny):
    model, tokenizer = tiny
    backend = IsolatedLoRABackend(model, tokenizer, LoRAConfig(rank=4, batch_size=2))
    backend.consolidate([Example("a", "p\n", "goto(cave)", 1)], night=1)
    backend.consolidate([Example("a", "p\n", "goto(cave)", 2)], night=2)
    assert backend.categories == ["a"]
    assert backend.stored_parameters() == backend.adapter_size()


def test_isolated_backend_ignores_replay(tiny):
    model, tokenizer = tiny
    backend = IsolatedLoRABackend(model, tokenizer, LoRAConfig(rank=4, batch_size=2))
    stats = backend.consolidate(
        [Example("a", "p\n", "goto(cave)", 1), Example("b", "p\n", "goto(river)", 1, True)],
        night=1,
    )
    assert stats.n_replay == 0
    assert backend.knows("b") == 0.0


def test_an_untrained_skill_is_not_known(tiny):
    model, tokenizer = tiny
    backend = LoRAAdapterBackend(model, tokenizer, LoRAConfig(rank=4))
    assert backend.knows("never_seen") == 0.0


def test_adapter_survives_a_save_load_round_trip(tiny, tmp_path):
    model, tokenizer = tiny
    backend = LoRAAdapterBackend(model, tokenizer, LoRAConfig(rank=4, batch_size=2))
    backend.consolidate(examples(2), night=1)
    backend.save(tmp_path / "adapter")

    restored = LoRAAdapterBackend(model, tokenizer, LoRAConfig(rank=4, batch_size=2))
    restored.load(tmp_path / "adapter")
    assert restored.targets == backend.targets
    assert len(restored.history) == 1


@pytest.mark.parametrize(
    "produced,expected",
    [
        ("goto(forest)\ngather(wood, 2)", 1.0),
        ("goto(forest)", 0.5),
        ("craft(table)", 0.0),
        ("goto(forest)\n# note\ngather(wood,2)", 1.0),
    ],
)
def test_program_agreement_scores_partial_recall(produced, expected):
    assert program_agreement(produced, "goto(forest)\ngather(wood, 2)") == expected
