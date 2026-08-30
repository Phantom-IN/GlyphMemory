"""T0 — strict freezing, exact parameter count, and no leakage between writers.

Three properties carry the experiment and each is asserted here rather than trusted:

- **exactly 80 trainable parameters** — a rung that quietly trains more is a different experiment
- **everything else frozen**, including `head.projection.weight`, the head LayerNorm, the BiGRU, the
  encoder, and every BatchNorm weight *and running statistic*
- **no parameter survives one writer into the next** — a leak would manufacture a gain that looks
  exactly like a result

The leakage test is the one that matters most. `fine_tuned_model` restores state in a `finally`, so
the guarantee holds even when the caller's block raises, and that path is exercised too.
"""

from __future__ import annotations

import copy

import pytest
import torch

from glyphmemory.config.schema import ModelConfig
from glyphmemory.ctc.tokenizer import Charset, Tokenizer
from glyphmemory.evaluation.rival_baselines import (
    DEFAULT_FT_WEIGHT_DECAY,
    PARAMETER_GROUPS,
    _select_parameters,
    fine_tuned_model,
    parameter_storage_bytes,
    trainable_parameter_count,
)
from glyphmemory.model.htr import GMBase

VOCAB_CHARS = "abcdefghijklmnopqrstuvwxyz "


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer(Charset.from_texts([VOCAB_CHARS], name="m11_t0_test"))


@pytest.fixture
def model(tokenizer: Tokenizer) -> GMBase:
    torch.manual_seed(0)
    config = ModelConfig(input_height=64, visual_dim=32, gru_hidden=32, gru_layers=1)
    return GMBase.from_config(config, tokenizer.vocab_size).eval()


@pytest.fixture
def lines() -> list[tuple[torch.Tensor, str]]:
    torch.manual_seed(1)
    return [(torch.rand(1, 64, 128), "abc"), (torch.rand(1, 64, 128), "cab")]


class TestGroupRegistration:
    def test_class_bias_is_registered(self) -> None:
        assert "class_bias" in PARAMETER_GROUPS

    def test_the_three_m8_groups_keep_their_identity(self) -> None:
        """Internal helper."""
        assert PARAMETER_GROUPS[:3] == ("head", "batchnorm", "full")


class TestParameterSelection:
    def test_selects_the_output_bias_and_nothing_else(self, model: GMBase) -> None:
        selected = _select_parameters(model, "class_bias")
        assert len(selected) == 1
        assert selected[0] is model.head.projection.bias

    def test_count_is_exactly_one_scalar_per_output_class(
        self, model: GMBase, tokenizer: Tokenizer
    ) -> None:
        assert trainable_parameter_count(model, "class_bias") == tokenizer.vocab_size

    def test_at_the_frozen_charset_this_is_80(self) -> None:
        """The pre-registered figure. A different vocabulary is a different experiment."""
        charset = Charset.english_v1()
        assert charset.size == 80
        config = ModelConfig(input_height=64, visual_dim=32, gru_hidden=32, gru_layers=1)
        assert trainable_parameter_count(GMBase.from_config(config, 80), "class_bias") == 80

    def test_storage_is_four_bytes_per_class(self, model: GMBase, tokenizer: Tokenizer) -> None:
        assert parameter_storage_bytes(model, "class_bias") == tokenizer.vocab_size * 4

    def test_excludes_the_projection_weight(self, model: GMBase) -> None:
        """`head.projection.weight` is 30,720 params at the real vocab — 384x the rung."""
        selected = {id(p) for p in _select_parameters(model, "class_bias")}
        assert id(model.head.projection.weight) not in selected

    def test_excludes_the_head_layernorm(self, model: GMBase) -> None:
        selected = {id(p) for p in _select_parameters(model, "class_bias")}
        assert id(model.head.norm.weight) not in selected
        assert id(model.head.norm.bias) not in selected

    def test_is_a_strict_subset_of_the_head_group(self, model: GMBase) -> None:
        head = {id(p) for p in _select_parameters(model, "head")}
        bias = {id(p) for p in _select_parameters(model, "class_bias")}
        assert bias < head


class TestStrictFreezing:
    def test_only_the_output_bias_receives_a_gradient(
        self, model: GMBase, tokenizer: Tokenizer, lines: list[tuple[torch.Tensor, str]]
    ) -> None:
        state = copy.deepcopy(model.state_dict())
        with fine_tuned_model(
            model, state, "class_bias", lines, tokenizer, steps=1, lr=1e-2, weight_decay=0.0
        ) as adapted:
            grads = {n: p.grad for n, p in adapted.named_parameters() if p.grad is not None}
        assert set(grads) == {"head.projection.bias"}

    def test_every_other_parameter_is_requires_grad_false_during_adaptation(
        self, model: GMBase, tokenizer: Tokenizer, lines: list[tuple[torch.Tensor, str]]
    ) -> None:
        state = copy.deepcopy(model.state_dict())
        with fine_tuned_model(
            model, state, "class_bias", lines, tokenizer, steps=1, lr=1e-2, weight_decay=0.0
        ) as adapted:
            trainable = [n for n, p in adapted.named_parameters() if p.requires_grad]
        assert trainable == ["head.projection.bias"]

    def test_batchnorm_running_statistics_do_not_move(
        self, model: GMBase, tokenizer: Tokenizer, lines: list[tuple[torch.Tensor, str]]
    ) -> None:
        """`.eval()` throughout, so BatchNorm keeps frozen statistics.

        If T0 silently re-estimated them, it would be carrying a mechanism this project already
        measured as harmful, and the result would not be about 80 biases at all.
        """
        state = copy.deepcopy(model.state_dict())
        before = {k: v.clone() for k, v in model.state_dict().items() if "running_" in k}
        assert before, "the model should contain BatchNorm running statistics"
        with fine_tuned_model(
            model, state, "class_bias", lines, tokenizer, steps=3, lr=1e-2, weight_decay=0.0
        ) as adapted:
            during = adapted.state_dict()
            for key, value in before.items():
                assert torch.equal(during[key], value), key

    def test_adaptation_actually_changes_the_bias(
        self, model: GMBase, tokenizer: Tokenizer, lines: list[tuple[torch.Tensor, str]]
    ) -> None:
        """A frozen-everything test suite would also pass if nothing trained at all."""
        state = copy.deepcopy(model.state_dict())
        before = model.head.projection.bias.detach().clone()
        with fine_tuned_model(
            model, state, "class_bias", lines, tokenizer, steps=5, lr=1e-2, weight_decay=0.0
        ) as adapted:
            assert not torch.allclose(adapted.head.projection.bias, before)

    def test_no_other_tensor_in_the_state_dict_changes(
        self, model: GMBase, tokenizer: Tokenizer, lines: list[tuple[torch.Tensor, str]]
    ) -> None:
        state = copy.deepcopy(model.state_dict())
        with fine_tuned_model(
            model, state, "class_bias", lines, tokenizer, steps=5, lr=1e-2, weight_decay=0.0
        ) as adapted:
            during = {k: v.clone() for k, v in adapted.state_dict().items()}
        changed = [k for k, v in during.items() if not torch.equal(v, state[k])]
        assert changed == ["head.projection.bias"]


class TestWeightDecay:
    def test_default_preserves_the_m8_behaviour(self) -> None:
        """Internal helper."""
        assert DEFAULT_FT_WEIGHT_DECAY == 0.01

    def test_zero_decay_leaves_an_unpushed_bias_alone(
        self, model: GMBase, tokenizer: Tokenizer
    ) -> None:
        """With decay, AdamW pulls the bias toward ZERO — away from gm-base-v0's trained value.

        Run with zero gradient signal (steps=0 is not enough to show it, so compare the two decay
        settings after real steps on the same data) and assert the decayed run moves the bias
        strictly closer to the origin than the undecayed one.
        """
        torch.manual_seed(2)
        lines = [(torch.rand(1, 64, 128), "abc")]
        state = copy.deepcopy(model.state_dict())
        # Bias starts large, so decay-toward-zero is visible against the gradient step.
        state["head.projection.bias"] = torch.full_like(state["head.projection.bias"], 5.0)

        norms = {}
        for decay in (0.0, 0.5):
            with fine_tuned_model(
                model, state, "class_bias", lines, tokenizer,
                steps=5, lr=1e-2, weight_decay=decay,
            ) as adapted:
                norms[decay] = float(adapted.head.projection.bias.detach().norm())
        assert norms[0.5] < norms[0.0]


class TestNoLeakageBetweenWriters:
    def test_state_is_restored_exactly_after_the_context_exits(
        self, model: GMBase, tokenizer: Tokenizer, lines: list[tuple[torch.Tensor, str]]
    ) -> None:
        """The guarantee the whole experiment rests on."""
        state = copy.deepcopy(model.state_dict())
        with fine_tuned_model(
            model, state, "class_bias", lines, tokenizer, steps=5, lr=1e-1, weight_decay=0.0
        ):
            pass
        for key, value in model.state_dict().items():
            assert torch.equal(value, state[key]), key

    def test_state_is_restored_even_when_the_caller_raises(
        self, model: GMBase, tokenizer: Tokenizer, lines: list[tuple[torch.Tensor, str]]
    ) -> None:
        state = copy.deepcopy(model.state_dict())
        with pytest.raises(RuntimeError, match="scoring blew up"), fine_tuned_model(
            model, state, "class_bias", lines, tokenizer, steps=3, lr=1e-1, weight_decay=0.0
        ):
            raise RuntimeError("scoring blew up")
        for key, value in model.state_dict().items():
            assert torch.equal(value, state[key]), key

    def test_two_writers_in_sequence_do_not_contaminate_each_other(
        self, model: GMBase, tokenizer: Tokenizer
    ) -> None:
        """The real failure mode: writer B's adaptation starting from writer A's biases.

        Writer B is adapted twice — once after A, once in isolation — and the two must agree
        exactly. If any state leaked, they would not.
        """
        torch.manual_seed(3)
        writer_a = [(torch.rand(1, 64, 128), "abc")]
        writer_b = [(torch.rand(1, 64, 128), "cab")]
        state = copy.deepcopy(model.state_dict())
        kwargs = dict(steps=5, lr=1e-1, weight_decay=0.0)

        with fine_tuned_model(model, state, "class_bias", writer_a, tokenizer, **kwargs):
            pass
        with fine_tuned_model(model, state, "class_bias", writer_b, tokenizer, **kwargs) as m:
            after_a_then_b = m.head.projection.bias.detach().clone()

        model.load_state_dict(state)
        with fine_tuned_model(model, state, "class_bias", writer_b, tokenizer, **kwargs) as m:
            isolated_b = m.head.projection.bias.detach().clone()

        assert torch.equal(after_a_then_b, isolated_b)

    def test_requires_grad_flags_are_restored(
        self, model: GMBase, tokenizer: Tokenizer, lines: list[tuple[torch.Tensor, str]]
    ) -> None:
        state = copy.deepcopy(model.state_dict())
        before = {n: p.requires_grad for n, p in model.named_parameters()}
        with fine_tuned_model(
            model, state, "class_bias", lines, tokenizer, steps=1, lr=1e-2, weight_decay=0.0
        ):
            pass
        assert {n: p.requires_grad for n, p in model.named_parameters()} == before


class TestDeterminism:
    def test_the_same_support_gives_bit_identical_adaptation(
        self, model: GMBase, tokenizer: Tokenizer, lines: list[tuple[torch.Tensor, str]]
    ) -> None:
        """Consequence worth asserting: in `.eval()` mode with fixed data, adaptation has no
        stochasticity at all. Seeds therefore vary only the SUPPORT DRAW, never the fit — which is
        why a writer whose support pool equals `n` shows exactly zero between-seed variance.
        """
        state = copy.deepcopy(model.state_dict())
        runs = []
        for _ in range(2):
            with fine_tuned_model(
                model, state, "class_bias", lines, tokenizer, steps=5, lr=1e-2, weight_decay=0.0
            ) as adapted:
                runs.append(adapted.head.projection.bias.detach().clone())
        assert torch.equal(runs[0], runs[1])
