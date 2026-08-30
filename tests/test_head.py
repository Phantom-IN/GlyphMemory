"""Character head tests.

The load-bearing one is :meth:`TestNoSoftmax.test_logits_are_not_a_distribution`. A softmax applied
inside the module would be applied again by the loss path, flattening the distribution so the loss
stops discriminating — and nothing would raise.
"""

from __future__ import annotations

import pytest
import torch

from glyphmemory.config.schema import ModelConfig
from glyphmemory.model.head import CharacterHead
from glyphmemory.model.model_info import parameter_count

VOCAB = 80


@pytest.fixture
def head() -> CharacterHead:
    torch.manual_seed(1337)
    return CharacterHead(vocab_size=VOCAB).eval()


class TestShapes:
    @pytest.mark.parametrize("time_steps", [1, 16, 128, 400])
    def test_output_shape(self, head: CharacterHead, time_steps: int) -> None:
        assert head(torch.randn(2, time_steps, 384)).shape == (2, time_steps, VOCAB)

    def test_vocab_dimension_is_last(self, head: CharacterHead) -> None:
        assert head(torch.randn(1, 8, 384)).shape[-1] == head.vocab_size

    def test_rejects_wrong_feature_width(self, head: CharacterHead) -> None:
        with pytest.raises(ValueError, match="Expected 384 features"):
            head(torch.randn(1, 8, 256))

    def test_rejects_wrong_rank(self, head: CharacterHead) -> None:
        with pytest.raises(ValueError, match=r"\[B, T, C\]"):
            head(torch.randn(8, 384))


class TestNoSoftmax:
    def test_logits_are_not_a_distribution(self, head: CharacterHead) -> None:
        """Raw logits: they must not sum to one, and must include negatives."""
        logits = head(torch.randn(4, 16, 384))
        sums = logits.exp().sum(dim=-1)
        assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-2)
        assert (logits < 0).any()

    def test_describe_states_it(self, head: CharacterHead) -> None:
        assert head.describe()["applies_softmax"] is False

    def test_module_contains_no_softmax_layer(self, head: CharacterHead) -> None:
        forbidden = (torch.nn.Softmax, torch.nn.LogSoftmax)
        assert not any(isinstance(m, forbidden) for m in head.modules())


class TestParameters:
    def test_measured_count_matches_hand_calculation(self) -> None:
        """LayerNorm(384) = 768; Linear(384, 80) = 384*80 + 80 = 30,800. Total 31,568."""
        assert parameter_count(CharacterHead(vocab_size=VOCAB)) == 31_568

    def test_from_config_derives_input_from_gru_hidden(self) -> None:
        head = CharacterHead.from_config(ModelConfig(gru_hidden=128), vocab_size=40)
        assert head.input_size == 256
        assert head.vocab_size == 40

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"input_size": 0, "vocab_size": 10}, "input_size"),
            ({"vocab_size": 1}, "at least 2"),
            ({"vocab_size": 10, "dropout": 1.0}, r"\[0, 1\)"),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            CharacterHead(**kwargs)


class TestTrainingBehaviour:
    def test_dropout_active_in_train_mode(self) -> None:
        torch.manual_seed(0)
        head = CharacterHead(vocab_size=VOCAB).train()
        features = torch.randn(2, 32, 384)
        assert not torch.allclose(head(features), head(features))

    def test_eval_is_deterministic(self, head: CharacterHead) -> None:
        features = torch.randn(2, 32, 384)
        with torch.no_grad():
            assert torch.equal(head(features), head(features))

    def test_gradients_flow(self) -> None:
        head = CharacterHead(vocab_size=VOCAB)
        features = torch.randn(2, 16, 384, requires_grad=True)
        head(features).sum().backward()
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()
