"""Sequence encoder tests.

:meth:`TestPacking.test_padding_does_not_change_a_samples_own_frames` is the reason this file
exists. Everything else is shape bookkeeping; that one proves the bidirectional GRU is not reading
its neighbours' padding — a defect that never raises, that width bucketing partially masks, and that
is invisible at batch size 1.
"""

from __future__ import annotations

import pytest
import torch

from glyphmemory.config.schema import ModelConfig
from glyphmemory.model.model_info import parameter_count
from glyphmemory.model.sequence import SequenceEncoder


@pytest.fixture
def encoder() -> SequenceEncoder:
    torch.manual_seed(1337)
    return SequenceEncoder().eval()


class TestShapes:
    @pytest.mark.parametrize("time_steps", [1, 4, 16, 128, 400])
    def test_output_shape(self, encoder: SequenceEncoder, time_steps: int) -> None:
        out = encoder(torch.randn(2, time_steps, 192))
        assert out.shape == (2, time_steps, 384)

    def test_output_size_is_twice_hidden(self, encoder: SequenceEncoder) -> None:
        assert encoder.output_size == encoder.hidden_size * 2 == 384

    def test_lengths_do_not_change_the_output_shape(self, encoder: SequenceEncoder) -> None:
        features = torch.randn(3, 64, 192)
        lengths = torch.tensor([64, 30, 7])
        assert encoder(features, lengths).shape == (3, 64, 384)

    def test_rejects_wrong_rank(self, encoder: SequenceEncoder) -> None:
        with pytest.raises(ValueError, match=r"\[B, T, C\]"):
            encoder(torch.randn(4, 192))


class TestPacking:
    def test_padding_does_not_change_a_samples_own_frames(self, encoder: SequenceEncoder) -> None:
        """The correctness claim of this module.

        One 20-frame line, run alone, must produce the same contextual features as the same line
        sitting in a batch padded to 80 frames. Without packing the backward direction integrates 60
        frames of padding first and the two disagree substantially.
        """
        torch.manual_seed(3)
        line = torch.randn(1, 20, 192)

        alone = encoder(line, torch.tensor([20]))

        padded = torch.zeros(1, 80, 192)
        padded[:, :20] = line
        in_batch = encoder(padded, torch.tensor([20]))

        assert torch.allclose(alone, in_batch[:, :20], atol=1e-5)

    def test_a_samples_frames_do_not_depend_on_its_neighbours(
        self, encoder: SequenceEncoder
    ) -> None:
        """The same line batched with different neighbours must encode identically."""
        torch.manual_seed(5)
        line = torch.randn(20, 192)

        def run(neighbour_frames: int) -> torch.Tensor:
            batch = torch.zeros(2, neighbour_frames, 192)
            batch[0, :20] = line
            batch[1] = torch.randn(neighbour_frames, 192)
            lengths = torch.tensor([20, neighbour_frames])
            return encoder(batch, lengths)[0, :20]

        assert torch.allclose(run(40), run(200), atol=1e-5)

    def test_without_lengths_padding_is_read(self, encoder: SequenceEncoder) -> None:
        """The bug this module prevents, demonstrated.

        Omitting ``input_lengths`` is documented as only correct for equal-length batches. This pins
        that it genuinely matters rather than being a stylistic preference — if this test ever
        starts passing, packing has silently stopped doing anything.
        """
        torch.manual_seed(3)
        line = torch.randn(1, 20, 192)
        padded = torch.zeros(1, 80, 192)
        padded[:, :20] = line

        packed = encoder(padded, torch.tensor([20]))[:, :20]
        unpacked = encoder(padded)[:, :20]

        assert not torch.allclose(packed, unpacked, atol=1e-3)

    def test_output_is_padded_back_to_total_length(self, encoder: SequenceEncoder) -> None:
        """``pad_packed_sequence`` pads to the longest *sequence*, not to ``T``.

        Without ``total_length`` this returns 30 frames for a 128-frame tensor, silently
        desynchronizing logits from the lengths that describe them.
        """
        out = encoder(torch.randn(2, 128, 192), torch.tensor([30, 12]))
        assert out.shape[1] == 128

    def test_frames_beyond_a_samples_length_are_zero(self, encoder: SequenceEncoder) -> None:
        out = encoder(torch.randn(2, 64, 192), torch.tensor([64, 10]))
        assert torch.count_nonzero(out[1, 10:]) == 0

    def test_unsorted_lengths_are_handled(self, encoder: SequenceEncoder) -> None:
        """``enforce_sorted=False``: batches arrive in width-bucket order, not sorted."""
        features = torch.randn(4, 50, 192)
        out = encoder(features, torch.tensor([12, 50, 31, 4]))
        assert out.shape == (4, 50, 384)
        assert torch.count_nonzero(out[3, 4:]) == 0

    def test_rejects_lengths_longer_than_the_tensor(self, encoder: SequenceEncoder) -> None:
        with pytest.raises(ValueError, match="exceeds T"):
            encoder(torch.randn(2, 10, 192), torch.tensor([10, 11]))

    def test_rejects_zero_length(self, encoder: SequenceEncoder) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            encoder(torch.randn(2, 10, 192), torch.tensor([5, 0]))

    def test_rejects_length_count_mismatch(self, encoder: SequenceEncoder) -> None:
        with pytest.raises(ValueError, match="entries for a batch"):
            encoder(torch.randn(3, 10, 192), torch.tensor([5, 5]))


class TestParameters:
    def test_measured_count_matches_hand_calculation(self) -> None:
        """2 layers, bidirectional, input 192, hidden 192.

        layer 1, per direction: 3*(192*192 + 192*192 + 192 + 192) = 222,336 layer 2 takes 384
        inputs: 3*(384*192 + 192*192 + 192 + 192) = 332,928 total: 2*(222,336 + 332,928) = 1,110,528
        """
        assert parameter_count(SequenceEncoder()) == 1_110_528

    def test_from_config(self) -> None:
        model = SequenceEncoder.from_config(ModelConfig(gru_hidden=128, gru_layers=1))
        assert model.hidden_size == 128
        assert model.num_layers == 1
        assert model.output_size == 256

    def test_single_layer_zeroes_dropout(self) -> None:
        """``nn.GRU`` warns when dropout is set with one layer; zero it rather than warn."""
        assert SequenceEncoder(num_layers=1, dropout=0.15).dropout == 0.0

    def test_describe(self, encoder: SequenceEncoder) -> None:
        described = encoder.describe()
        assert described["kind"] == "bigru"
        assert described["bidirectional"] is True
        assert described["output_size"] == 384

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"input_size": 0}, "positive"),
            ({"hidden_size": 0}, "positive"),
            ({"num_layers": 0}, "at least 1"),
            ({"dropout": 1.0}, r"\[0, 1\)"),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            SequenceEncoder(**kwargs)


class TestTrainingBehaviour:
    def test_dropout_is_active_in_train_mode(self) -> None:
        torch.manual_seed(0)
        model = SequenceEncoder().train()
        features = torch.randn(2, 32, 192)
        assert not torch.allclose(model(features), model(features))

    def test_eval_mode_is_deterministic(self, encoder: SequenceEncoder) -> None:
        features = torch.randn(2, 32, 192)
        with torch.no_grad():
            assert torch.equal(encoder(features), encoder(features))

    def test_gradients_flow_through_packing(self) -> None:
        model = SequenceEncoder()
        features = torch.randn(2, 40, 192, requires_grad=True)
        model(features, torch.tensor([40, 15])).sum().backward()
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"no gradient for {name}"
