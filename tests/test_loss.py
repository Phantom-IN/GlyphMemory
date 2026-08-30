"""CTC loss tests.

The `zero_infinity` tests matter most. That flag keeps a run alive by turning an unalignable
sample's infinite loss into zero — which also erases the evidence.
"""

from __future__ import annotations

import math

import pytest
import torch

from glyphmemory.ctc import BLANK_INDEX
from glyphmemory.model.loss import (
    BACKENDS_WITHOUT_CTC,
    count_infeasible,
    ctc_loss,
    required_alignment_lengths,
)

VOCAB = 12


def flat(*sequences: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """``(targets, target_lengths)`` in the flattened layout ``nn.CTCLoss`` expects."""
    targets = torch.tensor([label for sequence in sequences for label in sequence])
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    return targets, lengths


class TestRequiredAlignment:
    @pytest.mark.parametrize(
        ("labels", "expected"),
        [
            ([1, 2, 3], 3),
            ([1, 1], 3),
            ([1, 1, 1], 5),
            ([1, 2, 2, 3], 5),
            ([5], 1),
            ([], 0),
        ],
    )
    def test_adjacent_repeats_need_a_separating_blank(
        self, labels: list[int], expected: int
    ) -> None:
        targets, lengths = flat(labels)
        assert int(required_alignment_lengths(targets, lengths)[0]) == expected

    def test_matches_the_dataset_helper(self) -> None:
        """One rule, two call sites — they must not drift."""
        from glyphmemory.data.dataset import required_ctc_length

        for labels in ([1, 2, 3], [1, 1], [3, 3, 3], [1, 2, 2, 1], [7]):
            targets, lengths = flat(labels)
            assert int(required_alignment_lengths(targets, lengths)[0]) == required_ctc_length(
                labels
            )

    def test_counts_per_sample_in_a_batch(self) -> None:
        targets, lengths = flat([1, 1], [1, 2, 3], [4, 4, 4])
        assert required_alignment_lengths(targets, lengths).tolist() == [3, 3, 5]

    def test_count_infeasible_finds_the_offenders(self) -> None:
        targets, lengths = flat([1, 1], [1, 2])
        count, indices = count_infeasible(targets, lengths, torch.tensor([2, 5]))
        assert count == 1
        assert indices == [0]


class TestLoss:
    def test_finite_on_a_well_formed_batch(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(3, 40, VOCAB)
        targets, target_lengths = flat([1, 2, 3], [4, 5], [6, 7, 8, 9])
        loss, diag = ctc_loss(logits, targets, torch.tensor([40, 30, 20]), target_lengths)
        assert torch.isfinite(loss)
        assert diag.loss_is_finite
        assert diag.infeasible == 0

    def test_random_logits_land_near_the_uniform_expectation(self) -> None:
        """A sanity anchor on magnitude.

        With near-uniform logits, per-frame log-probability is about ``-log(C)``. A loss wildly away
        from ``target_length * log(C)`` means the reduction or the layout is wrong, which is
        otherwise hard to notice.
        """
        logits = torch.zeros(1, 60, VOCAB)
        targets, target_lengths = flat([1, 2, 3, 4, 5])
        loss, _ = ctc_loss(logits, targets, torch.tensor([60]), target_lengths)
        assert 0.2 < float(loss) / (5 * math.log(VOCAB)) < 5.0

    def test_perfect_alignment_gives_near_zero_loss(self) -> None:
        """Confident logits that spell the target must produce a very small loss.

        This is the sign-convention and blank-index test: get the blank wrong and this stays large.
        """
        labels = [1, 2, 3]
        logits = torch.full((1, 3, VOCAB), -20.0)
        for frame, label in enumerate(labels):
            logits[0, frame, label] = 20.0
        targets, target_lengths = flat(labels)
        loss, _ = ctc_loss(logits, targets, torch.tensor([3]), target_lengths)
        assert float(loss) < 1e-3

    def test_wrong_alignment_gives_large_loss(self) -> None:
        labels = [1, 2, 3]
        logits = torch.full((1, 3, VOCAB), -20.0)
        for frame in range(3):
            logits[0, frame, 9] = 20.0
        targets, target_lengths = flat(labels)
        loss, _ = ctc_loss(logits, targets, torch.tensor([3]), target_lengths)
        assert float(loss) > 10.0

    def test_blank_is_index_zero(self) -> None:
        assert BLANK_INDEX == 0

    def test_rejects_blank_outside_the_vocabulary(self) -> None:
        targets, target_lengths = flat([1])
        with pytest.raises(ValueError, match="outside the vocabulary"):
            ctc_loss(
                torch.randn(1, 5, VOCAB), targets, torch.tensor([5]), target_lengths, blank=VOCAB
            )

    def test_mixed_width_batch(self) -> None:
        """Gate 1 says length bugs hide in batches mixing a short and a long line."""
        torch.manual_seed(1)
        logits = torch.randn(2, 400, VOCAB)
        targets, target_lengths = flat([1, 2], [3, 4, 5, 6, 7, 8])
        loss, diag = ctc_loss(logits, targets, torch.tensor([400, 25]), target_lengths)
        assert torch.isfinite(loss)
        assert diag.input_length_min == 25
        assert diag.input_length_max == 400

    def test_gradients_are_finite(self) -> None:
        logits = torch.randn(2, 30, VOCAB, requires_grad=True)
        targets, target_lengths = flat([1, 2], [3, 4])
        loss, _ = ctc_loss(logits, targets, torch.tensor([30, 30]), target_lengths)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0


class TestLayoutGuards:
    def test_rejects_input_lengths_exceeding_t(self) -> None:
        """The padded-width bug, caught rather than trained on."""
        targets, target_lengths = flat([1, 2])
        with pytest.raises(ValueError, match="padded-width bug"):
            ctc_loss(torch.randn(1, 10, VOCAB), targets, torch.tensor([11]), target_lengths)

    def test_rejects_target_length_sum_mismatch(self) -> None:
        targets = torch.tensor([1, 2, 3])
        with pytest.raises(ValueError, match="flattened layout is inconsistent"):
            ctc_loss(torch.randn(1, 10, VOCAB), targets, torch.tensor([10]), torch.tensor([2]))

    def test_rejects_wrong_length_vector_shape(self) -> None:
        targets, target_lengths = flat([1, 2], [3, 4])
        with pytest.raises(ValueError, match="shape"):
            ctc_loss(torch.randn(2, 10, VOCAB), targets, torch.tensor([10]), target_lengths)

    def test_rejects_wrong_logits_rank(self) -> None:
        targets, target_lengths = flat([1])
        with pytest.raises(ValueError, match=r"\[B, T, C\]"):
            ctc_loss(torch.randn(10, VOCAB), targets, torch.tensor([10]), target_lengths)

    def test_transposed_logits_are_caught(self) -> None:
        """``[T, B, C]`` passed where ``[B, T, C]`` belongs — one of failure modes.

        Transposing ``[2, 30, C]`` yields ``[30, 2, C]``, so the loss reads B=30 and the length
        vectors (2 entries) no longer describe the batch. The shape guard fires.

        **This detection is structural, not semantic**: it works because B and T differ. A square
        batch — B frames for B samples — would transpose into a same-shaped tensor that no guard can
        distinguish. That case is unreachable in practice (T is hundreds of frames for batches of
        tens) but the limit is worth knowing rather than assuming the guard is total.
        """
        targets, target_lengths = flat([1, 2], [3, 4])
        transposed = torch.randn(2, 30, VOCAB).transpose(0, 1)
        with pytest.raises(ValueError, match="Expected input_lengths and target_lengths"):
            ctc_loss(transposed, targets, torch.tensor([30, 30]), target_lengths)


class TestZeroInfinity:
    def test_infeasible_sample_raises_by_default(self) -> None:
        """Internal helper."""
        targets, target_lengths = flat([1, 1, 1])  # needs 5 frames
        with pytest.raises(ValueError, match="cannot be aligned by CTC"):
            ctc_loss(torch.randn(1, 3, VOCAB), targets, torch.tensor([3]), target_lengths)

    def test_non_strict_counts_instead_of_raising(self) -> None:
        targets, target_lengths = flat([1, 1, 1])
        loss, diag = ctc_loss(
            torch.randn(1, 3, VOCAB), targets, torch.tensor([3]), target_lengths, strict=False
        )
        assert diag.infeasible == 1
        assert torch.isfinite(loss)  # zero_infinity did its job — and we still counted it

    def test_the_count_is_computed_from_lengths_not_from_the_loss(self) -> None:
        """``zero_infinity`` erases the loss evidence, so the count cannot depend on it."""
        targets, target_lengths = flat([1, 1, 1], [2, 3])
        _, diag = ctc_loss(
            torch.randn(2, 3, VOCAB),
            targets,
            torch.tensor([3, 3]),
            target_lengths,
            strict=False,
        )
        assert diag.infeasible == 1
        assert diag.loss_is_finite

    def test_a_feasible_batch_reports_zero(self) -> None:
        targets, target_lengths = flat([1, 1, 1])
        _, diag = ctc_loss(torch.randn(1, 5, VOCAB), targets, torch.tensor([5]), target_lengths)
        assert diag.infeasible == 0


class TestDiagnostics:
    def test_reports_length_ranges(self) -> None:
        targets, target_lengths = flat([1], [2, 3, 4])
        _, diag = ctc_loss(
            torch.randn(2, 20, VOCAB), targets, torch.tensor([20, 9]), target_lengths
        )
        assert diag.batch_size == 2
        assert diag.time_steps == 20
        assert (diag.input_length_min, diag.input_length_max) == (9, 20)
        assert (diag.target_length_min, diag.target_length_max) == (1, 3)

    def test_as_dict_is_json_shaped(self) -> None:
        targets, target_lengths = flat([1, 2])
        _, diag = ctc_loss(torch.randn(1, 10, VOCAB), targets, torch.tensor([10]), target_lengths)
        payload = diag.as_dict()
        assert payload["infeasible"] == 0
        assert payload["loss_is_finite"] is True


class TestDeviceParity:
    """Closes the item carried forward.

    Measured 2026-08-18, torch 2.13.0: **MPS has no `aten::_ctc_loss` kernel.** `ctc_loss` therefore
    computes on the CPU for backends in `BACKENDS_WITHOUT_CTC`, in code rather than behind
    `PYTORCH_ENABLE_MPS_FALLBACK=1` — a silent global env var is a worse contract than an explicit,
    logged fallback, and one the user could forget to set.
    """

    def test_cpu_is_not_in_the_fallback_set(self) -> None:
        assert "cpu" not in BACKENDS_WITHOUT_CTC
        assert "mps" in BACKENDS_WITHOUT_CTC

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="no MPS device on this machine"
    )
    def test_loss_matches_cpu_on_mps(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(3, 40, VOCAB)
        targets, target_lengths = flat([1, 2, 3], [4, 5], [6, 7, 8, 9])
        lengths = torch.tensor([40, 30, 20])

        cpu_loss, _ = ctc_loss(logits, targets, lengths, target_lengths)
        mps_loss, diagnostics = ctc_loss(
            logits.to("mps"), targets.to("mps"), lengths.to("mps"), target_lengths.to("mps")
        )
        assert mps_loss.device.type == "mps"
        assert float(mps_loss) == pytest.approx(float(cpu_loss), abs=1e-5)
        assert diagnostics.loss_is_finite

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="no MPS device on this machine"
    )
    def test_gradients_flow_back_through_the_cpu_hop(self) -> None:
        """The transfer is differentiable, so autograd returns the gradient to the device."""
        torch.manual_seed(0)
        logits = torch.randn(2, 30, VOCAB, device="mps", requires_grad=True)
        targets, target_lengths = flat([1, 2], [3, 4])
        loss, _ = ctc_loss(
            logits, targets.to("mps"), torch.tensor([30, 30]).to("mps"), target_lengths.to("mps")
        )
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.device.type == "mps"
        assert torch.isfinite(logits.grad).all()
        assert float(logits.grad.abs().sum()) > 0
