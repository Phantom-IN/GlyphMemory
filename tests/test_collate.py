"""Collation and CTC length bookkeeping.

The single most consequential bug available in this phase is padding leaking into ``input_lengths``.
Several tests below exist purely to make that impossible to reintroduce without a red suite.
"""

from __future__ import annotations

import pytest
import torch

from glyphmemory.ctc import Tokenizer
from glyphmemory.data import (
    Batch,
    CTCFeasibilityError,
    LineDataset,
    LineSample,
    RejectedSample,
    VariableWidthCollator,
    temporal_length,
)
from glyphmemory.data.validation import IntegrityCategory, IntegrityCounters

TOKENIZER = Tokenizer.english_v1()


def sample(
    text: str, width: int, *, sample_id: str = "synthetic/0", input_length: int | None = None
):
    """A synthetic LineSample with a chosen width — no image file needed."""
    targets = TOKENIZER.encode(text)
    padded = width + (-width % 16)
    return LineSample(
        image=torch.zeros(1, 64, padded, dtype=torch.float32),
        targets=torch.tensor(targets, dtype=torch.long),
        true_width=width,
        input_length=temporal_length(width) if input_length is None else input_length,
        text=text,
        sample_id=sample_id,
        writer_id="synthetic/w0",
        image_path=f"/tmp/{sample_id.replace('/', '_')}.png",
    )


# --------------------------------------------------------------------------- input lengths


def test_input_lengths_come_from_true_widths():
    items = [
        sample("short", 100, sample_id="s/0"),
        sample("a much longer line", 900, sample_id="s/1"),
    ]
    batch = VariableWidthCollator()(items)
    assert batch.input_lengths.tolist() == [temporal_length(100), temporal_length(900)]


def test_input_lengths_are_never_the_padded_width():
    """The bug this whole phase guards against."""
    items = [
        sample("short", 100, sample_id="s/0"),
        sample("longer text here", 900, sample_id="s/1"),
    ]
    batch = VariableWidthCollator()(items)
    padded_length = temporal_length(batch.max_width)
    assert batch.input_lengths[0].item() != padded_length
    assert all(int(length) <= padded_length for length in batch.input_lengths)


def test_mixed_width_batch_pads_to_the_maximum():
    items = [sample("a", 64, sample_id="s/0"), sample("bb", 1408, sample_id="s/1")]
    batch = VariableWidthCollator()(items)
    assert batch.images.shape[0] == 2
    assert batch.max_width >= 1408
    assert batch.max_width % 16 == 0


def test_padding_region_holds_the_pad_value():
    items = [sample("a", 64, sample_id="s/0"), sample("bb", 640, sample_id="s/1")]
    batch = VariableWidthCollator(pad_value=0.0)(items)
    tail = batch.images[0, :, :, 64:]
    assert torch.allclose(tail, torch.zeros_like(tail))


def test_custom_pad_value_is_used():
    items = [sample("a", 64, sample_id="s/0"), sample("bb", 640, sample_id="s/1")]
    batch = VariableWidthCollator(pad_value=-1.0)(items)
    assert batch.images[0, 0, 0, -1].item() == pytest.approx(-1.0)


# --------------------------------------------------------------------------- targets


def test_target_lengths_sum_to_the_flattened_length():
    items = [sample("hello", 400, sample_id="s/0"), sample("committee", 600, sample_id="s/1")]
    batch = VariableWidthCollator()(items)
    assert int(batch.target_lengths.sum()) == int(batch.targets.numel())


def test_targets_are_flattened_for_ctcloss():
    """nn.CTCLoss expects a 1-D concatenation plus a separate length vector."""
    batch = VariableWidthCollator()(
        [sample("ab", 400, sample_id="s/0"), sample("cde", 400, sample_id="s/1")]
    )
    assert batch.targets.ndim == 1
    assert batch.targets.numel() == 5
    assert batch.target_lengths.tolist() == [2, 3]


def test_targets_for_recovers_each_sample():
    texts = ["hello", "committee", "book"]
    items = [sample(text, 600, sample_id=f"s/{i}") for i, text in enumerate(texts)]
    batch = VariableWidthCollator()(items)
    for index, text in enumerate(texts):
        assert TOKENIZER.decode(batch.targets_for(index).tolist()) == text


def test_round_trip_decode_matches_text_for_every_sample(synthetic_corpus):
    dataset = LineDataset(records=synthetic_corpus.records, tokenizer=TOKENIZER)
    batch = VariableWidthCollator()([dataset[i] for i in range(len(dataset))])
    for index in range(batch.batch_size):
        assert TOKENIZER.decode(batch.targets_for(index).tolist()) == batch.texts[index]


@pytest.mark.parametrize(
    ("text", "expected_length"),
    [("hello", 5), ("book", 4), ("letter", 6), ("committee", 9)],
)
def test_repeated_character_target_lengths(text, expected_length):
    batch = VariableWidthCollator()([sample(text, 800)])
    assert int(batch.target_lengths[0]) == expected_length


def test_hand_computed_ctc_loss_layout_accepted_by_torch():
    """Verify the layout against nn.CTCLoss itself, before any model exists."""
    items = [sample("hello", 400, sample_id="s/0"), sample("book", 500, sample_id="s/1")]
    batch = VariableWidthCollator()(items)
    time_steps = int(batch.input_lengths.max())
    logits = torch.randn(time_steps, batch.batch_size, TOKENIZER.vocab_size).log_softmax(2)
    loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)(
        logits, batch.targets, batch.input_lengths, batch.target_lengths
    )
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------- feasibility


def test_infeasible_sample_dropped_and_counted_during_training(caplog):
    counters = IntegrityCounters()
    items = [
        sample("ok line", 800, sample_id="s/good"),
        sample("committee", 800, sample_id="s/bad", input_length=3),
    ]
    with caplog.at_level("WARNING", logger="glyphmemory.data.validation"):
        batch = VariableWidthCollator(training=True, counters=counters)(items)

    assert batch.batch_size == 1
    assert batch.sample_ids == ("s/good",)
    assert counters.count_of(IntegrityCategory.IMPOSSIBLE_CTC_LENGTH) == 1
    assert "s/bad" in caplog.text


def test_infeasible_rejection_names_the_lengths():
    counters = IntegrityCounters()
    VariableWidthCollator(training=True, counters=counters)(
        [
            sample("committee", 800, sample_id="s/bad", input_length=3),
            sample("ok", 800, sample_id="s/g"),
        ]
    )
    reason = counters.issues[0].reason
    assert "input_length 3" in reason
    assert "required 12" in reason


def test_infeasible_sample_raises_during_evaluation():
    """A silently dropped evaluation sample changes the denominator of a reported metric."""
    items = [sample("committee", 800, sample_id="s/bad", input_length=3)]
    with pytest.raises(CTCFeasibilityError, match="denominator"):
        VariableWidthCollator(training=False)(items)


def test_feasible_sample_at_the_exact_boundary_is_kept():
    exact = sample("hello", 800, sample_id="s/edge", input_length=6)
    assert VariableWidthCollator()([exact]).batch_size == 1


def test_rejected_samples_from_the_dataset_are_counted():
    counters = IntegrityCounters()
    items = [
        sample("fine", 400, sample_id="s/0"),
        RejectedSample("s/1", "/tmp/x.png", IntegrityCategory.UNREADABLE_IMAGE, "missing"),
    ]
    batch = VariableWidthCollator(counters=counters)(items)
    assert batch.batch_size == 1
    assert len(batch.rejected) == 1
    assert counters.count_of(IntegrityCategory.UNREADABLE_IMAGE) == 1


def test_all_rejected_yields_an_empty_batch(caplog):
    counters = IntegrityCounters()
    items = [RejectedSample("s/0", "/tmp/x.png", IntegrityCategory.UNREADABLE_IMAGE, "missing")]
    with caplog.at_level("WARNING", logger="glyphmemory.data.collate"):
        batch = VariableWidthCollator(counters=counters)(items)
    assert batch.is_empty
    assert batch.batch_size == 0
    assert len(batch.rejected) == 1
    assert "Every sample in this batch was rejected" in caplog.text


# --------------------------------------------------------------------------- batch helpers


def test_padding_efficiency_is_one_for_equal_widths():
    items = [sample("aa", 640, sample_id=f"s/{i}") for i in range(3)]
    assert VariableWidthCollator()(items).padding_efficiency == pytest.approx(1.0)


def test_padding_efficiency_drops_with_mixed_widths():
    items = [sample("a", 64, sample_id="s/0"), sample("b", 1600, sample_id="s/1")]
    assert VariableWidthCollator()(items).padding_efficiency < 0.6


def test_empty_batch_reports_zero_efficiency():
    from glyphmemory.data import empty_batch

    assert empty_batch().padding_efficiency == 0.0


def test_batch_moves_to_device_preserving_metadata():
    batch = VariableWidthCollator()([sample("hello", 400, sample_id="s/0")])
    moved = batch.to("cpu")
    assert isinstance(moved, Batch)
    assert moved.sample_ids == batch.sample_ids
    assert torch.equal(moved.input_lengths, batch.input_lengths)


def test_batch_metadata_is_aligned_with_rows(synthetic_corpus):
    dataset = LineDataset(records=synthetic_corpus.records, tokenizer=TOKENIZER)
    batch = VariableWidthCollator()([dataset[i] for i in range(4)])
    assert len(batch.texts) == batch.batch_size
    assert len(batch.writer_ids) == batch.batch_size
    assert len(batch.true_widths) == batch.batch_size
    assert len(batch.sample_ids) == batch.batch_size
