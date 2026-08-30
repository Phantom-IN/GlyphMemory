"""Offline metric training."""

from __future__ import annotations

import math

import pytest
import torch

from glyphmemory.memory.projection import GlyphProjection
from glyphmemory.memory.projection_training import (
    OccurrenceIndex,
    confusion_pairs_from_top_confusions,
    cosine_triplet_loss,
    sample_triplet_batch,
    train_projection,
)
from glyphmemory.probes.occurrences import CharacterOccurrence


def _occurrence(writer_id: str, character: str, feature: list[float]) -> CharacterOccurrence:
    vector = torch.tensor(feature, dtype=torch.float32)
    return CharacterOccurrence(
        writer_id=writer_id,
        sample_id=f"{writer_id}-{character}-sample",
        character=character,
        frame_index=0,
        visual_feature=torch.zeros(192),
        sequence_feature=vector,
        base_head_prediction=character,
        alignment_score=0.9,
    )


# ------------------------------------------------------------------ confusion_pairs


def test_confusion_pairs_are_symmetric():
    top_confusions = [{"reference": "r", "hypothesis": "v", "count": 10}]
    pairs = confusion_pairs_from_top_confusions(top_confusions)
    assert pairs["r"] == frozenset({"v"})
    assert pairs["v"] == frozenset({"r"})


def test_confusion_pairs_merge_across_multiple_entries():
    top_confusions = [
        {"reference": "r", "hypothesis": "v", "count": 10},
        {"reference": "r", "hypothesis": "n", "count": 5},
    ]
    pairs = confusion_pairs_from_top_confusions(top_confusions)
    assert pairs["r"] == frozenset({"v", "n"})


def test_confusion_pairs_ignore_self_confusions():
    top_confusions = [{"reference": "r", "hypothesis": "r", "count": 3}]
    pairs = confusion_pairs_from_top_confusions(top_confusions)
    assert pairs == {}


# ------------------------------------------------------------------ OccurrenceIndex


def test_occurrence_index_groups_by_character_and_writer_character():
    occurrences = [
        _occurrence("w1", "a", [1.0, 0.0]),
        _occurrence("w1", "a", [0.9, 0.1]),
        _occurrence("w2", "a", [0.0, 1.0]),
        _occurrence("w1", "b", [1.0, 1.0]),
    ]
    index = OccurrenceIndex.build(occurrences)

    assert len(index) == 4
    assert set(index.by_character["a"]) == {0, 1, 2}
    assert set(index.by_character["b"]) == {3}
    assert set(index.by_writer_character[("w1", "a")]) == {0, 1}
    assert set(index.by_writer_character[("w2", "a")]) == {2}


def test_occurrence_index_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        OccurrenceIndex.build([])


# ------------------------------------------------------------------ sample_triplet_batch


def _rich_occurrences() -> list[CharacterOccurrence]:
    # 3 writers x 3 characters x 3 occurrences each -- enough for both loss terms everywhere.
    occurrences = []
    for w in ("w1", "w2", "w3"):
        for c in ("a", "b", "c"):
            for i in range(3):
                occurrences.append(_occurrence(w, c, [hash((w, c, i)) % 100 / 100.0, 0.5]))
    return occurrences


def test_sample_triplet_batch_is_deterministic():
    index = OccurrenceIndex.build(_rich_occurrences())
    a = sample_triplet_batch(index, batch_size=10, seed=1337)
    b = sample_triplet_batch(index, batch_size=10, seed=1337)
    assert a == b


def test_sample_triplet_batch_char_triplets_share_character_for_anchor_and_positive():
    index = OccurrenceIndex.build(_rich_occurrences())
    batch = sample_triplet_batch(index, batch_size=20, seed=1337)
    for anchor, positive in zip(batch.char_anchor, batch.char_positive, strict=True):
        assert index.characters[anchor] == index.characters[positive]
        assert anchor != positive


def test_sample_triplet_batch_char_negative_has_a_different_character():
    index = OccurrenceIndex.build(_rich_occurrences())
    batch = sample_triplet_batch(index, batch_size=20, seed=1337)
    for anchor, negative in zip(batch.char_anchor, batch.char_negative, strict=True):
        assert index.characters[anchor] != index.characters[negative]


def test_sample_triplet_batch_writer_triplets_share_character_but_positive_shares_writer():
    index = OccurrenceIndex.build(_rich_occurrences())
    batch = sample_triplet_batch(index, batch_size=20, seed=1337)
    for anchor, positive, negative in zip(
        batch.writer_anchor, batch.writer_positive, batch.writer_negative, strict=True
    ):
        assert index.characters[anchor] == index.characters[positive] == index.characters[negative]
        assert index.writers[anchor] == index.writers[positive]
        assert index.writers[anchor] != index.writers[negative]


def test_sample_triplet_batch_prefers_confusion_pairs_for_char_negative():
    # Only "a" and "z" exist as confusable; if the sampler prefers confusions, every char negative
    # for an "a" anchor should be a "z" occurrence, deterministically, across many draws (not just
    # probabilistically likely).
    occurrences = [
        _occurrence("w1", "a", [1.0, 0.0]),
        _occurrence("w2", "a", [0.9, 0.1]),
        _occurrence("w1", "z", [0.0, 1.0]),
        _occurrence("w2", "z", [0.0, 0.9]),
        _occurrence("w1", "q", [0.5, 0.5]),
    ]
    index = OccurrenceIndex.build(occurrences)
    confusion_pairs = confusion_pairs_from_top_confusions(
        [{"reference": "a", "hypothesis": "z", "count": 99}]
    )

    batch = sample_triplet_batch(index, batch_size=30, seed=42, confusion_pairs=confusion_pairs)
    for anchor, negative in zip(batch.char_anchor, batch.char_negative, strict=True):
        if index.characters[anchor] == "a":
            assert index.characters[negative] == "z"


def test_sample_triplet_batch_counts_skips_when_no_writer_partner_exists():
    # A single writer for every character -> the writer loss can never find a cross-writer negative,
    # so every anchor should be skipped for the writer term, none for the char term.
    occurrences = [_occurrence("w1", "a", [1.0, 0.0]), _occurrence("w1", "a", [0.9, 0.1])]
    index = OccurrenceIndex.build(occurrences)
    batch = sample_triplet_batch(index, batch_size=5, seed=1337)
    assert batch.writer_skipped == 5
    assert len(batch.writer_anchor) == 0


# ------------------------------------------------------------------ cosine_triplet_loss


def test_cosine_triplet_loss_is_zero_when_positive_is_closer_than_negative_by_the_margin():
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])  # d_pos = 0
    negative = torch.tensor([[0.0, 1.0]])  # d_neg = 1
    loss = cosine_triplet_loss(anchor, positive, negative, margin=0.2)
    assert loss.item() == 0.0


def test_cosine_triplet_loss_is_positive_when_negative_is_closer_than_positive():
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[0.0, 1.0]])  # d_pos = 1
    negative = torch.tensor([[1.0, 0.0]])  # d_neg = 0
    loss = cosine_triplet_loss(anchor, positive, negative, margin=0.2)
    # relu(1 - 0 + 0.2) = 1.2
    assert loss.item() == pytest.approx(1.2)


def test_cosine_triplet_loss_matches_hand_computation():
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[math.sqrt(2) / 2, math.sqrt(2) / 2]])  # 45 degrees, d_pos ~ 0.293
    negative = torch.tensor([[-1.0, 0.0]])  # d_neg = 2
    loss = cosine_triplet_loss(anchor, positive, negative, margin=0.1)
    d_pos = 1 - (math.sqrt(2) / 2)
    d_neg = 2.0
    expected = max(0.0, d_pos - d_neg + 0.1)
    assert loss.item() == pytest.approx(expected)


# ------------------------------------------------------------------ train_projection


def _tiny_projection() -> GlyphProjection:
    # Matches _occurrence()'s 2D synthetic feature vectors -- the real (384D) default is exercised
    # separately in tests/test_memory_projection.py.
    return GlyphProjection(input_dim=2, hidden_dim=4, output_dim=3)


def test_train_projection_runs_end_to_end_on_synthetic_occurrences():
    occurrences = _rich_occurrences()
    model, log = train_projection(
        occurrences, steps=5, batch_size=16, seed=1337, projection=_tiny_projection()
    )

    assert isinstance(model, GlyphProjection)
    assert log.steps == 5
    output = model(torch.randn(3, model.input_dim))
    assert output.shape == (3, model.output_dim)


def test_train_projection_reports_skip_counts_not_silently():
    # Only one writer -> writer loss always skipped, but training still completes.
    occurrences = [
        _occurrence("w1", c, [float(i), 0.0]) for c in ("a", "b", "c") for i in range(3)
    ]
    _, log = train_projection(
        occurrences, steps=3, batch_size=8, seed=1337, projection=_tiny_projection()
    )
    assert log.total_writer_skipped > 0


def test_train_projection_continues_training_an_existing_model():
    occurrences = _rich_occurrences()
    model, _ = train_projection(
        occurrences, steps=2, batch_size=16, seed=1337, projection=_tiny_projection()
    )
    original_weight = model.net[0].weight.clone()

    model, log = train_projection(
        occurrences, steps=2, batch_size=16, seed=1338, projection=model
    )

    assert log.steps == 2
    assert not torch.equal(model.net[0].weight, original_weight)


def test_train_projection_rejects_zero_steps():
    with pytest.raises(ValueError, match="at least 1"):
        train_projection(_rich_occurrences(), steps=0, projection=_tiny_projection())


def test_train_projection_is_deterministic_given_the_same_seed():
    # Exercises the default (projection=None) path deliberately: that is where `seed` also has to
    # govern the fresh model's random initialization, not merely batch sampling -- a real bug caught
    # while writing this test (a pre-constructed GlyphProjection() is never seeded by `seed` at all,
    # so passing one explicitly would not exercise this property).
    occurrences = [
        _occurrence(w, c, [float(hash((w, c, i)) % 1000) / 1000.0] * 384)
        for w in ("w1", "w2", "w3")
        for c in ("a", "b", "c")
        for i in range(3)
    ]
    model_a, log_a = train_projection(occurrences, steps=4, batch_size=16, seed=1337)
    model_b, log_b = train_projection(occurrences, steps=4, batch_size=16, seed=1337)

    for (name, a), (_, b) in zip(
        model_a.state_dict().items(), model_b.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name
    assert log_a.char_losses == log_b.char_losses
