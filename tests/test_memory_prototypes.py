"""`GlyphAccumulator`: ``prototype(c) = L2_normalize(mean(normalized occurrences of c))`` formula,
streamed one observation at a time.
"""

from __future__ import annotations

import pytest
import torch

from glyphmemory.memory.prototypes import GlyphAccumulator, PrototypeAccumulator


def test_finalize_matches_the_hand_computed_two_stage_normalize_formula():
    acc = GlyphAccumulator()
    acc.observe("a", torch.tensor([3.0, 4.0]), confidence=0.8)  # normalizes to [0.6, 0.8]
    acc.observe("a", torch.tensor([0.0, 5.0]), confidence=0.6)  # normalizes to [0.0, 1.0]

    result = acc.finalize()

    mean = (torch.tensor([0.6, 0.8]) + torch.tensor([0.0, 1.0])) / 2
    expected_prototype = mean / mean.norm()
    prototype, count, confidence = result["a"]
    assert torch.allclose(prototype, expected_prototype, atol=1e-6)
    assert count == 2
    assert confidence == (0.8 + 0.6) / 2


def test_prototype_is_always_unit_norm():
    acc = GlyphAccumulator()
    acc.observe("r", torch.tensor([1.0, 2.0, 3.0]), confidence=0.5)
    acc.observe("r", torch.tensor([-1.0, 0.0, 9.0]), confidence=0.9)

    prototype, _, _ = acc.finalize()["r"]

    assert torch.isclose(prototype.norm(), torch.tensor(1.0), atol=1e-6)


def test_an_unobserved_character_produces_no_entry():
    acc = GlyphAccumulator()
    acc.observe("a", torch.tensor([1.0, 0.0]), confidence=1.0)

    result = acc.finalize()

    assert "z" not in result
    assert set(result) == {"a"}


def test_characters_and_count_reflect_observations_before_finalize():
    acc = GlyphAccumulator()
    acc.observe("a", torch.tensor([1.0, 0.0]), confidence=1.0)
    acc.observe("a", torch.tensor([0.0, 1.0]), confidence=1.0)
    acc.observe("b", torch.tensor([1.0, 0.0]), confidence=1.0)

    assert acc.characters == frozenset({"a", "b"})
    assert acc.count("a") == 2
    assert acc.count("b") == 1
    assert acc.count("z") == 0


def test_empty_accumulator_finalizes_to_empty():
    acc = GlyphAccumulator()
    assert acc.finalize() == {}
    assert acc.characters == frozenset()


def test_a_single_observation_normalizes_to_itself():
    acc = GlyphAccumulator()
    acc.observe("a", torch.tensor([3.0, 4.0]), confidence=0.42)

    prototype, count, confidence = acc.finalize()["a"]

    assert torch.allclose(prototype, torch.tensor([0.6, 0.8]), atol=1e-6)
    assert count == 1
    assert confidence == 0.42


def test_different_characters_do_not_interfere_with_each_other():
    acc = GlyphAccumulator()
    acc.observe("a", torch.tensor([1.0, 0.0]), confidence=1.0)
    acc.observe("b", torch.tensor([0.0, 1.0]), confidence=1.0)

    result = acc.finalize()

    assert torch.allclose(result["a"][0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(result["b"][0], torch.tensor([0.0, 1.0]))


# ------------------------------------------------------------------ PrototypeAccumulator (M9, P26)


def test_prototype_accumulator_mean_strategy_matches_glyph_accumulator_exactly():
    # Correctness cross-check: PrototypeAccumulator's "mean" strategy recomputes the identical
    # formula GlyphAccumulator streams -- from stored occurrences instead of a running sum.
    glyph_acc = GlyphAccumulator()
    proto_acc = PrototypeAccumulator()
    observations = [
        ("a", torch.tensor([3.0, 4.0]), 0.8),
        ("a", torch.tensor([0.0, 5.0]), 0.6),
        ("b", torch.tensor([1.0, 1.0]), 0.5),
    ]
    for character, vector, confidence in observations:
        glyph_acc.observe(character, vector, confidence)
        proto_acc.observe(character, vector, confidence)

    expected = glyph_acc.finalize()
    actual = proto_acc.finalize("mean")

    for character in expected:
        exp_prototype, exp_count, exp_confidence = expected[character]
        (act_prototype,), act_count, act_confidence = actual[character]
        assert torch.allclose(act_prototype, exp_prototype, atol=1e-6)
        assert act_count == exp_count
        assert act_confidence == exp_confidence


def test_confidence_weighted_matches_hand_computed_weighted_mean():
    acc = PrototypeAccumulator()
    acc.observe("a", torch.tensor([3.0, 4.0]), confidence=0.8)  # normalizes to [0.6, 0.8]
    acc.observe("a", torch.tensor([0.0, 5.0]), confidence=0.2)  # normalizes to [0.0, 1.0]

    (prototype,), count, confidence = acc.finalize("confidence_weighted")["a"]

    weighted_mean = 0.8 * torch.tensor([0.6, 0.8]) + 0.2 * torch.tensor([0.0, 1.0])
    expected = weighted_mean / weighted_mean.norm()
    assert torch.allclose(prototype, expected, atol=1e-6)
    assert count == 2
    assert confidence == (0.8 + 0.2) / 2


def test_confidence_weighted_falls_back_to_uniform_mean_at_negligible_confidence_mass():
    acc = PrototypeAccumulator()
    acc.observe("a", torch.tensor([1.0, 0.0]), confidence=0.0)
    acc.observe("a", torch.tensor([0.0, 1.0]), confidence=0.0)

    (prototype,), _, _ = acc.finalize("confidence_weighted")["a"]

    expected = torch.tensor([1.0, 1.0]) / torch.tensor([1.0, 1.0]).norm()
    assert torch.allclose(prototype, expected, atol=1e-6)


def test_medoid_picks_the_occurrence_closest_to_the_mean():
    acc = PrototypeAccumulator()
    acc.observe("a", torch.tensor([1.0, 0.0]), confidence=1.0)  # a: far from the eventual mean
    acc.observe("a", torch.tensor([0.8, 0.6]), confidence=1.0)  # b: closest to the mean
    acc.observe("a", torch.tensor([0.0, 1.0]), confidence=1.0)  # c: far from the eventual mean

    (prototype,), count, _ = acc.finalize("medoid")["a"]

    assert torch.allclose(prototype, torch.tensor([0.8, 0.6]), atol=1e-6)
    assert count == 3


def test_medoid_prototype_is_a_real_observed_vector_not_a_synthesized_average():
    acc = PrototypeAccumulator()
    acc.observe("a", torch.tensor([1.0, 0.0]), confidence=1.0)
    acc.observe("a", torch.tensor([0.0, 1.0]), confidence=1.0)

    (prototype,), _, _ = acc.finalize("medoid")["a"]

    observed = {(1.0, 0.0), (0.0, 1.0)}
    assert (round(prototype[0].item(), 6), round(prototype[1].item(), 6)) in observed


def test_top_k_selects_the_k_highest_confidence_occurrences_in_order():
    acc = PrototypeAccumulator()
    acc.observe("a", torch.tensor([1.0, 0.0]), confidence=0.9)
    acc.observe("a", torch.tensor([0.0, 1.0]), confidence=0.5)
    acc.observe("a", torch.tensor([0.6, 0.8]), confidence=0.7)
    acc.observe("a", torch.tensor([-1.0, 0.0]), confidence=0.99)

    prototypes, count, _ = acc.finalize("top_k", top_k=2)["a"]

    assert count == 4
    assert len(prototypes) == 2
    assert torch.allclose(prototypes[0], torch.tensor([-1.0, 0.0]), atol=1e-6)  # confidence 0.99
    assert torch.allclose(prototypes[1], torch.tensor([1.0, 0.0]), atol=1e-6)  # confidence 0.9


def test_top_k_returns_fewer_than_k_when_fewer_occurrences_exist():
    acc = PrototypeAccumulator()
    acc.observe("a", torch.tensor([1.0, 0.0]), confidence=0.9)
    acc.observe("a", torch.tensor([0.0, 1.0]), confidence=0.5)

    prototypes, count, _ = acc.finalize("top_k", top_k=5)["a"]

    assert count == 2
    assert len(prototypes) == 2


def test_top_k_breaks_confidence_ties_by_original_observation_order():
    acc = PrototypeAccumulator()
    acc.observe("a", torch.tensor([1.0, 0.0]), confidence=0.5)
    acc.observe("a", torch.tensor([0.0, 1.0]), confidence=0.5)

    prototypes, _, _ = acc.finalize("top_k", top_k=1)["a"]

    assert torch.allclose(prototypes[0], torch.tensor([1.0, 0.0]), atol=1e-6)


def test_prototype_accumulator_rejects_an_unknown_strategy():
    acc = PrototypeAccumulator()
    acc.observe("a", torch.tensor([1.0, 0.0]), confidence=1.0)

    with pytest.raises(ValueError, match="prototype_strategy"):
        acc.finalize("nonexistent")


def test_prototype_accumulator_characters_and_count_reflect_observations():
    acc = PrototypeAccumulator()
    acc.observe("a", torch.tensor([1.0, 0.0]), confidence=1.0)
    acc.observe("a", torch.tensor([0.0, 1.0]), confidence=1.0)
    acc.observe("b", torch.tensor([1.0, 0.0]), confidence=1.0)

    assert acc.characters == frozenset({"a", "b"})
    assert acc.count("a") == 2
    assert acc.count("b") == 1
    assert acc.count("z") == 0


def test_prototype_accumulator_empty_finalizes_to_empty():
    acc = PrototypeAccumulator()
    assert acc.finalize("mean") == {}
    assert acc.characters == frozenset()


def test_prototype_accumulator_an_unobserved_character_produces_no_entry():
    acc = PrototypeAccumulator()
    acc.observe("a", torch.tensor([1.0, 0.0]), confidence=1.0)

    result = acc.finalize("mean")

    assert "z" not in result
    assert set(result) == {"a"}
