"""Gated, blank-protected residual fusion: three non-negotiables, proven directly rather than only
asserted.
"""

from __future__ import annotations

import pytest
import torch

from glyphmemory.config.schema import MemoryConfig
from glyphmemory.ctc.tokenizer import Charset
from glyphmemory.memory.fusion import blank_emission_gate, correction, fuse, personalize
from glyphmemory.memory.profile import Glyph, WriterProfile
from glyphmemory.memory.projection import GlyphProjection
from glyphmemory.model.htr import HTROutput

CHARSET = Charset(symbols=("<blank>", "a", "b"))


def _profile(
    feature_dim: int = 2, feature_layer: str = "sequence", **glyphs: Glyph
) -> WriterProfile:
    return WriterProfile(
        schema_version="1",
        model_fingerprint="fp",
        feature_layer=feature_layer,
        feature_dim=feature_dim,
        glyphs=glyphs,
    )


def _glyph(prototype: list[float]) -> Glyph:
    vector = torch.tensor(prototype, dtype=torch.float32)
    return Glyph(
        character="a",
        prototype=vector / vector.norm(),
        number_of_observations=1,
        mean_alignment_confidence=1.0,
        feature_layer="sequence",
    )


# ------------------------------------------------------------------ blank_emission_gate


def test_gate_at_uniform_posterior_is_one_half():
    logits = torch.tensor([[0.0, 0.0]])  # 2-class: softmax = [0.5, 0.5]
    gate = blank_emission_gate(logits)
    assert torch.allclose(gate, torch.tensor([0.5]))


def test_gate_approaches_zero_when_blank_dominates():
    logits = torch.tensor([[10.0, -10.0]])
    gate = blank_emission_gate(logits)
    assert gate.item() < 0.01


def test_gate_approaches_one_when_blank_is_unlikely():
    logits = torch.tensor([[-10.0, 10.0]])
    gate = blank_emission_gate(logits)
    assert gate.item() > 0.99


def test_gate_works_on_a_batched_shape():
    logits = torch.tensor([[[0.0, 0.0], [10.0, -10.0]]])  # [B=1, T=2, C=2]
    gate = blank_emission_gate(logits)
    assert gate.shape == (1, 2)


# ------------------------------------------------------------------ correction / fuse


def test_correction_matches_hand_computed_values_with_gate_disabled():
    base_logits = torch.tensor([[1.0, 2.0, 0.5], [0.0, 0.0, 3.0]])
    scores = torch.tensor([[5.0, 0.9, -0.2], [0.0, 0.1, 0.4]])  # deliberate nonzero blank score

    result = correction(base_logits, scores, alpha=2.0, gate_by_emission=False, protect_blank=True)

    expected = torch.tensor([[0.0, 1.8, -0.4], [0.0, 0.2, 0.8]])
    assert torch.allclose(result, expected, atol=1e-5)


def test_fuse_matches_hand_computed_values_with_gate_disabled():
    base_logits = torch.tensor([[1.0, 2.0, 0.5], [0.0, 0.0, 3.0]])
    scores = torch.tensor([[5.0, 0.9, -0.2], [0.0, 0.1, 0.4]])

    result = fuse(base_logits, scores, alpha=2.0, gate_by_emission=False, protect_blank=True)

    expected = torch.tensor([[1.0, 3.8, 0.1], [0.0, 0.2, 3.8]])
    assert torch.allclose(result, expected, atol=1e-5)


def test_fuse_minus_base_equals_correction_exactly():
    base_logits = torch.tensor([[1.0, 2.0, 0.5], [3.0, -1.0, 0.0]])
    scores = torch.tensor([[0.2, 0.9, -0.4], [0.1, -0.3, 0.7]])

    for gate_by_emission in (True, False):
        for protect_blank in (True, False):
            fused = fuse(
                base_logits, scores, alpha=0.5,
                gate_by_emission=gate_by_emission, protect_blank=protect_blank,
            )
            expected_correction = correction(
                base_logits, scores, alpha=0.5,
                gate_by_emission=gate_by_emission, protect_blank=protect_blank,
            )
            assert torch.allclose(fused - base_logits, expected_correction, atol=1e-6)


def test_blank_logit_is_provably_untouched_when_protected():
    base_logits = torch.tensor([[1.0, 2.0, 0.5]])
    scores = torch.tensor([[9.0, 0.9, -0.2]])  # nonzero blank score, deliberately

    fused = fuse(base_logits, scores, alpha=1.0, protect_blank=True)

    assert torch.equal(fused[:, 0], base_logits[:, 0])


def test_protect_blank_false_actually_changes_behavior():
    # Fusion's own protection, not merely inherited from retrieval never producing a blank score —
    # proven by handing fuse a score tensor that *does* have one.
    base_logits = torch.tensor([[1.0, 2.0, 0.5]])
    scores = torch.tensor([[9.0, 0.9, -0.2]])

    protected = fuse(base_logits, scores, alpha=1.0, gate_by_emission=False, protect_blank=True)
    unprotected = fuse(base_logits, scores, alpha=1.0, gate_by_emission=False, protect_blank=False)

    assert torch.equal(protected[:, 0], base_logits[:, 0])
    assert not torch.equal(unprotected[:, 0], base_logits[:, 0])
    assert torch.allclose(unprotected[:, 0], torch.tensor([10.0]))


def test_zero_alpha_is_a_no_op():
    base_logits = torch.tensor([[1.0, 2.0, 0.5], [3.0, -1.0, 0.0]])
    scores = torch.tensor([[0.2, 0.9, -0.4], [0.1, -0.3, 0.7]])

    fused = fuse(base_logits, scores, alpha=0.0)

    assert torch.equal(fused, base_logits)


def test_correction_rejects_mismatched_shapes():
    base_logits = torch.zeros((2, 3))
    scores = torch.zeros((2, 4))
    with pytest.raises(ValueError, match="same shape"):
        correction(base_logits, scores, alpha=1.0)


# ------------------------------------------------------------------ personalize


def _output(logits, sequence_features, visual_features, input_lengths) -> HTROutput:
    return HTROutput(
        logits=torch.tensor(logits),
        sequence_features=torch.tensor(sequence_features),
        visual_features=torch.tensor(visual_features),
        input_lengths=torch.tensor(input_lengths),
    )


def test_personalize_matches_fuse_plus_memory_scores_directly():
    output = _output(
        logits=[[[1.0, 2.0, 0.5], [0.0, 0.0, 3.0]]],
        sequence_features=[[[1.0, 0.0], [0.0, 1.0]]],
        visual_features=[[[9.0, 9.0], [9.0, 9.0]]],
        input_lengths=[2],
    )
    profile = _profile(a=_glyph([1.0, 0.0]))
    config = MemoryConfig(
        enabled=True, feature_layer="sequence", alpha=1.0, gate_by_emission=False,
        protect_blank=True, pooling="posterior_weighted",
    )

    corrected = personalize(output, profile, CHARSET, config)

    from glyphmemory.memory.retrieval import memory_scores

    expected = fuse(
        output.logits[0],
        memory_scores(output.sequence_features[0], profile, CHARSET),
        alpha=1.0, gate_by_emission=False, protect_blank=True,
    )
    assert torch.allclose(corrected[0], expected, atol=1e-6)


def test_personalize_returns_base_logits_unchanged_when_disabled():
    output = _output(
        logits=[[[1.0, 2.0, 0.5]]],
        sequence_features=[[[1.0, 0.0]]],
        visual_features=[[[9.0, 9.0]]],
        input_lengths=[1],
    )
    profile = _profile(a=_glyph([1.0, 0.0]))
    config = MemoryConfig(enabled=False, feature_layer="sequence", alpha=1.0)

    corrected = personalize(output, profile, CHARSET, config)

    assert corrected is output.logits


def test_personalize_returns_base_logits_unchanged_when_profile_is_none():
    output = _output(
        logits=[[[1.0, 2.0, 0.5]]],
        sequence_features=[[[1.0, 0.0]]],
        visual_features=[[[9.0, 9.0]]],
        input_lengths=[1],
    )
    config = MemoryConfig(enabled=True, feature_layer="sequence", alpha=1.0)

    corrected = personalize(output, None, CHARSET, config)

    assert corrected is output.logits


def test_personalize_returns_base_logits_unchanged_when_profile_is_empty():
    output = _output(
        logits=[[[1.0, 2.0, 0.5]]],
        sequence_features=[[[1.0, 0.0]]],
        visual_features=[[[9.0, 9.0]]],
        input_lengths=[1],
    )
    profile = _profile()  # no glyphs
    config = MemoryConfig(enabled=True, feature_layer="sequence", alpha=1.0)

    corrected = personalize(output, profile, CHARSET, config)

    assert corrected is output.logits


def test_personalize_rejects_a_feature_layer_mismatch():
    output = _output(
        logits=[[[1.0, 2.0, 0.5]]],
        sequence_features=[[[1.0, 0.0]]],
        visual_features=[[[9.0, 9.0]]],
        input_lengths=[1],
    )
    profile = _profile(feature_layer="visual", feature_dim=2, a=_glyph([1.0, 0.0]))
    config = MemoryConfig(enabled=True, feature_layer="sequence", alpha=1.0)

    with pytest.raises(ValueError, match="feature"):
        personalize(output, profile, CHARSET, config)


def test_personalize_handles_a_batch_of_more_than_one():
    output = _output(
        logits=[
            [[1.0, 2.0, 0.5], [0.0, 0.0, 3.0]],
            [[0.5, 1.0, 2.0], [1.0, 1.0, 1.0]],
        ],
        sequence_features=[
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        visual_features=[
            [[9.0, 9.0], [9.0, 9.0]],
            [[9.0, 9.0], [9.0, 9.0]],
        ],
        input_lengths=[2, 2],
    )
    profile = _profile(a=_glyph([1.0, 0.0]))
    config = MemoryConfig(enabled=True, feature_layer="sequence", alpha=1.0, gate_by_emission=False)

    corrected = personalize(output, profile, CHARSET, config)

    assert corrected.shape == output.logits.shape
    assert not torch.equal(corrected, output.logits)


# ------------------------------------------------------------------ learned projection (M9)


def _tiny_projection() -> GlyphProjection:
    torch.manual_seed(0)
    return GlyphProjection(input_dim=2, hidden_dim=3, output_dim=2)


def test_personalize_applies_the_projection_before_retrieval():
    output = _output(
        logits=[[[1.0, 2.0, 0.5], [0.0, 0.0, 3.0]]],
        sequence_features=[[[1.0, 0.0], [0.0, 1.0]]],
        visual_features=[[[9.0, 9.0], [9.0, 9.0]]],
        input_lengths=[2],
    )
    projection = _tiny_projection()
    projected_prototype = projection(torch.tensor([1.0, 0.0]))
    profile = WriterProfile(
        schema_version="1",
        model_fingerprint="fp",
        feature_layer="learned_projection",
        feature_dim=2,
        glyphs={
            "a": Glyph(
                character="a",
                prototype=projected_prototype,
                number_of_observations=1,
                mean_alignment_confidence=1.0,
                feature_layer="learned_projection",
            )
        },
    )
    config = MemoryConfig(
        enabled=True, feature_layer="learned_projection", alpha=1.0, gate_by_emission=False,
    )

    corrected = personalize(output, profile, CHARSET, config, projection=projection)

    from glyphmemory.memory.retrieval import memory_scores

    projected_features = projection(output.sequence_features[0])
    expected = fuse(
        output.logits[0],
        memory_scores(projected_features, profile, CHARSET),
        alpha=1.0, gate_by_emission=False, protect_blank=True,
    )
    assert torch.allclose(corrected[0], expected, atol=1e-6)
    # And it must actually differ from what raw (unprojected) retrieval would have produced --
    # otherwise this test could pass even if personalize forgot to apply the projection at all.
    raw_scores = memory_scores(output.sequence_features[0], profile, CHARSET)
    raw_fused = fuse(output.logits[0], raw_scores, alpha=1.0, gate_by_emission=False)
    assert not torch.allclose(corrected[0], raw_fused, atol=1e-4)


def test_personalize_learned_projection_requires_a_projection():
    output = _output(
        logits=[[[1.0, 2.0, 0.5]]],
        sequence_features=[[[1.0, 0.0]]],
        visual_features=[[[9.0, 9.0]]],
        input_lengths=[1],
    )
    profile = _profile(feature_layer="learned_projection", a=_glyph([1.0, 0.0]))
    config = MemoryConfig(enabled=True, feature_layer="learned_projection", alpha=1.0)

    with pytest.raises(ValueError, match="requires a projection"):
        personalize(output, profile, CHARSET, config)


def test_personalize_rejects_a_projection_for_a_non_projected_layer():
    output = _output(
        logits=[[[1.0, 2.0, 0.5]]],
        sequence_features=[[[1.0, 0.0]]],
        visual_features=[[[9.0, 9.0]]],
        input_lengths=[1],
    )
    profile = _profile(a=_glyph([1.0, 0.0]))
    config = MemoryConfig(enabled=True, feature_layer="sequence", alpha=1.0)

    with pytest.raises(ValueError, match="does not use one"):
        personalize(output, profile, CHARSET, config, projection=_tiny_projection())
