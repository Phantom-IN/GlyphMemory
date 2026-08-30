"""`memory_scores`: cosine similarity against stored prototypes, masked to 0.0 for any character the
profile has no evidence for.
"""

from __future__ import annotations

import pytest
import torch

from glyphmemory.ctc.tokenizer import Charset
from glyphmemory.memory.profile import Glyph, WriterProfile
from glyphmemory.memory.retrieval import memory_scores

# blank, a, b, c
CHARSET = Charset(symbols=("<blank>", "a", "b", "c"))


def _glyph(character: str, prototype: list[float]) -> Glyph:
    vector = torch.tensor(prototype, dtype=torch.float32)
    return Glyph(
        character=character,
        prototype=vector / vector.norm(),
        number_of_observations=3,
        mean_alignment_confidence=0.9,
        feature_layer="sequence",
    )


def _profile(**glyphs: Glyph) -> WriterProfile:
    return WriterProfile(
        schema_version="1",
        model_fingerprint="fp",
        feature_layer="sequence",
        feature_dim=2,
        glyphs=glyphs,
    )


def test_scores_match_hand_computed_cosine_similarity():
    profile = _profile(a=_glyph("a", [1.0, 0.0]))
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    scores = memory_scores(features, profile, CHARSET)

    a_index = CHARSET.index_of("a")
    expected = torch.tensor([1.0, 0.0, 1.0 / (2**0.5)])
    assert torch.allclose(scores[:, a_index], expected, atol=1e-5)


def test_characters_absent_from_the_profile_score_zero():
    profile = _profile(a=_glyph("a", [1.0, 0.0]))
    features = torch.tensor([[1.0, 0.0]])

    scores = memory_scores(features, profile, CHARSET)

    assert scores[0, CHARSET.index_of("b")] == 0.0
    assert scores[0, CHARSET.index_of("c")] == 0.0


def test_blank_column_is_always_zero():
    profile = _profile(a=_glyph("a", [1.0, 0.0]), b=_glyph("b", [0.0, 1.0]))
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    scores = memory_scores(features, profile, CHARSET)

    assert torch.equal(scores[:, CHARSET.blank], torch.zeros(2))


def test_empty_profile_is_all_zero():
    profile = _profile()
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    scores = memory_scores(features, profile, CHARSET)

    assert torch.equal(scores, torch.zeros(2, CHARSET.size))


def test_output_shape_matches_charset_size():
    profile = _profile(a=_glyph("a", [1.0, 0.0]))
    features = torch.zeros((5, 2))

    scores = memory_scores(features, profile, CHARSET)

    assert scores.shape == (5, CHARSET.size)


def test_rejects_non_2d_features():
    profile = _profile(a=_glyph("a", [1.0, 0.0]))
    with pytest.raises(ValueError, match=r"\[T, D\]"):
        memory_scores(torch.zeros(2, 3, 2), profile, CHARSET)


def test_rejects_mismatched_feature_dim():
    profile = _profile(a=_glyph("a", [1.0, 0.0]))  # feature_dim=2
    features = torch.zeros((3, 5))
    with pytest.raises(ValueError, match=r"feature_dim|dim 5"):
        memory_scores(features, profile, CHARSET)


def test_multiple_characters_score_independently():
    profile = _profile(a=_glyph("a", [1.0, 0.0]), b=_glyph("b", [0.0, 1.0]))
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    scores = memory_scores(features, profile, CHARSET)

    a_index, b_index = CHARSET.index_of("a"), CHARSET.index_of("b")
    assert torch.allclose(scores[:, a_index], torch.tensor([1.0, 0.0]), atol=1e-5)
    assert torch.allclose(scores[:, b_index], torch.tensor([0.0, 1.0]), atol=1e-5)


def test_scores_are_invariant_to_feature_vector_scale():
    # Cosine similarity, not dot product: scaling a frame's feature must not change its score.
    profile = _profile(a=_glyph("a", [1.0, 0.0]))
    small = memory_scores(torch.tensor([[0.1, 0.0]]), profile, CHARSET)
    large = memory_scores(torch.tensor([[100.0, 0.0]]), profile, CHARSET)

    assert torch.allclose(small, large, atol=1e-5)


# ------------------------------------------------------------------ multi-prototype (M9, P26 top-K)


def _multi_glyph(character: str, prototypes: list[list[float]]) -> Glyph:
    vectors = [torch.tensor(p, dtype=torch.float32) for p in prototypes]
    normalized = [v / v.norm() for v in vectors]
    return Glyph(
        character=character,
        prototype=normalized[0],
        additional_prototypes=tuple(normalized[1:]),
        number_of_observations=3,
        mean_alignment_confidence=0.9,
        feature_layer="sequence",
    )


def test_a_character_with_no_additional_prototypes_behaves_exactly_as_before():
    single = _profile(a=_glyph("a", [1.0, 0.0]))
    multi = _profile(a=_multi_glyph("a", [[1.0, 0.0]]))  # one prototype, via the new code path
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    single_scores = memory_scores(features, single, CHARSET)
    multi_scores = memory_scores(features, multi, CHARSET)
    assert torch.equal(single_scores, multi_scores)


def test_multiple_prototypes_score_as_the_best_match_not_an_average():
    # Two prototypes for "a": one aligned with the query frame, one orthogonal to it. The best match
    # (cosine similarity 1.0) must win, not the average of 1.0 and 0.0.
    profile = _profile(a=_multi_glyph("a", [[1.0, 0.0], [0.0, 1.0]]))
    features = torch.tensor([[1.0, 0.0]])

    scores = memory_scores(features, profile, CHARSET)

    assert torch.isclose(scores[0, CHARSET.index_of("a")], torch.tensor(1.0), atol=1e-5)


def test_multiple_prototypes_per_character_score_independently_per_frame():
    profile = _profile(a=_multi_glyph("a", [[1.0, 0.0], [0.0, 1.0]]))
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    scores = memory_scores(features, profile, CHARSET)
    a_index = CHARSET.index_of("a")

    assert torch.isclose(scores[0, a_index], torch.tensor(1.0), atol=1e-5)  # matches prototype 0
    assert torch.isclose(scores[1, a_index], torch.tensor(1.0), atol=1e-5)  # matches prototype 1
    # [-1, 0] has cosine -1 against prototype 0 and 0 against prototype 1 -- the max (best-of-K) is
    # 0, not the -1 either single prototype alone would score.
    assert torch.isclose(scores[2, a_index], torch.tensor(0.0), atol=1e-5)


def test_a_character_with_additional_prototypes_does_not_affect_another_characters_score():
    profile = _profile(
        a=_multi_glyph("a", [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]),
        b=_glyph("b", [0.0, 1.0]),
    )
    features = torch.tensor([[0.0, 1.0]])

    scores = memory_scores(features, profile, CHARSET)

    assert torch.isclose(scores[0, CHARSET.index_of("b")], torch.tensor(1.0), atol=1e-5)
