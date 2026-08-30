"""Span pooling strategies: hand-built spans with a known expected vector, the same rigor used for
the aligner itself (4.1).
"""

from __future__ import annotations

import torch

from glyphmemory.alignment.spans import AlignmentSpan
from glyphmemory.memory.pooling import (
    MIN_WEIGHT_MASS,
    POOLING_STRATEGIES,
    peak_frame,
    posterior_weighted,
    uniform,
)

CLASS_INDEX = 1


def _log_probs(class1_probs: list[float]) -> torch.Tensor:
    """One row per frame: ``[P(blank), P(class1)]``, already log'd."""
    rows = [[1.0 - p, p] for p in class1_probs]
    return torch.log(torch.tensor(rows))


def test_posterior_weighted_matches_hand_computed_weights():
    # frame 0 outside the span; frames 1-2 in the span with P(class1) = 0.2, 0.8.
    log_probs = _log_probs([0.99, 0.2, 0.8, 0.99])
    features = torch.tensor([[9.0, 9.0], [1.0, 0.0], [0.0, 1.0], [9.0, 9.0]])
    span = AlignmentSpan(token="a", start_t=1, end_t=3, score=0.9)

    vector = posterior_weighted(span, log_probs, features, CLASS_INDEX)

    # weights = [0.2, 0.8] / 1.0 -> 0.2*[1,0] + 0.8*[0,1]
    assert torch.allclose(vector, torch.tensor([0.2, 0.8]), atol=1e-5)


def test_posterior_weighted_falls_back_to_uniform_on_negligible_mass():
    log_probs = _log_probs([0.99, 1e-30, 1e-30, 0.99])
    features = torch.tensor([[9.0, 9.0], [1.0, 0.0], [0.0, 1.0], [9.0, 9.0]])
    span = AlignmentSpan(token="a", start_t=1, end_t=3, score=0.01)

    vector = posterior_weighted(span, log_probs, features, CLASS_INDEX)

    assert torch.allclose(vector, torch.tensor([0.5, 0.5]), atol=1e-5)


def test_posterior_weighted_fallback_threshold_is_named_not_magic():
    # Documents the guard's boundary rather than re-deriving it: total mass exactly at the threshold
    # is treated as "trust it", only strictly below falls back.
    assert MIN_WEIGHT_MASS > 0


def test_peak_frame_selects_the_single_most_confident_frame():
    log_probs = _log_probs([0.99, 0.2, 0.8, 0.99])
    features = torch.tensor([[9.0, 9.0], [1.0, 0.0], [0.0, 1.0], [9.0, 9.0]])
    span = AlignmentSpan(token="a", start_t=1, end_t=3, score=0.9)

    vector = peak_frame(span, log_probs, features, CLASS_INDEX)

    assert torch.equal(vector, torch.tensor([0.0, 1.0]))


def test_peak_frame_ties_take_the_first_frame_argmax_convention():
    log_probs = _log_probs([0.99, 0.5, 0.5, 0.99])
    features = torch.tensor([[9.0, 9.0], [1.0, 0.0], [0.0, 1.0], [9.0, 9.0]])
    span = AlignmentSpan(token="a", start_t=1, end_t=3, score=0.5)

    vector = peak_frame(span, log_probs, features, CLASS_INDEX)

    assert torch.equal(vector, torch.tensor([1.0, 0.0]))


def test_uniform_is_the_plain_mean_over_the_span():
    log_probs = _log_probs([0.99, 0.2, 0.8, 0.99])
    features = torch.tensor([[9.0, 9.0], [1.0, 0.0], [0.0, 1.0], [9.0, 9.0]])
    span = AlignmentSpan(token="a", start_t=1, end_t=3, score=0.9)

    vector = uniform(span, log_probs, features, CLASS_INDEX)

    assert torch.allclose(vector, torch.tensor([0.5, 0.5]))


def test_uniform_ignores_the_posterior_entirely():
    # Same features, wildly different posteriors -> identical uniform output.
    features = torch.tensor([[9.0, 9.0], [1.0, 0.0], [0.0, 1.0], [9.0, 9.0]])
    span = AlignmentSpan(token="a", start_t=1, end_t=3, score=0.9)

    a = uniform(span, _log_probs([0.99, 0.01, 0.99, 0.99]), features, CLASS_INDEX)
    b = uniform(span, _log_probs([0.99, 0.99, 0.01, 0.99]), features, CLASS_INDEX)

    assert torch.allclose(a, b)


def test_single_frame_span_all_strategies_agree():
    log_probs = _log_probs([0.99, 0.7, 0.99])
    features = torch.tensor([[9.0, 9.0], [3.0, 4.0], [9.0, 9.0]])
    span = AlignmentSpan(token="a", start_t=1, end_t=2, score=0.7)

    for strategy in POOLING_STRATEGIES.values():
        vector = strategy(span, log_probs, features, CLASS_INDEX)
        assert torch.allclose(vector, torch.tensor([3.0, 4.0]))


def test_pooling_strategies_registry_has_exactly_the_three_named_strategies():
    assert set(POOLING_STRATEGIES) == {"posterior_weighted", "peak_frame", "uniform"}


def test_the_three_strategies_produce_visibly_different_vectors_on_a_skewed_span():
    log_probs = _log_probs([0.99, 0.05, 0.95, 0.99])
    features = torch.tensor([[9.0, 9.0], [1.0, 0.0], [0.0, 1.0], [9.0, 9.0]])
    span = AlignmentSpan(token="a", start_t=1, end_t=3, score=0.9)

    vectors = {
        name: strategy(span, log_probs, features, CLASS_INDEX)
        for name, strategy in POOLING_STRATEGIES.items()
    }

    assert not torch.allclose(vectors["posterior_weighted"], vectors["uniform"])
    assert not torch.allclose(vectors["peak_frame"], vectors["uniform"])
    assert not torch.allclose(vectors["posterior_weighted"], vectors["peak_frame"])
