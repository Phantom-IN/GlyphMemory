"""Span pooling: collapsing an `AlignmentSpan`'s frames into one feature vector.

Every strategy shares one signature — `(span, log_probs, features, class_index) -> Tensor[D]` — so
the compiler can dispatch by name via `POOLING_STRATEGIES` without a branch per caller. Strategies
that do not need every argument (`uniform`) still accept it, deliberately, so the compiler never has
to special-case a call.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from glyphmemory.alignment.spans import AlignmentSpan

#: Below this, a span's target-class posterior mass is too small to trust as a weighting — falls
#: back to a uniform average over the span rather than dividing by (near) zero.
MIN_WEIGHT_MASS = 1e-6


def posterior_weighted(
    span: AlignmentSpan, log_probs: Tensor, features: Tensor, class_index: int
) -> Tensor:
    """Weight each frame in the span by its posterior probability of emitting ``class_index``.

    A span with negligible total posterior mass for its own target class — possible on a
    low-confidence alignment — falls back to a uniform average rather than a near-zero-mass weighted
    mean that would be dominated by floating-point noise.
    """
    span_probs = log_probs[span.start_t : span.end_t, class_index].exp()
    total = span_probs.sum()
    span_features = features[span.start_t : span.end_t]
    if float(total.detach()) < MIN_WEIGHT_MASS:
        return span_features.mean(dim=0)
    weights = span_probs / total
    return (weights.unsqueeze(-1) * span_features).sum(dim=0)


def peak_frame(
    span: AlignmentSpan, log_probs: Tensor, features: Tensor, class_index: int
) -> Tensor:
    """The single frame within the span most confident about the target character.

    Argmax on log-probabilities is equivalent to argmax on probabilities (monotonic), so no
    ``exp()`` is needed just to find the peak.
    """
    span_log_probs = log_probs[span.start_t : span.end_t, class_index]
    peak_offset = int(torch.argmax(span_log_probs))
    return features[span.start_t + peak_offset].clone()


def uniform(span: AlignmentSpan, log_probs: Tensor, features: Tensor, class_index: int) -> Tensor:
    """Plain mean over every frame in the span — the naive baseline kept for the ablation.

    Ignores ``log_probs``/``class_index``, present only so every strategy in ``POOLING_STRATEGIES``
    shares one call signature.
    """
    return features[span.start_t : span.end_t].mean(dim=0)


PoolingStrategy = Callable[[AlignmentSpan, Tensor, Tensor, int], Tensor]

#: Name -> strategy, so `MemoryConfig.pooling` (a plain string, serializable in YAML) can select one
#: without the compiler branching on it.
POOLING_STRATEGIES: dict[str, PoolingStrategy] = {
    "posterior_weighted": posterior_weighted,
    "peak_frame": peak_frame,
    "uniform": uniform,
}
