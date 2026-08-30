"""Per-character prototype accumulation.

    prototype(c) = L2_normalize( mean( normalized occurrences of c ) )

Each pooled span vector is normalized *before* averaging, then the mean itself is normalized again —
the same two-stage recipe `probes/geometry.py`'s `class_means` already implements for offline
diagnostics. This module is the online, streaming version: one line's spans arrive at a time during
enrollment, not as one big batch tensor, so a running accumulator is the right shape rather than
reusing `class_means` directly.

**An unobserved character produces no entry, never a zero vector.** Inventing evidence for a
character nobody wrote is exactly what forbids for retrieval, and it would be no less wrong to bake
that invention into the profile one step earlier.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from glyphmemory.memory.pooling import MIN_WEIGHT_MASS
from glyphmemory.probes.geometry import l2_normalize

#: `MemoryConfig.top_k` default when a caller doesn't override it.
DEFAULT_TOP_K = 3


@dataclass(slots=True)
class GlyphAccumulator:
    """Running per-character sum of normalized vectors, confidence and observation count.

    Mutable by design — the compiler calls :meth:`observe` once per pooled span while walking a
    writer's enrollment lines, then :meth:`finalize` once at the end. Nothing here knows about
    images, models or alignment; it operates on `(character, vector, confidence)` triples exactly
    like `probes/geometry.py`'s functions operate on plain `(features, labels)`.
    """

    _sums: dict[str, Tensor] = field(default_factory=dict)
    _confidence_sums: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    @property
    def characters(self) -> frozenset[str]:
        return frozenset(self._counts)

    def count(self, character: str) -> int:
        return self._counts.get(character, 0)

    def observe(self, character: str, vector: Tensor, confidence: float) -> None:
        """Fold one pooled span occurrence of ``character`` into its running accumulator."""
        normalized = l2_normalize(vector.unsqueeze(0)).squeeze(0)
        if character not in self._sums:
            self._sums[character] = torch.zeros_like(normalized)
            self._confidence_sums[character] = 0.0
            self._counts[character] = 0
        self._sums[character] = self._sums[character] + normalized
        self._confidence_sums[character] += confidence
        self._counts[character] += 1

    def finalize(self) -> dict[str, tuple[Tensor, int, float]]:
        """One ``(prototype, observation_count, mean_confidence)`` per observed character.

        A character never passed to :meth:`observe` is simply absent from the result — never
        synthesized.
        """
        result: dict[str, tuple[Tensor, int, float]] = {}
        for character, count in self._counts.items():
            mean_vector = self._sums[character] / count
            prototype = l2_normalize(mean_vector.unsqueeze(0)).squeeze(0)
            mean_confidence = self._confidence_sums[character] / count
            result[character] = (prototype, count, mean_confidence)
        return result


# ------------------------------------------------------------------ prototype variants
#
# The named post-V0 ablation list: confidence-weighted mean, medoid, and top-K. None of
# the three are expressible as a running sum the way V0's plain mean
# is (confidence-weighting needs the per-occurrence weights kept alongside the vectors; medoid
# and top-K need the individual occurrence vectors themselves, not just their sum) -- so unlike
# `GlyphAccumulator`, `PrototypeAccumulator` stores every observed occurrence per character
# rather than folding each one into a running total. This is bounded by support size (a writer's
# enrollment set is a handful to tens of lines, not the diagnostic-scale data
# `probes/geometry.py`'s `MAX_POOL_SIZE` guards against), so the extra memory is not a concern in
# practice -- stated explicitly here rather than left implicit (this phase's own risk table).


def _stack(vectors: Sequence[Tensor]) -> Tensor:
    return torch.stack(list(vectors))


def _mean_strategy(
    vectors: Sequence[Tensor], confidences: Sequence[float], *, top_k: int
) -> tuple[Tensor, ...]:
    """V0's own formula, recomputed from stored occurrences instead of a running sum -- kept here
    (rather than only in `GlyphAccumulator`) so `PrototypeAccumulator` can serve as a correctness
    cross-check against it (`test_memory_prototypes.py`), and so every strategy in
    `PROTOTYPE_STRATEGIES` shares one call signature.
    """
    mean_vector = _stack(vectors).mean(dim=0)
    return (l2_normalize(mean_vector.unsqueeze(0)).squeeze(0),)


def _confidence_weighted_strategy(
    vectors: Sequence[Tensor], confidences: Sequence[float], *, top_k: int
) -> tuple[Tensor, ...]:
    """Weight each occurrence's contribution by its alignment confidence (confidence tracks
    correctness, Pearson r = -0.997 against per-line CER) -- occurrences the aligner trusted less
    contribute less to the prototype. Falls back to a uniform mean when the total confidence mass is
    negligible, the same guard `memory/pooling.py::posterior_weighted` uses for the same reason:
    dividing by (near) zero would be dominated by floating-point noise, not a meaningful weighting.
    """
    stacked = _stack(vectors)
    weights = torch.tensor(confidences, dtype=stacked.dtype)
    total = float(weights.sum())
    if total < MIN_WEIGHT_MASS:
        mean_vector = stacked.mean(dim=0)
    else:
        mean_vector = (weights.unsqueeze(-1) * stacked).sum(dim=0) / total
    return (l2_normalize(mean_vector.unsqueeze(0)).squeeze(0),)


def _medoid_strategy(
    vectors: Sequence[Tensor], confidences: Sequence[float], *, top_k: int
) -> tuple[Tensor, ...]:
    """The observed occurrence closest (by cosine similarity) to the class mean -- a prototype
    guaranteed to be a real observed instance, which a mean is not. Occurrences are already
    L2-normalized at `observe` time, so plain dot product against the (also normalized) mean is
    cosine similarity.
    """
    stacked = _stack(vectors)
    mean_vector = l2_normalize(stacked.mean(dim=0, keepdim=True)).squeeze(0)
    similarities = stacked @ mean_vector
    best_index = int(torch.argmax(similarities))
    return (stacked[best_index].clone(),)


def _top_k_strategy(
    vectors: Sequence[Tensor], confidences: Sequence[float], *, top_k: int
) -> tuple[Tensor, ...]:
    """The ``top_k`` highest-confidence occurrences, kept individually rather than collapsed into
    one vector -- the simplest defensible choice named in this phase's own work plan (K-means-style
    cluster centers are explicitly deferred unless this simpler option proves insufficient). Fewer
    than ``top_k`` occurrences observed is not an error: every occurrence is kept, same as
    everywhere else in this project's "never invent, never pad" discipline. ``sorted`` is stable, so
    tied confidences keep their original observation order rather than an arbitrary one.
    """
    ranked = sorted(range(len(vectors)), key=lambda i: confidences[i], reverse=True)
    return tuple(vectors[i].clone() for i in ranked[:top_k])


PrototypeStrategy = Callable[[Sequence[Tensor], Sequence[float], int], tuple[Tensor, ...]]

#: Name -> strategy, so `MemoryConfig.prototype_strategy` (a plain string, serializable in YAML) can
#: select one without `compile_profile` branching on it -- the same dispatch shape
#: `memory/pooling.py`'s `POOLING_STRATEGIES` and `memory/compiler.py`'s `FEATURE_ATTRIBUTES`
#: already use.
PROTOTYPE_STRATEGIES: dict[str, PrototypeStrategy] = {
    "mean": _mean_strategy,
    "confidence_weighted": _confidence_weighted_strategy,
    "medoid": _medoid_strategy,
    "top_k": _top_k_strategy,
}


@dataclass(slots=True)
class PrototypeAccumulator:
    """Stores every observed occurrence per character, so `finalize` can compute any strategy in
    `PROTOTYPE_STRATEGIES` -- including ones (confidence-weighted, medoid, top-K) that are not
    expressible as a running sum.
    """

    _vectors: dict[str, list[Tensor]] = field(default_factory=dict)
    _confidences: dict[str, list[float]] = field(default_factory=dict)

    @property
    def characters(self) -> frozenset[str]:
        return frozenset(self._vectors)

    def count(self, character: str) -> int:
        return len(self._vectors.get(character, ()))

    def observe(self, character: str, vector: Tensor, confidence: float) -> None:
        """Fold one pooled span occurrence of ``character`` into its stored occurrence list."""
        normalized = l2_normalize(vector.unsqueeze(0)).squeeze(0)
        self._vectors.setdefault(character, []).append(normalized)
        self._confidences.setdefault(character, []).append(confidence)

    def finalize(
        self, strategy: str, *, top_k: int = DEFAULT_TOP_K
    ) -> dict[str, tuple[tuple[Tensor, ...], int, float]]:
        """One ``(prototypes, observation_count, mean_confidence)`` per observed character.

        ``prototypes`` is a non-empty tuple: length 1 for every strategy except ``top_k`` (length
        ``min(top_k, observation_count)``). A character never passed to :meth:`observe` is simply
        absent — never synthesized, matching `GlyphAccumulator.finalize`.

        Raises:
            ValueError: ``strategy`` is not a key of `PROTOTYPE_STRATEGIES`.
        """
        if strategy not in PROTOTYPE_STRATEGIES:
            raise ValueError(
                f"Unknown prototype_strategy {strategy!r}; "
                f"expected one of {sorted(PROTOTYPE_STRATEGIES)}."
            )
        strategy_fn = PROTOTYPE_STRATEGIES[strategy]

        result: dict[str, tuple[tuple[Tensor, ...], int, float]] = {}
        for character, vectors in self._vectors.items():
            confidences = self._confidences[character]
            prototypes = strategy_fn(vectors, confidences, top_k=top_k)
            count = len(vectors)
            mean_confidence = sum(confidences) / count
            result[character] = (prototypes, count, mean_confidence)
        return result
