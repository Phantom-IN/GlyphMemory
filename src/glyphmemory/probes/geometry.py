"""Feature-space geometry: the primitives the feasibility probe is built from.

Whether that mechanism *can* work is a measurable property of the feature space, not an assumption,
and everything here is generic linear-algebra over ``(features, labels)`` pairs — it does not know
or care whether a label is a character or a writer, so the same functions answer both the
character-separability question and the writer-separability question.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
from torch import Tensor

L = TypeVar("L", bound=Hashable)

#: Pairwise-distance computations are O(N^2); cap the sample size rather than let a large probe
#: silently take minutes on a quadratic blowup.
MAX_PAIRWISE_SAMPLE = 2000

#: Functions that materialize a full [N, N] distance matrix subsample down to this many points first
#: — at real dataset scale (tens of thousands of occurrences) the full matrix does not fit in
#: memory, let alone run quickly. `MAX_PAIRWISE_SAMPLE` bounds *pair* counts for functions that only
#: ever look at individual pairs; this bounds *pool size* for functions whose result depends on the
#: whole neighbourhood of each point.
MAX_POOL_SIZE = 1500


def _subsample(
    features: Tensor, labels: Sequence[L], max_n: int, seed: int
) -> tuple[Tensor, list[L]]:
    n = features.shape[0]
    if n <= max_n:
        return features, list(labels)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(n), max_n))
    return features[indices], [labels[i] for i in indices]


def l2_normalize(features: Tensor) -> Tensor:
    """Row-wise L2 normalization, with a floor so an all-zero row does not divide by zero."""
    norm = features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return features / norm


def cosine_distance_matrix(a: Tensor, b: Tensor) -> Tensor:
    """``[N, M]`` pairwise cosine *distance* (``1 - cosine_similarity``), in ``[0, 2]``."""
    a_norm = l2_normalize(a)
    b_norm = l2_normalize(b)
    return 1.0 - a_norm @ b_norm.T


def class_means(features: Tensor, labels: Sequence[L]) -> dict[L, Tensor]:
    """L2-normalized mean feature per distinct label — the "class mean" in nearest-class-mean."""
    normalized = l2_normalize(features)
    means: dict[L, Tensor] = {}
    for label in set(labels):
        mask = torch.tensor([item == label for item in labels])
        means[label] = l2_normalize(normalized[mask].mean(dim=0, keepdim=True)).squeeze(0)
    return means


def ncm_predict(query_features: Tensor, means: dict[L, Tensor]) -> list[L]:
    """Nearest class mean by cosine similarity, over exactly the classes in ``means`` — predicting a
    class with zero support representatives is not possible by construction.
    """
    labels = list(means.keys())
    mean_matrix = torch.stack([means[label] for label in labels])
    distances = cosine_distance_matrix(query_features, mean_matrix)
    nearest = distances.argmin(dim=-1)
    return [labels[i] for i in nearest.tolist()]


@dataclass(frozen=True, slots=True)
class NCMResult:
    """Nearest-class-mean accuracy, with exactly how much of the query set could be scored.

    ``n_query_scored`` can be less than ``n_query_total``: a query example whose label never appears
    in the support set cannot be evaluated — there is no class mean to compare against, and
    inventing one would misrepresent what was actually measured.
    """

    accuracy: float | None
    n_query_scored: int
    n_query_total: int
    n_classes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "n_query_scored": self.n_query_scored,
            "n_query_total": self.n_query_total,
            "n_classes": self.n_classes,
        }


def ncm_accuracy(
    support_features: Tensor,
    support_labels: Sequence[L],
    query_features: Tensor,
    query_labels: Sequence[L],
) -> NCMResult:
    """Fit class means from the support set, score the query set against them."""
    means = class_means(support_features, support_labels)
    scorable = [label in means for label in query_labels]
    n_scored = sum(scorable)
    if n_scored == 0:
        return NCMResult(
            accuracy=None, n_query_scored=0, n_query_total=len(query_labels), n_classes=len(means)
        )

    mask = torch.tensor(scorable)
    predictions = ncm_predict(query_features[mask], means)
    scored_labels = [label for label, keep in zip(query_labels, scorable, strict=True) if keep]
    correct = sum(p == truth for p, truth in zip(predictions, scored_labels, strict=True))
    return NCMResult(
        accuracy=correct / n_scored,
        n_query_scored=n_scored,
        n_query_total=len(query_labels),
        n_classes=len(means),
    )


def _sampled_pair_indices(n: int, rng: random.Random, max_pairs: int) -> list[tuple[int, int]]:
    all_pairs_count = n * (n - 1) // 2
    if all_pairs_count <= max_pairs:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < max_pairs:
        i, j = rng.randrange(n), rng.randrange(n)
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    return list(pairs)


def intra_inter_class_distance(
    features: Tensor, labels: Sequence[L], *, seed: int = 0
) -> tuple[float | None, float | None]:
    """Mean intra-class and mean inter-class cosine distance.

    Real writer/character structure predicts intra < inter: a writer's own repeated glyph should sit
    closer to its own kind than to other characters. Pairs are sampled rather than enumerated
    exhaustively once ``N`` makes that quadratic (``MAX_PAIRWISE_SAMPLE``).
    """
    n = features.shape[0]
    if n < 2:
        return None, None
    rng = random.Random(seed)
    pairs = _sampled_pair_indices(n, rng, MAX_PAIRWISE_SAMPLE)

    # Distances for exactly the sampled pairs, never the full [N, N] matrix — at real dataset scale
    # (tens of thousands of occurrences) materializing that matrix is not just wasteful, it does not
    # fit in memory at all.
    normalized = l2_normalize(features)
    left = torch.stack([normalized[i] for i, _ in pairs])
    right = torch.stack([normalized[j] for _, j in pairs])
    pair_distances = (1.0 - (left * right).sum(dim=-1)).tolist()

    intra: list[float] = []
    inter: list[float] = []
    for (i, j), d in zip(pairs, pair_distances, strict=True):
        (intra if labels[i] == labels[j] else inter).append(d)

    intra_mean = sum(intra) / len(intra) if intra else None
    inter_mean = sum(inter) / len(inter) if inter else None
    return intra_mean, inter_mean


def leave_one_out_nn_accuracy(
    features: Tensor, labels: Sequence[L], *, seed: int = 0
) -> float | None:
    """1-nearest-neighbour classification accuracy, leave-one-out, by cosine distance.

    A pointier test than NCM: it asks whether *individual* occurrences cluster by label, not just
    whether the label's mean is in the right neighbourhood. Subsamples to ``MAX_POOL_SIZE`` first
    when given more than that — this builds a full pairwise distance matrix, which does not scale to
    a real dataset's occurrence count unbounded.
    """
    features, labels = _subsample(features, labels, MAX_POOL_SIZE, seed)
    n = features.shape[0]
    if n < 2:
        return None
    distances = cosine_distance_matrix(features, features)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.argmin(dim=-1)
    correct = sum(labels[i] == labels[int(nearest[i])] for i in range(n))
    return correct / n


@dataclass(frozen=True, slots=True)
class RetrievalComposition:
    """What a query occurrence's nearest neighbours actually are, on average.

    The three fractions sum to 1.0 (every neighbour is exactly one of them).
    """

    same_writer_same_character: float
    same_character_different_writer: float
    different_character: float
    k: int
    n_queries: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "same_writer_same_character": self.same_writer_same_character,
            "same_character_different_writer": self.same_character_different_writer,
            "different_character": self.different_character,
            "k": self.k,
            "n_queries": self.n_queries,
        }


def retrieval_composition(
    features: Tensor,
    writer_labels: Sequence[str],
    character_labels: Sequence[str],
    *,
    k: int = 5,
    seed: int = 0,
    max_queries: int = 500,
) -> RetrievalComposition | None:
    """Average composition of each query's ``k`` nearest neighbours.

    The candidate pool is subsampled to ``MAX_POOL_SIZE`` first (this builds a full pairwise
    distance matrix over it, which does not scale to a real dataset's occurrence count unbounded),
    then queries are drawn from within that pool, capped at ``max_queries``.
    """
    total = features.shape[0]
    if total > MAX_POOL_SIZE:
        pool_rng = random.Random(seed)
        keep = sorted(pool_rng.sample(range(total), MAX_POOL_SIZE))
        features = features[keep]
        writer_labels = [writer_labels[i] for i in keep]
        character_labels = [character_labels[i] for i in keep]
    n = features.shape[0]
    if n < 2:
        return None
    k = min(k, n - 1)

    rng = random.Random(seed)
    query_indices = list(range(n))
    if n > max_queries:
        query_indices = rng.sample(query_indices, max_queries)

    distances = cosine_distance_matrix(features, features)
    distances.fill_diagonal_(float("inf"))

    same_writer_same_char = same_char_diff_writer = diff_char = 0.0
    for query in query_indices:
        neighbours = torch.topk(distances[query], k, largest=False).indices.tolist()
        for neighbour in neighbours:
            if character_labels[neighbour] != character_labels[query]:
                diff_char += 1
            elif writer_labels[neighbour] == writer_labels[query]:
                same_writer_same_char += 1
            else:
                same_char_diff_writer += 1

    total = len(query_indices) * k
    return RetrievalComposition(
        same_writer_same_character=same_writer_same_char / total,
        same_character_different_writer=same_char_diff_writer / total,
        different_character=diff_char / total,
        k=k,
        n_queries=len(query_indices),
    )
