"""Feature-space geometry primitives: NCM, distance structure, nearest-neighbour, retrieval."""

from __future__ import annotations

import torch

from glyphmemory.probes.geometry import (
    NCMResult,
    RetrievalComposition,
    class_means,
    cosine_distance_matrix,
    intra_inter_class_distance,
    l2_normalize,
    leave_one_out_nn_accuracy,
    ncm_accuracy,
    ncm_predict,
    retrieval_composition,
)

# --------------------------------------------------------------------------- l2_normalize


def test_l2_normalize_produces_unit_vectors():
    features = torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])
    normed = l2_normalize(features)
    norms = normed.norm(dim=-1)
    assert torch.allclose(norms[:2], torch.tensor([1.0, 1.0]), atol=1e-6)
    assert norms[2] == 0.0  # the zero vector stays zero, never divides by zero


# --------------------------------------------------------------------------- distance matrix


def test_cosine_distance_of_identical_vectors_is_zero():
    features = torch.tensor([[1.0, 2.0, 3.0]])
    distances = cosine_distance_matrix(features, features)
    assert torch.allclose(distances, torch.zeros(1, 1), atol=1e-6)


def test_cosine_distance_of_opposite_vectors_is_two():
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[-1.0, 0.0]])
    distances = cosine_distance_matrix(a, b)
    assert torch.allclose(distances, torch.tensor([[2.0]]), atol=1e-6)


def test_cosine_distance_of_orthogonal_vectors_is_one():
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.0, 1.0]])
    distances = cosine_distance_matrix(a, b)
    assert torch.allclose(distances, torch.tensor([[1.0]]), atol=1e-6)


# --------------------------------------------------------------------------- class means / NCM


def test_class_means_separates_two_obvious_clusters():
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = ["a", "a", "b", "b"]
    means = class_means(features, labels)
    assert set(means.keys()) == {"a", "b"}
    assert means["a"][0] > means["a"][1]
    assert means["b"][1] > means["b"][0]


def test_ncm_predict_picks_the_nearer_mean():
    means = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])}
    query = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    assert ncm_predict(query, means) == ["a", "b"]


def test_ncm_accuracy_perfect_separation():
    support_features = torch.tensor([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]])
    support_labels = ["a", "a", "b", "b"]
    query_features = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    query_labels = ["a", "b"]

    result = ncm_accuracy(support_features, support_labels, query_features, query_labels)
    assert isinstance(result, NCMResult)
    assert result.accuracy == 1.0
    assert result.n_query_scored == 2
    assert result.n_classes == 2


def test_ncm_accuracy_skips_query_labels_with_no_support():
    support_features = torch.tensor([[1.0, 0.0]])
    support_labels = ["a"]
    query_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    query_labels = ["a", "unseen"]

    result = ncm_accuracy(support_features, support_labels, query_features, query_labels)
    assert result.n_query_scored == 1
    assert result.n_query_total == 2
    assert result.accuracy == 1.0  # the one scorable example was correct


def test_ncm_accuracy_returns_none_when_nothing_is_scorable():
    support_features = torch.tensor([[1.0, 0.0]])
    support_labels = ["a"]
    query_features = torch.tensor([[0.0, 1.0]])
    query_labels = ["unseen"]

    result = ncm_accuracy(support_features, support_labels, query_features, query_labels)
    assert result.accuracy is None
    assert result.n_query_scored == 0


# --------------------------------------------------------------------------- intra/inter distance


def test_intra_inter_distance_is_smaller_within_class_for_separated_clusters():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.96, 0.04],
            [0.0, 1.0],
            [0.02, 0.98],
            [0.04, 0.96],
        ]
    )
    labels = ["a", "a", "a", "b", "b", "b"]
    intra, inter = intra_inter_class_distance(features, labels)
    assert intra is not None and inter is not None
    assert intra < inter


def test_intra_inter_distance_none_below_two_points():
    intra, inter = intra_inter_class_distance(torch.zeros(1, 4), ["a"])
    assert intra is None
    assert inter is None


# --------------------------------------------------------------------------- leave-one-out NN


def test_leave_one_out_nn_accuracy_perfect_for_tight_clusters():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ]
    )
    labels = ["a", "a", "b", "b"]
    accuracy = leave_one_out_nn_accuracy(features, labels)
    assert accuracy == 1.0


def test_leave_one_out_nn_accuracy_none_for_a_single_point():
    assert leave_one_out_nn_accuracy(torch.zeros(1, 3), ["a"]) is None


# --------------------------------------------------------------------------- retrieval composition


def test_retrieval_composition_sums_to_one():
    torch.manual_seed(0)
    features = torch.randn(20, 8)
    writers = [f"w{i % 4}" for i in range(20)]
    characters = [f"c{i % 3}" for i in range(20)]
    result = retrieval_composition(features, writers, characters, k=5, seed=0)
    assert isinstance(result, RetrievalComposition)
    total = (
        result.same_writer_same_character
        + result.same_character_different_writer
        + result.different_character
    )
    assert total == 1.0 or abs(total - 1.0) < 1e-9


def test_retrieval_composition_all_same_writer_same_character_when_uniform():
    features = torch.randn(6, 4)
    writers = ["w0"] * 6
    characters = ["a"] * 6
    result = retrieval_composition(features, writers, characters, k=3)
    assert result.same_writer_same_character == 1.0
    assert result.same_character_different_writer == 0.0
    assert result.different_character == 0.0


def test_retrieval_composition_none_below_two_points():
    assert retrieval_composition(torch.zeros(1, 4), ["w"], ["a"]) is None


# --------------------------------------------------------------------------- large-N safety
#
# A real probe run pools tens of thousands of occurrences. Functions that would otherwise
# materialize a full [N, N] distance matrix must subsample first — these tests exist because
# that subsampling was originally missing for `retrieval_composition`'s candidate pool and
# would have hung/OOM'd the first time this ran on real data instead of a small unit test.


def test_retrieval_composition_handles_a_pool_larger_than_the_cap():
    from glyphmemory.probes.geometry import MAX_POOL_SIZE

    torch.manual_seed(0)
    n = MAX_POOL_SIZE + 500
    features = torch.randn(n, 16)
    writers = [f"w{i % 10}" for i in range(n)]
    characters = [f"c{i % 5}" for i in range(n)]

    result = retrieval_composition(features, writers, characters, k=5, seed=1)

    assert result is not None
    assert result.n_queries <= n
    total = (
        result.same_writer_same_character
        + result.same_character_different_writer
        + result.different_character
    )
    assert abs(total - 1.0) < 1e-9


def test_retrieval_composition_labels_stay_aligned_to_features_after_subsampling():
    """The bug this guards against: subsampling `writer_labels` and `character_labels` independently
    (or not at all) desynchronizes them from `features`, so every neighbour lookup reads the wrong
    label. Construct a case where writer and character are perfectly correlated with the feature's
    own cluster, so misalignment would show up as a wrong (non-1.0) same-writer-same-character
    fraction.
    """
    from glyphmemory.probes.geometry import MAX_POOL_SIZE

    torch.manual_seed(0)
    n = MAX_POOL_SIZE + 200
    cluster_centers = torch.randn(20, 8) * 10  # far apart, tight clusters below
    cluster_id = [i % 20 for i in range(n)]
    features = torch.stack([cluster_centers[c] + torch.randn(8) * 0.01 for c in cluster_id])
    writers = [f"w{c}" for c in cluster_id]
    characters = [f"ch{c}" for c in cluster_id]

    result = retrieval_composition(features, writers, characters, k=3, seed=2)
    assert result is not None
    assert result.same_writer_same_character > 0.99


def test_leave_one_out_nn_accuracy_handles_a_pool_larger_than_the_cap():
    from glyphmemory.probes.geometry import MAX_POOL_SIZE

    torch.manual_seed(0)
    n = MAX_POOL_SIZE + 500
    cluster_centers = torch.randn(10, 8) * 10
    cluster_id = [i % 10 for i in range(n)]
    features = torch.stack([cluster_centers[c] + torch.randn(8) * 0.01 for c in cluster_id])
    labels = [f"c{c}" for c in cluster_id]

    accuracy = leave_one_out_nn_accuracy(features, labels, seed=3)
    assert accuracy is not None
    assert accuracy > 0.99  # tight, well-separated clusters even after subsampling
