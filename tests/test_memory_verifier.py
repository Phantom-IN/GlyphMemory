"""Two tests here carry more weight than the rest. ``test_parameter_count_matches_the_registration``
holds the pre-registered 284,752 so a widened architecture cannot quietly ship as the registered
one.
"""

from __future__ import annotations

import pytest
import torch

from glyphmemory.memory.verifier import (
    DEFAULT_EMBED,
    GlyphVerifier,
    char_loss,
    compile_character_embeddings,
    score_candidates,
    writer_loss,
)


@pytest.fixture
def verifier() -> GlyphVerifier:
    torch.manual_seed(0)
    return GlyphVerifier()


class TestArchitecture:
    def test_parameter_count_matches_the_registration(self, verifier: GlyphVerifier) -> None:
        """284,752 is pre-registered. A different number is a different experiment."""
        assert verifier.parameter_count == 284_752

    def test_inside_the_stated_envelope(self, verifier: GlyphVerifier) -> None:
        assert 150_000 <= verifier.parameter_count <= 350_000

    def test_system_total_stays_under_the_research_objective(
        self, verifier: GlyphVerifier
    ) -> None:
        """Objective 1 caps the *system* at 3M; GM-Base is 1,544,560."""
        assert 1_544_560 + verifier.parameter_count < 3_000_000

    def test_shape(self, verifier: GlyphVerifier) -> None:
        assert verifier(torch.zeros(5, 2, 64, 40)).shape == (5, DEFAULT_EMBED)

    def test_output_is_always_unit_norm(self, verifier: GlyphVerifier) -> None:
        """Every consumer compares by cosine; an unnormalized row would silently rescale it."""
        out = verifier(torch.randn(8, 2, 64, 40))
        assert torch.allclose(out.norm(dim=-1), torch.ones(8), atol=1e-5)

    def test_accepts_the_widths_the_line_pipeline_produces(self, verifier: GlyphVerifier) -> None:
        assert verifier(torch.zeros(2, 2, 64, 32)).shape == (2, DEFAULT_EMBED)

    def test_rejects_wrong_channel_count(self, verifier: GlyphVerifier) -> None:
        with pytest.raises(ValueError, match=r"expected \[B, 2"):
            verifier(torch.zeros(4, 1, 64, 40))

    def test_rejects_unbatched_input(self, verifier: GlyphVerifier) -> None:
        with pytest.raises(ValueError):
            verifier(torch.zeros(2, 64, 40))

    def test_no_pretrained_weights_are_loaded(self) -> None:
        """Invariant 2. Two instances differ, so construction cannot be loading a fixture."""
        torch.manual_seed(1)
        a = GlyphVerifier()
        torch.manual_seed(2)
        b = GlyphVerifier()
        assert not torch.equal(a.proj.weight, b.proj.weight)


class TestEnrollment:
    def test_produces_one_unit_norm_embedding_per_observed_character(
        self, verifier: GlyphVerifier
    ) -> None:
        crops = torch.randn(6, 2, 64, 40)
        profile = compile_character_embeddings(verifier, crops, list("aabbcc"))
        assert sorted(profile) == ["a", "b", "c"]
        for embedding in profile.values():
            assert embedding.norm() == pytest.approx(1.0, abs=1e-5)

    def test_enrollment_takes_no_gradients(self, verifier: GlyphVerifier) -> None:
        """Invariant 4, asserted rather than assumed: no parameter may accumulate a gradient."""
        for parameter in verifier.parameters():
            parameter.grad = None
        profile = compile_character_embeddings(verifier, torch.randn(4, 2, 64, 40), list("abab"))
        assert all(p.grad is None for p in verifier.parameters())
        assert all(not e.requires_grad for e in profile.values())

    def test_unobserved_characters_are_absent_rather_than_zero(
        self, verifier: GlyphVerifier
    ) -> None:
        """A zero vector has cosine 0 against everything and would read as 'somewhat similar'."""
        profile = compile_character_embeddings(verifier, torch.randn(2, 2, 64, 40), ["a", "a"])
        assert "z" not in profile

    def test_batching_does_not_change_the_result(self, verifier: GlyphVerifier) -> None:
        crops = torch.randn(10, 2, 64, 40)
        chars = list("abcabcabca")
        one = compile_character_embeddings(verifier, crops, chars, batch_size=64)
        many = compile_character_embeddings(verifier, crops, chars, batch_size=3)
        for key in one:
            assert torch.allclose(one[key], many[key], atol=1e-6)

    def test_mismatched_lengths_raise(self, verifier: GlyphVerifier) -> None:
        with pytest.raises(ValueError, match="crops but"):
            compile_character_embeddings(verifier, torch.randn(3, 2, 64, 40), ["a"])

    def test_empty_support(self, verifier: GlyphVerifier) -> None:
        assert compile_character_embeddings(verifier, torch.zeros(0, 2, 64, 40), []) == {}


class TestScoreCandidates:
    def test_absent_candidates_are_omitted_not_sentinel_scored(self) -> None:
        """Internal helper."""
        query = torch.tensor([1.0, 0.0])
        profile = {"a": torch.tensor([1.0, 0.0])}
        scores = score_candidates(query, profile, ["a", "b"])
        assert set(scores) == {"a"}
        assert scores["a"] == pytest.approx(1.0)


class TestCharLoss:
    def test_tracks_the_inference_decision_it_is_meant_to_train(self) -> None:
        """The property that makes this the specified loss.

        The repair is not a cleverer arithmetic combination at inference; it is training the
        representation on the decision that is actually made. So the assertion is that this loss is
        a monotone function of the target's margin over its best competitor, and is minimised
        exactly when the inference argmax is right.

        Cosines are set exactly: with a unit query along e1 and prototypes ``[c, sqrt(1-c^2), 0]``,
        the cosine against the query is exactly ``c``.
        """

        def prototypes_with_cosines(cosines: list[float]) -> torch.Tensor:
            rows = [
                [c, float(torch.sqrt(torch.tensor(1.0 - c * c))), 0.0] for c in cosines
            ]
            return torch.tensor(rows).unsqueeze(0)

        query = torch.tensor([[1.0, 0.0, 0.0]])
        target = torch.tensor([0])
        mask = torch.ones(1, 3, dtype=torch.bool)

        losses = [
            float(char_loss(query, prototypes_with_cosines([0.9, r, r]), target, mask))
            for r in (0.85, 0.6, 0.3, 0.0)
        ]
        assert losses == sorted(losses, reverse=True), losses
        assert losses[-1] == pytest.approx(0.0, abs=1e-3)

        # And it is genuinely a decision loss: when a competitor wins, the loss is large.
        losing = float(char_loss(query, prototypes_with_cosines([0.2, 0.95, 0.1]), target, mask))
        assert losing > losses[0]

    def test_masked_candidates_cannot_be_chosen(self) -> None:
        queries = torch.tensor([[1.0, 0.0]])
        prototypes = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])  # index 1 matches perfectly
        targets = torch.tensor([0])
        unmasked = char_loss(queries, prototypes, targets, torch.ones(1, 2, dtype=torch.bool))
        masked = char_loss(
            queries, prototypes, targets, torch.tensor([[True, False]])
        )
        assert masked < unmasked
        assert masked == pytest.approx(0.0, abs=1e-6)

    def test_rows_whose_target_is_masked_out_are_excluded(self) -> None:
        """Scoring an impossible target would inject an arbitrary constant into the loss."""
        queries = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
        prototypes = torch.nn.functional.normalize(torch.randn(2, 3, 4), dim=-1)
        targets = torch.tensor([0, 1])
        mask = torch.tensor([[True, True, True], [False, False, True]])
        loss = char_loss(queries, prototypes, targets, mask)
        only_valid = char_loss(queries[:1], prototypes[:1], targets[:1], mask[:1])
        assert loss == pytest.approx(float(only_valid))

    def test_all_rows_invalid_returns_zero_rather_than_nan(self) -> None:
        loss = char_loss(
            torch.randn(2, 4),
            torch.randn(2, 3, 4),
            torch.tensor([0, 0]),
            torch.zeros(2, 3, dtype=torch.bool),
        )
        assert float(loss) == 0.0

    def test_a_perfect_match_scores_near_zero(self) -> None:
        queries = torch.tensor([[1.0, 0.0]])
        prototypes = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        assert char_loss(
            queries, prototypes, torch.tensor([0]), torch.ones(1, 2, dtype=torch.bool)
        ) == pytest.approx(0.0, abs=1e-6)


class TestWriterLoss:
    def test_prefers_the_query_own_writer(self) -> None:
        queries = torch.tensor([[1.0, 0.0]])
        writer_prototypes = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        right = writer_loss(queries, writer_prototypes, torch.tensor([0]), mask)
        wrong = writer_loss(queries, writer_prototypes, torch.tensor([1]), mask)
        assert right < wrong

    def test_a_single_available_writer_contributes_nothing(self) -> None:
        """With one writer the softmax is degenerate — there is no Type-A negative to learn from."""
        loss = writer_loss(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            torch.tensor([0]),
            torch.tensor([[True, False]]),
        )
        assert float(loss) == 0.0

    def test_writers_lacking_the_character_are_masked_out(self) -> None:
        queries = torch.nn.functional.normalize(torch.randn(1, 4), dim=-1)
        protos = torch.nn.functional.normalize(torch.randn(1, 3, 4), dim=-1)
        full = writer_loss(queries, protos, torch.tensor([0]), torch.ones(1, 3, dtype=torch.bool))
        partial_mask = torch.tensor([[True, True, False]])
        partial = writer_loss(queries, protos, torch.tensor([0]), partial_mask)
        assert float(full) != float(partial)


class TestGradientFlow:
    def test_both_terms_reach_the_encoder(self, verifier: GlyphVerifier) -> None:
        """A loss that does not reach the stem would train silently and learn nothing."""
        crops = torch.randn(4, 2, 64, 40)
        embeddings = verifier(crops)
        prototypes = torch.nn.functional.normalize(torch.randn(4, 3, DEFAULT_EMBED), dim=-1)
        loss = char_loss(
            embeddings, prototypes, torch.tensor([0, 1, 2, 0]), torch.ones(4, 3, dtype=torch.bool)
        ) + writer_loss(
            embeddings,
            torch.nn.functional.normalize(torch.randn(4, 2, DEFAULT_EMBED), dim=-1),
            torch.tensor([0, 1, 0, 1]),
            torch.ones(4, 2, dtype=torch.bool),
        )
        loss.backward()
        assert verifier.stem[0].weight.grad is not None
        assert verifier.stem[0].weight.grad.abs().sum() > 0
