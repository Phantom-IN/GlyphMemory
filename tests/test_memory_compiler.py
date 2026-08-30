"""`compile_profile`: real plumbing (preprocessed image -> model -> alignment -> pooled
prototypes) on the synthetic corpus with a small randomly-initialized model, plus the
gradient-free and graceful-degradation properties the design requires directly.
"""

from __future__ import annotations

import pytest
import torch

from glyphmemory.config.schema import MemoryConfig
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.data.preprocessing import preprocess_path
from glyphmemory.memory.compiler import compile_profile
from glyphmemory.memory.projection import GlyphProjection
from glyphmemory.model import GMBase

MODEL_FINGERPRINT = "deadbeefcafef00d"


def _model_and_charset():
    torch.manual_seed(0)
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    model = GMBase(vocab_size=tokenizer.vocab_size)
    return model, tokenizer.charset


def _lines_for(synthetic_corpus, writer_id: str, n: int | None = None):
    records = synthetic_corpus.records_for(writer_id)
    if n is not None:
        records = records[:n]
    return [(preprocess_path(r.image).tensor, r.text) for r in records]


def test_compile_profile_produces_only_observed_characters(synthetic_corpus):
    model, charset = _model_and_charset()
    writer = synthetic_corpus.writers[0]
    lines = _lines_for(synthetic_corpus, writer)

    profile = compile_profile(model, charset, lines, model_fingerprint=MODEL_FINGERPRINT)

    observed_characters = {c for _, text in lines for c in text}
    assert profile.characters <= observed_characters
    assert profile.characters, "at least some characters should be alignable"
    for character in profile.characters:
        assert character in charset


def test_compile_profile_never_invents_an_unobserved_character(synthetic_corpus):
    model, charset = _model_and_charset()
    writer = synthetic_corpus.writers[0]
    lines = _lines_for(synthetic_corpus, writer, n=1)

    profile = compile_profile(model, charset, lines, model_fingerprint=MODEL_FINGERPRINT)

    observed_characters = {c for _, text in lines for c in text}
    unseen = set(charset.characters) - observed_characters
    assert not (profile.characters & unseen)


def test_compile_profile_prototypes_are_unit_norm(synthetic_corpus):
    model, charset = _model_and_charset()
    writer = synthetic_corpus.writers[0]
    lines = _lines_for(synthetic_corpus, writer)

    profile = compile_profile(model, charset, lines, model_fingerprint=MODEL_FINGERPRINT)

    for character in profile.characters:
        prototype = profile.prototype_for(character)
        assert torch.isclose(prototype.norm(), torch.tensor(1.0), atol=1e-4)


def test_compile_profile_more_support_lines_never_decreases_observation_counts(synthetic_corpus):
    model, charset = _model_and_charset()
    writer = synthetic_corpus.writers[0]
    all_lines = _lines_for(synthetic_corpus, writer)
    assert len(all_lines) >= 2

    small_profile = compile_profile(
        model, charset, all_lines[:1], model_fingerprint=MODEL_FINGERPRINT
    )
    large_profile = compile_profile(
        model, charset, all_lines, model_fingerprint=MODEL_FINGERPRINT
    )

    for character, count in small_profile.counts.items():
        assert large_profile.counts.get(character, 0) >= count


def test_compile_profile_records_the_requested_feature_layer_and_dim(synthetic_corpus):
    model, charset = _model_and_charset()
    writer = synthetic_corpus.writers[0]
    lines = _lines_for(synthetic_corpus, writer)

    sequence_profile = compile_profile(
        model,
        charset,
        lines,
        model_fingerprint=MODEL_FINGERPRINT,
        config=MemoryConfig(feature_layer="sequence"),
    )
    visual_profile = compile_profile(
        model,
        charset,
        lines,
        model_fingerprint=MODEL_FINGERPRINT,
        config=MemoryConfig(feature_layer="visual"),
    )

    assert sequence_profile.feature_layer == "sequence"
    assert sequence_profile.feature_dim == 384
    assert visual_profile.feature_layer == "visual"
    assert visual_profile.feature_dim == 192
    for character in sequence_profile.characters:
        glyph = sequence_profile.glyphs[character]
        assert glyph.feature_layer == "sequence"


@pytest.mark.parametrize("pooling", ["posterior_weighted", "peak_frame", "uniform"])
def test_compile_profile_works_with_every_pooling_strategy(synthetic_corpus, pooling):
    model, charset = _model_and_charset()
    writer = synthetic_corpus.writers[0]
    lines = _lines_for(synthetic_corpus, writer)

    profile = compile_profile(
        model,
        charset,
        lines,
        model_fingerprint=MODEL_FINGERPRINT,
        config=MemoryConfig(pooling=pooling),
    )

    assert profile.characters


def test_compile_profile_stores_the_given_model_fingerprint(synthetic_corpus):
    model, charset = _model_and_charset()
    lines = _lines_for(synthetic_corpus, synthetic_corpus.writers[0], n=1)

    profile = compile_profile(model, charset, lines, model_fingerprint="some-fingerprint")

    assert profile.model_fingerprint == "some-fingerprint"


def test_compile_profile_rejects_empty_lines(synthetic_corpus):
    model, charset = _model_and_charset()

    with pytest.raises(ValueError, match="at least one"):
        compile_profile(model, charset, [], model_fingerprint=MODEL_FINGERPRINT)


def test_compile_profile_rejects_unknown_pooling_strategy(synthetic_corpus):
    model, charset = _model_and_charset()
    lines = _lines_for(synthetic_corpus, synthetic_corpus.writers[0], n=1)

    with pytest.raises(ValueError, match="pooling"):
        compile_profile(
            model,
            charset,
            lines,
            model_fingerprint=MODEL_FINGERPRINT,
            config=MemoryConfig(pooling="nonexistent"),
        )


def test_compile_profile_rejects_unknown_feature_layer(synthetic_corpus):
    model, charset = _model_and_charset()
    lines = _lines_for(synthetic_corpus, synthetic_corpus.writers[0], n=1)

    with pytest.raises(ValueError, match="feature_layer"):
        compile_profile(
            model,
            charset,
            lines,
            model_fingerprint=MODEL_FINGERPRINT,
            config=MemoryConfig(feature_layer="nonexistent"),
        )


# ------------------------------------------------------------------ gradient-free (invariant 4)


def test_compile_profile_output_carries_no_grad_tracking(synthetic_corpus):
    model, charset = _model_and_charset()
    lines = _lines_for(synthetic_corpus, synthetic_corpus.writers[0])

    with torch.enable_grad():
        profile = compile_profile(model, charset, lines, model_fingerprint=MODEL_FINGERPRINT)

    for glyph in profile.glyphs.values():
        assert glyph.prototype.requires_grad is False
        assert glyph.prototype.grad_fn is None


def test_compile_profile_builds_no_computation_graph_even_when_grad_is_enabled(synthetic_corpus):
    model, charset = _model_and_charset()
    lines = _lines_for(synthetic_corpus, synthetic_corpus.writers[0])

    for parameter in model.parameters():
        assert parameter.requires_grad

    with torch.enable_grad():
        compile_profile(model, charset, lines, model_fingerprint=MODEL_FINGERPRINT)

    # Parameters keep requiring grad (nothing was frozen) -- compile_profile just never builds a
    # graph in the first place, so there is nothing to backprop through afterward either.
    for parameter in model.parameters():
        assert parameter.requires_grad


# ------------------------------------------------------------------ learned projection (M9)


def test_compile_profile_learned_projection_produces_96d_unit_norm_prototypes(synthetic_corpus):
    model, charset = _model_and_charset()
    torch.manual_seed(1)
    projection = GlyphProjection()
    lines = _lines_for(synthetic_corpus, synthetic_corpus.writers[0])

    profile = compile_profile(
        model,
        charset,
        lines,
        model_fingerprint=MODEL_FINGERPRINT,
        config=MemoryConfig(feature_layer="learned_projection"),
        projection=projection,
        projection_fingerprint="proj-fingerprint",
    )

    assert profile.feature_layer == "learned_projection"
    assert profile.feature_dim == 96
    assert profile.projection_fingerprint == "proj-fingerprint"
    for character in profile.characters:
        prototype = profile.prototype_for(character)
        assert prototype.shape == (96,)
        assert torch.isclose(prototype.norm(), torch.tensor(1.0), atol=1e-4)


def test_compile_profile_learned_projection_requires_a_projection(synthetic_corpus):
    model, charset = _model_and_charset()
    lines = _lines_for(synthetic_corpus, synthetic_corpus.writers[0], n=1)

    with pytest.raises(ValueError, match="requires a projection"):
        compile_profile(
            model,
            charset,
            lines,
            model_fingerprint=MODEL_FINGERPRINT,
            config=MemoryConfig(feature_layer="learned_projection"),
        )


def test_compile_profile_rejects_a_projection_for_a_non_projected_layer(synthetic_corpus):
    model, charset = _model_and_charset()
    lines = _lines_for(synthetic_corpus, synthetic_corpus.writers[0], n=1)

    with pytest.raises(ValueError, match="does not use one"):
        compile_profile(
            model,
            charset,
            lines,
            model_fingerprint=MODEL_FINGERPRINT,
            config=MemoryConfig(feature_layer="sequence"),
            projection=GlyphProjection(),
        )


def test_compile_profile_learned_projection_defaults_to_none_projection_fingerprint(
    synthetic_corpus,
):
    model, charset = _model_and_charset()
    lines = _lines_for(synthetic_corpus, synthetic_corpus.writers[0], n=1)

    profile = compile_profile(
        model,
        charset,
        lines,
        model_fingerprint=MODEL_FINGERPRINT,
        config=MemoryConfig(feature_layer="learned_projection"),
        projection=GlyphProjection(),
    )

    assert profile.projection_fingerprint is None


# ------------------------------------------------------------------ prototype variants (M9, P26)


@pytest.mark.parametrize(
    "strategy", ["mean", "confidence_weighted", "medoid", "top_k"]
)
def test_compile_profile_works_with_every_prototype_strategy(synthetic_corpus, strategy):
    model, charset = _model_and_charset()
    lines = _lines_for(synthetic_corpus, synthetic_corpus.writers[0])

    profile = compile_profile(
        model,
        charset,
        lines,
        model_fingerprint=MODEL_FINGERPRINT,
        config=MemoryConfig(prototype_strategy=strategy),
    )

    assert profile.characters
    for character in profile.characters:
        glyph = profile.glyphs[character]
        assert torch.isclose(glyph.prototype.norm(), torch.tensor(1.0), atol=1e-4)
        for extra in glyph.additional_prototypes:
            assert torch.isclose(extra.norm(), torch.tensor(1.0), atol=1e-4)


def test_compile_profile_top_k_produces_at_most_top_k_prototypes_per_character(synthetic_corpus):
    model, charset = _model_and_charset()
    lines = _lines_for(synthetic_corpus, synthetic_corpus.writers[0])

    profile = compile_profile(
        model,
        charset,
        lines,
        model_fingerprint=MODEL_FINGERPRINT,
        config=MemoryConfig(prototype_strategy="top_k", top_k=2),
    )

    for character in profile.characters:
        glyph = profile.glyphs[character]
        total_prototypes = 1 + len(glyph.additional_prototypes)
        assert total_prototypes <= 2
        assert total_prototypes <= glyph.number_of_observations


def test_compile_profile_non_top_k_strategies_produce_no_additional_prototypes(synthetic_corpus):
    model, charset = _model_and_charset()
    lines = _lines_for(synthetic_corpus, synthetic_corpus.writers[0])

    for strategy in ("mean", "confidence_weighted", "medoid"):
        profile = compile_profile(
            model,
            charset,
            lines,
            model_fingerprint=MODEL_FINGERPRINT,
            config=MemoryConfig(prototype_strategy=strategy),
        )
        for character in profile.characters:
            assert profile.glyphs[character].additional_prototypes == ()


def test_compile_profile_rejects_unknown_prototype_strategy(synthetic_corpus):
    model, charset = _model_and_charset()
    lines = _lines_for(synthetic_corpus, synthetic_corpus.writers[0], n=1)

    with pytest.raises(ValueError, match="prototype_strategy"):
        compile_profile(
            model,
            charset,
            lines,
            model_fingerprint=MODEL_FINGERPRINT,
            config=MemoryConfig(prototype_strategy="nonexistent"),
        )


def test_compile_profile_restores_the_models_training_mode(synthetic_corpus):
    model, charset = _model_and_charset()
    lines = _lines_for(synthetic_corpus, synthetic_corpus.writers[0], n=1)

    model.train()
    compile_profile(model, charset, lines, model_fingerprint=MODEL_FINGERPRINT)
    assert model.training is True

    model.eval()
    compile_profile(model, charset, lines, model_fingerprint=MODEL_FINGERPRINT)
    assert model.training is False
