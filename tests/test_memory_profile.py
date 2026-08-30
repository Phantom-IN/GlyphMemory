"""`Glyph` / `WriterProfile`: schema, the counts/confidences derived views, and the save/load
roundtrip with fingerprint-mismatch rejection.
"""

from __future__ import annotations

import pytest
import torch

from glyphmemory.memory.profile import (
    PROFILE_SCHEMA_VERSION,
    Glyph,
    ProfileCompatibilityError,
    WriterProfile,
)


def _glyph(character: str, *, dim: int = 4, observations: int = 3, confidence: float = 0.9):
    vector = torch.zeros(dim)
    vector[0] = 1.0
    return Glyph(
        character=character,
        prototype=vector,
        number_of_observations=observations,
        mean_alignment_confidence=confidence,
        feature_layer="sequence",
    )


def _profile(**glyphs: Glyph) -> WriterProfile:
    return WriterProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        model_fingerprint="deadbeefcafef00d",
        feature_layer="sequence",
        feature_dim=4,
        glyphs=glyphs,
    )


# ------------------------------------------------------------------ schema / derived views


def test_counts_and_confidences_are_derived_from_glyphs():
    profile = _profile(
        a=_glyph("a", observations=5, confidence=0.7),
        b=_glyph("b", observations=2, confidence=0.4),
    )

    assert profile.counts == {"a": 5, "b": 2}
    assert profile.confidences == {"a": 0.7, "b": 0.4}


def test_characters_and_prototype_for():
    profile = _profile(a=_glyph("a"))

    assert profile.characters == frozenset({"a"})
    assert profile.prototype_for("a") is not None
    assert profile.prototype_for("z") is None


def test_estimated_bytes_grows_with_glyph_count_and_dimension():
    small = _profile(a=_glyph("a", dim=4))
    bigger = _profile(a=_glyph("a", dim=4), b=_glyph("b", dim=4))
    higher_dim = _profile(a=_glyph("a", dim=384))

    assert bigger.estimated_bytes() > small.estimated_bytes()
    assert higher_dim.estimated_bytes() > small.estimated_bytes()


def test_describe_reports_sorted_characters_and_counts():
    profile = _profile(b=_glyph("b"), a=_glyph("a"))

    description = profile.describe()

    assert description["characters"] == "ab"
    assert description["num_characters"] == 2
    assert description["has_global_style"] is False


def test_global_style_defaults_to_none_and_is_optional():
    profile = _profile(a=_glyph("a"))
    assert profile.global_style is None


# ------------------------------------------------------------------ persistence


def test_save_load_roundtrip_preserves_every_field(tmp_path):
    original = _profile(
        a=_glyph("a", observations=3, confidence=0.8),
        r=_glyph("r", observations=1, confidence=0.5),
    )

    path = original.save(tmp_path / "writer.profile.pt")
    loaded = WriterProfile.load(path)

    assert loaded.schema_version == original.schema_version
    assert loaded.model_fingerprint == original.model_fingerprint
    assert loaded.feature_layer == original.feature_layer
    assert loaded.feature_dim == original.feature_dim
    assert loaded.characters == original.characters
    for character in original.characters:
        assert torch.allclose(loaded.prototype_for(character), original.prototype_for(character))
    assert loaded.counts == original.counts
    assert loaded.confidences == original.confidences


def test_load_with_matching_fingerprint_succeeds(tmp_path):
    original = _profile(a=_glyph("a"))
    path = original.save(tmp_path / "writer.profile.pt")

    loaded = WriterProfile.load(path, expected_model_fingerprint="deadbeefcafef00d")

    assert loaded.model_fingerprint == "deadbeefcafef00d"


def test_load_with_mismatched_fingerprint_raises(tmp_path):
    original = _profile(a=_glyph("a"))
    path = original.save(tmp_path / "writer.profile.pt")

    with pytest.raises(ProfileCompatibilityError, match="different one"):
        WriterProfile.load(path, expected_model_fingerprint="0000000000000000")


def test_load_with_mismatched_fingerprint_and_strict_false_does_not_raise(tmp_path):
    original = _profile(a=_glyph("a"))
    path = original.save(tmp_path / "writer.profile.pt")

    loaded = WriterProfile.load(
        path, expected_model_fingerprint="0000000000000000", strict=False
    )

    assert loaded.model_fingerprint == "deadbeefcafef00d"


def test_load_rejects_a_future_schema_version(tmp_path):
    original = _profile(a=_glyph("a"))
    path = original.save(tmp_path / "writer.profile.pt")

    payload = torch.load(path, weights_only=False)
    payload["schema_version"] = "999"
    torch.save(payload, path)

    with pytest.raises(ProfileCompatibilityError, match="schema"):
        WriterProfile.load(path)


def test_load_rejects_a_file_that_is_not_a_writer_profile(tmp_path):
    path = tmp_path / "not_a_profile.pt"
    torch.save({"unrelated": "data"}, path)

    with pytest.raises(ProfileCompatibilityError):
        WriterProfile.load(path)


def test_save_is_atomic_no_temp_file_left_behind(tmp_path):
    original = _profile(a=_glyph("a"))
    path = original.save(tmp_path / "writer.profile.pt")

    assert path.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_global_style_roundtrips_when_present(tmp_path):
    profile = WriterProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        model_fingerprint="deadbeefcafef00d",
        feature_layer="sequence",
        feature_dim=4,
        glyphs={"a": _glyph("a")},
        global_style=torch.tensor([1.0, 2.0, 3.0]),
    )
    path = profile.save(tmp_path / "writer.profile.pt")

    loaded = WriterProfile.load(path)

    assert torch.allclose(loaded.global_style, torch.tensor([1.0, 2.0, 3.0]))


# ------------------------------------------------------------------ projection identity (M9)


def test_projection_fingerprint_defaults_to_none():
    profile = _profile(a=_glyph("a"))
    assert profile.projection_fingerprint is None


def test_projection_fingerprint_roundtrips_through_save_load(tmp_path):
    profile = WriterProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        model_fingerprint="deadbeefcafef00d",
        feature_layer="sequence",
        feature_dim=4,
        glyphs={"a": _glyph("a")},
        projection_fingerprint="projectionabc123",
    )
    path = profile.save(tmp_path / "writer.profile.pt")

    loaded = WriterProfile.load(path)

    assert loaded.projection_fingerprint == "projectionabc123"


def test_a_profile_saved_without_the_field_loads_with_projection_fingerprint_none(tmp_path):
    profile = _profile(a=_glyph("a"))
    path = profile.save(tmp_path / "writer.profile.pt")
    payload = torch.load(path, weights_only=False)
    del payload["projection_fingerprint"]
    torch.save(payload, path)

    loaded = WriterProfile.load(path)

    assert loaded.projection_fingerprint is None


def test_require_projection_passes_when_fingerprints_match():
    profile = WriterProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        model_fingerprint="deadbeefcafef00d",
        feature_layer="sequence",
        feature_dim=4,
        glyphs={"a": _glyph("a")},
        projection_fingerprint="proj123",
    )
    profile.require_projection("proj123")  # must not raise


def test_require_projection_passes_when_both_are_none():
    profile = _profile(a=_glyph("a"))  # projection_fingerprint defaults to None
    profile.require_projection(None)  # must not raise


def test_require_projection_raises_on_mismatch():
    profile = WriterProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        model_fingerprint="deadbeefcafef00d",
        feature_layer="sequence",
        feature_dim=4,
        glyphs={"a": _glyph("a")},
        projection_fingerprint="proj123",
    )
    with pytest.raises(ProfileCompatibilityError, match="projection"):
        profile.require_projection("a-different-projection")


def test_require_projection_raises_when_profile_is_raw_but_projection_expected():
    profile = _profile(a=_glyph("a"))  # raw features, projection_fingerprint=None
    with pytest.raises(ProfileCompatibilityError, match="projection"):
        profile.require_projection("proj123")


def test_require_projection_raises_when_projection_expected_none_but_profile_has_one():
    profile = WriterProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        model_fingerprint="deadbeefcafef00d",
        feature_layer="sequence",
        feature_dim=4,
        glyphs={"a": _glyph("a")},
        projection_fingerprint="proj123",
    )
    with pytest.raises(ProfileCompatibilityError, match="projection"):
        profile.require_projection(None)


def test_require_projection_with_strict_false_does_not_raise():
    profile = WriterProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        model_fingerprint="deadbeefcafef00d",
        feature_layer="sequence",
        feature_dim=4,
        glyphs={"a": _glyph("a")},
        projection_fingerprint="proj123",
    )
    profile.require_projection("mismatched", strict=False)  # must not raise


def test_describe_includes_projection_fingerprint():
    profile = WriterProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        model_fingerprint="deadbeefcafef00d",
        feature_layer="sequence",
        feature_dim=4,
        glyphs={"a": _glyph("a")},
        projection_fingerprint="proj123",
    )
    assert profile.describe()["projection_fingerprint"] == "proj123"
