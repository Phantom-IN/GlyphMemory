"""Internal helper."""

from __future__ import annotations

import pytest
import torch

from glyphmemory.memory.projection import (
    PROJECTION_SCHEMA_VERSION,
    GlyphProjection,
    ProjectionCompatibilityError,
    ProjectionMeta,
    load_projection,
    save_projection,
)
from glyphmemory.model.model_info import parameter_count


def _meta(**overrides) -> ProjectionMeta:
    defaults = dict(
        schema_version=PROJECTION_SCHEMA_VERSION,
        base_model_fingerprint="deadbeefcafef00d",
        input_dim=384,
        hidden_dim=128,
        output_dim=96,
        training_steps=10,
        final_char_loss=0.1,
        final_writer_loss=0.2,
        char_loss_weight=1.0,
        writer_loss_weight=1.0,
    )
    defaults.update(overrides)
    return ProjectionMeta(**defaults)


def test_forward_produces_the_configured_output_dim():
    model = GlyphProjection()
    output = model(torch.randn(5, 384))
    assert output.shape == (5, 96)


def test_forward_output_is_unit_norm():
    model = GlyphProjection()
    output = model(torch.randn(10, 384))
    norms = output.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(10), atol=1e-5)


def test_forward_accepts_a_single_unbatched_vector():
    model = GlyphProjection()
    output = model(torch.randn(384))
    assert output.shape == (96,)
    assert torch.isclose(output.norm(), torch.tensor(1.0), atol=1e-5)


def test_forward_rejects_the_wrong_input_dimension():
    model = GlyphProjection()
    with pytest.raises(ValueError, match="384"):
        model(torch.randn(5, 10))


def test_custom_dimensions_are_respected():
    model = GlyphProjection(input_dim=8, hidden_dim=4, output_dim=2)
    output = model(torch.randn(3, 8))
    assert output.shape == (3, 2)


def test_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="positive"):
        GlyphProjection(input_dim=0)


def test_parameter_count_matches_the_two_linear_layers():
    model = GlyphProjection(input_dim=384, hidden_dim=128, output_dim=96)
    expected = (384 * 128 + 128) + (128 * 96 + 96)
    assert parameter_count(model) == expected


def test_parameter_count_is_far_below_gm_base_ceiling():
    model = GlyphProjection()
    assert parameter_count(model) < 100_000


# ------------------------------------------------------------------ save / load


def test_save_load_roundtrip_preserves_weights_and_meta(tmp_path):
    model = GlyphProjection()
    meta = _meta()
    path = save_projection(tmp_path / "projection.pt", model=model, meta=meta)

    loaded_model, loaded_meta = load_projection(path)

    for (name, original), (_, restored) in zip(
        model.state_dict().items(), loaded_model.state_dict().items(), strict=True
    ):
        assert torch.equal(original, restored), name
    assert loaded_meta.base_model_fingerprint == "deadbeefcafef00d"
    assert loaded_meta.training_steps == 10


def test_loaded_model_produces_identical_output_to_the_original(tmp_path):
    model = GlyphProjection()
    path = save_projection(tmp_path / "projection.pt", model=model, meta=_meta())
    loaded_model, _ = load_projection(path)

    x = torch.randn(4, 384)
    with torch.no_grad():
        assert torch.equal(model(x), loaded_model(x))


def test_load_with_matching_base_fingerprint_succeeds(tmp_path):
    path = save_projection(tmp_path / "p.pt", model=GlyphProjection(), meta=_meta())
    _, meta = load_projection(path, expected_base_model_fingerprint="deadbeefcafef00d")
    assert meta.base_model_fingerprint == "deadbeefcafef00d"


def test_load_with_mismatched_base_fingerprint_raises(tmp_path):
    path = save_projection(tmp_path / "p.pt", model=GlyphProjection(), meta=_meta())
    with pytest.raises(ProjectionCompatibilityError, match="different"):
        load_projection(path, expected_base_model_fingerprint="0000000000000000")


def test_load_with_mismatched_fingerprint_and_strict_false_does_not_raise(tmp_path):
    path = save_projection(tmp_path / "p.pt", model=GlyphProjection(), meta=_meta())
    _, meta = load_projection(
        path, expected_base_model_fingerprint="0000000000000000", strict=False
    )
    assert meta.base_model_fingerprint == "deadbeefcafef00d"


def test_load_rejects_a_future_schema_version(tmp_path):
    path = save_projection(tmp_path / "p.pt", model=GlyphProjection(), meta=_meta())
    payload = torch.load(path, weights_only=False)
    payload["meta"]["schema_version"] = "999"
    torch.save(payload, path)
    with pytest.raises(ProjectionCompatibilityError, match="schema"):
        load_projection(path)


def test_load_rejects_a_file_that_is_not_a_projection_artifact(tmp_path):
    path = tmp_path / "not_a_projection.pt"
    torch.save({"unrelated": "data"}, path)
    with pytest.raises(ProjectionCompatibilityError):
        load_projection(path)


def test_save_is_atomic_no_temp_file_left_behind(tmp_path):
    path = save_projection(tmp_path / "p.pt", model=GlyphProjection(), meta=_meta())
    assert path.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_projection_meta_roundtrips_through_as_dict_and_from_dict():
    meta = _meta(seed=1337, git_commit="abc123")
    restored = ProjectionMeta.from_dict(meta.as_dict())
    assert restored == meta
