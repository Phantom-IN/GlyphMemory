"""Augmentation.

The load-bearing tests are the negative ones: that flips and large rotations cannot be built from
any config, and that evaluation is never augmented. A transform that alters textual identity
corrupts labels rather than regularising the model.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from glyphmemory.config import AugmentationConfig, Config, load_config
from glyphmemory.data import (
    AugmentationPipeline,
    Identity,
    augment_deterministically,
    build_augmentation,
    to_uint8_tensor,
)
from glyphmemory.data.transforms import (
    FORBIDDEN_TRANSFORMS,
    MAX_SAFE_ROTATION_DEGREES,
    MAX_SAFE_SHEAR_DEGREES,
)

FAMILY_FLAGS = (
    "rotation",
    "translate",
    "scale",
    "shear",
    "brightness",
    "contrast",
    "blur",
    "perspective",
    "elastic",
)


def sample_image(width: int = 200, height: int = 64) -> torch.Tensor:
    image = Image.new("L", (width, height), 240)
    for x in range(20, width - 20, 7):
        for y in range(16, height - 16):
            image.putpixel((x, y), 15)
    return to_uint8_tensor(image)


def all_off(**overrides) -> AugmentationConfig:
    off = dict.fromkeys(FAMILY_FLAGS, False)
    return AugmentationConfig(**{**off, **overrides})


# --------------------------------------------------------------------------- forbidden


def test_no_flip_is_reachable_from_the_default_config():
    assert not any(name in repr(build_augmentation().transform) for name in FORBIDDEN_TRANSFORMS)


def test_no_flip_is_reachable_from_any_family_combination():
    """Exhaustive over single-family pipelines — no config path can produce a flip."""
    for family in FAMILY_FLAGS:
        pipeline = build_augmentation(all_off(**{family: True}))
        rendered = repr(pipeline.transform)
        for forbidden in FORBIDDEN_TRANSFORMS:
            assert forbidden not in rendered, f"{forbidden} reachable via {family}"


def test_all_families_together_contain_no_forbidden_transform():
    config = AugmentationConfig(**dict.fromkeys(FAMILY_FLAGS, True))
    rendered = repr(build_augmentation(config).transform)
    assert not any(name in rendered for name in FORBIDDEN_TRANSFORMS)


def test_config_has_no_flip_option_at_all():
    """Flips are not merely disabled — they are inexpressible."""
    fields = set(AugmentationConfig.__dataclass_fields__)
    assert not any("flip" in name for name in fields)
    assert not any("crop" in name for name in fields)


def test_large_rotation_rejected():
    with pytest.raises(ValueError, match="exceeds the safe maximum"):
        build_augmentation(AugmentationConfig(rotation_degrees=MAX_SAFE_ROTATION_DEGREES + 1))


def test_large_shear_rejected():
    with pytest.raises(ValueError, match="exceeds the safe maximum"):
        build_augmentation(AugmentationConfig(shear_degrees=MAX_SAFE_SHEAR_DEGREES + 1))


def test_inverted_scale_range_rejected():
    with pytest.raises(ValueError, match="must not exceed"):
        build_augmentation(AugmentationConfig(scale_min=1.2, scale_max=0.9))


@pytest.mark.parametrize("probability", [-0.1, 1.5])
def test_out_of_range_probability_rejected(probability):
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        build_augmentation(AugmentationConfig(affine_probability=probability))


# --------------------------------------------------------------------------- identity paths


def test_evaluation_is_never_augmented():
    pipeline = build_augmentation(AugmentationConfig(), training=False)
    assert pipeline.is_identity
    assert isinstance(pipeline.transform, Identity)


def test_evaluation_identity_is_bitwise():
    image = sample_image()
    pipeline = build_augmentation(AugmentationConfig(), training=False)
    assert torch.equal(pipeline(image), image)


def test_disabled_config_is_identity():
    image = sample_image()
    pipeline = build_augmentation(AugmentationConfig(enabled=False))
    assert pipeline.is_identity
    assert torch.equal(pipeline(image), image)


def test_all_families_off_is_identity():
    image = sample_image()
    pipeline = build_augmentation(all_off())
    assert pipeline.is_identity
    assert torch.equal(pipeline(image), image)


def test_none_config_uses_defaults():
    assert build_augmentation(None).families


# --------------------------------------------------------------------------- family toggles


@pytest.mark.parametrize("family", FAMILY_FLAGS)
def test_each_family_toggles_independently(family):
    """Ablation requires every family to be switchable on its own."""
    enabled = build_augmentation(all_off(**{family: True}))
    assert family in enabled.families
    assert len(enabled.families) == 1

    disabled = build_augmentation(AugmentationConfig(**{family: False}))
    assert family not in disabled.families


def test_default_families_match_the_documented_set():
    families = set(build_augmentation().families)
    assert {
        "rotation",
        "translate",
        "scale",
        "shear",
        "brightness",
        "contrast",
        "blur",
        "perspective",
    } <= families
    # Elastic is off by default: most likely to distort glyph identity.
    assert "elastic" not in families


def test_elastic_is_opt_in():
    assert not AugmentationConfig().elastic
    assert "elastic" in build_augmentation(AugmentationConfig(elastic=True)).families


def test_geometric_families_compose_into_one_affine():
    """Four separate affines would resample four times; one keeps interpolation blur down."""
    rendered = repr(build_augmentation(all_off(rotation=True, shear=True, scale=True)).transform)
    assert rendered.count("RandomAffine") == 1


def test_pipeline_describes_itself():
    described = build_augmentation().describe()
    assert described["identity"] is False
    assert "shear" in described["families"]


# --------------------------------------------------------------------------- behaviour


def test_augmentation_changes_the_image():
    image = sample_image()
    pipeline = build_augmentation(AugmentationConfig(affine_probability=1.0))
    changed = any(
        not torch.equal(augment_deterministically(pipeline, image, seed=seed), image)
        for seed in range(5)
    )
    assert changed


def test_augmentation_preserves_shape_and_dtype():
    image = sample_image()
    pipeline = build_augmentation(AugmentationConfig(affine_probability=1.0))
    result = augment_deterministically(pipeline, image, seed=1)
    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_deterministic_augmentation_is_reproducible():
    image = sample_image()
    pipeline = build_augmentation(AugmentationConfig(affine_probability=1.0))
    first = augment_deterministically(pipeline, image, seed=7)
    second = augment_deterministically(pipeline, image, seed=7)
    assert torch.equal(first, second)


def test_different_seeds_give_different_augmentations():
    image = sample_image()
    pipeline = build_augmentation(AugmentationConfig(affine_probability=1.0))
    outputs = {
        augment_deterministically(pipeline, image, seed=seed).numpy().tobytes() for seed in range(6)
    }
    assert len(outputs) > 1


def test_deterministic_augmentation_restores_global_rng():
    """A preview must not perturb the training RNG stream."""
    torch.manual_seed(123)
    before = torch.rand(3)
    torch.manual_seed(123)
    pipeline = build_augmentation(AugmentationConfig(affine_probability=1.0))
    augment_deterministically(pipeline, sample_image(), seed=999)
    assert torch.equal(torch.rand(3), before)


def test_geometric_fill_uses_the_page_background():
    """Corners exposed by a warp must be paper, not black — black would read as ink."""
    image = sample_image()
    pipeline = build_augmentation(all_off(rotation=True, affine_probability=1.0))
    result = augment_deterministically(pipeline, image, seed=2)
    assert result.max().item() >= 200


# --------------------------------------------------------------------------- config wiring


def test_augmentation_section_loads_from_the_shipped_config():
    config = load_config("configs/default.yaml")
    assert config.data.augmentation.enabled is True
    assert config.data.augmentation.elastic is False


def test_rotation_default_is_conservative():
    """A wide line rotated about its centre displaces its ends by (W/2)*tan(theta)."""
    assert Config().data.augmentation.rotation_degrees <= 2.0


def test_pipeline_type_is_exported():
    assert isinstance(build_augmentation(), AugmentationPipeline)
