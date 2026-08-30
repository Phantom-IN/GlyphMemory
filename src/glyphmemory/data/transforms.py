"""Training augmentation.

Defaults are deliberately conservative: augmentation that alters textual identity does not
regularise the model, it **corrupts the labels**.

Forbidden outright, and asserted unreachable by test:

```text
horizontal flip · vertical flip · large rotation · aggressive crop
```

A flipped line is not a harder example of the same text — it is a different, wrong label.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torchvision.transforms import v2

from glyphmemory.config.schema import AugmentationConfig
from glyphmemory.data.preprocessing import BACKGROUND_LEVEL
from glyphmemory.runtime.logging import get_logger

logger = get_logger("data.transforms")

#: Hard ceiling on rotation. Beyond this a line is no longer plausibly the same writing.
MAX_SAFE_ROTATION_DEGREES = 5.0

#: Hard ceiling on shear.
MAX_SAFE_SHEAR_DEGREES = 10.0

#: Transform class names that must never appear in a built pipeline.
FORBIDDEN_TRANSFORMS = (
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomCrop",
    "RandomResizedCrop",
)


class Identity:
    """Explicit no-op, returned when augmentation is disabled.

    A named class rather than ``None`` so the caller always has something to invoke, and so
    ``transforms`` in a run record reads ``[]`` rather than being absent.
    """

    def __call__(self, image: Tensor) -> Tensor:
        return image

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "Identity()"


@dataclass(frozen=True, slots=True)
class AugmentationPipeline:
    """A built pipeline plus the description of what it contains."""

    transform: Callable[[Tensor], Tensor]
    families: tuple[str, ...]

    def __call__(self, image: Tensor) -> Tensor:
        return self.transform(image)

    @property
    def is_identity(self) -> bool:
        return not self.families

    def describe(self) -> dict[str, Any]:
        return {"families": list(self.families), "identity": self.is_identity}


def _validate(config: AugmentationConfig) -> None:
    """Reject magnitudes that would change what the text says."""
    if config.rotation_degrees > MAX_SAFE_ROTATION_DEGREES:
        raise ValueError(
            f"rotation_degrees={config.rotation_degrees} exceeds the safe maximum of "
            f"{MAX_SAFE_ROTATION_DEGREES}. Large rotation is forbidden."
        )
    if config.shear_degrees > MAX_SAFE_SHEAR_DEGREES:
        raise ValueError(
            f"shear_degrees={config.shear_degrees} exceeds the safe maximum of "
            f"{MAX_SAFE_SHEAR_DEGREES}."
        )
    if config.scale_min > config.scale_max:
        raise ValueError(
            f"scale_min={config.scale_min} must not exceed scale_max={config.scale_max}"
        )
    for name in (
        "affine_probability",
        "blur_probability",
        "perspective_probability",
        "elastic_probability",
        "photometric_probability",
    ):
        value = getattr(config, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}={value} must be in [0, 1]")


def build_augmentation(
    config: AugmentationConfig | None = None, *, training: bool = True
) -> AugmentationPipeline:
    """Build the augmentation pipeline for one split.

    Args:
        config: Which families to enable and how strongly.
        training: ``False`` returns :class:`Identity`. **Evaluation is never augmented** — a metric
            computed on perturbed inputs measures something nobody asked for.

    The geometric families (rotation, translation, scale, shear) are composed into a *single*
    ``RandomAffine`` rather than chained. They stay independently toggleable, but one resampling
    step avoids the cumulative interpolation blur of four.
    """
    config = config or AugmentationConfig()

    if not training or not config.enabled:
        return AugmentationPipeline(transform=Identity(), families=())

    _validate(config)

    families: list[str] = []
    stages: list[Any] = []

    degrees = config.rotation_degrees if config.rotation else 0.0
    translate = (config.translate_fraction, config.translate_fraction) if config.translate else None
    scale = (config.scale_min, config.scale_max) if config.scale else None
    shear = (-config.shear_degrees, config.shear_degrees) if config.shear else None

    if config.rotation:
        families.append("rotation")
    if config.translate:
        families.append("translate")
    if config.scale:
        families.append("scale")
    if config.shear:
        families.append("shear")

    if degrees or translate or scale or shear:
        affine = v2.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            fill=BACKGROUND_LEVEL,
        )
        stages.append(v2.RandomApply([affine], p=config.affine_probability))

    if config.perspective:
        families.append("perspective")
        stages.append(
            v2.RandomPerspective(
                distortion_scale=config.perspective_scale,
                p=config.perspective_probability,
                fill=BACKGROUND_LEVEL,
            )
        )

    if config.elastic:
        families.append("elastic")
        stages.append(
            v2.RandomApply(
                [v2.ElasticTransform(alpha=config.elastic_alpha, fill=BACKGROUND_LEVEL)],
                p=config.elastic_probability,
            )
        )

    if config.brightness or config.contrast:
        if config.brightness:
            families.append("brightness")
        if config.contrast:
            families.append("contrast")
        stages.append(
            v2.RandomApply(
                [
                    v2.ColorJitter(
                        brightness=config.brightness_strength if config.brightness else 0.0,
                        contrast=config.contrast_strength if config.contrast else 0.0,
                    )
                ],
                p=config.photometric_probability,
            )
        )

    if config.blur:
        families.append("blur")
        stages.append(
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=3, sigma=(0.05, config.blur_sigma_max))],
                p=config.blur_probability,
            )
        )

    if not stages:
        return AugmentationPipeline(transform=Identity(), families=())

    pipeline = v2.Compose(stages)
    _assert_no_forbidden_transforms(pipeline)
    logger.debug("Augmentation families enabled: %s", families)
    return AugmentationPipeline(transform=pipeline, families=tuple(families))


def _assert_no_forbidden_transforms(pipeline: Any) -> None:
    """Fail loudly if a label-destroying transform ever reaches a pipeline."""
    found = [name for name in FORBIDDEN_TRANSFORMS if name in repr(pipeline)]
    if found:
        raise AssertionError(
            f"Forbidden transform(s) {found} in the augmentation pipeline. Flips and crops "
            "change what the text says — they corrupt labels rather than regularise."
        )


def augment_deterministically(
    pipeline: AugmentationPipeline, image: Tensor, *, seed: int
) -> Tensor:
    """Apply ``pipeline`` with a fixed RNG state, for reproducible previews and tests."""
    generator_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        return pipeline(image)
    finally:
        torch.random.set_rng_state(generator_state)
