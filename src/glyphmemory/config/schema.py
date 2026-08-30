"""Configuration schema.

    data  model  training  evaluation  memory  runtime

Fields are added as milestones need them. Unknown keys are rejected by the loader, because a
silently ignored typo in a research config produces a wrong experiment that still looks successful.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


class ConfigError(ValueError):
    """Raised for malformed or unrecognised configuration."""


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution environment."""

    device: str = "auto"
    seed: int = 1337
    num_workers: int = 4
    log_level: str = "INFO"


@dataclass(frozen=True)
class AugmentationConfig:
    """Training augmentation. Every family toggles independently for ablation.

    Flips and aggressive crops are not expressible here at all — there is deliberately no flag for
    them.

    Rotation defaults low for a geometric reason: a line is very wide relative to its height, so
    rotating about the centre displaces the ends vertically by ``(W/2)·tan θ``. On a 1200 px line
    even 2 degrees moves the ends ~21 px on a 64 px-tall image. Shear expresses writer slant far
    more safely and is enabled more strongly.
    """

    enabled: bool = True

    rotation: bool = True
    rotation_degrees: float = 1.0
    translate: bool = True
    translate_fraction: float = 0.02
    scale: bool = True
    scale_min: float = 0.96
    scale_max: float = 1.04
    shear: bool = True
    shear_degrees: float = 4.0
    affine_probability: float = 0.7

    brightness: bool = True
    brightness_strength: float = 0.2
    contrast: bool = True
    contrast_strength: float = 0.2
    photometric_probability: float = 0.5

    blur: bool = True
    blur_sigma_max: float = 0.8
    blur_probability: float = 0.15

    perspective: bool = True
    perspective_scale: float = 0.05
    perspective_probability: float = 0.15

    # Off by default: the most expensive family and the most likely to distort glyph identity.
    # Enable deliberately for an ablation.
    elastic: bool = False
    elastic_alpha: float = 12.0
    elastic_probability: float = 0.05


@dataclass(frozen=True)
class DataConfig:
    """Dataset selection and preprocessing."""

    dataset: str = "synthetic"
    manifest: str | None = None
    image_height: int = 64
    max_width: int = 1600
    width_multiple: int = 16

    # Pixel normalization.
    invert_pixels: bool = True
    pixel_mean: float = 0.0
    pixel_std: float = 1.0

    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass(frozen=True)
class ModelConfig:
    """GM-Base architecture."""

    name: str = "gm_base"
    input_height: int = 64
    visual_dim: int = 192
    gru_hidden: int = 192
    gru_layers: int = 2
    gru_dropout: float = 0.15
    head_dropout: float = 0.1
    max_parameters: int = 3_000_000


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation."""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    min_learning_rate: float = 1e-6
    grad_clip_norm: float = 5.0
    amp: bool = False
    # Validations without an improvement in validation CER before stopping.
    patience: int = 10


@dataclass(frozen=True)
class EvaluationConfig:
    """Metrics and decoding."""

    decoder: str = "greedy"
    beam_width: int | None = None
    report_wer: bool = True


@dataclass(frozen=True)
class MemoryConfig:
    """Writer memory."""

    enabled: bool = False
    feature_layer: str = "sequence"
    alpha: float = 0.5
    gate_by_emission: bool = True
    protect_blank: bool = True
    pooling: str = "posterior_weighted"
    #: How occurrences of one character become its stored prototype(s): ``mean`` (V0 default),
    #: ``confidence_weighted``, ``medoid``, or ``top_k``.
    prototype_strategy: str = "mean"
    #: Prototypes kept per character when ``prototype_strategy == "top_k"``. Unused otherwise.
    top_k: int = 3


@dataclass(frozen=True)
class Config:
    """Root configuration."""

    name: str = "unnamed"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def to_dict(obj: Any) -> Any:
    """Recursively convert a config dataclass to plain dictionaries for serialisation."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {key: to_dict(value) for key, value in obj.items()}
    return obj
