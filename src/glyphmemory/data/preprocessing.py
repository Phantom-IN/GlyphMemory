"""Image preprocessing — the model's input contract.

```text
load -> grayscale -> [augment] -> resize to height 64 -> pad width to a multiple of 16
     -> float, invert, normalize  ->  [1, 64, W]
```

Two things here matter more than image quality.

**The true unpadded width is carried separately from the padded width.** ``input_lengths`` for CTC
is derived from the true width; losing it is precisely how padding gets counted as valid input,
which trains a model that decodes truncated text and never errors.

**:func:`temporal_length` is the single source of truth for ``T``.** The visual encoder must
be *tested against* this function rather than reimplementing the arithmetic; two independent
computations of ``T`` that drift apart is the highest-value bug this phase can prevent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torchvision.transforms.v2 import functional as TF

from glyphmemory.runtime.logging import get_logger

logger = get_logger("data.preprocessing")

#: Target line height.
DEFAULT_HEIGHT = 64

#: Widths are padded to a multiple of this for batching and runtime convenience.
DEFAULT_WIDTH_MULTIPLE = 16

#: Horizontal downsampling performed by the visual encoder.
HORIZONTAL_DOWNSAMPLE = 4

#: Oversized lines are flagged, never crushed.
DEFAULT_MAX_WIDTH = 1600

#: Background level in the loaded 8-bit grayscale image: paper is white.
BACKGROUND_LEVEL = 255


class UnreadableImageError(OSError):
    """An image file could not be opened or decoded.

    Recorded as ``IntegrityCategory.UNREADABLE_IMAGE`` rather than crashing a training run.
    """

    def __init__(self, path: str | Path, reason: str) -> None:
        self.path = str(path)
        self.reason = reason
        super().__init__(f"Cannot read image {self.path!r}: {reason}")


@dataclass(frozen=True, slots=True)
class PixelNormalization:
    """How pixel values become model input.

    **Decision: fixed constants, never per-image statistics.**.

    Per-image standardisation would rescale every line to the same mean and variance, which erases
    exactly the signal this project is built on — a writer with faint, thin strokes and one with
    heavy, dark strokes would be normalised into each other. Ink density is writer identity, so it
    must survive preprocessing intact.

    ``invert`` maps paper to 0 and ink towards 1. That makes the background match the zero that
    convolutions and batch padding already use, so padded regions contribute nothing rather than a
    bright constant.

    ``mean``/``std`` are a **documented placeholder**: with ``0.0``/``1.0`` this is plain ``[0, 1]``
    scaling.
    """

    invert: bool = True
    mean: float = 0.0
    std: float = 1.0

    def __post_init__(self) -> None:
        if self.std <= 0:
            raise ValueError(f"Normalization std must be positive, got {self.std}")

    @property
    def background_value(self) -> float:
        """Value a padded pixel takes after normalization."""
        raw = 0.0 if self.invert else 1.0
        return (raw - self.mean) / self.std

    def apply(self, image: Tensor) -> Tensor:
        """uint8 ``[1, H, W]`` -> float32 ``[1, H, W]``."""
        result = image.to(torch.float32) / 255.0
        if self.invert:
            result = 1.0 - result
        return (result - self.mean) / self.std


@dataclass(frozen=True, slots=True)
class PreprocessedLine:
    """A preprocessed line and the widths that CTC depends on."""

    tensor: Tensor
    true_width: int
    padded_width: int
    original_size: tuple[int, int]
    oversized: bool = False

    @property
    def height(self) -> int:
        return int(self.tensor.shape[-2])

    @property
    def input_length(self) -> int:
        """Valid CTC time steps — derived from the **true** width, never the padded one."""
        return temporal_length(self.true_width)

    @property
    def padded_length(self) -> int:
        """Time steps the model actually emits for this tensor."""
        return temporal_length(self.padded_width)


def temporal_length(width: int, downsample: int = HORIZONTAL_DOWNSAMPLE) -> int:
    """Time steps produced by the visual encoder for an input of ``width`` pixels.

    **The single source of truth for ``T``.** The encoder is tested against this; it must not
    the value independently.

    Uses ceiling division to match stride-2 convolutions with padding, whose output is ``ceil(in /
    2)`` per stage. Since widths are padded to a multiple of 16 before reaching the model, ceiling
    and floor coincide in practice — the distinction only matters for unpadded probes.
    """
    if width < 0:
        raise ValueError(f"Width must be non-negative, got {width}")
    if downsample < 1:
        raise ValueError(f"Downsample must be at least 1, got {downsample}")
    return math.ceil(width / downsample)


def resized_width(source_width: int, source_height: int, height: int = DEFAULT_HEIGHT) -> int:
    """Width a source image will have after height normalization, without decoding it.

    Mirrors :func:`resize_to_height` exactly — same rounding, same minimum of 1 — so a caller can
    predict the post-preprocessing width from manifest metadata alone.

    This exists because **the width a batch is padded to is not the width recorded in the
    manifest**. Line heights vary between samples (44-176 px in CVL), so aspect ratios differ and
    the two quantities are only weakly related. Bucketing on the source width therefore groups the
    wrong things: measured on CVL, Spearman 0.468 between the two, and 73.9% padding efficiency
    against 92.2% achievable.
    """
    if source_height < 1 or source_width < 1:
        raise ValueError(f"Source dimensions must be positive, got {source_width}x{source_height}")
    if height < 1:
        raise ValueError(f"Height must be at least 1, got {height}")
    if source_height == height:
        return source_width
    return max(1, round(source_width * height / source_height))


def load_line_image(path: str | Path) -> Image.Image:
    """Load a line image as 8-bit grayscale.

    Raises:
        UnreadableImageError: The file is missing, not an image, or truncated.
    """
    path = Path(path)
    try:
        with Image.open(path) as handle:
            handle.load()
            return handle.convert("L")
    except FileNotFoundError as exc:
        raise UnreadableImageError(path, "file does not exist") from exc
    except UnidentifiedImageError as exc:
        raise UnreadableImageError(path, "not a recognisable image format") from exc
    except OSError as exc:
        raise UnreadableImageError(path, f"{type(exc).__name__}: {exc}") from exc


def to_uint8_tensor(image: Image.Image) -> Tensor:
    """PIL grayscale -> uint8 ``[1, H, W]``."""
    if image.mode != "L":
        image = image.convert("L")
    return TF.pil_to_tensor(image)


def resize_to_height(image: Tensor, height: int = DEFAULT_HEIGHT) -> Tensor:
    """Scale to ``height`` rows, **preserving aspect ratio**."""
    if height < 1:
        raise ValueError(f"Height must be at least 1, got {height}")
    _, current_height, current_width = image.shape
    if current_height == height:
        return image
    width = max(1, round(current_width * height / current_height))
    return TF.resize(image, [height, width], antialias=True)


def pad_width_to_multiple(
    image: Tensor,
    multiple: int = DEFAULT_WIDTH_MULTIPLE,
    *,
    fill: int = BACKGROUND_LEVEL,
) -> tuple[Tensor, int]:
    """Right-pad the width to a multiple of ``multiple``.

    **Right-padding, not centring.** It keeps the real content prefix-aligned with
    ``input_lengths``, so time step *t* means the same thing whether or not padding is present.

    Returns:
        ``(padded_image, true_width)`` — the true width is the caller's responsibility to
        carry into ``input_lengths``.
    """
    if multiple < 1:
        raise ValueError(f"Multiple must be at least 1, got {multiple}")
    true_width = int(image.shape[-1])
    remainder = true_width % multiple
    if remainder == 0:
        return image, true_width
    padding = multiple - remainder
    return TF.pad(image, [0, 0, padding, 0], fill=fill), true_width


def check_max_width(
    width: int,
    *,
    max_width: int = DEFAULT_MAX_WIDTH,
    sample_id: str | None = None,
    path: str | None = None,
) -> bool:
    """Whether ``width`` is within the guard. Logs and returns ``False`` when it is not.

    **Never resizes.** An oversized line is flagged so it can be counted and, if necessary,
    segmented externally — horizontally crushing it would destroy the aspect ratio that makes the
    glyphs readable.
    """
    if width <= max_width:
        return True
    logger.warning(
        "[oversized_width] sample_id=%s path=%s: width %d exceeds max_width %d; flagged, "
        "not resized",
        sample_id,
        path,
        width,
        max_width,
    )
    return False


def preprocess_image(
    image: Image.Image | Tensor,
    *,
    height: int = DEFAULT_HEIGHT,
    width_multiple: int = DEFAULT_WIDTH_MULTIPLE,
    max_width: int = DEFAULT_MAX_WIDTH,
    normalization: PixelNormalization | None = None,
    augmentation: object | None = None,
    sample_id: str | None = None,
    path: str | None = None,
) -> PreprocessedLine:
    """Run the full pipeline on one line.

    Augmentation is applied at **native resolution, before height normalisation**, so a geometric
    warp is resampled once by the subsequent resize rather than twice.

    Args:
        image: A PIL grayscale image or a uint8 ``[1, H, W]`` tensor.
        augmentation: Callable applied to the uint8 tensor. Pass ``None`` for evaluation —
            evaluation is never augmented.
    """
    normalization = normalization or PixelNormalization()
    tensor = to_uint8_tensor(image) if isinstance(image, Image.Image) else image
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"Expected a [1, H, W] grayscale tensor, got shape {tuple(tensor.shape)}")

    original_size = (int(tensor.shape[-1]), int(tensor.shape[-2]))

    if augmentation is not None:
        tensor = augmentation(tensor)

    tensor = resize_to_height(tensor, height)
    within_guard = check_max_width(
        int(tensor.shape[-1]), max_width=max_width, sample_id=sample_id, path=path
    )
    padded, true_width = pad_width_to_multiple(tensor, width_multiple)

    return PreprocessedLine(
        tensor=normalization.apply(padded),
        true_width=true_width,
        padded_width=int(padded.shape[-1]),
        original_size=original_size,
        oversized=not within_guard,
    )


def preprocess_path(path: str | Path, **kwargs) -> PreprocessedLine:
    """Load and preprocess in one call. Convenience for the dataset and the CLI."""
    kwargs.setdefault("path", str(path))
    return preprocess_image(load_line_image(path), **kwargs)
