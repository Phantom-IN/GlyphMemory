"""Glyph pixel regions — reconstructing what a character actually occupies.

The downsample factor is right and the inference is not.

    a character occupies                       20.5 px  (p50 20.0)
    a forced-aligned span covers                6.0 px  (p50  4.0)
    a base-emitted span, in the eligible pool   1 frame = 4 px for 91.9% of slots
    midpoint tiling between adjacent centers   20.05 px  <- recovers the true width

CTC is peaky: it emits a character on one frame and blanks around it. The *span* is where the model
committed, not where the ink is. So a glyph region is **reconstructed** from the midpoint between
neighbouring glyph centers, never sliced from the span.

Two channels leave this module, and the second is not decoration. The window is a fixed 40 px so a
convolutional network has a fixed canvas, but the measured cell is only p50 19 px — every window
therefore contains parts of the neighbouring glyphs. Without a mask marking which columns are *this*
glyph, the network cannot know which character it is being asked about.

**The support and query sides use different center sources, deliberately and identically at training
and inference.** Enrollment has the support transcription by protocol, so forced alignment is
available there in deployment. Query text is not available in deployment, so query centers come from
the base's own emission. Training uses exactly this asymmetry, so no segmentation mismatch exists
between training and inference.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from glyphmemory.alignment.spans import AlignmentSpan
from glyphmemory.data.preprocessing import HORIZONTAL_DOWNSAMPLE
from glyphmemory.evaluation.emitted_spans import EmittedOccurrence

#: Width of the extracted window, in pixels. The measured midpoint cell is p50 19 px / p90 30 px /
#: p99 43 px, so 40 px holds well past p90 with genuine neighbouring context.
CROP_WIDTH = 40

#: Half-width, in frames, of the cell assumed at a line's first and last glyph, where one side has
#: no neighbour to take a midpoint against. 2.5 frames = 10 px each side, near the measured p50 cell
#: of 19 px.
EDGE_HALF_FRAMES = 2.5


def emitted_centers(occurrences: Sequence[EmittedOccurrence]) -> list[float]:
    """Glyph centers, in frames, for the query side.

    The peak frame is the argmax position within the emitted run — where the model actually
    committed to the character, which is the best available estimate of the glyph's center when no
    transcription exists.
    """
    return [float(occurrence.peak) for occurrence in occurrences]


def aligned_centers(spans: Sequence[AlignmentSpan]) -> list[float]:
    """Glyph centers, in frames, for the support side.

    ``end_t`` is exclusive (:class:`~glyphmemory.alignment.spans.AlignmentSpan`), so the last frame
    of the span is ``end_t - 1`` and the center is the midpoint of the closed interval.
    """
    return [(span.start_t + span.end_t - 1) / 2.0 for span in spans]


def midpoint_cells(centers: Sequence[float]) -> list[tuple[float, float]]:
    """Tile the line into one cell per glyph, in frames.

    Each cell runs from the midpoint with the previous center to the midpoint with the next. The
    first and last glyphs have only one neighbour, so their open side extends
    :data:`EDGE_HALF_FRAMES`.

    Returns ``[(lo, hi), ...]``, one per center, in the order given. Cells tile without gaps or
    overlaps, which is what makes the mask channel unambiguous.
    """
    if not centers:
        return []
    cells: list[tuple[float, float]] = []
    for index, center in enumerate(centers):
        low = (centers[index - 1] + center) / 2.0 if index else center - EDGE_HALF_FRAMES
        high = (
            (center + centers[index + 1]) / 2.0
            if index + 1 < len(centers)
            else center + EDGE_HALF_FRAMES
        )
        cells.append((low, high))
    return cells


def extract_crops(
    line: Tensor,
    centers: Sequence[float],
    *,
    true_width: int | None = None,
    crop_width: int = CROP_WIDTH,
) -> Tensor:
    """Extract one ``[2, H, crop_width]`` region per center.

    Args:
        line: The preprocessed line, ``[1, H, W]`` or ``[H, W]`` — exactly the tensor
            :class:`~glyphmemory.data.preprocessing.PreprocessedLine` carries, in its own
            normalization.
        centers: Glyph centers in *frames*, from :func:`emitted_centers` or :func:`aligned_centers`.
        true_width: Unpadded line width in pixels. Columns beyond it are padding and are treated as
            background. Defaults to the tensor's full width.
        crop_width: Window width in pixels.

    Shape:
        ``-> [len(centers), 2, H, crop_width]``. Channel 0 is pixels, channel 1 is the cell mask.

    Windows are clipped at the line edges and padded with the line's own background value — the
    minimum over the real columns, which is the correct background under either polarity, rather
    than a hard-coded zero that would be ink under one of them.
    """
    if line.ndim == 3:
        if line.shape[0] != 1:
            raise ValueError(f"expected a single-channel line, got shape {tuple(line.shape)}")
        line = line[0]
    if line.ndim != 2:
        raise ValueError(f"expected [1, H, W] or [H, W], got shape {tuple(line.shape)}")

    height, width = int(line.shape[0]), int(line.shape[1])
    limit = width if true_width is None else max(1, min(int(true_width), width))
    if not centers:
        return line.new_zeros((0, 2, height, crop_width))

    background = line[:, :limit].min()
    cells = midpoint_cells(centers)
    crops = line.new_full((len(centers), 2, height, crop_width), 0.0)

    for index, (center, (low, high)) in enumerate(zip(centers, cells, strict=True)):
        # Pixel column of the glyph center. A frame covers [4t, 4t+4), so its center is 4t + 2.
        center_px = center * HORIZONTAL_DOWNSAMPLE + HORIZONTAL_DOWNSAMPLE / 2.0
        start = round(center_px - crop_width / 2.0)

        pixels = crops[index, 0]
        pixels.fill_(float(background))
        source_start, source_end = max(0, start), min(limit, start + crop_width)
        if source_end > source_start:
            offset = source_start - start
            pixels[:, offset : offset + (source_end - source_start)] = line[
                :, source_start:source_end
            ]

        # Cell mask, in the window's own coordinates. Clamped to the window, so a glyph whose cell
        # runs past the edge is still marked over the part that is visible.
        low_px = round(low * HORIZONTAL_DOWNSAMPLE) - start
        high_px = round(high * HORIZONTAL_DOWNSAMPLE) - start
        low_px, high_px = max(0, low_px), min(crop_width, high_px)
        if high_px > low_px:
            crops[index, 1, :, low_px:high_px] = 1.0

    return crops


def crops_from_occurrences(
    line: Tensor,
    occurrences: Sequence[EmittedOccurrence],
    *,
    true_width: int | None = None,
) -> Tensor:
    """Query-side convenience: base-emitted peaks -> ``[N, 2, H, 40]``."""
    return extract_crops(line, emitted_centers(occurrences), true_width=true_width)


def crops_from_spans(
    line: Tensor,
    spans: Sequence[AlignmentSpan],
    *,
    true_width: int | None = None,
) -> Tensor:
    """Support-side convenience: forced-aligned span centers -> ``[N, 2, H, 40]``."""
    return extract_crops(line, aligned_centers(spans), true_width=true_width)


def region_statistics(centers: Sequence[float]) -> dict[str, float]:
    """Cell widths in pixels, for logging what a run actually extracted.

    A run whose cells look nothing like that has a segmentation problem, and this is how it becomes
    visible.
    """
    cells = midpoint_cells(centers)
    if not cells:
        return {"count": 0.0, "mean_px": 0.0, "p50_px": 0.0, "p90_px": 0.0}
    widths = sorted((high - low) * HORIZONTAL_DOWNSAMPLE for low, high in cells)
    return {
        "count": float(len(widths)),
        "mean_px": float(torch.tensor(widths).mean()),
        "p50_px": widths[len(widths) // 2],
        "p90_px": widths[min(len(widths) - 1, int(0.9 * len(widths)))],
    }
