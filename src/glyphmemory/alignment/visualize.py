"""Overlay recovered alignment spans on the line image — a saved artifact, not a GUI.

Saved to disk, matching ``data preview-augmentations``'s existing pattern (``cli.py``) rather than
opening a window.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from glyphmemory.alignment.spans import AlignmentSpan
from glyphmemory.data.preprocessing import HORIZONTAL_DOWNSAMPLE

#: Alternating span colors, cycled — enough to visually separate adjacent characters without needing
#: a colormap library this project does not otherwise depend on.
_SPAN_COLORS = (
    (255, 99, 71),  # tomato
    (65, 179, 255),  # sky blue
    (154, 205, 50),  # yellowgreen
    (238, 130, 238),  # violet
    (255, 165, 0),  # orange
)

_LABEL_BAND_HEIGHT = 18
_SCORE_BAND_HEIGHT = 10


def _to_pil(image: Tensor | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("L")
    if image.dim() == 3:
        image = image[0]
    array = image.detach().cpu()
    if array.dtype.is_floating_point:
        array = (array.clamp(0, 1) * 255).to(torch.uint8)
    return Image.fromarray(array.numpy(), mode="L")


def render_alignment(
    image: Tensor | Image.Image,
    spans: Sequence[AlignmentSpan],
    *,
    downsample: int = HORIZONTAL_DOWNSAMPLE,
) -> Image.Image:
    """Render ``image`` with ``spans`` overlaid as colored bands, character labels, and a per-span
    confidence strip.

    Frame index → pixel column is the **approximate** mapping ``column = frame * downsample`` — a
    visualization aid, not a claim of exact sub-pixel correctness. The encoder's true receptive
    field means a frame's "true" source pixels form a window, not a point; this marks the window's
    start, which is precise enough to catch a badly wrong alignment by eye, which is this function's
    entire job.

    Returns a single composite image: the line image, a translucent span overlay, character labels
    below it, and a confidence strip below that (darker = lower ``span.score``).
    """
    base = _to_pil(image).convert("RGB")
    width, height = base.size

    canvas = Image.new(
        "RGB", (width, height + _LABEL_BAND_HEIGHT + _SCORE_BAND_HEIGHT), color=(255, 255, 255)
    )
    canvas.paste(base, (0, 0))

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    label_draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, span in enumerate(spans):
        color = _SPAN_COLORS[index % len(_SPAN_COLORS)]
        x0 = min(span.start_t * downsample, width)
        x1 = min(span.end_t * downsample, width)
        if x1 <= x0:
            x1 = min(x0 + 1, width)

        draw.rectangle([x0, 0, x1, height], fill=(*color, 70))
        draw.line([(x0, 0), (x0, height)], fill=(*color, 200), width=1)

        label = span.token if span.token != " " else "·"
        label_x = (x0 + x1) // 2
        label_draw.text(
            (label_x, height + 2), label, fill=(0, 0, 0), font=font, anchor="ma"
        )

        score_intensity = int(255 * max(0.0, min(1.0, span.score)))
        score_color = (255 - score_intensity, score_intensity, 60)
        label_draw.rectangle(
            [x0, height + _LABEL_BAND_HEIGHT, x1, height + _LABEL_BAND_HEIGHT + _SCORE_BAND_HEIGHT],
            fill=score_color,
        )

    composite = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    canvas.paste(composite, (0, 0))
    return canvas


def save_alignment(
    image: Tensor | Image.Image,
    spans: Sequence[AlignmentSpan],
    path: str | Path,
    *,
    downsample: int = HORIZONTAL_DOWNSAMPLE,
) -> Path:
    """Render and save an alignment visualization. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_alignment(image, spans, downsample=downsample)
    rendered.save(path)
    return path
