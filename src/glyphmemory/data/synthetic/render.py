"""Deterministic line rendering.

A synthetic "writer" is a :class:`WriterStyle` — a fixed vector of visual parameters — paired with a
typeface. Each rendered line perturbs that vector slightly, so a writer is a *distribution over
renderings* rather than one constant image. That distinction matters: if every line from a writer
were pixel-identical, prototype averaging would be trivially perfect and the oracle would prove
nothing.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from glyphmemory.data.synthetic.fonts import FontSource

DEFAULT_HEIGHT = 64
BACKGROUND = 255

# Per-writer sampling ranges. Recorded in the phase handoff; changing them changes every synthetic
# fixture, so treat them as part of the generator's identity.
FONT_SIZE_RANGE = (30.0, 46.0)
LETTER_SPACING_RANGE = (-1.0, 3.0)
SLANT_RANGE = (-14.0, 14.0)
STROKE_WIDTH_CHOICES = (0, 0, 1)
BASELINE_OFFSET_RANGE = (-3.0, 3.0)
INK_LEVEL_RANGE = (0, 70)

# Per-line jitter, deliberately much smaller than the between-writer spread so writer identity stays
# dominant.
JITTER_FONT_SIZE = 1.2
JITTER_LETTER_SPACING = 0.35
JITTER_SLANT = 1.5
JITTER_BASELINE = 1.2
JITTER_INK = 12


@dataclass(frozen=True, slots=True)
class WriterStyle:
    """The visual signature of one synthetic writer."""

    font_index: int = 0
    font_size: float = 38.0
    letter_spacing: float = 0.0
    slant_degrees: float = 0.0
    stroke_width: int = 0
    baseline_offset: float = 0.0
    ink_level: int = 0

    @classmethod
    def sample(
        cls, rng: random.Random, *, writer_index: int, n_writers: int, n_fonts: int
    ) -> WriterStyle:
        """Draw a writer's style.

        Slant is **stratified across writers** rather than sampled independently: it is the most
        visually dominant axis, and stratifying guarantees two writers are separated even on an
        unlucky draw. Everything else is sampled freely.
        """
        low, high = SLANT_RANGE
        if n_writers > 1:
            band = (high - low) / n_writers
            slant = low + band * (writer_index + rng.random())
        else:
            slant = rng.uniform(low, high)

        return cls(
            font_index=writer_index % max(1, n_fonts),
            font_size=rng.uniform(*FONT_SIZE_RANGE),
            letter_spacing=rng.uniform(*LETTER_SPACING_RANGE),
            slant_degrees=slant,
            stroke_width=rng.choice(STROKE_WIDTH_CHOICES),
            baseline_offset=rng.uniform(*BASELINE_OFFSET_RANGE),
            ink_level=rng.randint(*INK_LEVEL_RANGE),
        )

    def jitter(self, rng: random.Random) -> WriterStyle:
        """A slightly perturbed copy, for one line."""
        return replace(
            self,
            font_size=max(12.0, self.font_size + rng.uniform(-JITTER_FONT_SIZE, JITTER_FONT_SIZE)),
            letter_spacing=self.letter_spacing
            + rng.uniform(-JITTER_LETTER_SPACING, JITTER_LETTER_SPACING),
            slant_degrees=self.slant_degrees + rng.uniform(-JITTER_SLANT, JITTER_SLANT),
            baseline_offset=self.baseline_offset + rng.uniform(-JITTER_BASELINE, JITTER_BASELINE),
            ink_level=max(0, min(120, self.ink_level + rng.randint(-JITTER_INK, JITTER_INK))),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "font_index": self.font_index,
            "font_size": round(self.font_size, 3),
            "letter_spacing": round(self.letter_spacing, 3),
            "slant_degrees": round(self.slant_degrees, 3),
            "stroke_width": self.stroke_width,
            "baseline_offset": round(self.baseline_offset, 3),
            "ink_level": self.ink_level,
        }


def render_line(
    text: str,
    font_source: FontSource,
    style: WriterStyle,
    *,
    height: int = DEFAULT_HEIGHT,
    margin_ratio: float = 0.12,
) -> Image.Image:
    """Render one line of text.

    Pure and deterministic: identical arguments always yield identical pixels. Per-line variation
    comes from passing a jittered style, not from randomness in here.

    Returns:
        A grayscale ``L`` image of exactly ``height`` rows and variable width.
    """
    if not text:
        raise ValueError("Cannot render an empty line.")

    font = font_source.load(style.font_size)
    fill = int(style.ink_level)

    # Measure per-character so letter spacing can be applied. Drawing the whole string at once would
    # ignore the spacing parameter entirely.
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    advances = [probe.textlength(character, font=font) for character in text]
    text_width = sum(advances) + style.letter_spacing * max(0, len(text) - 1)

    pad = int(style.font_size * 1.2) + style.stroke_width * 3 + 4
    canvas_width = max(1, int(text_width)) + 2 * pad
    canvas_height = int(style.font_size * 2.4) + 2 * pad

    canvas = Image.new("L", (canvas_width, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    pen_x = float(pad)
    pen_y = float(pad) + style.baseline_offset
    for character, advance in zip(text, advances, strict=True):
        if character != " ":
            draw.text(
                (pen_x, pen_y),
                character,
                font=font,
                fill=fill,
                stroke_width=style.stroke_width,
                stroke_fill=fill,
            )
        pen_x += advance + style.letter_spacing

    if abs(style.slant_degrees) > 1e-6:
        canvas = _shear(canvas, style.slant_degrees)

    return _crop_and_normalise_height(canvas, height=height, margin_ratio=margin_ratio)


def _shear(image: Image.Image, degrees: float) -> Image.Image:
    """Slant an image horizontally, widening the canvas so nothing is clipped."""
    shear = math.tan(math.radians(degrees))
    width, height = image.size
    extra = int(abs(shear) * height) + 1
    widened = Image.new("L", (width + 2 * extra, height), BACKGROUND)
    widened.paste(image, (extra, 0))

    # AFFINE maps output (x, y) to input (x + a*y + c, y). Offsetting by shear*height keeps the
    # sheared content centred in the widened canvas.
    return widened.transform(
        widened.size,
        Image.AFFINE,
        (1.0, shear, -shear * height / 2.0, 0.0, 1.0, 0.0),
        resample=Image.BICUBIC,
        fillcolor=BACKGROUND,
    )


def _crop_and_normalise_height(
    image: Image.Image, *, height: int, margin_ratio: float
) -> Image.Image:
    """Trim to the inked region, add a proportional margin, and scale to exactly ``height``."""
    ink = ImageOps.invert(image).getbbox()
    if ink is None:
        # Whitespace-only render (e.g. a line of spaces). Emit a blank strip rather than crashing;
        # the corpus does not produce these, but a caller might.
        return Image.new("L", (max(1, height // 2), height), BACKGROUND)

    left, top, right, bottom = ink
    margin = max(1, int((bottom - top) * margin_ratio))
    box = (
        max(0, left - margin),
        max(0, top - margin),
        min(image.width, right + margin),
        min(image.height, bottom + margin),
    )
    cropped = image.crop(box)

    scale = height / cropped.height
    width = max(1, round(cropped.width * scale))
    return cropped.resize((width, height), Image.LANCZOS)


def image_difference(a: Image.Image, b: Image.Image) -> float:
    """Mean absolute pixel difference in [0, 1], after matching widths.

    Used by tests to assert that two writers render measurably differently, and that one writer's
    repeated renders differ only slightly.
    """
    width = min(a.width, b.width)
    height = min(a.height, b.height)
    left = a.convert("L").resize((width, height), Image.LANCZOS).tobytes()
    right = b.convert("L").resize((width, height), Image.LANCZOS).tobytes()
    total = sum(abs(x - y) for x, y in zip(left, right, strict=True))
    return total / (len(left) * 255)
