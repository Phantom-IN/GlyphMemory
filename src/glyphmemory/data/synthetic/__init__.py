"""Synthetic line generation — a correctness harness, never a performance benchmark."""

from glyphmemory.data.synthetic.corpus import (
    CORPUS_MODES,
    DEFAULT_MIN_OCCURRENCES,
    WORDS,
    coverage_counts,
    missing_coverage,
    sample_lines,
)
from glyphmemory.data.synthetic.fonts import (
    BUNDLED_FONT_NAME,
    FontSource,
    bundled_font,
    discover_fonts,
    resolve_fonts,
)
from glyphmemory.data.synthetic.render import (
    DEFAULT_HEIGHT,
    WriterStyle,
    image_difference,
    render_line,
)

__all__ = [
    "BUNDLED_FONT_NAME",
    "CORPUS_MODES",
    "DEFAULT_HEIGHT",
    "DEFAULT_MIN_OCCURRENCES",
    "WORDS",
    "FontSource",
    "WriterStyle",
    "bundled_font",
    "coverage_counts",
    "discover_fonts",
    "image_difference",
    "missing_coverage",
    "render_line",
    "resolve_fonts",
    "sample_lines",
]
