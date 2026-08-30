"""Recognition metrics."""

from glyphmemory.metrics.edit import EditCounts, edit_counts, edit_distance
from glyphmemory.metrics.text import (
    MACRO,
    MICRO,
    MetricResult,
    SampleMetric,
    cer,
    character_counts,
    corpus_cer,
    corpus_wer,
    macro_cer,
    wer,
    word_counts,
)

__all__ = [
    "MACRO",
    "MICRO",
    "EditCounts",
    "MetricResult",
    "SampleMetric",
    "cer",
    "character_counts",
    "corpus_cer",
    "corpus_wer",
    "edit_counts",
    "edit_distance",
    "macro_cer",
    "wer",
    "word_counts",
]
