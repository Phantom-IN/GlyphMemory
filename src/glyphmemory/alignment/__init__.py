"""CTC forced alignment — recovering where each character sits in time."""

from glyphmemory.alignment.checks import (
    AlignmentSanityReport,
    check_monotonic_and_nonoverlapping,
    non_blank_argmax_fraction,
    sanity_report,
    span_coverage_fraction,
    span_width_stats,
)
from glyphmemory.alignment.forced_align import (
    AlignedPath,
    AlignmentInfeasibleError,
    ForcedAlignment,
    forced_align,
    viterbi_align,
)
from glyphmemory.alignment.spans import AlignmentSpan

__all__ = [
    "AlignedPath",
    "AlignmentInfeasibleError",
    "AlignmentSanityReport",
    "AlignmentSpan",
    "ForcedAlignment",
    "check_monotonic_and_nonoverlapping",
    "forced_align",
    "non_blank_argmax_fraction",
    "sanity_report",
    "span_coverage_fraction",
    "span_width_stats",
    "viterbi_align",
]
