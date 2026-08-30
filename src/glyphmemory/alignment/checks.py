"""Sanity checks for a recovered alignment, beyond eyeballing it.

With no per-character ground-truth boxes available for real handwriting
(`src/glyphmemory/data/adapters/iam.py` deliberately never retains IAM's word geometry), these
checks are what "sane" can mean without one: internal consistency (monotonic, non-overlapping —
guaranteed by construction for a *correct* Viterbi path, so a violation here would mean a bug, not
just a bad model) and plausibility against independent signals the model itself provides (the
non-blank argmax rate, the ~2.5-3 frames/character design estimate).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from torch import Tensor

from glyphmemory.alignment.spans import AlignmentSpan
from glyphmemory.ctc.tokenizer import BLANK_INDEX


def check_monotonic_and_nonoverlapping(spans: Sequence[AlignmentSpan]) -> tuple[str, ...]:
    """Violations of temporal order — spans must never go backward or overlap.

    A correct Viterbi path guarantees this by construction (lattice-state index is non-decreasing),
    so any violation reported here means either a bug in the aligner or that ``spans`` did not
    actually come from :func:`~glyphmemory.alignment.forced_align.forced_align`. Returns the empty
    tuple when everything is in order.
    """
    violations: list[str] = []
    for earlier, later in pairwise(spans):
        if later.start_t < earlier.start_t:
            violations.append(
                f"{later.token!r} (start {later.start_t}) starts before "
                f"{earlier.token!r} (start {earlier.start_t})"
            )
        elif later.start_t < earlier.end_t:
            violations.append(
                f"{earlier.token!r} (ends {earlier.end_t}) overlaps "
                f"{later.token!r} (starts {later.start_t})"
            )
    return tuple(violations)


def non_blank_argmax_fraction(log_probs: Tensor, *, blank: int = BLANK_INDEX) -> float:
    """Fraction of frames whose most likely class is not blank — an independent signal for how much
    of the line the model itself believes carries a character, to compare span coverage against.
    """
    if log_probs.shape[0] == 0:
        return 0.0
    return float((log_probs.argmax(dim=-1) != blank).float().mean())


def span_coverage_fraction(spans: Sequence[AlignmentSpan], num_frames: int) -> float:
    """Fraction of the line's frames claimed by some span. Not expected to equal
    :func:`non_blank_argmax_fraction` exactly — a span always claims at least one frame even for a
    character the model is unsure about, while argmax only counts frames where blank loses outright
    — but a large, unexplained gap between the two is a red flag worth a look.
    """
    if num_frames == 0:
        return 0.0
    covered = sum(span.length for span in spans)
    return covered / num_frames


def span_width_stats(spans: Sequence[AlignmentSpan]) -> dict[str, float]:
    """Mean/median/min/max span width in frames, to sanity-check against the ~2.5-3 frames/character
    design estimate at 4x downsampling.
    """
    if not spans:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    widths = [span.length for span in spans]
    return {
        "mean": statistics.mean(widths),
        "median": statistics.median(widths),
        "min": min(widths),
        "max": max(widths),
    }


@dataclass(frozen=True, slots=True)
class AlignmentSanityReport:
    """Everything checks about one recovered alignment, beyond eyeballing it."""

    violations: tuple[str, ...]
    span_coverage: float
    non_blank_argmax: float
    width_stats: dict[str, float] = field(default_factory=dict)
    mean_score: float = 0.0

    @property
    def is_clean(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, Any]:
        return {
            "violations": list(self.violations),
            "is_clean": self.is_clean,
            "span_coverage": self.span_coverage,
            "non_blank_argmax": self.non_blank_argmax,
            "width_stats": self.width_stats,
            "mean_score": self.mean_score,
        }


def sanity_report(
    spans: Sequence[AlignmentSpan], log_probs: Tensor, *, blank: int = BLANK_INDEX
) -> AlignmentSanityReport:
    """Run every check in this module against one alignment and its source log-probabilities."""
    num_frames = log_probs.shape[0]
    mean_score = statistics.mean(span.score for span in spans) if spans else 0.0
    return AlignmentSanityReport(
        violations=check_monotonic_and_nonoverlapping(spans),
        span_coverage=span_coverage_fraction(spans, num_frames),
        non_blank_argmax=non_blank_argmax_fraction(log_probs, blank=blank),
        width_stats=span_width_stats(spans),
        mean_score=mean_score,
    )
