"""Sanity checks for a recovered alignment: monotonicity, coverage, width plausibility."""

from __future__ import annotations

import pytest
import torch

from glyphmemory.alignment import (
    AlignmentSpan,
    check_monotonic_and_nonoverlapping,
    non_blank_argmax_fraction,
    sanity_report,
    span_coverage_fraction,
    span_width_stats,
)

# --------------------------------------------------------------------------- monotonic/overlap


def test_clean_spans_have_no_violations():
    spans = [
        AlignmentSpan(token="a", start_t=0, end_t=3, score=0.9),
        AlignmentSpan(token="b", start_t=3, end_t=6, score=0.9),
        AlignmentSpan(token="c", start_t=6, end_t=9, score=0.9),
    ]
    assert check_monotonic_and_nonoverlapping(spans) == ()


def test_overlapping_spans_are_flagged():
    spans = [
        AlignmentSpan(token="a", start_t=0, end_t=5, score=0.9),
        AlignmentSpan(token="b", start_t=3, end_t=8, score=0.9),
    ]
    violations = check_monotonic_and_nonoverlapping(spans)
    assert len(violations) == 1
    assert "overlaps" in violations[0]


def test_out_of_order_spans_are_flagged():
    spans = [
        AlignmentSpan(token="a", start_t=5, end_t=9, score=0.9),
        AlignmentSpan(token="b", start_t=0, end_t=3, score=0.9),
    ]
    violations = check_monotonic_and_nonoverlapping(spans)
    assert len(violations) == 1
    assert "starts before" in violations[0]


def test_adjacent_touching_spans_are_not_a_violation():
    # end_t is exclusive, so a span ending at 3 and the next starting at 3 do not overlap.
    spans = [
        AlignmentSpan(token="a", start_t=0, end_t=3, score=0.9),
        AlignmentSpan(token="b", start_t=3, end_t=6, score=0.9),
    ]
    assert check_monotonic_and_nonoverlapping(spans) == ()


def test_empty_and_single_span_are_trivially_clean():
    assert check_monotonic_and_nonoverlapping([]) == ()
    assert check_monotonic_and_nonoverlapping(
        [AlignmentSpan(token="a", start_t=0, end_t=1, score=1.0)]
    ) == ()


# --------------------------------------------------------------------------- coverage


def test_non_blank_argmax_fraction_counts_correctly():
    # blank=0. Frames: blank, label, label, blank -> 2/4 non-blank.
    log_probs = torch.log(
        torch.tensor(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.7, 0.3],
            ]
        )
    )
    assert non_blank_argmax_fraction(log_probs, blank=0) == 0.5


def test_non_blank_argmax_fraction_handles_zero_frames():
    assert non_blank_argmax_fraction(torch.zeros(0, 3)) == 0.0


def test_span_coverage_fraction():
    spans = [
        AlignmentSpan(token="a", start_t=0, end_t=2, score=0.9),
        AlignmentSpan(token="b", start_t=4, end_t=8, score=0.9),
    ]
    assert span_coverage_fraction(spans, num_frames=10) == (2 + 4) / 10


def test_span_coverage_fraction_handles_zero_frames():
    assert span_coverage_fraction([], num_frames=0) == 0.0


# --------------------------------------------------------------------------- width stats


def test_span_width_stats():
    spans = [
        AlignmentSpan(token="a", start_t=0, end_t=2, score=0.9),  # width 2
        AlignmentSpan(token="b", start_t=2, end_t=6, score=0.9),  # width 4
        AlignmentSpan(token="c", start_t=6, end_t=9, score=0.9),  # width 3
    ]
    stats = span_width_stats(spans)
    assert stats["mean"] == 3.0
    assert stats["median"] == 3.0
    assert stats["min"] == 2
    assert stats["max"] == 4


def test_span_width_stats_empty():
    stats = span_width_stats([])
    assert stats == {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}


# --------------------------------------------------------------------------- full report


def test_sanity_report_bundles_everything():
    spans = [
        AlignmentSpan(token="a", start_t=0, end_t=2, score=0.9),
        AlignmentSpan(token="b", start_t=2, end_t=4, score=0.8),
    ]
    log_probs = torch.log(torch.full((4, 3), 1 / 3))
    report = sanity_report(spans, log_probs, blank=0)
    assert report.is_clean
    assert report.mean_score == pytest.approx(0.85)
    assert report.width_stats["mean"] == 2.0
    payload = report.as_dict()
    assert payload["is_clean"] is True
    assert payload["violations"] == []


def test_sanity_report_surfaces_a_dirty_alignment():
    spans = [
        AlignmentSpan(token="a", start_t=0, end_t=5, score=0.9),
        AlignmentSpan(token="b", start_t=2, end_t=7, score=0.9),
    ]
    log_probs = torch.log(torch.full((7, 3), 1 / 3))
    report = sanity_report(spans, log_probs)
    assert not report.is_clean
    assert len(report.violations) == 1
