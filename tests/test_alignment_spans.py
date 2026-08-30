"""AlignmentSpan: the shape every downstream phase reads."""

from __future__ import annotations

import pytest

from glyphmemory.alignment import AlignmentSpan


def test_span_holds_its_fields():
    span = AlignmentSpan(token="r", start_t=12, end_t=16, score=0.87)
    assert span.token == "r"
    assert span.start_t == 12
    assert span.end_t == 16
    assert span.score == 0.87


def test_length_is_end_minus_start():
    span = AlignmentSpan(token="a", start_t=3, end_t=7, score=1.0)
    assert span.length == 4


def test_end_t_must_exceed_start_t():
    with pytest.raises(ValueError, match="end_t"):
        AlignmentSpan(token="a", start_t=5, end_t=5, score=1.0)
    with pytest.raises(ValueError, match="end_t"):
        AlignmentSpan(token="a", start_t=5, end_t=3, score=1.0)


def test_as_dict_round_trips_the_fields():
    span = AlignmentSpan(token="x", start_t=0, end_t=2, score=0.5)
    assert span.as_dict() == {"token": "x", "start_t": 0, "end_t": 2, "score": 0.5}


def test_span_is_immutable():
    span = AlignmentSpan(token="a", start_t=0, end_t=1, score=1.0)
    with pytest.raises(AttributeError):
        span.start_t = 5  # type: ignore[misc]
