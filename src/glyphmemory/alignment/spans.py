"""The unit every downstream writer-memory phase reads: one character's location in time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AlignmentSpan:
    """One target character's recovered temporal span.

    Attributes:
        token: The character this span was aligned to (not the CTC blank — blank never produces a
            span; it is the gap between them).
        start_t: First frame (inclusive) assigned to this character.
        end_t: Last frame **exclusive** — ``end_t - start_t`` is the span's frame count, and
            ``features[start_t:end_t]`` is the Python-idiomatic way to slice it out.
        score: Mean posterior probability of ``token`` over its span's frames, in ``[0, 1]``. Not
            the raw Viterbi path log-probability — that is a property of the *whole* alignment, not
            this one character, and is carried separately on
            :class:`~glyphmemory.alignment.forced_align.ForcedAlignment`.
    """

    token: str
    start_t: int
    end_t: int
    score: float

    def __post_init__(self) -> None:
        if self.end_t <= self.start_t:
            raise ValueError(
                f"AlignmentSpan for {self.token!r} has end_t ({self.end_t}) <= "
                f"start_t ({self.start_t}); every span must cover at least one frame."
            )

    @property
    def length(self) -> int:
        return self.end_t - self.start_t

    def as_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "start_t": self.start_t,
            "end_t": self.end_t,
            "score": self.score,
        }
