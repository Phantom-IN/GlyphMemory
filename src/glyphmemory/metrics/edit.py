"""Levenshtein distance with a substitution / insertion / deletion breakdown.

The breakdown is not decoration. Building it here, while it costs a few extra lines in an algorithm
being written anyway, is much cheaper than retrofitting it into a metric that already produced
numbers.

Two rows of the DP matrix are kept rather than the whole thing, so memory is ``O(min(m, n))``. A
backtrace would need the full matrix; instead each cell carries its running S/I/D counts, which
costs three extra integers per cell and no second pass.

Orientation, since it is the thing that gets mixed up: **the reference is what should have been
written, the hypothesis is what the model produced.** An *insertion* is a character the model added;
a *deletion* is one it dropped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EditCounts:
    """Edit operations turning a reference into a hypothesis."""

    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    reference_length: int = 0

    @property
    def total(self) -> int:
        """Levenshtein distance: the three operation counts sum to it by construction."""
        return self.substitutions + self.insertions + self.deletions

    @property
    def error_rate(self) -> float | None:
        """``total / reference_length``.

        ``None`` when the reference is empty — the rate is genuinely undefined there, and returning
        ``1.0`` or ``inf`` would silently poison any average taken over it. The corpus-level figure
        handles empty references correctly by summing numerator and denominator separately; see
        :mod:`glyphmemory.metrics.text`.
        """
        if self.reference_length == 0:
            return None
        return self.total / self.reference_length

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(
            substitutions=self.substitutions + other.substitutions,
            insertions=self.insertions + other.insertions,
            deletions=self.deletions + other.deletions,
            reference_length=self.reference_length + other.reference_length,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "total": self.total,
            "reference_length": self.reference_length,
            "error_rate": self.error_rate,
        }


def edit_counts(reference: Sequence[Any], hypothesis: Sequence[Any]) -> EditCounts:
    """Levenshtein distance from ``reference`` to ``hypothesis``, split by operation.

    Works on any sequence of comparable items: a string for CER, a list of words for WER.

    Ties are broken **substitution first, then deletion, then insertion**. Any consistent rule gives
    the same total distance; the split between operations can differ, so the rule is fixed here
    rather than left to whichever comparison Python evaluates first.
    """
    m, n = len(reference), len(hypothesis)
    if m == 0:
        return EditCounts(insertions=n, reference_length=0)
    if n == 0:
        return EditCounts(deletions=m, reference_length=m)

    # Row j of the previous reference position. Each cell holds (total, S, I, D).
    previous: list[tuple[int, int, int, int]] = [(j, 0, j, 0) for j in range(n + 1)]

    for i in range(1, m + 1):
        current: list[tuple[int, int, int, int]] = [(i, 0, 0, i)]
        for j in range(1, n + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                current.append(previous[j - 1])
                continue

            sub_total, sub_s, sub_i, sub_d = previous[j - 1]
            del_total, del_s, del_i, del_d = previous[j]
            ins_total, ins_s, ins_i, ins_d = current[j - 1]

            best = (sub_total + 1, sub_s + 1, sub_i, sub_d)
            if del_total + 1 < best[0]:
                best = (del_total + 1, del_s, del_i, del_d + 1)
            if ins_total + 1 < best[0]:
                best = (ins_total + 1, ins_s, ins_i + 1, ins_d)
            current.append(best)
        previous = current

    _, substitutions, insertions, deletions = previous[n]
    return EditCounts(
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        reference_length=m,
    )


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Levenshtein distance only, for callers that do not need the breakdown."""
    return edit_counts(reference, hypothesis).total
