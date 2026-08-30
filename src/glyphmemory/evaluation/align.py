"""Character-level alignment with a full backtrace.

``glyphmemory.metrics.edit`` deliberately keeps only two DP rows — O(min(m, n)) memory — because it
only ever needs the S/I/D *counts*. Both require the actual edit sequence, not just its totals, so
this module keeps the full matrix and backtraces it. It exists alongside ``metrics.edit`` rather
than replacing it: full backtrace storage is unnecessary cost for the millions of calls a training
loop's CER makes, and is cheap here because evaluation runs once per line on a single split.

The tie-break rule is copied from ``metrics.edit.edit_counts`` exactly — substitution preferred over
deletion preferred over insertion — so this module's S/I/D totals agree with the trainer's CER on
the same input. ``test_evaluation_align.py`` asserts that agreement directly; the two
implementations drifting apart silently would mean the taxonomy describes a different set of errors
than the CER it is supposed to explain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

MATCH = "match"
SUBSTITUTION = "sub"
INSERTION = "ins"
DELETION = "del"


@dataclass(frozen=True, slots=True)
class AlignOp:
    """One step of the alignment between a reference and a hypothesis string.

    ``ref_char`` is ``None`` for an insertion (nothing in the reference), ``hyp_char`` is ``None``
    for a deletion (nothing in the hypothesis). Both are set for a match or a substitution.
    """

    kind: str
    ref_char: str | None
    hyp_char: str | None


def align_ops(reference: Sequence[str], hypothesis: Sequence[str]) -> list[AlignOp]:
    """The edit sequence turning ``reference`` into ``hypothesis``, in reference order.

    Same tie-break as :func:`glyphmemory.metrics.edit.edit_counts`: substitution over deletion over
    insertion. ``O(m*n)`` time and space — used once per evaluation line, not in a hot loop.
    """
    m, n = len(reference), len(hypothesis)
    if m == 0:
        return [AlignOp(INSERTION, None, hypothesis[j]) for j in range(n)]
    if n == 0:
        return [AlignOp(DELETION, reference[i], None) for i in range(m)]

    # cost[i][j] = edit distance between reference[:i] and hypothesis[:j]. back[i][j] = the op that
    # reached (i, j) from its predecessor cell.
    cost = [[0] * (n + 1) for _ in range(m + 1)]
    back: list[list[str]] = [[MATCH] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        cost[i][0] = i
        back[i][0] = DELETION
    for j in range(1, n + 1):
        cost[0][j] = j
        back[0][j] = INSERTION

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
                back[i][j] = MATCH
                continue

            sub_cost = cost[i - 1][j - 1] + 1
            del_cost = cost[i - 1][j] + 1
            ins_cost = cost[i][j - 1] + 1

            best_cost, best_op = sub_cost, SUBSTITUTION
            if del_cost < best_cost:
                best_cost, best_op = del_cost, DELETION
            if ins_cost < best_cost:
                best_cost, best_op = ins_cost, INSERTION

            cost[i][j] = best_cost
            back[i][j] = best_op

    ops: list[AlignOp] = []
    i, j = m, n
    while i > 0 or j > 0:
        op = back[i][j]
        if op == MATCH:
            ops.append(AlignOp(MATCH, reference[i - 1], hypothesis[j - 1]))
            i, j = i - 1, j - 1
        elif op == SUBSTITUTION:
            ops.append(AlignOp(SUBSTITUTION, reference[i - 1], hypothesis[j - 1]))
            i, j = i - 1, j - 1
        elif op == DELETION:
            ops.append(AlignOp(DELETION, reference[i - 1], None))
            i -= 1
        else:  # INSERTION
            ops.append(AlignOp(INSERTION, None, hypothesis[j - 1]))
            j -= 1
    ops.reverse()
    return ops
