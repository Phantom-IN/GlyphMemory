"""CTC forced alignment: a Viterbi search over the CTC lattice.

Given per-frame log-probabilities and a **known** target sequence, recover the single most likely
path through the CTC extended-label lattice, then read off each target character's contiguous frame
span from that path.

**Why hand-rolled rather than a library call.** This is short enough to own outright, and owning it
means the span/score shape is exactly what the rest of the project needs.

**The extended lattice.** A target of ``L`` tokens becomes a sequence of ``S = 2L + 1`` lattice
states by interleaving blanks: ``[blank, t_1, blank, t_2, blank, ..., t_L, blank]``. A valid path is
non-decreasing in lattice-state index, advances by at most 2 per frame, and occupies every state at
least one frame — which is exactly why alignment is infeasible when ``S > T``
(``AlignmentInfeasibleError``), the same feasibility boundary ``model/loss.py`` already enforces for
the loss itself.

**Repeated characters are the constraint that shapes the recursion.** Skipping directly from one
label state to the label two positions later (``s-2 -> s``, bypassing the blank between them) is
only legal when the two labels differ. Two adjacent identical target characters (``"ll"``) therefore
**require** the blank between their extended-lattice states to be visited explicitly — omit this
check and repeated characters silently collapse into one span.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from glyphmemory.alignment.spans import AlignmentSpan
from glyphmemory.ctc.tokenizer import BLANK_INDEX, Charset

NEG_INF = float("-inf")


class AlignmentInfeasibleError(ValueError):
    """Raised when the target cannot possibly fit in the available frames.

    The extended lattice needs ``2*len(target) + 1`` states, each occupied at least one frame, so
    alignment is impossible once that exceeds ``T``. This is the same boundary ``model/loss.py``'s
    CTC feasibility check enforces for the loss — this module names it explicitly rather than
    letting the DP fail silently with every state at ``-inf``.
    """


@dataclass(frozen=True, slots=True)
class AlignedPath:
    """The raw Viterbi output: one extended-lattice state per input frame.

    Low-level and index-only — no character identity, no charset. :func:`forced_align` builds on
    this to produce the character-labeled :class:`~glyphmemory.alignment.spans.AlignmentSpan`
    sequence callers actually want.
    """

    states: tuple[int, ...]
    log_prob: float


def _extended_sequence(target_indices: Sequence[int], blank: int) -> list[int]:
    extended = [blank]
    for index in target_indices:
        extended.append(index)
        extended.append(blank)
    return extended


def viterbi_align(
    log_probs: Tensor, target_indices: Sequence[int], *, blank: int = BLANK_INDEX
) -> AlignedPath:
    """The Viterbi core: best path through the CTC extended lattice.

    Args:
        log_probs: ``[T, C]``, already ``log_softmax``'d — the same convention ``model/loss.py``'s
            ``ctc_loss`` uses when it computes the loss itself. Applying ``log_softmax`` again here
            would silently double-normalize.
        target_indices: Class indices of the target sequence, in order. **Not** including the blank
            — blanks are inserted automatically to build the extended lattice.
        blank: The CTC blank class index.

    Tie-break, fixed and deterministic: at each lattice state, the incoming transition is checked in
    the order *stay* → *advance by 1* → *advance by 2*, and only a **strictly** higher score
    displaces the current best. Ties therefore always prefer the transition that advances least —
    the same "smallest well-defined rule beats an implicit one" reasoning
    ``metrics.edit.edit_counts`` applies to its own tie-break. At the final frame, if both the last
    label and the trailing blank are equally likely end states, the label wins — it is the more
    useful endpoint for span extraction and is no less valid a CTC path.
    """
    if log_probs.dim() != 2:
        raise ValueError(f"log_probs must be [T, C], got shape {tuple(log_probs.shape)}")

    t_total = log_probs.shape[0]
    extended = _extended_sequence(target_indices, blank)
    s_total = len(extended)

    if s_total > t_total:
        raise AlignmentInfeasibleError(
            f"target of {len(target_indices)} tokens needs {s_total} lattice states "
            f"(including blanks) but only {t_total} frames are available; forced alignment "
            "is infeasible at this width."
        )

    lp = log_probs.tolist()  # plain floats: the DP is control-flow heavy, not tensor-heavy

    alpha: list[list[float]] = [[NEG_INF] * s_total for _ in range(t_total)]
    backptr: list[list[int]] = [[0] * s_total for _ in range(t_total)]

    alpha[0][0] = lp[0][extended[0]]
    if s_total > 1:
        alpha[0][1] = lp[0][extended[1]]

    for t in range(1, t_total):
        row = lp[t]
        prev = alpha[t - 1]
        for s in range(s_total):
            best_val = prev[s]
            best_from = 0
            if s - 1 >= 0 and prev[s - 1] > best_val:
                best_val = prev[s - 1]
                best_from = 1
            if (
                s - 2 >= 0
                and s % 2 == 1
                and extended[s] != extended[s - 2]
                and prev[s - 2] > best_val
            ):
                best_val = prev[s - 2]
                best_from = 2
            backptr[t][s] = best_from
            alpha[t][s] = best_val + row[extended[s]] if best_val != NEG_INF else NEG_INF

    if s_total >= 2 and alpha[t_total - 1][s_total - 2] >= alpha[t_total - 1][s_total - 1]:
        end_state = s_total - 2
    else:
        end_state = s_total - 1
    end_log_prob = alpha[t_total - 1][end_state]

    if end_log_prob == NEG_INF:
        raise AlignmentInfeasibleError(
            "no valid path exists through the lattice for this target and these log-"
            "probabilities (every terminal state has probability zero)."
        )

    states = [0] * t_total
    s = end_state
    for t in range(t_total - 1, -1, -1):
        states[t] = s
        offset = backptr[t][s]
        s -= offset

    return AlignedPath(states=tuple(states), log_prob=end_log_prob)


@dataclass(frozen=True, slots=True)
class ForcedAlignment:
    """A complete alignment: one span per target character, plus the path's overall score."""

    spans: tuple[AlignmentSpan, ...]
    log_prob: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "spans": [span.as_dict() for span in self.spans],
            "log_prob": self.log_prob,
        }


def _spans_from_path(
    path: AlignedPath, target_text: str, log_probs: Tensor, extended: list[int]
) -> tuple[AlignmentSpan, ...]:
    spans: list[AlignmentSpan] = []
    probs = log_probs.exp()
    for k, character in enumerate(target_text):
        label_state = 2 * k + 1
        frames = [t for t, s in enumerate(path.states) if s == label_state]
        if not frames:
            # Unreachable given a feasible path (every lattice state is visited at least once once S
            # <= T holds), but guarded rather than silently producing a bad span.
            raise AlignmentInfeasibleError(
                f"character {k} ({character!r}) was never visited by the recovered path — "
                "this indicates a bug in the Viterbi recursion, not a data problem."
            )
        start_t, end_t = frames[0], frames[-1] + 1
        class_index = extended[label_state]
        score = float(probs[start_t:end_t, class_index].mean())
        spans.append(AlignmentSpan(token=character, start_t=start_t, end_t=end_t, score=score))
    return tuple(spans)


def forced_align(
    log_probs: Tensor, target_text: str, charset: Charset, *, blank: int = BLANK_INDEX
) -> ForcedAlignment:
    """Align ``log_probs`` against the **known** ``target_text``, character by character.

    The convenience layer over :func:`viterbi_align`: encodes ``target_text`` through ``charset``,
    runs the Viterbi search, and reads off one :class:`AlignmentSpan` per character with its mean
    posterior probability as a confidence score. ``target_text`` is assumed already normalized (the
    same text a batch's collator would have encoded) — this module does not apply normalization
    itself, matching the low-level, policy-free role ``metrics.edit`` plays relative to
    ``metrics.text``.
    """
    target_indices = [charset.index_of(character) for character in target_text]
    extended = _extended_sequence(target_indices, blank)
    path = viterbi_align(log_probs, target_indices, blank=blank)
    spans = _spans_from_path(path, target_text, log_probs, extended)
    return ForcedAlignment(spans=spans, log_prob=path.log_prob)
