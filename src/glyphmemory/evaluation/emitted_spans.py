"""Character occurrences as the **base model itself emits them**, and their alignment to truth.

Every personalization number this project has produced was measured on spans obtained by
force-aligning the model's posteriors to the *reference transcript*. Enrollment may legitimately do
that — it has the transcript. **Inference may not.** A deployed system knows only what it emitted,
so a reranking mechanism has exactly the character slots its own decoding produced: no slot for a
character it never emitted, and a slot in the wrong place when it segmented badly.

This module builds that inference-time view:

    emitted_occurrences()  greedy CTC decoding -> one span per emitted character, with the
                           candidate set and confidence available at that span
    align_operations()     Levenshtein backtrace between emitted text and reference, so each
                           emitted occurrence can be labelled CORRECT / SUBSTITUTION / INSERTION,
                           and reference characters with no emitted slot as DELETION

**The reference is used only to label outcomes, never to define spans.** That separation is the
entire point of Gate C2, and it is why `emitted_occurrences` never sees the transcript.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from torch import Tensor

from glyphmemory.ctc.decode import BLANK_INDEX
from glyphmemory.ctc.tokenizer import Charset


class Outcome(StrEnum):
    """How an aligned position turned out for the base model."""

    CORRECT = "correct"
    SUBSTITUTION = "substitution"
    INSERTION = "insertion"
    DELETION = "deletion"


@dataclass(frozen=True, slots=True)
class EmittedOccurrence:
    """One character the base model actually emitted, and what it knew at that moment.

    ``start``/``end`` are inclusive frame indices of the argmax run that produced this character.
    ``candidates`` is the base head's own top-k at ``peak`` — the slot's candidate set, which is
    what a reranking mechanism is allowed to choose among.
    """

    index: int
    character: str
    start: int
    end: int
    peak: int
    confidence: float
    margin: float
    entropy: float
    candidates: tuple[str, ...]

    @property
    def frames(self) -> int:
        return self.end - self.start + 1


def emitted_occurrences(
    logits: Tensor,
    charset: Charset,
    *,
    length: int | None = None,
    top_k: int = 5,
    blank: int = BLANK_INDEX,
) -> list[EmittedOccurrence]:
    """Greedy-decode ``[T, C]`` logits into one occurrence per emitted character.

    Follows CTC collapsing exactly: a run of identical non-blank argmax labels is **one** character,
    and a blank closes the run, so ``a a <blank> a`` emits two ``a``s rather than one. Getting this
    wrong would silently merge or split slots and corrupt every downstream count.

    Args:
        length: Valid frames; frames at or beyond it are padding and are not read.
        top_k: Size of the candidate set recorded per occurrence.

    Raises:
        ValueError: ``logits`` is not 2-D, ``length`` is out of range, or ``top_k`` < 1.
    """
    if logits.dim() != 2:
        raise ValueError(f"Expected [T, C], got shape {tuple(logits.shape)}")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    time_steps, classes = logits.shape
    if length is None:
        length = time_steps
    if not 0 <= length <= time_steps:
        raise ValueError(f"length {length} outside [0, {time_steps}]")
    if length == 0:
        return []

    valid = logits[:length]
    argmax = valid.argmax(dim=-1).tolist()
    probabilities = valid.softmax(dim=-1)
    confidence = probabilities.max(dim=-1).values.tolist()
    entropy = (-(probabilities * (probabilities + 1e-12).log()).sum(-1)).tolist()
    k = min(top_k, classes)
    top = valid.topk(k, dim=-1)
    top_indices = top.indices.tolist()
    margins = (
        (top.values[:, 0] - top.values[:, 1]).tolist()
        if k > 1
        else [0.0] * length
    )

    runs: list[list[int]] = []
    previous = blank
    for frame, label in enumerate(argmax):
        if label == blank:
            previous = blank
            continue
        if label != previous:
            runs.append([frame])
        else:
            runs[-1].append(frame)
        previous = label

    occurrences: list[EmittedOccurrence] = []
    for position, frames in enumerate(runs):
        peak = max(frames, key=lambda f: confidence[f])
        occurrences.append(
            EmittedOccurrence(
                index=position,
                character=charset.char_at(argmax[frames[0]]),
                start=frames[0],
                end=frames[-1],
                peak=peak,
                confidence=confidence[peak],
                margin=margins[peak],
                entropy=entropy[peak],
                candidates=tuple(charset.char_at(i) for i in top_indices[peak]),
            )
        )
    return occurrences


@dataclass(frozen=True, slots=True)
class AlignedPosition:
    """One aligned position between the emitted sequence and the reference."""

    outcome: Outcome
    #: Index into the emitted sequence, or ``None`` for a deletion (no emitted slot exists).
    emitted_index: int | None
    #: Index into the reference, or ``None`` for an insertion.
    reference_index: int | None
    emitted: str | None
    reference: str | None


def align_operations(
    reference: Sequence[Any], hypothesis: Sequence[Any]
) -> list[AlignedPosition]:
    """Levenshtein alignment with backtrace, reference against hypothesis.

    `metrics/edit.py::edit_counts` deliberately avoids a backtrace (it carries running counts
    instead, which is cheaper and enough for CER). R2 needs the correspondence itself — which
    emitted character stands where a reference character should be — so this computes the full
    matrix and walks it back.

    **A deletion produces an `AlignedPosition` with `emitted_index=None`**, which is the whole
    point: there is no slot for a reranker to act on, and fabricating one would invent headroom that
    a deployed system cannot reach.
    """
    n, m = len(reference), len(hypothesis)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                cost[i][j] = 1 + min(cost[i - 1][j - 1], cost[i - 1][j], cost[i][j - 1])

    positions: list[AlignedPosition] = []
    i, j = n, m
    while i > 0 or j > 0:
        matched = (
            i > 0
            and j > 0
            and reference[i - 1] == hypothesis[j - 1]
            and cost[i][j] == cost[i - 1][j - 1]
        )
        if matched:
            positions.append(AlignedPosition(Outcome.CORRECT, j - 1, i - 1,
                                             hypothesis[j - 1], reference[i - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            positions.append(AlignedPosition(Outcome.SUBSTITUTION, j - 1, i - 1,
                                             hypothesis[j - 1], reference[i - 1]))
            i, j = i - 1, j - 1
        elif j > 0 and cost[i][j] == cost[i][j - 1] + 1:
            positions.append(AlignedPosition(Outcome.INSERTION, j - 1, None,
                                             hypothesis[j - 1], None))
            j -= 1
        else:
            positions.append(AlignedPosition(Outcome.DELETION, None, i - 1,
                                             None, reference[i - 1]))
            i -= 1
    positions.reverse()
    return positions


def outcome_counts(positions: Sequence[AlignedPosition]) -> dict[str, int]:
    """Tally of outcomes, for the reachable/unreachable accounting R2 reports."""
    counts = {outcome.value: 0 for outcome in Outcome}
    for position in positions:
        counts[position.outcome.value] += 1
    return counts
