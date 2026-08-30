"""Is writer memory *complementary* to the base head, or *nested* inside it?

A nearest-class-mean readout over these features scores slightly below the base head.
**Aggregate accuracy cannot support the inference that fusion therefore cannot help**, in
either direction: two classifiers of similar accuracy can be

    nested         -- the weaker is right only where the stronger already is.
                      No readout can ever recover anything. Headroom is zero.
    complementary  -- the weaker is right on cases the stronger gets wrong.
                      A readout has something real to recover.

Distinguishing them needs the *joint* outcome per frame, which no earlier probe computed. This
module is that statistic, kept deliberately small and pure so it can be reasoned about and
tested without a model: :func:`complementarity` consumes two boolean sequences and nothing else.

**Two senses in which `oracle_accuracy` is optimistic, and neither may be dropped when quoting it**:

1. It is an **oracle combiner** -- it credits a frame whenever *either* source is right, which
   requires knowing which one to believe. No deployed system has that.
2. Frame labels come from **ground-truth forced alignment**. Enrollment has the transcript, so
   support-side alignment is legitimate; a query line at inference does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass(frozen=True, slots=True)
class Complementarity:
    """The joint outcome of two per-frame classifiers over the same frames.

    Every frame falls in exactly one of the four counts, so they sum to ``n_frames`` by construction
    -- asserted in this module's own tests rather than assumed.
    """

    #: base right, memory right
    both_correct: int
    #: base wrong, memory right -- the headroom a better readout could recover
    rescues: int
    #: base right, memory wrong -- what an indiscriminate readout would import
    damages: int
    #: neither right -- beyond the reach of any combination of these two
    both_wrong: int

    @property
    def n_frames(self) -> int:
        return self.both_correct + self.rescues + self.damages + self.both_wrong

    @property
    def base_correct(self) -> int:
        return self.both_correct + self.damages

    @property
    def memory_correct(self) -> int:
        return self.both_correct + self.rescues

    @property
    def base_accuracy(self) -> float | None:
        return self.base_correct / self.n_frames if self.n_frames else None

    @property
    def memory_accuracy(self) -> float | None:
        return self.memory_correct / self.n_frames if self.n_frames else None

    @property
    def oracle_accuracy(self) -> float | None:
        """Fraction where *either* source is right. **Upper bound only** -- see the module
        docstring's two caveats.
        """
        return (self.n_frames - self.both_wrong) / self.n_frames if self.n_frames else None

    @property
    def headroom(self) -> float | None:
        """``oracle_accuracy - base_accuracy``: what a perfect readout could add, at most."""
        if self.n_frames == 0:
            return None
        return (self.rescues) / self.n_frames

    @property
    def rescue_rate(self) -> float | None:
        """Of the frames the base head gets wrong, the fraction memory gets right."""
        wrong = self.rescues + self.both_wrong
        return self.rescues / wrong if wrong else None

    @property
    def damage_rate(self) -> float | None:
        """Of the frames the base head gets right, the fraction memory gets wrong."""
        return self.damages / self.base_correct if self.base_correct else None

    @property
    def rescue_damage_ratio(self) -> float | None:
        """``rescues / damages``. Below 1.0, trusting memory indiscriminately loses frames.

        ``None`` when there are no damages *and* no rescues (nothing to compare); when there are
        rescues but no damages the ratio is infinite, reported as ``float("inf")`` rather than
        silently clamped -- a real, if unlikely, outcome.
        """
        if self.damages == 0:
            return None if self.rescues == 0 else float("inf")
        return self.rescues / self.damages

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_frames": self.n_frames,
            "both_correct": self.both_correct,
            "rescues": self.rescues,
            "damages": self.damages,
            "both_wrong": self.both_wrong,
            "base_accuracy": self.base_accuracy,
            "memory_accuracy": self.memory_accuracy,
            "oracle_accuracy": self.oracle_accuracy,
            "headroom": self.headroom,
            "rescue_rate": self.rescue_rate,
            "damage_rate": self.damage_rate,
            "rescue_damage_ratio": self.rescue_damage_ratio,
        }


def complementarity(
    base_correct: Sequence[bool], memory_correct: Sequence[bool]
) -> Complementarity:
    """Joint outcome of two classifiers over the *same* frames, in the same order.

    Raises:
        ValueError: The two sequences differ in length -- which would silently pair frame ``i`` of
            one source with a different frame of the other and produce a meaningless result.
    """
    if len(base_correct) != len(memory_correct):
        raise ValueError(
            f"base_correct has {len(base_correct)} entries and memory_correct has "
            f"{len(memory_correct)}; they must describe the same frames in the same order."
        )
    both = rescues = damages = neither = 0
    for base_ok, memory_ok in zip(base_correct, memory_correct, strict=True):
        if base_ok and memory_ok:
            both += 1
        elif memory_ok:
            rescues += 1
        elif base_ok:
            damages += 1
        else:
            neither += 1
    return Complementarity(
        both_correct=both, rescues=rescues, damages=damages, both_wrong=neither
    )


@dataclass(frozen=True, slots=True)
class ReadoutScale:
    """The dynamic range V0 fusion has to work with, on one set of frames.

    `memory/fusion.py` computes ``base + alpha * gate(t) * score``. Whether that can change a
    decision depends on the size of ``alpha * gate * score`` against the base head's **top-2 logit
    margin** -- the gap the correction must close to flip an argmax.
    """

    n_frames: int
    score_min: float
    score_max: float
    logit_min: float
    logit_max: float
    margin_mean: float
    margin_median: float
    gate_mean: float
    alpha: float

    @property
    def max_correction(self) -> float:
        """The largest correction fusion can apply at ``alpha``: ``alpha * max_gate * max_score``.

        Uses ``gate <= 1`` by construction (``gate = 1 - P_blank``), so this is an upper bound that
        flatters the mechanism rather than a typical value.
        """
        return self.alpha * max(abs(self.score_min), abs(self.score_max))

    @property
    def margin_to_correction_ratio(self) -> float | None:
        """How many times larger the median decision margin is than the largest correction.

        Above ~1 the correction cannot flip a typical decision at all; the further above, the more
        structurally inert the readout is.
        """
        return self.margin_median / self.max_correction if self.max_correction else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_frames": self.n_frames,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "logit_min": self.logit_min,
            "logit_max": self.logit_max,
            "margin_mean": self.margin_mean,
            "margin_median": self.margin_median,
            "gate_mean": self.gate_mean,
            "alpha": self.alpha,
            "max_correction": self.max_correction,
            "margin_to_correction_ratio": self.margin_to_correction_ratio,
        }


def readout_scale(logits: Tensor, scores: Tensor, *, alpha: float) -> ReadoutScale:
    """Measure the readout's dynamic range on ``[T, V]`` logits and matching memory scores.

    Args:
        logits: Base head logits, ``[T, V]``.
        scores: Memory scores aligned to the same frames and classes, ``[T, V]``.
        alpha: The fusion strength the measurement is reported at.

    Raises:
        ValueError: Shapes disagree, or ``logits`` has fewer than two classes (a top-2 margin is
            undefined with one class).
    """
    if logits.shape != scores.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and scores {tuple(scores.shape)} differ.")
    if logits.ndim != 2 or logits.shape[-1] < 2:
        raise ValueError(f"logits must be [T, V] with V >= 2, got {tuple(logits.shape)}.")

    top2 = logits.topk(2, dim=-1).values
    margins = top2[:, 0] - top2[:, 1]
    probabilities = logits.softmax(dim=-1)
    gate = 1.0 - probabilities[:, 0]  # blank is index 0
    return ReadoutScale(
        n_frames=int(logits.shape[0]),
        score_min=float(scores.min()),
        score_max=float(scores.max()),
        logit_min=float(logits.min()),
        logit_max=float(logits.max()),
        margin_mean=float(margins.mean()),
        margin_median=float(margins.median()),
        gate_mean=float(gate.mean()),
        alpha=alpha,
    )
