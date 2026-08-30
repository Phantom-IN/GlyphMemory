"""Per-writer CER distribution and the heavy-tail condition.

Each writer's own CER is **micro**-averaged (edits summed over that writer's lines, divided by that
writer's total reference length) — the same convention `metrics.text` uses for the corpus figure, so
a writer's number is computed the same way the number it is compared against is. The **worst-decile
CER** is the pooled micro CER over the worst ~10% of writers' lines combined, not the mean of their
individual rates, for the same reason: pooling before dividing avoids a single two-line writer
swinging the aggregate as much as a fifty-line one.

A writer's CER computed from very few lines is noisy.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from glyphmemory.ctc.normalization import NFC_V1, NormalizationPolicy
from glyphmemory.metrics.edit import EditCounts
from glyphmemory.metrics.text import character_counts


@dataclass(frozen=True, slots=True)
class WriterCER:
    """One writer's line count and micro-averaged CER."""

    writer_id: str
    lines: int
    counts: EditCounts

    @property
    def cer(self) -> float | None:
        return self.counts.error_rate

    def as_dict(self) -> dict[str, Any]:
        return {
            "writer_id": self.writer_id,
            "lines": self.lines,
            "cer": self.cer,
            **self.counts.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class PerWriterDistribution:
    """The per-writer CER distribution and the heavy-tail statistic.

    ``worst_decile_ratio`` is ``None`` when the median is zero or undefined — a ratio against zero
    is not meaningful, and reporting one silently would misstate a perfect-median result as an
    infinite tail.
    """

    writers: tuple[WriterCER, ...]
    median_cer: float | None
    worst_decile_writers: tuple[str, ...]
    worst_decile_cer: float | None
    worst_decile_ratio: float | None
    tail_threshold: float = 2.0

    @property
    def n_writers(self) -> int:
        return len(self.writers)

    @property
    def passes_tail_condition(self) -> bool | None:
        """``True`` iff worst-decile CER >= ``tail_threshold`` x median. ``None`` if undefined."""
        if self.worst_decile_ratio is None:
            return None
        return self.worst_decile_ratio >= self.tail_threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_writers": self.n_writers,
            "median_cer": self.median_cer,
            "worst_decile_writers": list(self.worst_decile_writers),
            "worst_decile_cer": self.worst_decile_cer,
            "worst_decile_ratio": self.worst_decile_ratio,
            "tail_threshold": self.tail_threshold,
            "passes_tail_condition": self.passes_tail_condition,
            "by_writer": [w.as_dict() for w in self.writers],
        }

    def format(self) -> str:
        ratio = "n/a" if self.worst_decile_ratio is None else f"{self.worst_decile_ratio:.2f}x"
        median = "n/a" if self.median_cer is None else f"{self.median_cer:.4f}"
        worst = "n/a" if self.worst_decile_cer is None else f"{self.worst_decile_cer:.4f}"
        verdict = self.passes_tail_condition
        verdict_text = "n/a" if verdict is None else ("PASS" if verdict else "FAIL")
        return (
            f"per-writer CER   {self.n_writers} writers\n"
            f"  median               {median}\n"
            f"  worst-decile ({len(self.worst_decile_writers)} writers)   {worst}\n"
            f"  worst-decile / median   {ratio}   "
            f"(>= {self.tail_threshold}x required)   {verdict_text}"
        )


def per_writer_distribution(
    records: Iterable[tuple[str, str, str]], *, policy: NormalizationPolicy = NFC_V1
) -> PerWriterDistribution:
    """Build the per-writer CER distribution from ``(writer_id, reference, hypothesis)`` triples."""
    pooled: dict[str, EditCounts] = defaultdict(EditCounts)
    line_counts: dict[str, int] = defaultdict(int)

    for writer_id, reference, hypothesis in records:
        counts = character_counts(reference, hypothesis, policy=policy)
        pooled[writer_id] = pooled[writer_id] + counts
        line_counts[writer_id] += 1

    writers = tuple(
        WriterCER(writer_id=writer_id, lines=line_counts[writer_id], counts=pooled[writer_id])
        for writer_id in sorted(pooled)
    )

    scored = [w for w in writers if w.cer is not None]
    if not scored:
        return PerWriterDistribution(
            writers=writers,
            median_cer=None,
            worst_decile_writers=(),
            worst_decile_cer=None,
            worst_decile_ratio=None,
        )

    median_cer = statistics.median(w.cer for w in scored)  # type: ignore[misc]

    ranked = sorted(scored, key=lambda w: w.cer, reverse=True)  # type: ignore[arg-type]
    decile_n = max(1, math.ceil(0.1 * len(ranked)))
    worst = ranked[:decile_n]
    worst_counts = EditCounts()
    for w in worst:
        worst_counts = worst_counts + w.counts
    worst_decile_cer = worst_counts.error_rate

    worst_decile_ratio = None
    if worst_decile_cer is not None and median_cer is not None and median_cer > 0:
        worst_decile_ratio = worst_decile_cer / median_cer

    return PerWriterDistribution(
        writers=writers,
        median_cer=median_cer,
        worst_decile_writers=tuple(w.writer_id for w in worst),
        worst_decile_cer=worst_decile_cer,
        worst_decile_ratio=worst_decile_ratio,
    )
