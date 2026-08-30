"""Corpus statistics that constrain the research design.

The reason is specific — a writer needs enough lines to supply ``n`` support lines **and** a
meaningful query pool, and per-writer line counts are heavily skewed. The number of writers that can
actually sustain ``CER@10`` is therefore much smaller than the writer count, and reporting
``CER@10`` without knowing that number means reporting a figure whose noise floor is unknown.

Nothing here decides anything. It measures, so that a decision made later is made on evidence.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from glyphmemory.data.manifest import ManifestRecord

#: Shot counts a few-shot protocol reports at.
DEFAULT_THRESHOLDS: tuple[int, ...] = (1, 3, 5, 10, 15, 20)


@dataclass(frozen=True, slots=True)
class WriterHistogram:
    """Lines per writer, and how many writers clear each shot threshold."""

    lines_per_writer: dict[str, int]
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS

    @property
    def n_writers(self) -> int:
        return len(self.lines_per_writer)

    @property
    def total_lines(self) -> int:
        return sum(self.lines_per_writer.values())

    @property
    def counts(self) -> list[int]:
        """Line counts, sorted ascending."""
        return sorted(self.lines_per_writer.values())

    @property
    def minimum(self) -> int:
        return min(self.counts, default=0)

    @property
    def maximum(self) -> int:
        return max(self.counts, default=0)

    @property
    def mean(self) -> float:
        return self.total_lines / self.n_writers if self.n_writers else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.counts) if self.counts else 0.0

    def quartiles(self) -> tuple[float, float]:
        """``(q1, q3)``.

        ``statistics.quantiles`` needs at least two data points; with fewer, the single value is
        both quartiles. Returning something wrong-but-plausible for tiny inputs is how a
        fixture-sized run quietly produces a misleading report.
        """
        counts = self.counts
        if not counts:
            return (0.0, 0.0)
        if len(counts) < 2:
            return (float(counts[0]), float(counts[0]))
        q1, _, q3 = statistics.quantiles(counts, n=4, method="inclusive")
        return (float(q1), float(q3))

    @property
    def iqr(self) -> float:
        q1, q3 = self.quartiles()
        return q3 - q1

    def writers_with_at_least(self, n_lines: int) -> int:
        """How many writers have at least ``n_lines`` lines."""
        return sum(1 for count in self.lines_per_writer.values() if count >= n_lines)

    def support_capacity(self, *, query_size: int) -> dict[int, int]:
        """Writers able to supply ``n`` support lines while reserving ``query_size`` queries."""
        return {n: self.writers_with_at_least(n + query_size) for n in self.thresholds if n > 0}

    def as_dict(self) -> dict[str, Any]:
        q1, q3 = self.quartiles()
        return {
            "writers": self.n_writers,
            "lines": self.total_lines,
            "min": self.minimum,
            "max": self.maximum,
            "mean": round(self.mean, 2),
            "median": self.median,
            "q1": q1,
            "q3": q3,
            "iqr": self.iqr,
            "writers_with_at_least": {
                str(n): self.writers_with_at_least(n) for n in self.thresholds
            },
        }

    def format(self, *, query_size: int | None = None) -> str:
        q1, q3 = self.quartiles()
        lines = [
            f"writers          {self.n_writers:>8,}",
            f"lines            {self.total_lines:>8,}",
            f"lines/writer     min {self.minimum}  q1 {q1:g}  median {self.median:g}  "
            f"q3 {q3:g}  max {self.maximum}   (IQR {self.iqr:g})",
            "writers with at least n lines:",
        ]
        for n in self.thresholds:
            lines.append(f"  n>={n:<3} {self.writers_with_at_least(n):>8,}")
        if query_size is not None:
            lines.append(f"writers able to supply n support lines and {query_size} queries:")
            for n, count in self.support_capacity(query_size=query_size).items():
                lines.append(f"  CER@{n:<3} {count:>8,}")
        return "\n".join(lines)


def writer_histogram(
    records: Iterable[ManifestRecord],
    *,
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS,
) -> WriterHistogram:
    """Lines per writer across ``records``."""
    counts: Counter[str] = Counter()
    for record in records:
        counts[record.writer_id] += 1
    return WriterHistogram(lines_per_writer=dict(counts), thresholds=thresholds)


@dataclass(frozen=True, slots=True)
class PassageDistribution:
    """How source texts are distributed over writers.

    ``passages_per_writer`` is the ceiling on passage-disjoint support/query splitting: a writer
    with one passage cannot have support and query drawn from different ones.
    """

    lines_per_passage: dict[str, int]
    writers_per_passage: dict[str, int]
    passages_per_writer: dict[str, int]
    unlabelled_lines: int = 0

    @property
    def n_passages(self) -> int:
        return len(self.lines_per_passage)

    def writers_with_at_least_passages(self, n: int) -> int:
        return sum(1 for count in self.passages_per_writer.values() if count >= n)

    def as_dict(self) -> dict[str, Any]:
        counts = sorted(self.passages_per_writer.values())
        return {
            "passages": self.n_passages,
            "unlabelled_lines": self.unlabelled_lines,
            "lines_per_passage": dict(sorted(self.lines_per_passage.items())),
            "writers_per_passage": dict(sorted(self.writers_per_passage.items())),
            "passages_per_writer": {
                "min": min(counts, default=0),
                "median": statistics.median(counts) if counts else 0,
                "max": max(counts, default=0),
            },
            "writers_with_at_least_2_passages": self.writers_with_at_least_passages(2),
        }

    def format(self) -> str:
        counts = sorted(self.passages_per_writer.values())
        median = statistics.median(counts) if counts else 0
        lines = [
            f"passages         {self.n_passages:>8,}",
            f"unlabelled lines {self.unlabelled_lines:>8,}",
            f"passages/writer  min {min(counts, default=0)}  median {median:g}  "
            f"max {max(counts, default=0)}",
            f"writers with >=2 passages   {self.writers_with_at_least_passages(2):>8,}",
            "lines per passage:",
        ]
        for name, count in sorted(self.lines_per_passage.items()):
            writers = self.writers_per_passage.get(name, 0)
            lines.append(f"  {name:<6} {count:>8,} line(s)   {writers:>6,} writer(s)")
        return "\n".join(lines)


def passage_distribution(records: Iterable[ManifestRecord]) -> PassageDistribution:
    """Passage coverage per writer and writer coverage per passage.

    Records without a ``passage_id`` are counted as ``unlabelled_lines`` rather than being bucketed
    under ``None``, so a dataset that carries no passage metadata reports that fact instead of
    appearing to have one giant passage.
    """
    lines: Counter[str] = Counter()
    writers_by_passage: dict[str, set[str]] = defaultdict(set)
    passages_by_writer: dict[str, set[str]] = defaultdict(set)
    unlabelled = 0

    for record in records:
        if record.passage_id is None:
            unlabelled += 1
            continue
        lines[record.passage_id] += 1
        writers_by_passage[record.passage_id].add(record.writer_id)
        passages_by_writer[record.writer_id].add(record.passage_id)

    return PassageDistribution(
        lines_per_passage=dict(lines),
        writers_per_passage={name: len(w) for name, w in writers_by_passage.items()},
        passages_per_writer={w: len(p) for w, p in passages_by_writer.items()},
        unlabelled_lines=unlabelled,
    )
