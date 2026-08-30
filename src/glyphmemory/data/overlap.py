"""Text-overlap measurement between two record sets.

Removing overlapping lines would shrink an already writer-constrained split, so the trade must be
made with numbers in hand rather than out of concern.

Two conventions make these numbers mean something.

**The normalization is recorded in the report.** Overlap figures are meaningless without it, and the
normalization used here need not be the one used for CER — a comparison that folds case would report
higher overlap than a comparison that does not, and neither is wrong as long as which one ran is
written down.

**Direction is explicit.** The leakage-relevant question is not "how similar are these two sets" but
"how much of B was already seen in A". :attr:`OverlapCounts.instance_coverage_of_b` answers that at
the level of actual lines, which is what a CER denominator is made of; Jaccard is reported too but
treats a set of 5,000 training lines and 50 test lines as symmetric, which for leakage they are not.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from glyphmemory.ctc.normalization import NFC_V1, NormalizationPolicy, normalize
from glyphmemory.data.manifest import ManifestRecord
from glyphmemory.data.splits import SupportQuerySplit

#: Word n-gram orders reported alongside exact-line and word overlap.
DEFAULT_NGRAM_ORDERS: tuple[int, ...] = (3, 5)


@dataclass(frozen=True, slots=True)
class OverlapCounts:
    """Overlap at one granularity.

    ``unique_*`` count distinct items; ``instances_b`` counts occurrences in B, so a phrase repeated
    by 200 writers weighs 200 rather than 1.
    """

    granularity: str
    unique_a: int
    unique_b: int
    shared: int
    instances_b: int
    instances_b_shared: int

    @property
    def jaccard(self) -> float:
        union = self.unique_a + self.unique_b - self.shared
        return self.shared / union if union else 0.0

    @property
    def unique_coverage_of_b(self) -> float:
        """Fraction of B's distinct items that also occur in A."""
        return self.shared / self.unique_b if self.unique_b else 0.0

    @property
    def instance_coverage_of_b(self) -> float:
        """Fraction of B's occurrences whose item also occurs in A.

        This is the leakage figure: for exact-line granularity it is literally the share of B's
        lines whose transcript the model may have seen in A.
        """
        return self.instances_b_shared / self.instances_b if self.instances_b else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "granularity": self.granularity,
            "unique_a": self.unique_a,
            "unique_b": self.unique_b,
            "shared": self.shared,
            "instances_b": self.instances_b,
            "instances_b_shared": self.instances_b_shared,
            "jaccard": round(self.jaccard, 6),
            "unique_coverage_of_b": round(self.unique_coverage_of_b, 6),
            "instance_coverage_of_b": round(self.instance_coverage_of_b, 6),
        }


def _counts(
    granularity: str,
    items_a: Sequence[Sequence[str]],
    items_b: Sequence[Sequence[str]],
) -> OverlapCounts:
    """Overlap between two collections of per-record item sequences."""
    set_a = {item for items in items_a for item in items}
    counter_b: Counter[str] = Counter()
    for items in items_b:
        counter_b.update(items)

    shared_keys = set_a & set(counter_b)
    return OverlapCounts(
        granularity=granularity,
        unique_a=len(set_a),
        unique_b=len(counter_b),
        shared=len(shared_keys),
        instances_b=sum(counter_b.values()),
        instances_b_shared=sum(counter_b[key] for key in shared_keys),
    )


@dataclass(frozen=True, slots=True)
class OverlapReport:
    """Overlap between two labelled record sets, under a named normalization."""

    label_a: str
    label_b: str
    normalization: str
    n_records_a: int
    n_records_b: int
    lines: OverlapCounts
    words: OverlapCounts
    ngrams: dict[int, OverlapCounts] = field(default_factory=dict)
    passages: OverlapCounts | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "a": self.label_a,
            "b": self.label_b,
            "normalization": self.normalization,
            "records_a": self.n_records_a,
            "records_b": self.n_records_b,
            "lines": self.lines.as_dict(),
            "words": self.words.as_dict(),
            "ngrams": {str(n): counts.as_dict() for n, counts in sorted(self.ngrams.items())},
        }
        if self.passages is not None:
            payload["passages"] = self.passages.as_dict()
        return payload

    def format(self) -> str:
        rows: list[tuple[str, OverlapCounts]] = [
            ("exact line", self.lines),
            ("word", self.words),
            *[(f"{n}-gram", counts) for n, counts in sorted(self.ngrams.items())],
        ]
        if self.passages is not None:
            rows.append(("passage", self.passages))

        out = [
            f"{self.label_a} -> {self.label_b}   "
            f"({self.n_records_a:,} vs {self.n_records_b:,} records, "
            f"normalization {self.normalization!r})",
            f"  {'granularity':<12} {'uniq A':>9} {'uniq B':>9} {'shared':>9} "
            f"{'jaccard':>8} {'% of B lines':>13}",
        ]
        for name, counts in rows:
            out.append(
                f"  {name:<12} {counts.unique_a:>9,} {counts.unique_b:>9,} "
                f"{counts.shared:>9,} {counts.jaccard:>8.3f} "
                f"{counts.instance_coverage_of_b:>12.1%}"
            )
        return "\n".join(out)


def _word_ngrams(words: Sequence[str], n: int) -> list[str]:
    """Word n-grams as space-joined strings. Shorter lines yield none."""
    if n < 1 or len(words) < n:
        return []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def text_overlap(
    records_a: Iterable[ManifestRecord],
    records_b: Iterable[ManifestRecord],
    *,
    label_a: str = "a",
    label_b: str = "b",
    policy: NormalizationPolicy = NFC_V1,
    ngram_orders: tuple[int, ...] = DEFAULT_NGRAM_ORDERS,
    extra_normalizer: Callable[[str], str] | None = None,
) -> OverlapReport:
    """Measure how much of ``records_b`` already appears in ``records_a``.

    Args:
        policy: Normalization applied to every transcript before comparison. Recorded in the report
            by name.
        extra_normalizer: Optional additional transform applied after ``policy`` — e.g.
            ``str.lower`` to measure case-insensitive overlap. The report's normalization string
            names it, because an unrecorded transform makes the figure unquotable.
        ngram_orders: Word n-gram orders to report.

    Passage overlap is reported only when **both** sides carry ``passage_id``; a dataset without
    passage metadata produces no passage row rather than a misleading zero.
    """
    list_a = list(records_a)
    list_b = list(records_b)

    def prepare(text: str) -> str:
        normalized = normalize(text, policy)
        return extra_normalizer(normalized) if extra_normalizer else normalized

    texts_a = [prepare(record.text) for record in list_a]
    texts_b = [prepare(record.text) for record in list_b]
    words_a = [text.split() for text in texts_a]
    words_b = [text.split() for text in texts_b]

    normalization = policy.name
    if extra_normalizer is not None:
        normalization = f"{policy.name}+{getattr(extra_normalizer, '__name__', 'custom')}"

    passages = None
    if any(r.passage_id for r in list_a) and any(r.passage_id for r in list_b):
        passages = _counts(
            "passage",
            [[r.passage_id] for r in list_a if r.passage_id],
            [[r.passage_id] for r in list_b if r.passage_id],
        )

    return OverlapReport(
        label_a=label_a,
        label_b=label_b,
        normalization=normalization,
        n_records_a=len(list_a),
        n_records_b=len(list_b),
        lines=_counts("exact line", [[text] for text in texts_a], [[text] for text in texts_b]),
        words=_counts("word", words_a, words_b),
        ngrams={
            n: _counts(
                f"{n}-gram",
                [_word_ngrams(words, n) for words in words_a],
                [_word_ngrams(words, n) for words in words_b],
            )
            for n in ngram_orders
        },
        passages=passages,
    )


def split_overlaps(
    records: Iterable[ManifestRecord],
    *,
    policy: NormalizationPolicy = NFC_V1,
    ngram_orders: tuple[int, ...] = DEFAULT_NGRAM_ORDERS,
) -> list[OverlapReport]:
    """Train->val and train->test overlap for a manifest whose splits are populated.

    Pairs with an empty side are skipped: reporting 0% overlap against an empty validation split
    would look like a clean result rather than a missing one.
    """
    by_split: dict[str, list[ManifestRecord]] = {}
    for record in records:
        by_split.setdefault(record.split, []).append(record)

    reports = []
    for target in ("val", "test"):
        if by_split.get("train") and by_split.get(target):
            reports.append(
                text_overlap(
                    by_split["train"],
                    by_split[target],
                    label_a="train",
                    label_b=target,
                    policy=policy,
                    ngram_orders=ngram_orders,
                )
            )
    return reports


@dataclass(frozen=True, slots=True)
class PerWriterOverlapCounts:
    """Support/query overlap at one granularity, accumulated per writer."""

    granularity: str
    instances_query: int
    instances_shared: int
    writers_affected: int

    @property
    def fraction_shared(self) -> float:
        return self.instances_shared / self.instances_query if self.instances_query else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "granularity": self.granularity,
            "instances_query": self.instances_query,
            "instances_shared": self.instances_shared,
            "writers_affected": self.writers_affected,
            "fraction_shared": round(self.fraction_shared, 6),
        }


@dataclass(frozen=True, slots=True)
class PerWriterOverlapReport:
    """Support/query leakage, measured **within each writer** and then summed.

    Pooling across writers would be meaningless here: writer A's query passage is writer B's support
    passage in any corpus where everyone copies the same texts, so a pooled measurement reports
    ~100% overlap for a split that leaks nothing.
    """

    normalization: str
    n_writers: int
    n_support: int
    n_query: int
    granularities: dict[str, PerWriterOverlapCounts] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "measurement": "per-writer support -> query",
            "normalization": self.normalization,
            "writers": self.n_writers,
            "support_lines": self.n_support,
            "query_lines": self.n_query,
            "granularities": {
                name: counts.as_dict() for name, counts in self.granularities.items()
            },
        }

    def format(self) -> str:
        out = [
            f"support -> query, per writer   "
            f"({self.n_writers:,} writers, {self.n_support:,} support / "
            f"{self.n_query:,} query lines, normalization {self.normalization!r})",
            f"  {'granularity':<12} {'query items':>12} {'shared':>10} {'% shared':>10} "
            f"{'writers hit':>12}",
        ]
        for name, counts in self.granularities.items():
            out.append(
                f"  {name:<12} {counts.instances_query:>12,} {counts.instances_shared:>10,} "
                f"{counts.fraction_shared:>9.1%} {counts.writers_affected:>12,}"
            )
        return "\n".join(out)


def support_query_overlap(
    records: Iterable[ManifestRecord],
    support: SupportQuerySplit,
    *,
    policy: NormalizationPolicy = NFC_V1,
    ngram_orders: tuple[int, ...] = DEFAULT_NGRAM_ORDERS,
) -> PerWriterOverlapReport:
    """Measure, per writer, how much of their query text they already enrolled on."""
    by_id = {record.sample_id: record for record in records if record.sample_id is not None}

    def items_for(record: ManifestRecord) -> dict[str, list[str]]:
        text = normalize(record.text, policy)
        words = text.split()
        granularities: dict[str, list[str]] = {"exact line": [text], "word": words}
        for order in ngram_orders:
            granularities[f"{order}-gram"] = _word_ngrams(words, order)
        if record.passage_id is not None:
            granularities["passage"] = [record.passage_id]
        return granularities

    names = ["exact line", "word", *[f"{n}-gram" for n in ngram_orders], "passage"]
    query_totals: Counter[str] = Counter()
    shared_totals: Counter[str] = Counter()
    affected: Counter[str] = Counter()
    n_support = n_query = 0

    for writer in sorted(support.writers):
        support_records = [by_id[i] for i in support.support_for(writer) if i in by_id]
        query_records = [by_id[i] for i in support.query_for(writer) if i in by_id]
        n_support += len(support_records)
        n_query += len(query_records)

        enrolled: dict[str, set[str]] = {name: set() for name in names}
        for record in support_records:
            for name, items in items_for(record).items():
                enrolled[name].update(items)

        writer_hits: Counter[str] = Counter()
        for record in query_records:
            for name, items in items_for(record).items():
                query_totals[name] += len(items)
                writer_hits[name] += sum(1 for item in items if item in enrolled[name])

        for name, hits in writer_hits.items():
            shared_totals[name] += hits
            if hits:
                affected[name] += 1

    return PerWriterOverlapReport(
        normalization=policy.name,
        n_writers=len(support.writers),
        n_support=n_support,
        n_query=n_query,
        granularities={
            name: PerWriterOverlapCounts(
                granularity=name,
                instances_query=query_totals[name],
                instances_shared=shared_totals[name],
                writers_affected=affected[name],
            )
            for name in names
            if query_totals[name]
        },
    )
