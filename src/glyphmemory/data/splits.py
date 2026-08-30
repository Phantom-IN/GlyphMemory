"""Writer-disjoint splits and support/query pools.

A leak here silently invalidates every personalization result the project will ever produce and is
close to undetectable afterwards, so the assertion is cheap insurance against the most expensive
possible failure.

Splits partition **writers**, not lines. Two lines by the same person must never land on opposite
sides of a train/test boundary.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from glyphmemory.data.manifest import VALID_SPLITS, ManifestRecord
from glyphmemory.runtime.logging import get_logger

logger = get_logger("data.splits")

# How many offending IDs to name in an assertion message before truncating.
MAX_REPORTED_LEAKS = 20


class SplitLeakError(AssertionError):
    """A writer or sample appears in more than one split. Never catch this to continue."""


@dataclass(frozen=True, slots=True)
class WriterSplit:
    """Writer IDs assigned to each split."""

    train: frozenset[str]
    val: frozenset[str]
    test: frozenset[str]
    seed: int | None = None

    def writers_for(self, split: str) -> frozenset[str]:
        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {list(VALID_SPLITS)}")
        return getattr(self, split)

    def split_for(self, writer_id: str) -> str | None:
        """Which split a writer belongs to, or ``None`` if unassigned."""
        for split in VALID_SPLITS:
            if writer_id in self.writers_for(split):
                return split
        return None

    @property
    def all_writers(self) -> frozenset[str]:
        return self.train | self.val | self.test

    def sizes(self) -> dict[str, int]:
        return {split: len(self.writers_for(split)) for split in VALID_SPLITS}

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            **{split: sorted(self.writers_for(split)) for split in VALID_SPLITS},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WriterSplit:
        return cls(
            train=frozenset(payload.get("train", ())),
            val=frozenset(payload.get("val", ())),
            test=frozenset(payload.get("test", ())),
            seed=payload.get("seed"),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> WriterSplit:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _describe_leak(label: str, a: str, b: str, shared: Iterable[str]) -> str:
    offenders = sorted(shared)
    shown = offenders[:MAX_REPORTED_LEAKS]
    hidden = len(offenders) - MAX_REPORTED_LEAKS
    suffix = f" (+{hidden} more)" if hidden > 0 else ""
    return f"{label} appear in both {a!r} and {b!r}: {shown}{suffix}"


def assert_writer_disjoint(split: WriterSplit) -> None:
    """Raise :class:`SplitLeakError` if any writer appears in more than one split."""
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    problems = [
        _describe_leak("Writer(s)", a, b, shared)
        for a, b in pairs
        if (shared := split.writers_for(a) & split.writers_for(b))
    ]
    if problems:
        raise SplitLeakError(
            "Writer-disjointness violated — personalization results would be invalid. "
            + " | ".join(problems)
        )


def writers_in(records: Iterable[ManifestRecord]) -> list[str]:
    """Unique writer IDs in deterministic (sorted) order."""
    return sorted({record.writer_id for record in records})


def _allocate(total: int, ratios: Sequence[float]) -> list[int]:
    """Distribute ``total`` writers across splits according to ``ratios``.

    Three properties, in priority order:

    1. every writer is assigned — the counts always sum to ``total``;
    2. every split with a non-zero ratio receives at least one writer, borrowing from the largest
       split if flooring left it empty;
    3. the allocation is otherwise proportional, with the remainder going to the splits with the
       largest fractional parts.

    Property 2 matters at small writer counts: flooring ``(0.8, 0.1, 0.1)`` over 3 writers yields
    ``[2, 0, 0]``, and an empty validation split is a silently broken experiment.
    """
    counts = [int(total * ratio) for ratio in ratios]
    requested = [index for index, ratio in enumerate(ratios) if ratio > 0]

    remainder = total - sum(counts)
    if remainder and requested:
        by_fraction = sorted(requested, key=lambda i: (total * ratios[i]) - counts[i], reverse=True)
        for step in range(remainder):
            counts[by_fraction[step % len(by_fraction)]] += 1

    for index in requested:
        if counts[index] == 0:
            donor = max(requested, key=lambda i: counts[i])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[index] += 1

    return counts


def make_writer_disjoint_split(
    records: Iterable[ManifestRecord],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 1337,
) -> WriterSplit:
    """Partition writers into train/val/test.

    Deterministic under ``seed``: writers are sorted before shuffling, so the result does not depend
    on manifest order.

    Partitioning is by **writer count**, which does not guarantee balanced *line* counts — writers
    contribute very unequal numbers of lines.
    """
    if len(ratios) != 3:
        raise ValueError(f"Expected three ratios (train, val, test), got {len(ratios)}")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError(f"Ratios must be non-negative, got {ratios}")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    writers = writers_in(records)
    requested = sum(1 for ratio in ratios if ratio > 0)
    if len(writers) < requested:
        raise ValueError(
            f"Cannot build a writer-disjoint split: {len(writers)} writer(s) available but "
            f"{requested} non-empty split(s) requested. A split that reuses writers is not a "
            "split."
        )

    shuffled = list(writers)
    random.Random(seed).shuffle(shuffled)

    n_train, n_val, n_test = _allocate(len(shuffled), ratios)

    split = WriterSplit(
        train=frozenset(shuffled[:n_train]),
        val=frozenset(shuffled[n_train : n_train + n_val]),
        test=frozenset(shuffled[n_train + n_val : n_train + n_val + n_test]),
        seed=seed,
    )
    assert_writer_disjoint(split)
    logger.info("Writer-disjoint split (seed=%d): %s", seed, split.sizes())
    return split


def apply_writer_split(
    records: Iterable[ManifestRecord], split: WriterSplit
) -> list[ManifestRecord]:
    """Return records with ``split`` set from the writer assignment.

    Writers absent from the split are dropped **loudly** — a warning naming the count, since an
    unassigned writer means the split and the manifest disagree.
    """
    assigned: list[ManifestRecord] = []
    unassigned: set[str] = set()
    for record in records:
        target = split.split_for(record.writer_id)
        if target is None:
            unassigned.add(record.writer_id)
            continue
        assigned.append(
            ManifestRecord(**{**record.to_dict(), "split": target})
            if record.split != target
            else record
        )
    if unassigned:
        logger.warning(
            "%d writer(s) in the manifest are absent from the split and were excluded: %s",
            len(unassigned),
            sorted(unassigned)[:MAX_REPORTED_LEAKS],
        )
    return assigned


def split_statistics(
    records: Iterable[ManifestRecord], split: WriterSplit
) -> dict[str, dict[str, int]]:
    """Writer and line counts per split, for checking line balance after partitioning."""
    lines: Counter[str] = Counter()
    writers: dict[str, set[str]] = defaultdict(set)
    for record in records:
        target = split.split_for(record.writer_id)
        if target is None:
            continue
        lines[target] += 1
        writers[target].add(record.writer_id)
    return {name: {"writers": len(writers[name]), "lines": lines[name]} for name in VALID_SPLITS}


@dataclass(frozen=True, slots=True)
class SupportQuerySplit:
    """Per-writer support and query pools for few-shot evaluation.

    Deriving these pools at evaluation time instead of storing them is how denominators silently
    drift between runs.
    """

    support: dict[str, tuple[str, ...]]
    query: dict[str, tuple[str, ...]]
    seed: int | None = None

    @property
    def writers(self) -> frozenset[str]:
        return frozenset(self.support) | frozenset(self.query)

    def support_for(self, writer_id: str) -> tuple[str, ...]:
        return self.support.get(writer_id, ())

    def query_for(self, writer_id: str) -> tuple[str, ...]:
        return self.query.get(writer_id, ())

    def writers_supporting(self, n_shot: int) -> frozenset[str]:
        """Writers with at least ``n_shot`` support lines *and* a non-empty query pool."""
        return frozenset(
            writer
            for writer in self.writers
            if len(self.support_for(writer)) >= n_shot and self.query_for(writer)
        )

    def assert_disjoint(self) -> None:
        """Raise if any sample appears in both pools for the same writer."""
        problems = [
            _describe_leak(f"Sample(s) for writer {writer!r}", "support", "query", shared)
            for writer in sorted(self.writers)
            if (shared := set(self.support_for(writer)) & set(self.query_for(writer)))
        ]
        if problems:
            raise SplitLeakError(
                "Support/query overlap — adaptation gain would be measured on enrolled "
                "lines. " + " | ".join(problems)
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "support": {writer: list(ids) for writer, ids in sorted(self.support.items())},
            "query": {writer: list(ids) for writer, ids in sorted(self.query.items())},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SupportQuerySplit:
        return cls(
            support={w: tuple(ids) for w, ids in payload.get("support", {}).items()},
            query={w: tuple(ids) for w, ids in payload.get("query", {}).items()},
            seed=payload.get("seed"),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> SupportQuerySplit:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def make_support_query_split(
    records: Sequence[ManifestRecord],
    *,
    query_size: int,
    seed: int = 1337,
    group_of: Any = None,
) -> SupportQuerySplit:
    """Reserve a fixed query pool per writer; everything else becomes the support pool.

    Args:
        records: Records for the evaluation writers. Each must have a ``sample_id``.
        query_size: Lines reserved for querying, per writer. Reserved first and never re-drawn, so
            the denominator of every ``CER@n`` is identical.
        seed: Determinism. Records are sorted by ``sample_id`` before shuffling.
        group_of: Optional callable mapping a record to a group key (e.g. ``passage_id``). When
            given, support and query are drawn from **disjoint groups**, so a writer cannot be
            queried on text they enrolled on.

    Writers with too few lines are skipped with a warning rather than silently producing an empty
    pool.
    """
    if query_size < 1:
        raise ValueError(f"query_size must be at least 1, got {query_size}")

    by_writer: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        if record.sample_id is None:
            raise ValueError(
                f"Record {record.image!r} has no sample_id; support/query pools are stored as "
                "sample IDs so every record needs one."
            )
        by_writer[record.writer_id].append(record)

    support: dict[str, tuple[str, ...]] = {}
    query: dict[str, tuple[str, ...]] = {}
    skipped: list[str] = []

    for writer, writer_records in sorted(by_writer.items()):
        ordered = sorted(writer_records, key=lambda r: r.sample_id or "")
        rng = random.Random(f"{seed}:{writer}")

        if group_of is None:
            shuffled = list(ordered)
            rng.shuffle(shuffled)
            query_records = shuffled[:query_size]
            support_records = shuffled[query_size:]
        else:
            groups: dict[Any, list[ManifestRecord]] = defaultdict(list)
            for record in ordered:
                groups[group_of(record)].append(record)
            group_keys = sorted(groups, key=str)
            rng.shuffle(group_keys)

            query_records, support_records, filling = [], [], True
            for key in group_keys:
                if filling and len(query_records) < query_size:
                    query_records.extend(groups[key])
                    filling = len(query_records) < query_size
                else:
                    support_records.extend(groups[key])

        if not query_records or not support_records:
            skipped.append(writer)
            continue

        query[writer] = tuple(r.sample_id for r in query_records if r.sample_id)
        support[writer] = tuple(r.sample_id for r in support_records if r.sample_id)

    if skipped:
        logger.warning(
            "%d writer(s) had too few lines (or too few groups) for a support/query split "
            "and were excluded: %s",
            len(skipped),
            sorted(skipped)[:MAX_REPORTED_LEAKS],
        )

    split = SupportQuerySplit(support=support, query=query, seed=seed)
    split.assert_disjoint()
    logger.info(
        "Support/query split (seed=%d): %d writer(s); query_size=%d",
        seed,
        len(split.writers),
        query_size,
    )
    return split
