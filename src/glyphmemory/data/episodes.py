"""Writer episodes for episodic training.

Episodic *training* needs a genuinely different access pattern than `data/splits.py`'s
`SupportQuerySplit`: many `(support, query)` draws per writer across many training steps and epochs,
with support size varying draw-to-draw — not one query pool reserved once per writer and reused for
the life of an evaluation run.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from glyphmemory.data.manifest import ManifestRecord
from glyphmemory.data.splits import SplitLeakError

#: "randomly vary support count across 1, 3, 5, 10.".
DEFAULT_SUPPORT_SIZES: tuple[int, ...] = (1, 3, 5, 10)

#: Training's own query size — deliberately not eval's `query_size=5` (which sizes a single, large,
#: reliable `CER@n` measurement reused across a whole few-shot curve, a different job). A training
#: episode only needs enough query lines to give the CTC loss a non-degenerate per-step signal; on
#: IAM's real 525 train writers (median 11 lines/writer), `query_size=2` keeps `support_n` in `{1,
#: 3, 5}` feasible for 95%+ of writers.
DEFAULT_QUERY_SIZE = 2


@dataclass(frozen=True, slots=True)
class Episode:
    """One `(support, query)` draw for one writer — sample IDs only, the same "IDs in, IDs out"
    shape `SupportQuerySplit` already uses.
    """

    writer_id: str
    support_ids: tuple[str, ...]
    query_ids: tuple[str, ...]

    @property
    def support_size(self) -> int:
        return len(self.support_ids)


class EpisodeSampler:
    """Draws `Episode`s from IAM train-split writers, one per call, support size varying.

    Unlike `SupportQuerySplit` (one query pool per writer, reserved once and reused for every `n`),
    every call to :meth:`sample` can return a different draw for the same writer: deterministic
    given ``(seed, writer_id, draw_index)``, not given ``(seed, writer_id)`` alone.

    Args:
        records: Candidate lines. Every record must have ``split == "train"`` — a record from any
            other split raises :class:`~glyphmemory.data.splits.SplitLeakError` at construction
            time, a re-check rather than trusting the caller filtered correctly.
        query_size: Query lines per episode. See `DEFAULT_QUERY_SIZE`.
        support_sizes: Candidate support sizes an episode may draw. See `DEFAULT_SUPPORT_SIZES`.
        group_of: Optionally groups records so support and query are drawn from **disjoint groups**
            within a writer, one whole group to a side (`data/splits.py: make_support_query_split`
            uses `passage_id` for exactly this reason at eval time —: lines from the same scanned
            page/passage share ink, pen and scanning conditions). **Defaults to `None` (one
            ungrouped pool per writer) for a measured, not assumed, reason**: on real IAM
            train-split data, 282 of 525 (53.7%) writers have **exactly one** passage covering every
            one of their lines — for such a writer no grouping can ever split support from query at
            all (the single group has nowhere to go but entirely to one side), so
            `group_of=passage_id` here would make the majority of the training writer population
            permanently undrawable, not merely occasionally infeasible. Eval's one-time
            `SupportQuerySplit` absorbs this by quietly skipping such writers from the split once;
            episodic training draws an episode thousands of times, so silently losing over half the
            population by default is a far larger cost there. Pass `lambda r: r.passage_id`
            explicitly to opt into passage-disjoint episodes for writers whose structure supports it
            — `sample` raises per-call (not silently skips) when a specific draw cannot satisfy it,
            so a caller that opts in must be prepared to catch and retry, unlike eval's own silent
            skip.
        seed: Determinism.

    Raises:
        SplitLeakError: A non-train-split record was passed.
        ValueError: ``query_size`` or any of ``support_sizes`` is not a positive integer, or a
            record has no ``sample_id``.
    """

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        *,
        query_size: int = DEFAULT_QUERY_SIZE,
        support_sizes: Sequence[int] = DEFAULT_SUPPORT_SIZES,
        group_of: Callable[[ManifestRecord], Any] | None = None,
        seed: int = 1337,
    ) -> None:
        if query_size < 1:
            raise ValueError(f"query_size must be at least 1, got {query_size}")
        if not support_sizes or any(n < 1 for n in support_sizes):
            raise ValueError(
                f"support_sizes must be a non-empty sequence of positive integers, got "
                f"{support_sizes!r}"
            )
        non_train = sorted({record.split for record in records if record.split != "train"})
        if non_train:
            raise SplitLeakError(
                "EpisodeSampler received non-train-split record(s) "
                f"(split(s) found: {non_train}) -- episodic training may only draw from IAM "
                "train-split writers."
            )

        self._query_size = query_size
        self._support_sizes = tuple(sorted(support_sizes))
        self._group_of = group_of
        self._seed = seed

        by_writer: dict[str, list[ManifestRecord]] = defaultdict(list)
        for record in records:
            if record.sample_id is None:
                raise ValueError(
                    f"Record {record.image!r} has no sample_id; episodes are stored as sample "
                    "IDs so every record needs one."
                )
            by_writer[record.writer_id].append(record)
        self._by_writer: dict[str, tuple[ManifestRecord, ...]] = {
            writer: tuple(sorted(recs, key=lambda r: r.sample_id or ""))
            for writer, recs in by_writer.items()
        }

    @property
    def writers(self) -> frozenset[str]:
        return frozenset(self._by_writer)

    def feasible_support_sizes(self, writer_id: str) -> tuple[int, ...]:
        """Support sizes drawable for ``writer_id``, given its real line count and query size."""
        available = len(self._by_writer.get(writer_id, ()))
        return tuple(n for n in self._support_sizes if available >= n + self._query_size)

    def sample(self, writer_id: str, draw_index: int) -> Episode:
        """One deterministic draw for ``writer_id``.

        ``draw_index`` distinguishes repeated calls: the same ``(writer_id, draw_index)`` always
        returns the same episode; a different ``draw_index`` generally returns a different one
        (support size and/or the specific lines drawn) — the "many draws, not one reservation"
        property `SupportQuerySplit` does not have.

        Raises:
            ValueError: ``writer_id`` is not a writer this sampler was built with, has too few lines
                for any support size given ``query_size``, or (only possible with ``group_of`` set)
                passage-disjoint grouping leaves too few support candidates for the drawn
                ``support_size`` even though the writer's raw line count seemed sufficient — never
                silently returns fewer lines than requested.
        """
        if writer_id not in self._by_writer:
            raise ValueError(f"{writer_id!r} is not a train-split writer known to this sampler.")
        feasible = self.feasible_support_sizes(writer_id)
        if not feasible:
            raise ValueError(
                f"writer {writer_id!r} has only {len(self._by_writer[writer_id])} line(s), too "
                f"few for query_size={self._query_size} plus the smallest support size "
                f"{min(self._support_sizes)}."
            )

        rng = random.Random(f"{self._seed}:{writer_id}:{draw_index}")
        support_size = rng.choice(feasible)
        ordered = self._by_writer[writer_id]

        if self._group_of is None:
            shuffled = list(ordered)
            rng.shuffle(shuffled)
            query_records = shuffled[: self._query_size]
            support_records = shuffled[self._query_size : self._query_size + support_size]
        else:
            groups: dict[Any, list[ManifestRecord]] = defaultdict(list)
            for record in ordered:
                groups[self._group_of(record)].append(record)
            group_keys = sorted(groups, key=str)
            rng.shuffle(group_keys)

            query_records: list[ManifestRecord] = []
            support_pool: list[ManifestRecord] = []
            filling_query = True
            for group_key in group_keys:
                if filling_query and len(query_records) < self._query_size:
                    query_records.extend(groups[group_key])
                    filling_query = len(query_records) < self._query_size
                else:
                    support_pool.extend(groups[group_key])

            if len(support_pool) < support_size:
                raise ValueError(
                    f"writer {writer_id!r}, draw {draw_index}: passage-disjoint grouping left "
                    f"only {len(support_pool)} support candidate(s) after reserving the query "
                    f"group(s), fewer than the requested support_size={support_size}. Retry "
                    "with a different draw_index."
                )
            rng.shuffle(support_pool)
            support_records = support_pool[:support_size]

        return Episode(
            writer_id=writer_id,
            support_ids=tuple(r.sample_id for r in support_records if r.sample_id),
            query_ids=tuple(r.sample_id for r in query_records if r.sample_id),
        )


def iter_writer_cycle(writer_ids: Sequence[str], *, seed: int) -> Iterator[str]:
    """An endless cycle over ``writer_ids``, reshuffled every full pass.

    Every writer appears exactly once per pass — a real spread across the full train-writer
    population, not a fixed small subset or a fixed sequence repeated identically forever (which
    would pair the same writers adjacently every epoch). Determinism given ``seed``; each pass
    reshuffles with a seed derived from the pass index, so pass order is reproducible but not a
    single static permutation.
    """
    ordered = sorted(set(writer_ids))
    if not ordered:
        raise ValueError("writer_ids must be non-empty.")
    epoch = 0
    while True:
        shuffled = list(ordered)
        random.Random(f"{seed}:epoch:{epoch}").shuffle(shuffled)
        yield from shuffled
        epoch += 1
