"""Width-aware batch sampling.

Naive random batching over variable-width lines spends most of its compute on padding: one 1400 px
line drags a batch of 200 px lines up to 1400 px each. Bucketing groups similar-width samples so
batches stay dense.

The scheme is shuffle → pool → sort within pool → batch → shuffle batches.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from glyphmemory.runtime.logging import get_logger

logger = get_logger("data.samplers")

#: Pool size as a multiple of the batch size. Larger means tighter width grouping and less
#: randomness; 8 is a conventional middle.
DEFAULT_BUCKET_MULTIPLIER = 8


def padding_efficiency(true_widths: Sequence[int], max_width: int | None = None) -> float:
    """Fraction of a padded batch that is real content, in ``(0, 1]``.

    ``1.0`` means every sample is the batch maximum and nothing is wasted.
    """
    if not true_widths:
        return 0.0
    ceiling = max_width if max_width is not None else max(true_widths)
    if ceiling <= 0:
        return 0.0
    return sum(true_widths) / (len(true_widths) * ceiling)


class WidthBucketSampler(Sampler[list[int]]):
    """Yields batches of indices grouped by similar width.

    Deterministic under ``seed`` for a given epoch. Call :meth:`set_epoch` each epoch so the
    grouping changes; without it every epoch sees identical batches, which correlates batch content
    with width permanently.
    """

    def __init__(
        self,
        widths: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        bucket_multiplier: int = DEFAULT_BUCKET_MULTIPLIER,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if bucket_multiplier < 1:
            raise ValueError(f"bucket_multiplier must be at least 1, got {bucket_multiplier}")

        self.widths = list(widths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_multiplier = bucket_multiplier
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change the shuffle for a new epoch."""
        self.epoch = epoch

    def __len__(self) -> int:
        total = len(self.widths)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(len(self.widths)))
        if not indices:
            return

        rng = random.Random(f"{self.seed}:{self.epoch}")
        if self.shuffle:
            rng.shuffle(indices)

        pool_size = self.batch_size * self.bucket_multiplier
        batches: list[list[int]] = []
        for start in range(0, len(indices), pool_size):
            pool = indices[start : start + pool_size]
            # Sort inside the pool only: width-homogeneous batches, still stochastic.
            pool.sort(key=lambda index: self.widths[index])
            for offset in range(0, len(pool), self.batch_size):
                batch = pool[offset : offset + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


class SequentialBatchSampler(Sampler[list[int]]):
    """Plain in-order batching, for comparing against bucketing and for stable evaluation."""

    def __init__(self, count: int, batch_size: int, *, drop_last: bool = False) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.count = count
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self) -> int:
        if self.drop_last:
            return self.count // self.batch_size
        return (self.count + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        for start in range(0, self.count, self.batch_size):
            batch = list(range(start, min(start + self.batch_size, self.count)))
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch
