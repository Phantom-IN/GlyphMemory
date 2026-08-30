"""DataLoader construction.

Wires dataset, sampler, collator and worker seeding into one place so a training run and an
evaluation run cannot accidentally disagree about batching policy.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from glyphmemory.config.schema import Config
from glyphmemory.ctc.tokenizer import Tokenizer
from glyphmemory.data.collate import VariableWidthCollator
from glyphmemory.data.dataset import LineDataset
from glyphmemory.data.preprocessing import PixelNormalization
from glyphmemory.data.samplers import (
    DEFAULT_BUCKET_MULTIPLIER,
    SequentialBatchSampler,
    WidthBucketSampler,
)
from glyphmemory.data.transforms import build_augmentation
from glyphmemory.data.validation import IntegrityCounters
from glyphmemory.runtime.seed import seed_worker


def normalization_from(config: Config) -> PixelNormalization:
    return PixelNormalization(
        invert=config.data.invert_pixels,
        mean=config.data.pixel_mean,
        std=config.data.pixel_std,
    )


def build_dataset(
    manifest: str | Path,
    tokenizer: Tokenizer,
    config: Config,
    *,
    split: str | None = None,
    training: bool = True,
) -> LineDataset:
    """Construct a dataset from a manifest, honouring the config's preprocessing settings.

    Augmentation is built with ``training`` so an evaluation dataset is bitwise unaugmented.
    """
    return LineDataset.from_manifest(
        manifest,
        tokenizer,
        split=split,
        augmentation=build_augmentation(config.data.augmentation, training=training),
        height=config.data.image_height,
        width_multiple=config.data.width_multiple,
        max_width=config.data.max_width,
        normalization=normalization_from(config),
    )


def build_dataloader(
    dataset: LineDataset,
    config: Config,
    *,
    training: bool = True,
    batch_size: int | None = None,
    bucket: bool = True,
    bucket_multiplier: int = DEFAULT_BUCKET_MULTIPLIER,
    counters: IntegrityCounters | None = None,
    num_workers: int | None = None,
    drop_last: bool = False,
) -> DataLoader:
    """Build a ``DataLoader`` over ``dataset``.

    Args:
        training: Controls the CTC feasibility policy — drop-with-counter when training, raise when
            evaluating (see :class:`VariableWidthCollator`).
        bucket: Width bucketing. Sequential batching is available for A/B comparison; it must not
            change any metric, only speed.
        counters: Shared integrity counters, so rejections from every batch accumulate in one place
            rather than being scattered per batch.
    """
    batch_size = batch_size or config.training.batch_size
    workers = config.runtime.num_workers if num_workers is None else num_workers

    if bucket:
        batch_sampler = WidthBucketSampler(
            dataset.widths,
            batch_size,
            shuffle=training,
            seed=config.runtime.seed,
            bucket_multiplier=bucket_multiplier,
            drop_last=drop_last,
        )
    else:
        batch_sampler = SequentialBatchSampler(len(dataset), batch_size, drop_last=drop_last)

    collator = VariableWidthCollator(
        training=training,
        counters=counters,
        pad_value=dataset.pad_value,
        width_multiple=config.data.width_multiple,
    )

    generator = torch.Generator()
    generator.manual_seed(config.runtime.seed)

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=workers > 0,
    )
