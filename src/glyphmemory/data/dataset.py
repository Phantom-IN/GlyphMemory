"""Line dataset.

Turns manifest records into model-ready samples. Holds **paths and text only** — never open file
handles or PIL objects — so the dataset stays picklable and survives the ``spawn`` start method
DataLoader workers use on macOS.

A sample that cannot be used is returned as a :class:`RejectedSample` rather than raising.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from glyphmemory.ctc.tokenizer import Tokenizer, UnsupportedCharacterError
from glyphmemory.data.manifest import ManifestRecord, read_manifest
from glyphmemory.data.preprocessing import (
    DEFAULT_HEIGHT,
    DEFAULT_MAX_WIDTH,
    DEFAULT_WIDTH_MULTIPLE,
    PixelNormalization,
    UnreadableImageError,
    preprocess_path,
    resized_width,
)
from glyphmemory.data.validation import IntegrityCategory
from glyphmemory.runtime.logging import get_logger

logger = get_logger("data.dataset")


def required_ctc_length(targets: Sequence[int]) -> int:
    """Minimum number of CTC time steps needed to emit ``targets``.

    CTC must insert a blank between two identical consecutive labels, so the requirement is
    ``len(targets) + adjacent_repeats``::

        "hello"     -> 5 + 1 (ll)         = 6
        "committee" -> 9 + 3 (mm, tt, ee) = 12
    """
    repeats = sum(1 for a, b in itertools.pairwise(targets) if a == b)
    return len(targets) + repeats


@dataclass(frozen=True, slots=True)
class LineSample:
    """One usable line, ready for batching."""

    image: Tensor
    targets: Tensor
    true_width: int
    input_length: int
    text: str
    sample_id: str
    writer_id: str
    image_path: str
    oversized: bool = False

    @property
    def target_length(self) -> int:
        return int(self.targets.numel())

    @property
    def required_length(self) -> int:
        return required_ctc_length(self.targets.tolist())

    @property
    def is_ctc_feasible(self) -> bool:
        """Whether there are enough time steps to align this transcript."""
        return self.input_length >= self.required_length


@dataclass(frozen=True, slots=True)
class RejectedSample:
    """A record that could not become a sample. Always carries a countable reason."""

    sample_id: str
    image_path: str
    category: IntegrityCategory
    reason: str


@dataclass
class LineDataset(Dataset):
    """Manifest-backed dataset of handwritten lines.

    Args:
        records: Parsed manifest records.
        tokenizer: Encodes transcripts. Normalization travels with it, so ``decode(targets) ==
            text``.
        augmentation: Applied to the uint8 tensor before height normalisation. Pass ``None`` for
            evaluation — evaluation is never augmented.
    """

    records: tuple[ManifestRecord, ...]
    tokenizer: Tokenizer
    augmentation: Callable[[Tensor], Tensor] | None = None
    height: int = DEFAULT_HEIGHT
    width_multiple: int = DEFAULT_WIDTH_MULTIPLE
    max_width: int = DEFAULT_MAX_WIDTH
    normalization: PixelNormalization | None = None

    def __post_init__(self) -> None:
        self.records = tuple(self.records)
        self.normalization = self.normalization or PixelNormalization()

    def take(self, n: int) -> LineDataset:
        """A dataset over the first ``n`` records, sharing this one's settings.

        Takes a *prefix* rather than a random sample deliberately: the gate needs the same lines on
        every run, and a frozen prefix of a manifest whose order is already deterministic is the
        simplest thing that guarantees it.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        return replace(self, records=self.records[:n])

    def select(self, sample_ids: Sequence[str]) -> LineDataset:
        """A dataset over exactly ``sample_ids``, in the order given.

        Raises:
            KeyError: An ID is not present. Silently returning a smaller set would let the
                tiny-overfit gate run on fewer lines than it claims.
        """
        by_id = {record.sample_id: record for record in self.records}
        missing = [sid for sid in sample_ids if sid not in by_id]
        if missing:
            raise KeyError(
                f"{len(missing)} requested sample_id(s) are not in this manifest, e.g. "
                f"{missing[:5]}. Regenerate the manifest, or fix the sample list."
            )
        return replace(self, records=tuple(by_id[sid] for sid in sample_ids))

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_manifest(
        cls,
        manifest: str | Path,
        tokenizer: Tokenizer,
        *,
        split: str | None = None,
        **kwargs,
    ) -> LineDataset:
        """Load a manifest, optionally keeping only one split."""
        records = tuple(read_manifest(manifest))
        if split is not None:
            records = tuple(record for record in records if record.split == split)
        return cls(records=records, tokenizer=tokenizer, **kwargs)

    # ------------------------------------------------------------------ access

    def __len__(self) -> int:
        return len(self.records)

    @property
    def widths(self) -> list[int]:
        """Width each sample will occupy **after preprocessing**, for the bucket sampler.

        Not the manifest width. The collator pads to the height-normalized width, and source line
        heights vary — 44-176 px in CVL — so aspect ratios differ and the manifest width predicts
        the padded width only weakly.

        Falls back to the manifest width when height is missing, and to ``0`` when both are:
        bucketing degrades to arbitrary grouping rather than failing, and the sampler still produces
        valid batches. Efficiency is a performance property, never a correctness one —
        ``input_lengths`` masks padding regardless of how batches are grouped.
        """
        estimates: list[int] = []
        for record in self.records:
            if record.width and record.height:
                estimates.append(resized_width(record.width, record.height, self.height))
            else:
                estimates.append(record.width or 0)
        return estimates

    @property
    def pad_value(self) -> float:
        """Value batch padding must use, so it matches within-sample padding."""
        assert self.normalization is not None
        return self.normalization.background_value

    def __getitem__(self, index: int) -> LineSample | RejectedSample:
        record = self.records[index]
        sample_id = record.sample_id or record.image

        try:
            targets = self.tokenizer.encode(record.text)
        except UnsupportedCharacterError as exc:
            return RejectedSample(
                sample_id, record.image, IntegrityCategory.UNSUPPORTED_CHARACTER, str(exc)
            )

        if not targets:
            return RejectedSample(
                sample_id,
                record.image,
                IntegrityCategory.MISSING_TRANSCRIPT,
                f"transcript is empty after {self.tokenizer.policy.name} normalization",
            )

        try:
            processed = preprocess_path(
                record.image,
                height=self.height,
                width_multiple=self.width_multiple,
                max_width=self.max_width,
                normalization=self.normalization,
                augmentation=self.augmentation,
                sample_id=sample_id,
            )
        except UnreadableImageError as exc:
            return RejectedSample(
                sample_id, record.image, IntegrityCategory.UNREADABLE_IMAGE, exc.reason
            )

        return LineSample(
            image=processed.tensor,
            targets=torch.tensor(targets, dtype=torch.long),
            true_width=processed.true_width,
            input_length=processed.input_length,
            text=self.tokenizer.decode(targets),
            sample_id=sample_id,
            writer_id=record.writer_id,
            image_path=record.image,
            oversized=processed.oversized,
        )
