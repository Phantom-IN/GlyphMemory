"""Variable-width batching and CTC length bookkeeping.

This is where CTC correctness is won or lost. Two rules carry the whole phase:

1. **``input_lengths`` come from the true unpadded width, never the padded width.** Counting padding
   as valid input trains a model that decodes truncated text and never raises.
2. **Targets are flattened** into ``[sum(L)]`` with a separate ``target_lengths`` vector — the
   layout :class:`torch.nn.CTCLoss` expects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from glyphmemory.data.dataset import LineSample, RejectedSample
from glyphmemory.data.preprocessing import DEFAULT_WIDTH_MULTIPLE
from glyphmemory.data.validation import IntegrityCategory, IntegrityCounters, IntegrityIssue
from glyphmemory.runtime.logging import get_logger

logger = get_logger("data.collate")


class CTCFeasibilityError(ValueError):
    """A sample cannot be aligned by CTC and evaluation refuses to skip it silently.

    Raised only outside training. During evaluation a dropped sample changes the denominator of a
    reported metric, which turns a data problem into a wrong number.
    """


@dataclass(frozen=True, slots=True)
class Batch:
    """A collated batch, plus everything error analysis needs afterwards."""

    images: Tensor
    input_lengths: Tensor
    targets: Tensor
    target_lengths: Tensor
    texts: tuple[str, ...] = ()
    sample_ids: tuple[str, ...] = ()
    writer_ids: tuple[str, ...] = ()
    image_paths: tuple[str, ...] = ()
    true_widths: tuple[int, ...] = ()
    rejected: tuple[RejectedSample, ...] = field(default=())

    @property
    def batch_size(self) -> int:
        return int(self.images.shape[0])

    @property
    def max_width(self) -> int:
        return int(self.images.shape[-1])

    @property
    def is_empty(self) -> bool:
        return self.batch_size == 0

    @property
    def padding_efficiency(self) -> float:
        """Fraction of the batch tensor that is real content rather than padding.

        1.0 means no waste. This is how a bucketing regression is noticed without reading the
        loader.
        """
        if self.is_empty or self.max_width == 0:
            return 0.0
        return sum(self.true_widths) / (self.batch_size * self.max_width)

    def targets_for(self, index: int) -> Tensor:
        """Unflatten one sample's targets back out of the concatenated tensor."""
        start = int(self.target_lengths[:index].sum().item())
        end = start + int(self.target_lengths[index].item())
        return self.targets[start:end]

    def to(self, device: torch.device | str) -> Batch:
        """Move tensors to a device, leaving metadata untouched."""
        return Batch(
            images=self.images.to(device),
            input_lengths=self.input_lengths.to(device),
            targets=self.targets.to(device),
            target_lengths=self.target_lengths.to(device),
            texts=self.texts,
            sample_ids=self.sample_ids,
            writer_ids=self.writer_ids,
            image_paths=self.image_paths,
            true_widths=self.true_widths,
            rejected=self.rejected,
        )


def empty_batch(height: int = 64, pad_value: float = 0.0) -> Batch:
    """A zero-sample batch. Returned when every sample in a batch was rejected."""
    return Batch(
        images=torch.full((0, 1, height, 0), pad_value, dtype=torch.float32),
        input_lengths=torch.zeros(0, dtype=torch.long),
        targets=torch.zeros(0, dtype=torch.long),
        target_lengths=torch.zeros(0, dtype=torch.long),
    )


@dataclass
class VariableWidthCollator:
    """Pads a list of samples into a batch and enforces the CTC feasibility policy.

    **Decision: drop-with-counter during training, raise during evaluation.**.

    In training, one unalignable line should not stop a run — it is counted under
    ``impossible_ctc_length`` and logged with its ``sample_id``, ``path`` and reason. In evaluation
    there is no acceptable silent drop: removing a sample changes the denominator of a reported CER,
    so :class:`CTCFeasibilityError` is raised instead.
    """

    training: bool = True
    counters: IntegrityCounters | None = None
    pad_value: float = 0.0
    width_multiple: int = DEFAULT_WIDTH_MULTIPLE

    def __post_init__(self) -> None:
        if self.counters is None:
            self.counters = IntegrityCounters()

    def __call__(self, items: Sequence[LineSample | RejectedSample]) -> Batch:
        usable: list[LineSample] = []
        rejected: list[RejectedSample] = []

        for item in items:
            if isinstance(item, RejectedSample):
                rejected.append(item)
                continue
            if not item.is_ctc_feasible:
                rejected.append(self._infeasible(item))
                continue
            usable.append(item)

        for rejection in rejected:
            self._record(rejection)

        if not usable:
            if rejected:
                logger.warning(
                    "Every sample in this batch was rejected (%d); emitting an empty batch.",
                    len(rejected),
                )
            blank = empty_batch(pad_value=self.pad_value)
            return Batch(
                images=blank.images,
                input_lengths=blank.input_lengths,
                targets=blank.targets,
                target_lengths=blank.target_lengths,
                rejected=tuple(rejected),
            )

        return self._pack(usable, tuple(rejected))

    # ------------------------------------------------------------------ internals

    def _infeasible(self, sample: LineSample) -> RejectedSample:
        reason = (
            f"input_length {sample.input_length} < required {sample.required_length} for "
            f"{sample.target_length} target(s) (true_width={sample.true_width}); "
            f"text={sample.text!r}"
        )
        if not self.training:
            raise CTCFeasibilityError(
                f"Sample {sample.sample_id!r} cannot be aligned by CTC: {reason}. Refusing to "
                "drop it during evaluation — that would change the denominator of a reported "
                "metric."
            )
        return RejectedSample(
            sample.sample_id, sample.image_path, IntegrityCategory.IMPOSSIBLE_CTC_LENGTH, reason
        )

    def _record(self, rejection: RejectedSample) -> None:
        assert self.counters is not None
        self.counters.record(
            IntegrityIssue(
                category=rejection.category,
                sample_id=rejection.sample_id,
                path=rejection.image_path,
                reason=rejection.reason,
            )
        )

    def _pack(self, samples: list[LineSample], rejected: tuple[RejectedSample, ...]) -> Batch:
        height = int(samples[0].image.shape[-2])
        max_width = max(int(sample.image.shape[-1]) for sample in samples)
        if max_width % self.width_multiple:
            max_width += self.width_multiple - (max_width % self.width_multiple)

        images = torch.full(
            (len(samples), 1, height, max_width), self.pad_value, dtype=torch.float32
        )
        for position, sample in enumerate(samples):
            width = int(sample.image.shape[-1])
            images[position, :, :, :width] = sample.image

        # input_lengths come from the TRUE width. If this ever reads padded widths the model trains
        # happily and decodes truncated text.
        input_lengths = torch.tensor([sample.input_length for sample in samples], dtype=torch.long)
        target_lengths = torch.tensor(
            [sample.target_length for sample in samples], dtype=torch.long
        )
        targets = torch.cat([sample.targets for sample in samples])

        return Batch(
            images=images,
            input_lengths=input_lengths,
            targets=targets,
            target_lengths=target_lengths,
            texts=tuple(sample.text for sample in samples),
            sample_ids=tuple(sample.sample_id for sample in samples),
            writer_ids=tuple(sample.writer_id for sample in samples),
            image_paths=tuple(sample.image_path for sample in samples),
            true_widths=tuple(sample.true_width for sample in samples),
            rejected=rejected,
        )
