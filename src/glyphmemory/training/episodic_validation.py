"""A small, cheap, periodic generic-recognition probe for episodic training.

That answers *how much* damage the run did and says nothing about *when* it happened. This module is
the cheap instrument that answers the "when": the same decode-and-score methodology
`evaluation/report.py::evaluate_checkpoint` uses, on a deliberately tiny fixed subset, run every
``k`` steps during training.

Three properties this probe is built to have, each for a stated reason:

1. **Disjoint from the few-shot harness writers.** The probe subset never contains `HARNESS_WRITERS`
   — the 3 validation writers harness (and therefore every adaptation-gain number) is measured on.
2. **Cheap enough not to distort the run it is diagnosing.** Batches are preprocessed and collated
   **once**, at construction, and reused for every check — evaluation is unaugmented and
   deterministic, so every check would otherwise redo identical image I/O and resizing.
   `EpisodicTrainingLog` reports probe time separately from training time so the diagnostic's own
   cost is visible rather than silently folded into throughput.
3. **Loud, not silent, about an unusable line.** Batches are built with the evaluation collator
   (``training=False``), which raises on a CTC-infeasible sample rather than dropping it and
   changing a reported metric's denominator. Because batching happens at construction, such a line
   fails immediately, not 200 steps into a training run.

**This probe is a diagnostic, not a verdict.** Its subset is ~5% of the validation split, sized for
speed; its own sampling noise is large relative to the ~1pp effect is chasing (see ->
`m10-episodic-diagnostics-001`). It exists to show the *shape* of a curve over training steps.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from glyphmemory.config.schema import Config
from glyphmemory.ctc.decode import DEFAULT_DECODER, decode_output
from glyphmemory.ctc.tokenizer import Tokenizer
from glyphmemory.data.collate import Batch, VariableWidthCollator
from glyphmemory.data.dataset import LineDataset
from glyphmemory.data.loader import normalization_from
from glyphmemory.data.manifest import ManifestRecord
from glyphmemory.data.splits import SplitLeakError
from glyphmemory.metrics.text import corpus_cer, corpus_wer
from glyphmemory.model.htr import GMBase
from glyphmemory.runtime.logging import get_logger

logger = get_logger("training.episodic_validation")

#: The 3 IAM validation writers few-shot harness measures adaptation gain on (->
#: `few-shot-harness-validation-001`, reused unchanged by every - gain number).
HARNESS_WRITERS: frozenset[str] = frozenset({"iam/025", "iam/058", "iam/061"})

#: Probe subset shape. Spread thin across many writers rather than deep into a few: this measures
#: *generic* recognition, so writer coverage matters more than lines-per-writer.
DEFAULT_PROBE_WRITERS = 24
DEFAULT_LINES_PER_WRITER = 5

#: Steps between checks.
DEFAULT_PROBE_EVERY = 25

#: Probe batching. Small and fixed so a probe check costs the same on every device this project runs
#: on, independent of whatever batch size the surrounding training config happens to use.
DEFAULT_PROBE_BATCH_SIZE = 8


@dataclass(frozen=True, slots=True)
class ProbeCheck:
    """One periodic generic-recognition check, at a known training step."""

    step: int
    cer: float | None
    wer: float | None
    n_lines: int
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "cer": self.cer,
            "wer": self.wer,
            "n_lines": self.n_lines,
            "seconds": round(self.seconds, 3),
        }


def select_probe_records(
    records: Iterable[ManifestRecord],
    *,
    split: str = "val",
    exclude_writers: Iterable[str] = HARNESS_WRITERS,
    n_writers: int = DEFAULT_PROBE_WRITERS,
    lines_per_writer: int = DEFAULT_LINES_PER_WRITER,
    seed: int = 1337,
) -> tuple[ManifestRecord, ...]:
    """Pick a small, deterministic, harness-disjoint probe subset out of ``records``.

    Writers are chosen from those with at least ``lines_per_writer`` lines in ``split``, so every
    chosen writer contributes the same number of lines — a writer contributing 1 line and another
    contributing 40 would make the probe's CER mostly one writer's handwriting.

    Args:
        exclude_writers: Writers that must not appear. Defaults to `HARNESS_WRITERS`; pass an empty
            collection only with a stated reason.

    Returns:
        Records, ordered deterministically by ``(writer_id, sample_id)``.

    Raises:
        SplitLeakError: An excluded writer survived selection — a re-check of the result, not trust
            in the filter above it (discipline, applied to a different pair of populations).
        ValueError: ``n_writers``/``lines_per_writer`` are not positive, or ``split`` has fewer than
            ``n_writers`` eligible writers — never silently returns a smaller probe than asked for.
    """
    if n_writers < 1 or lines_per_writer < 1:
        raise ValueError(
            f"n_writers and lines_per_writer must both be positive, got "
            f"{n_writers} and {lines_per_writer}"
        )
    excluded = frozenset(exclude_writers)

    by_writer: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        if record.split != split or record.writer_id in excluded:
            continue
        if record.sample_id is None:
            raise ValueError(f"Record {record.image!r} has no sample_id; the probe needs one.")
        by_writer[record.writer_id].append(record)

    eligible = sorted(w for w, recs in by_writer.items() if len(recs) >= lines_per_writer)
    if len(eligible) < n_writers:
        raise ValueError(
            f"split={split!r} has only {len(eligible)} writer(s) with at least "
            f"{lines_per_writer} line(s) after excluding {sorted(excluded)}; asked for "
            f"{n_writers}."
        )

    rng = random.Random(f"{seed}:probe:{split}")
    chosen_writers = sorted(rng.sample(eligible, n_writers))

    selected: list[ManifestRecord] = []
    for writer_id in chosen_writers:
        ordered = sorted(by_writer[writer_id], key=lambda r: r.sample_id or "")
        picked = random.Random(f"{seed}:probe:{writer_id}").sample(ordered, lines_per_writer)
        selected.extend(picked)

    leaked = sorted({r.writer_id for r in selected} & excluded)
    if leaked:
        raise SplitLeakError(
            f"probe subset contains excluded writer(s) {leaked} -- the probe must stay disjoint "
            "from the few-shot harness writers so checkpoint selection on it never tunes toward "
            "the writers the final adaptation-gain verdict is measured on."
        )
    return tuple(sorted(selected, key=lambda r: (r.writer_id, r.sample_id or "")))


class ValidationProbe:
    """A fixed probe subset, batched once, scored on demand.

    Args:
        records: The probe lines, typically from `select_probe_records`. Never re-filtered here — a
            caller that hand-picks records is taking responsibility for their disjointness, and
            :meth:`assert_disjoint_from` exists to check it explicitly.
        batch_size: See `DEFAULT_PROBE_BATCH_SIZE`.

    Raises:
        ValueError: ``records`` is empty.
        CTCFeasibilityError: A probe line cannot be aligned by CTC — raised here, at construction,
            rather than mid-training (`VariableWidthCollator`'s evaluation policy).
    """

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        tokenizer: Tokenizer,
        config: Config,
        *,
        device: torch.device | str = "cpu",
        batch_size: int = DEFAULT_PROBE_BATCH_SIZE,
    ) -> None:
        if not records:
            raise ValueError("ValidationProbe needs at least one record.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self._records = tuple(records)
        self._tokenizer = tokenizer
        self._device = device if isinstance(device, torch.device) else torch.device(device)

        dataset = LineDataset(
            records=self._records,
            tokenizer=tokenizer,
            augmentation=None,  # evaluation is never augmented
            height=config.data.image_height,
            width_multiple=config.data.width_multiple,
            max_width=config.data.max_width,
            normalization=normalization_from(config),
        )
        collator = VariableWidthCollator(
            training=False,  # raises on an infeasible line instead of changing the denominator
            pad_value=dataset.pad_value,
            width_multiple=config.data.width_multiple,
        )
        samples = [dataset[i] for i in range(len(dataset))]
        self._batches: tuple[Batch, ...] = tuple(
            collator(samples[start : start + batch_size]).to(self._device)
            for start in range(0, len(samples), batch_size)
        )

    @property
    def records(self) -> tuple[ManifestRecord, ...]:
        return self._records

    @property
    def n_lines(self) -> int:
        return len(self._records)

    @property
    def writers(self) -> frozenset[str]:
        return frozenset(r.writer_id for r in self._records)

    def assert_disjoint_from(self, writers: Iterable[str]) -> None:
        """Raise if this probe touches any of ``writers``."""
        shared = sorted(self.writers & frozenset(writers))
        if shared:
            raise SplitLeakError(f"probe subset overlaps {shared}.")

    def evaluate(self, model: GMBase, *, step: int) -> ProbeCheck:
        """Decode and score the probe subset with ``model``'s current weights.

        Runs in ``model.eval()`` under ``torch.no_grad()`` and restores the model's previous
        training mode afterward — the same mode discipline `memory/compiler.py::compile_profile` and
        `training/episodic.py::episodic_step_v1` already established, so a probe call cannot leave
        the surrounding training loop in eval mode.
        """
        started = time.perf_counter()
        was_training = model.training
        model.eval()
        pairs: list[tuple[str, str]] = []
        sample_ids: list[str] = []
        try:
            with torch.no_grad():
                for batch in self._batches:
                    if batch.is_empty:
                        continue
                    output = model(batch.images, batch.input_lengths)
                    predictions = decode_output(output, self._tokenizer)
                    for sample_id, reference, hypothesis in zip(
                        batch.sample_ids, batch.texts, predictions, strict=True
                    ):
                        pairs.append((reference, hypothesis))
                        sample_ids.append(sample_id)
        finally:
            model.train(was_training)

        cer = corpus_cer(
            pairs, policy=self._tokenizer.policy, decoder=DEFAULT_DECODER, sample_ids=sample_ids
        )
        wer = corpus_wer(
            pairs, policy=self._tokenizer.policy, decoder=DEFAULT_DECODER, sample_ids=sample_ids
        )
        return ProbeCheck(
            step=step,
            cer=cer.value,
            wer=wer.value,
            n_lines=len(pairs),
            seconds=time.perf_counter() - started,
        )
