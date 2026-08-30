"""The training loop.

The loop itself is ordinary. Four things about it are not conveniences:

**Selection is by validation CER, never by loss.** CTC loss and CER are only loosely coupled — a
model can improve its loss while its decoded output degrades, because the loss rewards probability
mass on the correct alignment while CER only cares what the argmax says. Selecting on loss is how a
project ships its second-best checkpoint and never finds out.

**Gradient clipping is counted, not just applied.** Clipping that fires on nearly every step means
the run is unstable, and the correct response is a lower learning rate or a longer warmup, never a
higher ``max_norm``. A clip rate that is never measured is a diagnosis never made.

**A non-finite loss is skipped and counted, not absorbed.** One ``inf`` must not poison the weights,
and it must not be invisible either — the batch's ``sample_ids`` are logged.

**Validation decodes a fixed set of samples every time and prints them beside ground truth.**
Unchanging nonsense means a length or layout bug, near-correct-but-shifted means downsampling
arithmetic. Fixed samples, so successive epochs are comparable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from glyphmemory.config.schema import Config
from glyphmemory.ctc.decode import DEFAULT_DECODER, decode_output
from glyphmemory.ctc.tokenizer import Tokenizer
from glyphmemory.metrics.text import MetricResult, corpus_cer, corpus_wer
from glyphmemory.model.htr import GMBase
from glyphmemory.model.loss import ctc_loss_for
from glyphmemory.runtime.logging import get_logger
from glyphmemory.training.checkpoint import (
    BEST_FILENAME,
    LAST_FILENAME,
    SELECTION_METRIC,
    CheckpointMeta,
    is_better,
    save_checkpoint,
)

logger = get_logger("training.trainer")

#: How many fixed samples to decode and print at every validation.
DEFAULT_PREVIEW_SAMPLES = 4


@dataclass(frozen=True, slots=True)
class EpochStats:
    """One training epoch, as recorded in ``metrics.jsonl``."""

    epoch: int
    steps: int
    loss: float
    learning_rate: float
    clipped_steps: int
    skipped_steps: int
    samples: int
    seconds: float

    @property
    def clip_rate(self) -> float:
        return self.clipped_steps / self.steps if self.steps else 0.0

    @property
    def samples_per_second(self) -> float:
        return self.samples / self.seconds if self.seconds > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "steps": self.steps,
            "train_loss": self.loss,
            "learning_rate": self.learning_rate,
            "clipped_steps": self.clipped_steps,
            "clip_rate": round(self.clip_rate, 6),
            "skipped_steps": self.skipped_steps,
            "samples": self.samples,
            "seconds": round(self.seconds, 3),
            "samples_per_second": round(self.samples_per_second, 2),
        }


@dataclass(frozen=True, slots=True)
class ValidationStats:
    """One validation pass."""

    epoch: int
    loss: float
    cer: MetricResult
    wer: MetricResult
    samples: int
    seconds: float
    previews: tuple[tuple[str, str], ...] = ()

    @property
    def cer_value(self) -> float | None:
        return self.cer.value

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "val_loss": self.loss,
            SELECTION_METRIC: self.cer.value,
            "val_wer": self.wer.value,
            "val_exact_matches": self.cer.exact_matches,
            "val_samples": self.samples,
            "seconds": round(self.seconds, 3),
            "normalization": self.cer.normalization,
            "decoder": self.cer.decoder.label,
        }

    def format(self) -> str:
        cer = "n/a" if self.cer.value is None else f"{self.cer.value:.4f}"
        wer = "n/a" if self.wer.value is None else f"{self.wer.value:.4f}"
        lines = [
            f"  val loss {self.loss:.4f}   CER {cer}   WER {wer}   "
            f"exact {self.cer.exact_matches}/{self.samples}",
        ]
        for reference, hypothesis in self.previews:
            lines.append(f"    truth {reference[:64]!r}")
            lines.append(f"    pred  {hypothesis[:64]!r}")
        return "\n".join(lines)


@dataclass
class Trainer:
    """Trains :class:`GMBase` and records everything needed to reproduce the run.

    Args:
        model: The recognizer.
        tokenizer: Supplies the vocabulary and the normalization the metrics report.
        optimizer: Built by the caller, so the run record can describe it.
        scheduler: Stepped **per optimizer step**, not per epoch.
        train_loader: Training batches. Augmentation on.
        val_loader: Validation batches. Augmentation off — asserted, not assumed.
        config: Full config; hyperparameters come from ``config.training``.
        device: Resolved device.
        experiment: Where ``metrics.jsonl`` and checkpoints are written. ``None`` runs without
            persistence, which is what the tests and ``--max-steps`` probes use.
    """

    model: GMBase
    tokenizer: Tokenizer
    optimizer: Optimizer
    scheduler: LRScheduler | None
    train_loader: DataLoader
    val_loader: DataLoader | None
    config: Config
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    experiment: Any = None
    checkpoint_meta: dict[str, Any] = field(default_factory=dict)
    preview_samples: int = DEFAULT_PREVIEW_SAMPLES
    #: When set, every validation appends its decoded-versus-truth previews here as JSONL.
    trace_path: Path | None = None

    step: int = field(default=0, init=False)
    best_value: float | None = field(default=None, init=False)
    history: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.model.to(self.device)
        self._assert_validation_is_unaugmented()

    # ------------------------------------------------------------------ guards

    def _assert_validation_is_unaugmented(self) -> None:
        """Augmentation during validation invalidates the number it produces.

        Checked against what the pipeline *does*, not against the flag that built it. A disabled
        pipeline is an ``AugmentationPipeline`` wrapping ``Identity`` rather than ``None`` —
        deliberately, so callers always have something to invoke — so the meaningful question is
        ``is_identity``, and a ``None`` check would pass a fully augmenting pipeline straight
        through.
        """
        if self.val_loader is None:
            return
        augmentation = getattr(self.val_loader.dataset, "augmentation", None)
        if augmentation is None:
            return
        if not getattr(augmentation, "is_identity", False):
            families = getattr(augmentation, "families", ("unknown",))
            raise ValueError(
                f"The validation dataset augments {list(families)}. Evaluation is never "
                "augmented — a metric computed on perturbed "
                "inputs measures something nobody asked for. Build it with training=False."
            )

    # ------------------------------------------------------------------ training

    def train_epoch(self, epoch: int) -> EpochStats:
        """One pass over the training loader."""
        self.model.train()
        started = time.perf_counter()

        total_loss = 0.0
        steps = clipped = skipped = samples = 0
        max_norm = self.config.training.grad_clip_norm

        for batch in self.train_loader:
            if batch.is_empty:
                continue
            batch = batch.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(batch.images, batch.input_lengths)
            loss, _ = ctc_loss_for(output, batch.targets, batch.target_lengths)

            if not torch.isfinite(loss):
                skipped += 1
                logger.warning(
                    "Non-finite loss at step %d; skipping. sample_ids=%s",
                    self.step,
                    list(batch.sample_ids[:8]),
                )
                continue

            loss.backward()
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            if float(norm) > max_norm:
                clipped += 1
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += float(loss.detach())
            steps += 1
            samples += batch.batch_size
            self.step += 1

        elapsed = time.perf_counter() - started
        stats = EpochStats(
            epoch=epoch,
            steps=steps,
            loss=total_loss / steps if steps else float("nan"),
            learning_rate=self.optimizer.param_groups[0]["lr"],
            clipped_steps=clipped,
            skipped_steps=skipped,
            samples=samples,
            seconds=elapsed,
        )

        if stats.steps and stats.clip_rate > 0.5:
            logger.warning(
                "Gradient clipping fired on %.0f%% of steps this epoch. That is instability, "
                "not a threshold problem — lower the learning rate or lengthen warmup rather "
                "than raising grad_clip_norm.",
                100 * stats.clip_rate,
            )
        if stats.skipped_steps:
            logger.warning(
                "%d step(s) skipped this epoch on a non-finite loss.", stats.skipped_steps
            )
        return stats

    # ------------------------------------------------------------------ validation

    @torch.no_grad()
    def validate(self, epoch: int) -> ValidationStats | None:
        """Loss, CER and WER over the validation loader, plus fixed decoded previews."""
        if self.val_loader is None:
            return None

        self.model.eval()
        started = time.perf_counter()

        total_loss = 0.0
        batches = 0
        pairs: list[tuple[str, str]] = []
        sample_ids: list[str] = []

        for batch in self.val_loader:
            if batch.is_empty:
                continue
            batch = batch.to(self.device)
            output = self.model(batch.images, batch.input_lengths)
            loss, _ = ctc_loss_for(output, batch.targets, batch.target_lengths)
            if torch.isfinite(loss):
                total_loss += float(loss)
                batches += 1
            predictions = decode_output(output, self.tokenizer)
            pairs.extend(zip(batch.texts, predictions, strict=True))
            sample_ids.extend(batch.sample_ids)

        cer = corpus_cer(
            pairs, policy=self.tokenizer.policy, decoder=DEFAULT_DECODER, sample_ids=sample_ids
        )
        wer = corpus_wer(
            pairs, policy=self.tokenizer.policy, decoder=DEFAULT_DECODER, sample_ids=sample_ids
        )

        # The first N pairs in loader order — the same lines every epoch, so successive previews are
        # comparable rather than a fresh random sample each time.
        previews = tuple(pairs[: self.preview_samples])

        return ValidationStats(
            epoch=epoch,
            loss=total_loss / batches if batches else float("nan"),
            cer=cer,
            wer=wer,
            samples=len(pairs),
            seconds=time.perf_counter() - started,
            previews=previews,
        )

    # ------------------------------------------------------------------ run

    def fit(self, epochs: int | None = None, *, validate_every: int = 1) -> list[dict[str, Any]]:
        """Train for ``epochs``, validating and checkpointing along the way.

        Returns the metrics history, which is also appended to ``metrics.jsonl``.
        """
        epochs = epochs if epochs is not None else self.config.training.epochs
        patience = 0

        for epoch in range(1, epochs + 1):
            train_stats = self.train_epoch(epoch)
            record: dict[str, Any] = {**train_stats.as_dict()}

            validation = None
            if self.val_loader is not None and epoch % validate_every == 0:
                validation = self.validate(epoch)
                if validation is not None:
                    record.update(validation.as_dict())

            logger.info(
                "epoch %d/%d  loss %.4f  lr %.2e  %.1f samples/s  clip %.0f%%",
                epoch,
                epochs,
                train_stats.loss,
                train_stats.learning_rate,
                train_stats.samples_per_second,
                100 * train_stats.clip_rate,
            )
            if validation is not None:
                logger.info("%s", validation.format())

            self.history.append(record)
            if self.experiment is not None:
                self.experiment.append_jsonl(self.experiment.metrics_stream_path, record)
            if validation is not None:
                self._append_trace(epoch, validation)

            improved = self._checkpoint(epoch, validation)
            patience = 0 if improved else patience + 1
            if validation is not None and patience >= self.config.training.patience:
                logger.info(
                    "Stopping early: %s has not improved for %d validation(s).",
                    SELECTION_METRIC,
                    patience,
                )
                break

        return self.history

    def _append_trace(self, epoch: int, validation: ValidationStats) -> None:
        """Persist this epoch's decoded previews.

        One epoch in isolation tells you none of them, which is why this is a file rather than a
        print.
        """
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            for index, (reference, hypothesis) in enumerate(validation.previews):
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "index": index,
                            "reference": reference,
                            "hypothesis": hypothesis,
                            "cer": self.cer_of(reference, hypothesis),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    def cer_of(self, reference: str, hypothesis: str) -> float | None:
        """Per-line CER under the tokenizer's policy, for the trace."""
        from glyphmemory.metrics.text import cer

        return cer(reference, hypothesis, policy=self.tokenizer.policy)

    def _checkpoint(self, epoch: int, validation: ValidationStats | None) -> bool:
        """Write ``last.pt`` always and ``best.pt`` on improvement. Returns whether it improved.

        Selection uses :data:`SELECTION_METRIC` — validation CER — in this one place, so "best"
        cannot drift into meaning "lowest loss".
        """
        if self.experiment is None:
            return False

        metrics = {SELECTION_METRIC: validation.cer.value} if validation is not None else {}
        meta = CheckpointMeta(
            epoch=epoch,
            step=self.step,
            metrics={k: v for k, v in metrics.items() if v is not None},
            charset_fingerprint=self.tokenizer.charset.fingerprint(),
            tokenizer_fingerprint=self.tokenizer.fingerprint(),
            **self.checkpoint_meta,
        )

        directory = Path(self.experiment.checkpoints_dir)
        save_checkpoint(
            directory / LAST_FILENAME,
            model=self.model,
            meta=meta,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        candidate = validation.cer.value if validation is not None else None
        if is_better(candidate, self.best_value):
            self.best_value = candidate
            save_checkpoint(
                directory / BEST_FILENAME,
                model=self.model,
                meta=meta,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            logger.info("New best %s: %.4f (epoch %d)", SELECTION_METRIC, candidate, epoch)
            return True
        return False
