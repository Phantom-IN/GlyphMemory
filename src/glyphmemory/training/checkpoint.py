"""Checkpoint saving and loading.

A checkpoint is not just weights. So the charset fingerprint travels inside the file and
:func:`load_checkpoint` **refuses** to load against a different one. That refusal is the single most
important line in this module.

Two further properties, both cheap and both about not losing evidence:

**Writes are atomic.** Save to a temporary file in the same directory, then ``os.replace``. A run
interrupted mid-save otherwise leaves a truncated file that still has a plausible name, and the next
person to load it gets an unpickling error days later rather than a missing file immediately.

**Selection metadata travels with the weights.** Epoch, metrics, config, manifest fingerprints, git
commit, seed and parameter count.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from glyphmemory.runtime.logging import get_logger

logger = get_logger("training.checkpoint")

#: Bumped when the stored key set changes, so an old checkpoint fails loudly rather than loading
#: with silently missing fields.
CHECKPOINT_SCHEMA_VERSION = "1"

BEST_FILENAME = "best.pt"
LAST_FILENAME = "last.pt"

#: The metric the best checkpoint is selected on.
SELECTION_METRIC = "val_cer"


class CheckpointCompatibilityError(ValueError):
    """A checkpoint cannot be loaded into the current setup. Never catch this to continue."""


@dataclass(frozen=True, slots=True)
class CheckpointMeta:
    """Everything about a checkpoint except the tensors."""

    epoch: int
    step: int
    metrics: dict[str, float]
    charset_fingerprint: str
    tokenizer_fingerprint: str
    manifest_fingerprints: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    parameter_count: int = 0
    git_commit: str | None = None
    seed: int | None = None
    run_id: str | None = None
    schema_version: str = CHECKPOINT_SCHEMA_VERSION

    @property
    def selection_value(self) -> float | None:
        """The metric ``best.pt`` is chosen by."""
        return self.metrics.get(SELECTION_METRIC)

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "metrics": dict(self.metrics),
            "charset_fingerprint": self.charset_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "manifest_fingerprints": dict(self.manifest_fingerprints),
            "config": self.config,
            "parameter_count": self.parameter_count,
            "git_commit": self.git_commit,
            "seed": self.seed,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CheckpointMeta:
        return cls(
            epoch=payload["epoch"],
            step=payload["step"],
            metrics=dict(payload.get("metrics", {})),
            charset_fingerprint=payload["charset_fingerprint"],
            tokenizer_fingerprint=payload.get("tokenizer_fingerprint", ""),
            manifest_fingerprints=dict(payload.get("manifest_fingerprints", {})),
            config=payload.get("config", {}),
            parameter_count=payload.get("parameter_count", 0),
            git_commit=payload.get("git_commit"),
            seed=payload.get("seed"),
            run_id=payload.get("run_id"),
            schema_version=payload.get("schema_version", "0"),
        )


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """A checkpoint read from disk, before anything is restored into a module."""

    meta: CheckpointMeta
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    meta: CheckpointMeta,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> Path:
    """Write a checkpoint atomically.

    The temporary file is created in the destination directory, not in ``/tmp``, so the final
    ``os.replace`` is a same-filesystem rename and therefore atomic.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "meta": meta.as_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }

    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def load_checkpoint(
    path: str | Path,
    *,
    charset_fingerprint: str | None = None,
    strict_charset: bool = True,
) -> LoadedCheckpoint:
    """Read a checkpoint, refusing one trained against a different vocabulary.

    Args:
        charset_fingerprint: The fingerprint of the charset the caller intends to use. When given
            and different from the stored one, loading raises.
        strict_charset: Set ``False`` only for inspecting an incompatible checkpoint deliberately —
            never in a training or evaluation path.

    Raises:
        CheckpointCompatibilityError: Schema version or charset mismatch.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if "meta" not in payload or "model_state" not in payload:
        raise CheckpointCompatibilityError(
            f"{path} does not look like a GlyphMemory checkpoint (missing 'meta' or 'model_state')."
        )

    meta = CheckpointMeta.from_dict(payload["meta"])
    if meta.schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            f"{path} uses checkpoint schema {meta.schema_version!r}, but this build expects "
            f"{CHECKPOINT_SCHEMA_VERSION!r}."
        )

    if charset_fingerprint is not None and meta.charset_fingerprint != charset_fingerprint:
        message = (
            f"{path} was trained against charset {meta.charset_fingerprint[:12]} but the "
            f"current charset is {charset_fingerprint[:12]}. A checkpoint is meaningless "
            "without the vocabulary it was trained against — the "
            "same logits decode to different text."
        )
        if strict_charset:
            raise CheckpointCompatibilityError(message)
        logger.warning("%s Loading anyway because strict_charset=False.", message)

    return LoadedCheckpoint(
        meta=meta,
        model_state=payload["model_state"],
        optimizer_state=payload.get("optimizer_state"),
        scheduler_state=payload.get("scheduler_state"),
    )


def restore_model(model: nn.Module, checkpoint: LoadedCheckpoint, *, strict: bool = True) -> None:
    """Load weights into ``model`` in place."""
    model.load_state_dict(checkpoint.model_state, strict=strict)


def is_better(candidate: float | None, incumbent: float | None) -> bool:
    """Whether ``candidate`` should replace ``incumbent`` as the best checkpoint.

    Lower is better — the selection metric is a CER. ``None`` never wins, so a validation that
    produced no value (every reference empty) cannot displace a real result.
    """
    if candidate is None:
        return False
    if incumbent is None:
        return True
    return candidate < incumbent
