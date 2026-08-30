"""The learned glyph projection.

    384D sequence feature -> Linear(384,128) -> ReLU -> Linear(128,96) -> L2 normalize

A projection trained against one `gm-base-v0` fingerprint must not silently be used against a
different one, or against no checkpoint at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from glyphmemory.probes.geometry import l2_normalize

#: Bumped when the stored key set changes -- the same reason `training/checkpoint.py` and
#: `memory/profile.py` version their own schemas.
PROJECTION_SCHEMA_VERSION = "1"

DEFAULT_INPUT_DIM = 384
DEFAULT_HIDDEN_DIM = 128
DEFAULT_OUTPUT_DIM = 96


class ProjectionCompatibilityError(ValueError):
    """A projection artifact cannot be loaded against the active base model. Never catch to
    continue.
    """


class GlyphProjection(nn.Module):
    """`Linear(384, 128) -> ReLU -> Linear(128, 96) -> L2 normalize`."""

    def __init__(
        self,
        *,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        output_dim: int = DEFAULT_OUTPUT_DIM,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or output_dim < 1:
            raise ValueError(
                f"input_dim, hidden_dim and output_dim must all be positive, got "
                f"{input_dim}, {hidden_dim}, {output_dim}"
            )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        """``[..., input_dim] -> [..., output_dim]``, L2-normalized along the last dimension.

        Accepts any leading shape (a single vector, `[N, D]`, or `[B, T, D]`) -- `l2_normalize` and
        `nn.Linear` both operate on the last dimension only.
        """
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension {self.input_dim}, got {features.shape[-1]}"
            )
        return l2_normalize(self.net(features))

    def describe(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        }


@dataclass(frozen=True, slots=True)
class ProjectionMeta:
    """Everything about a trained projection except the weights."""

    schema_version: str
    base_model_fingerprint: str
    input_dim: int
    hidden_dim: int
    output_dim: int
    training_steps: int
    final_char_loss: float | None
    final_writer_loss: float | None
    char_loss_weight: float
    writer_loss_weight: float
    seed: int | None = None
    git_commit: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_model_fingerprint": self.base_model_fingerprint,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "training_steps": self.training_steps,
            "final_char_loss": self.final_char_loss,
            "final_writer_loss": self.final_writer_loss,
            "char_loss_weight": self.char_loss_weight,
            "writer_loss_weight": self.writer_loss_weight,
            "seed": self.seed,
            "git_commit": self.git_commit,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectionMeta:
        return cls(
            schema_version=payload["schema_version"],
            base_model_fingerprint=payload["base_model_fingerprint"],
            input_dim=payload["input_dim"],
            hidden_dim=payload["hidden_dim"],
            output_dim=payload["output_dim"],
            training_steps=payload["training_steps"],
            final_char_loss=payload.get("final_char_loss"),
            final_writer_loss=payload.get("final_writer_loss"),
            char_loss_weight=payload["char_loss_weight"],
            writer_loss_weight=payload["writer_loss_weight"],
            seed=payload.get("seed"),
            git_commit=payload.get("git_commit"),
        )


def save_projection(path: str | Path, *, model: GlyphProjection, meta: ProjectionMeta) -> Path:
    """Write a projection artifact atomically -- same same-filesystem temp-file-then-rename
    `training/checkpoint.py` and `memory/profile.py` both use.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {"meta": meta.as_dict(), "model_state": model.state_dict()}

    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def load_projection(
    path: str | Path,
    *,
    expected_base_model_fingerprint: str | None = None,
    strict: bool = True,
) -> tuple[GlyphProjection, ProjectionMeta]:
    """Read a projection artifact, refusing one trained against a different base checkpoint.

    Args:
        expected_base_model_fingerprint: The active `gm-base-v0` checkpoint's fingerprint
            (`runtime.checkpoint_fingerprint`). When given and different from the projection's own,
            loading raises -- a projection trained on one base model's feature space is meaningless
            applied to another's, the same reasoning `WriterProfile.load`'s
            `expected_model_fingerprint` already applies one layer up.
        strict: Set `False` only to inspect an incompatible projection deliberately -- never on a
            real enrollment/transcription path.

    Raises:
        ProjectionCompatibilityError: Schema version or base-model-fingerprint mismatch.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if "meta" not in payload or "model_state" not in payload:
        raise ProjectionCompatibilityError(
            f"{path} does not look like a GlyphMemory projection artifact "
            "(missing 'meta' or 'model_state')."
        )

    meta = ProjectionMeta.from_dict(payload["meta"])
    if meta.schema_version != PROJECTION_SCHEMA_VERSION:
        raise ProjectionCompatibilityError(
            f"{path} uses projection schema {meta.schema_version!r}, but this build expects "
            f"{PROJECTION_SCHEMA_VERSION!r}."
        )

    if (
        expected_base_model_fingerprint is not None
        and meta.base_model_fingerprint != expected_base_model_fingerprint
    ):
        message = (
            f"{path} was trained against base model {meta.base_model_fingerprint[:12]} but the "
            f"active model is {expected_base_model_fingerprint[:12]}. A projection's weights "
            "are meaningless applied to a different model's feature space."
        )
        if strict:
            raise ProjectionCompatibilityError(message)

    model = GlyphProjection(
        input_dim=meta.input_dim, hidden_dim=meta.hidden_dim, output_dim=meta.output_dim
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, meta
