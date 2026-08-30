"""`WriterProfile`: the compiled, serializable output of enrollment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

#: Bumped when the stored key set changes, so an old profile fails loudly rather than loading with
#: silently missing fields — the same reason `training/checkpoint.py` versions its schema.
PROFILE_SCHEMA_VERSION = "1"


class ProfileCompatibilityError(ValueError):
    """A `WriterProfile` cannot be loaded against the active model. Never catch to continue."""


@dataclass(frozen=True, slots=True)
class Glyph:
    """One character's compiled evidence for one writer."""

    character: str
    prototype: Tensor
    number_of_observations: int
    mean_alignment_confidence: float
    feature_layer: str
    #: Prototypes beyond `prototype` (top-K variant). `retrieval.py::memory_scores` scores a
    #: character against `prototype` plus every vector here and keeps the best match, so a profile
    #: with no additional prototypes behaves exactly like a single-prototype one always has.
    additional_prototypes: tuple[Tensor, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "character": self.character,
            "prototype": self.prototype,
            "number_of_observations": self.number_of_observations,
            "mean_alignment_confidence": self.mean_alignment_confidence,
            "feature_layer": self.feature_layer,
            "additional_prototypes": self.additional_prototypes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Glyph:
        return cls(
            character=payload["character"],
            prototype=payload["prototype"],
            number_of_observations=payload["number_of_observations"],
            mean_alignment_confidence=payload["mean_alignment_confidence"],
            feature_layer=payload["feature_layer"],
            additional_prototypes=tuple(payload.get("additional_prototypes", ())),
        )


@dataclass(frozen=True, slots=True)
class WriterProfile:
    """Compiled per-writer memory.

    ``counts`` and ``confidences`` are the flat per-character views the section's pseudocode shows
    alongside ``glyphs`` — exposed here as properties derived from ``glyphs`` rather than stored a
    second time, so there is exactly one source of truth and no way for the two to drift apart (the
    opposite direction of `model/htr.py`'s `HTROutput.input_lengths`, which is stored precisely
    *because* recomputing it wrongly is the likelier bug — here storing it twice is the risk
    instead).
    """

    schema_version: str
    model_fingerprint: str
    feature_layer: str
    feature_dim: int
    glyphs: dict[str, Glyph] = field(default_factory=dict)
    #: Present in the schema now so adding it later is not a schema-version bump.
    global_style: Tensor | None = None
    #: Identity of the `GlyphProjection` (`memory/projection.py`) that produced these prototypes, if
    #: any -- `None` means raw, unprojected features (every V0 profile). `require_projection` is the
    #: load-time compatibility check one layer up from `model_fingerprint`'s: a profile compiled
    #: through one projection's feature space is as meaningless loaded against a different one as
    #: against a different base model.
    projection_fingerprint: str | None = None

    @property
    def counts(self) -> dict[str, int]:
        return {character: glyph.number_of_observations for character, glyph in self.glyphs.items()}

    @property
    def confidences(self) -> dict[str, float]:
        return {
            character: glyph.mean_alignment_confidence for character, glyph in self.glyphs.items()
        }

    @property
    def characters(self) -> frozenset[str]:
        return frozenset(self.glyphs)

    def prototype_for(self, character: str) -> Tensor | None:
        """The compiled prototype for ``character``, or ``None`` if it was never observed."""
        glyph = self.glyphs.get(character)
        return glyph.prototype if glyph is not None else None

    def require_projection(
        self, expected_projection_fingerprint: str | None, *, strict: bool = True
    ) -> None:
        """Raise if this profile's projection identity does not match the active one.

        Args:
            expected_projection_fingerprint: The active `GlyphProjection` artifact's content
                fingerprint (`runtime.checkpoint_fingerprint`), or ``None`` to require a **raw**,
                unprojected profile. Comparing against ``None`` is deliberate, not a skip: a profile
                compiled with a projection is exactly as wrong to read as raw features as one
                compiled without a projection is to read as if it had one.
            strict: Set ``False`` only to inspect a mismatched profile deliberately.

        Raises:
            ProfileCompatibilityError: The profile's `projection_fingerprint` disagrees with
                `expected_projection_fingerprint`.
        """
        if self.projection_fingerprint == expected_projection_fingerprint:
            return
        message = (
            f"profile projection identity {self.projection_fingerprint!r} does not match the "
            f"active projection {expected_projection_fingerprint!r}. A profile's prototypes "
            "live in one specific feature space -- raw, or one particular GlyphProjection's "
            "output -- and reading it as if it lived in a different one silently scores "
            "cosine similarities in the wrong space, the same failure mode "
            "`model_fingerprint` mismatches guard against one layer down."
        )
        if strict:
            raise ProfileCompatibilityError(message)

    def estimated_bytes(self) -> int:
        """Rough serialized size: prototype tensors plus a fixed per-glyph metadata overhead."""
        tensor_bytes = sum(
            glyph.prototype.element_size() * glyph.prototype.nelement()
            + sum(t.element_size() * t.nelement() for t in glyph.additional_prototypes)
            for glyph in self.glyphs.values()
        )
        # count (int) + confidence (float) + a short character key, per glyph — a rough, explicitly
        # labeled estimate, not a byte-exact accounting.
        metadata_bytes = len(self.glyphs) * 24
        return tensor_bytes + metadata_bytes

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_fingerprint": self.model_fingerprint,
            "feature_layer": self.feature_layer,
            "feature_dim": self.feature_dim,
            "num_characters": len(self.glyphs),
            "characters": "".join(sorted(self.glyphs)),
            "estimated_bytes": self.estimated_bytes(),
            "has_global_style": self.global_style is not None,
            "projection_fingerprint": self.projection_fingerprint,
        }

    # ------------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> Path:
        """Write atomically: the same same-filesystem temp-file-then-rename `training/checkpoint.py`
        uses, so a save interrupted mid-write never leaves a truncated profile with a plausible
        filename.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "model_fingerprint": self.model_fingerprint,
            "feature_layer": self.feature_layer,
            "feature_dim": self.feature_dim,
            "glyphs": {character: glyph.as_dict() for character, glyph in self.glyphs.items()},
            "global_style": self.global_style,
            "projection_fingerprint": self.projection_fingerprint,
        }

        temporary = path.with_name(f".{path.name}.tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_model_fingerprint: str | None = None,
        strict: bool = True,
    ) -> WriterProfile:
        """Read a profile, refusing one compiled against a different model.

        Args:
            expected_model_fingerprint: The active checkpoint's fingerprint
                (`runtime.checkpoint_fingerprint`). When given and different from the profile's own,
                loading raises — a profile's prototypes live in *this specific model's* feature
                space, and cosine similarity against features from a different model is not a
                compatibility warning, it is silently wrong personalization.
            strict: Set ``False`` only to inspect an incompatible profile deliberately — never on a
                real enrollment/transcription path.

        Raises:
            ProfileCompatibilityError: Schema version or model-fingerprint mismatch.
        """
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)

        if "schema_version" not in payload or "glyphs" not in payload:
            raise ProfileCompatibilityError(
                f"{path} does not look like a GlyphMemory WriterProfile "
                "(missing 'schema_version' or 'glyphs')."
            )
        if payload["schema_version"] != PROFILE_SCHEMA_VERSION:
            raise ProfileCompatibilityError(
                f"{path} uses profile schema {payload['schema_version']!r}, but this build "
                f"expects {PROFILE_SCHEMA_VERSION!r}."
            )

        stored_fingerprint = payload["model_fingerprint"]
        if (
            expected_model_fingerprint is not None
            and stored_fingerprint != expected_model_fingerprint
        ):
            message = (
                f"{path} was compiled against model {stored_fingerprint[:12]} but the active "
                f"model is {expected_model_fingerprint[:12]}. A WriterProfile's prototypes live "
                "in that specific model's feature space — "
                "loading it against a different one would silently score real cosine "
                "similarities in the wrong space."
            )
            if strict:
                raise ProfileCompatibilityError(message)

        return cls(
            schema_version=payload["schema_version"],
            model_fingerprint=stored_fingerprint,
            feature_layer=payload["feature_layer"],
            feature_dim=payload["feature_dim"],
            glyphs={
                character: Glyph.from_dict(glyph_payload)
                for character, glyph_payload in payload["glyphs"].items()
            },
            global_style=payload.get("global_style"),
            projection_fingerprint=payload.get("projection_fingerprint"),
        )
