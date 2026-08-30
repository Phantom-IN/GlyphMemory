"""Gated, blank-protected residual fusion.

    final_logit[c, t] = base_logit[c, t] + alpha * gate(t) * memory_score[c, t] gate(t) = 1 -
    P_blank(t)

1. **The blank logit is never modified.** It has no prototype (`memory/retrieval.py` already
   guarantees this structurally), and this module enforces it a second time, defensively, so the
   guarantee does not depend on retrieval never changing.
2. **Gated by emission probability.** Memory acts only on frames the base model already believes are
   emitting *some* character — without this, positive bias on blank-dominated frames inserts
   spurious characters.
3. **Residual by construction.** `final = base + correction`, never regenerated logits — so generic
   recognition survives even when memory is disabled, absent, or wrong.

`fuse` and `correction` are pure, per-sample (`[T, C]`) functions, testable with hand-built logits.
`personalize` is the batched orchestration that reads an `HTROutput` and a `WriterProfile` and calls
them — kept outside `GMBase.forward`, because fusion is a post-hoc transform of a frozen model's
output, not a change to the model (`gm-base-v0` stays frozen, ADR-0008).
"""

from __future__ import annotations

import torch
from torch import Tensor

from glyphmemory.config.schema import MemoryConfig
from glyphmemory.ctc.tokenizer import BLANK_INDEX, Charset
from glyphmemory.memory.compiler import FEATURE_ATTRIBUTES, PROJECTED_FEATURE_LAYERS
from glyphmemory.memory.profile import WriterProfile
from glyphmemory.memory.projection import GlyphProjection
from glyphmemory.memory.retrieval import memory_scores
from glyphmemory.model.htr import HTROutput


def blank_emission_gate(base_logits: Tensor, *, blank_index: int = BLANK_INDEX) -> Tensor:
    """``1 - P(blank)`` at every frame, from the base model's own softmax over its own logits.

    Args:
        base_logits: ``[..., C]`` — any number of leading dims, so the same function serves a single
            sample ``[T, C]`` or a batch ``[B, T, C]``.

    Returns:
        ``[...]`` (the same shape minus the class dimension), in ``[0, 1]``.
    """
    probabilities = torch.softmax(base_logits, dim=-1)
    return 1.0 - probabilities[..., blank_index]


def correction(
    base_logits: Tensor,
    scores: Tensor,
    *,
    alpha: float,
    gate_by_emission: bool = True,
    protect_blank: bool = True,
    blank_index: int = BLANK_INDEX,
) -> Tensor:
    """The additive term fusion applies: ``alpha * gate(t) * scores[c, t]``, blank zeroed.

    Kept separate from :func:`fuse` so the residual property (``fuse(...) - base_logits ==
    correction(...)``) is checkable independently rather than only by construction.
    """
    if base_logits.shape != scores.shape:
        raise ValueError(
            f"base_logits {tuple(base_logits.shape)} and scores {tuple(scores.shape)} must "
            "have the same shape."
        )

    if gate_by_emission:
        gate = blank_emission_gate(base_logits, blank_index=blank_index)
    else:
        gate = torch.ones(
            base_logits.shape[:-1], dtype=base_logits.dtype, device=base_logits.device
        )

    result = alpha * gate.unsqueeze(-1) * scores
    if protect_blank:
        result = result.clone()
        result[..., blank_index] = 0.0
    return result


def fuse(
    base_logits: Tensor,
    scores: Tensor,
    *,
    alpha: float,
    gate_by_emission: bool = True,
    protect_blank: bool = True,
    blank_index: int = BLANK_INDEX,
) -> Tensor:
    """``base_logits + correction(...)`` — the residual fusion itself.

    Args:
        base_logits: ``[..., C]``, the frozen model's own logits. Never mutated in place.
        scores: ``[..., C]``, `retrieval.memory_scores`'s output (or a hand-built tensor of the same
            shape, for tests).
    """
    return base_logits + correction(
        base_logits,
        scores,
        alpha=alpha,
        gate_by_emission=gate_by_emission,
        protect_blank=protect_blank,
        blank_index=blank_index,
    )


def personalize(
    output: HTROutput,
    profile: WriterProfile | None,
    charset: Charset,
    config: MemoryConfig,
    *,
    projection: GlyphProjection | None = None,
) -> Tensor:
    """``[B, T, C]`` corrected logits — the batched entry point `GMBase.forward` never calls.

    Falls back to ``output.logits`` **unchanged, the same tensor** — not a recomputed functional
    equivalent — whenever memory is disabled or has nothing to contribute: ``config.enabled`` is
    ``False``, ``profile`` is ``None``, or ``profile`` has no glyphs.

    Args:
        projection: A trained `GlyphProjection`. Required when ``config.feature_layer in
            PROJECTED_FEATURE_LAYERS``; applied per frame to the raw ``sequence_features`` before
            retrieval, the same transform and the same order (per frame, before pooling/comparison)
            `memory/compiler.py::compile_profile` applies when it compiled ``profile`` — both sides
            of every cosine score must live in the same projected space.

    Raises:
        ValueError: ``profile.feature_layer`` disagrees with ``config.feature_layer`` — fusing
            frames from one feature space against prototypes compiled from another would silently
            score nonsense rather than raise. Also raised when ``projection`` is given/omitted
            inconsistently with ``config.feature_layer``.
    """
    if not config.enabled or profile is None or not profile.glyphs:
        return output.logits

    if profile.feature_layer != config.feature_layer:
        raise ValueError(
            f"profile was compiled on {profile.feature_layer!r} features but config requests "
            f"{config.feature_layer!r}; fusing them would compare mismatched feature spaces."
        )

    uses_projection = config.feature_layer in PROJECTED_FEATURE_LAYERS
    if uses_projection and projection is None:
        raise ValueError(
            f"feature_layer {config.feature_layer!r} requires a projection, got none."
        )
    if not uses_projection and projection is not None:
        raise ValueError(
            f"a projection was given but feature_layer {config.feature_layer!r} does not use one."
        )

    feature_attribute = FEATURE_ATTRIBUTES[config.feature_layer]
    features = getattr(output, feature_attribute)  # [B, T, D]
    if projection is not None:
        features = projection.to(features.device)(features)

    corrected = torch.stack(
        [
            fuse(
                output.logits[b],
                memory_scores(features[b], profile, charset),
                alpha=config.alpha,
                gate_by_emission=config.gate_by_emission,
                protect_blank=config.protect_blank,
            )
            for b in range(output.batch_size)
        ]
    )
    return corrected
