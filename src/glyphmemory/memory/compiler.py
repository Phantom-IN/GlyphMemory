"""The prototype compiler, turned into code.

    image -> frozen base recognizer -> feature sequence
          -> forced CTC alignment against the known transcript
          -> per-character feature span
          -> pool span (posterior-weighted / peak-frame / uniform)
          -> L2 normalize -> accumulate per character

Enrollment is forward passes, alignment and averaging, nothing else, and this module is the one
place that has to prove it rather than just claim it (`test_memory_compiler.py`'s gradient-free
test).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from glyphmemory.alignment import forced_align
from glyphmemory.config.schema import MemoryConfig
from glyphmemory.ctc.normalization import NFC_V1, normalize
from glyphmemory.ctc.tokenizer import Charset
from glyphmemory.memory.pooling import POOLING_STRATEGIES
from glyphmemory.memory.profile import PROFILE_SCHEMA_VERSION, Glyph, WriterProfile
from glyphmemory.memory.projection import GlyphProjection
from glyphmemory.memory.prototypes import (
    PROTOTYPE_STRATEGIES,
    GlyphAccumulator,
    PrototypeAccumulator,
)
from glyphmemory.memory.style import compile_global_style
from glyphmemory.model.htr import GMBase

#: `MemoryConfig.feature_layer` -> which `HTROutput` tensor to read.
FEATURE_ATTRIBUTES = {
    "sequence": "sequence_features",
    "visual": "visual_features",
    "learned_projection": "sequence_features",
}

#: `feature_layer` values whose raw features must be passed through a trained `GlyphProjection`
#: before pooling/accumulation. Retrieval (`memory/fusion.py::personalize`) applies it the same way,
#: per frame, before the cosine comparison, so both sides of every score live in the same 96D space.
PROJECTED_FEATURE_LAYERS = frozenset({"learned_projection"})


def compile_profile(
    model: GMBase,
    charset: Charset,
    lines: Sequence[tuple[Tensor, str]],
    *,
    model_fingerprint: str,
    config: MemoryConfig | None = None,
    device: torch.device | str = "cpu",
    projection: GlyphProjection | None = None,
    projection_fingerprint: str | None = None,
    with_global_style: bool = False,
) -> WriterProfile:
    """Compile a `WriterProfile` from `[(image, transcript), ...]`.

    Args:
        model: The frozen recognizer. Only ever run in eval mode under `no_grad` here, and restored
            to whatever training mode it was in before this call returns — enrollment never changes
            the model, not even its mode.
        charset: Must match ``model``'s vocabulary. `forced_align` raises
            :class:`~glyphmemory.ctc.tokenizer.UnsupportedCharacterError` on a transcript character
            the charset does not define, before it could silently produce a wrong span for it.
        lines: One ``(image, transcript)`` pair per enrollment line. ``image`` is a single,
            already-preprocessed ``[1, H, W]`` tensor (unbatched); this function does not reach into
            a `DataLoader`, so any source of paired images/transcripts — a batch, a CLI-loaded file,
            a test fixture — can call it the same way. Must be non-empty: a profile compiled from
            zero lines is not a meaningful object.
        model_fingerprint: Identity of the checkpoint ``model``'s weights came from
            (`runtime.checkpoint_fingerprint` over the checkpoint file), stored on the profile so a
            mismatched profile refuses to load later (`WriterProfile.load`) rather than silently
            personalizing a different recognizer's logits.
        config: Feature layer and pooling strategy. Defaults to `MemoryConfig()`'s V0 defaults
            (``sequence`` features, ``posterior_weighted`` pooling).
        device: Where to run the forward passes.
        projection: A trained `GlyphProjection`. Required when ``config.feature_layer in
            PROJECTED_FEATURE_LAYERS``, applied per frame to the raw ``sequence_features`` before
            pooling; rejected otherwise, since a projection given for a ``feature_layer`` that does
            not use one is a likely caller mistake, not a harmless no-op.
        projection_fingerprint: Identity of ``projection`` (`runtime.checkpoint_fingerprint` over
            the projection artifact file), stored on the returned profile so a later
            `WriterProfile.require_projection` can refuse a mismatched projection the same way
            `model_fingerprint` already guards the base model.

    Raises:
        ValueError: ``lines`` is empty, ``config.pooling``/``config.feature_layer`` names an unknown
            strategy/layer, or ``projection`` is given/omitted inconsistently with
            ``config.feature_layer``.
    """
    if not lines:
        raise ValueError("compile_profile requires at least one enrollment line, got zero.")

    resolved_config = config or MemoryConfig()
    if resolved_config.pooling not in POOLING_STRATEGIES:
        raise ValueError(
            f"Unknown pooling strategy {resolved_config.pooling!r}; "
            f"expected one of {sorted(POOLING_STRATEGIES)}."
        )
    if resolved_config.feature_layer not in FEATURE_ATTRIBUTES:
        raise ValueError(
            f"Unknown feature_layer {resolved_config.feature_layer!r}; "
            f"expected one of {sorted(FEATURE_ATTRIBUTES)}."
        )
    if resolved_config.prototype_strategy not in PROTOTYPE_STRATEGIES:
        raise ValueError(
            f"Unknown prototype_strategy {resolved_config.prototype_strategy!r}; "
            f"expected one of {sorted(PROTOTYPE_STRATEGIES)}."
        )
    uses_projection = resolved_config.feature_layer in PROJECTED_FEATURE_LAYERS
    if uses_projection and projection is None:
        raise ValueError(
            f"feature_layer {resolved_config.feature_layer!r} requires a projection, got none."
        )
    if not uses_projection and projection is not None:
        raise ValueError(
            f"a projection was given but feature_layer {resolved_config.feature_layer!r} "
            "does not use one."
        )
    pool = POOLING_STRATEGIES[resolved_config.pooling]
    feature_attribute = FEATURE_ATTRIBUTES[resolved_config.feature_layer]
    resolved_device = device if isinstance(device, torch.device) else torch.device(device)
    if projection is not None:
        projection = projection.to(resolved_device)

    uses_prototype_variant = resolved_config.prototype_strategy != "mean"

    was_training = model.training
    model.eval()
    accumulator: GlyphAccumulator | PrototypeAccumulator = (
        PrototypeAccumulator() if uses_prototype_variant else GlyphAccumulator()
    )
    feature_dim: int | None = None

    try:
        with torch.no_grad():
            for image, transcript in lines:
                reference = normalize(transcript, NFC_V1)
                batch = image.unsqueeze(0).to(resolved_device)
                output = model(batch)
                length = int(output.input_lengths[0])

                features = getattr(output, feature_attribute)[0, :length]
                if projection is not None:
                    features = projection(features)
                if feature_dim is None:
                    feature_dim = int(features.shape[-1])
                logits = output.logits[0, :length]
                log_probs = torch.log_softmax(logits, dim=-1)

                alignment = forced_align(log_probs, reference, charset)
                for span in alignment.spans:
                    class_index = charset.index_of(span.token)
                    vector = pool(span, log_probs, features, class_index)
                    accumulator.observe(span.token, vector.detach().cpu(), span.score)
    finally:
        model.train(was_training)

    assert feature_dim is not None  # `lines` is non-empty, so the loop ran at least once.

    if isinstance(accumulator, PrototypeAccumulator):
        glyphs = {
            character: Glyph(
                character=character,
                prototype=prototypes[0],
                additional_prototypes=prototypes[1:],
                number_of_observations=count,
                mean_alignment_confidence=confidence,
                feature_layer=resolved_config.feature_layer,
            )
            for character, (prototypes, count, confidence) in accumulator.finalize(
                resolved_config.prototype_strategy, top_k=resolved_config.top_k
            ).items()
        }
    else:
        glyphs = {
            character: Glyph(
                character=character,
                prototype=prototype,
                number_of_observations=count,
                mean_alignment_confidence=confidence,
                feature_layer=resolved_config.feature_layer,
            )
            for character, (prototype, count, confidence) in accumulator.finalize().items()
        }

    global_style = (
        compile_global_style(model, [image for image, _ in lines], device=device)
        if with_global_style
        else None
    )

    return WriterProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        model_fingerprint=model_fingerprint,
        feature_layer=resolved_config.feature_layer,
        feature_dim=feature_dim,
        glyphs=glyphs,
        global_style=global_style,
        projection_fingerprint=projection_fingerprint,
    )
