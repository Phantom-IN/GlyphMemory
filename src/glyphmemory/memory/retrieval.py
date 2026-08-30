"""Cosine retrieval against a `WriterProfile`.

For every frame in a feature sequence, cosine similarity against every stored prototype becomes one
score per character class, scattered into a vector aligned to the full charset — so it can be added
directly to the base model's logits without either side having to know which classes the other has
evidence for.

**A character absent from the profile gets a neutral 0.0 score, never invented evidence** — the same
principle states for retrieval and already applies to prototype compilation
(`memory/prototypes.py`). The blank class never has a prototype (the compiler only ever aligns
non-blank characters — `alignment/forced_align.py`), so it is always 0.0 here structurally, before
`memory/fusion.py`'s own blank protection is even reached.

Unlike `probes/geometry.py`'s `cosine_distance_matrix` — built for offline diagnostic batches at
O(N²) pairwise scale — this is inference-time retrieval against ``K`` stored prototypes: one ``[T,
D] @ [D, K]`` matmul, not a pairwise-distance problem. Only the normalization primitive is shared.
"""

from __future__ import annotations

import torch
from torch import Tensor

from glyphmemory.ctc.tokenizer import Charset
from glyphmemory.memory.profile import WriterProfile
from glyphmemory.probes.geometry import l2_normalize


def memory_scores(features: Tensor, profile: WriterProfile, charset: Charset) -> Tensor:
    """Cosine similarity of every frame in ``features`` against every prototype in ``profile``.

    Args:
        features: ``[T, D]``, one writer-memory feature vector per frame. ``D`` must equal
            ``profile.feature_dim`` — comparing frames from one feature layer against prototypes
            compiled from another would silently score nonsense.
        profile: The compiled writer memory. An empty profile (no glyphs — never enrolled, or
            enrolled with no alignable characters) produces an all-neutral result.
        charset: Defines the output's column order and size. A profile character not present in
            ``charset`` raises via `Charset.index_of` — a real inconsistency, not a case to paper
            over.

    Returns:
        ``[T, C]``, ``C == charset.size``. Every column the profile has no prototype for
        (including blank, always) is exactly ``0.0``. A character with more than one stored
        prototypes at each frame, not an average -- "compare against the best of K", per

    Raises:
        ValueError: ``features`` is not ``[T, D]``, or its ``D`` disagrees with
            ``profile.feature_dim``.
    """
    if features.dim() != 2:
        raise ValueError(f"features must be [T, D], got shape {tuple(features.shape)}")
    if features.shape[-1] != profile.feature_dim:
        raise ValueError(
            f"features has dim {features.shape[-1]} but profile was compiled at "
            f"{profile.feature_dim} ({profile.feature_layer} features)."
        )

    time_steps = features.shape[0]
    scores = torch.zeros((time_steps, charset.size), dtype=features.dtype, device=features.device)
    if not profile.glyphs:
        return scores

    characters = sorted(profile.glyphs)  # deterministic column order, not load-bearing
    # Every character contributes `1 + len(additional_prototypes)` columns, appended in `characters`
    # order -- so each character's block is contiguous and can be sliced by (start, end) below,
    # rather than needing a per-column character lookup.
    all_prototypes: list[Tensor] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for character in characters:
        glyph = profile.glyphs[character]
        variants = (glyph.prototype, *glyph.additional_prototypes)
        all_prototypes.extend(variants)
        spans.append((cursor, cursor + len(variants)))
        cursor += len(variants)

    # WriterProfile prototypes are always stored on CPU (compile_profile detaches them there);
    # features may be on any resolved device, so move the prototypes to match rather than the other
    # way round -- there is exactly one of these per call, never a batch of them.
    prototypes = torch.stack(all_prototypes).to(device=features.device, dtype=features.dtype)
    similarities = l2_normalize(features) @ l2_normalize(prototypes).T  # [T, sum(1+len(extra))]

    for character, (start, end) in zip(characters, spans, strict=True):
        column = charset.index_of(character)
        # `.max(dim=1)` on a width-1 slice is that single column unchanged -- identical to the
        # pre-Phase-26 single-prototype behavior whenever no glyph has additional prototypes.
        scores[:, column] = similarities[:, start:end].max(dim=1).values
    return scores
