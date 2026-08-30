"""GlyphVerifier — a writer-conditioned glyph embedding over pixel regions.

**Why an embedding network and not a pairwise relation comparator.** Two measured constraints, not a
stylistic preference:

- Enrollment must be forward passes only. Storing raw crops for a relation comparator costs 70
  characters x 3 exemplars x 2,560 B ~= 537 KB against a current ~59 KB mean profile. One 112-d fp16
  embedding per character costs 15.3 KB.
- Inference cost. A relation network re-runs a comparator per candidate; an embedding network
  encodes the query once and takes cosines against stored vectors.

**Why the loss is a normalised softmax and not a similarity difference.** measured the intuitive
formulation — similarity to memory's candidate minus similarity to the base's own character — and
found it *worse than either arm alone* (AUROC 0.510 against 0.653 for memory similarity; the
base-similarity arm scores 0.402, below chance). Both similarities are dominated by a common "how
clean is this span" factor, so subtracting cancels the informative common mode and keeps the noise.
A softmax over candidates cancels that common mode in the gradient instead, which is why
:func:`char_loss` normalises rather than subtracts.

The encoder reuses GM-Base's own :class:`~glyphmemory.model.blocks.InvertedResidual2D`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from glyphmemory.model.blocks import InvertedResidual2D
from glyphmemory.model.model_info import parameter_count

#: Stage widths. Chosen as the largest configuration inside the 150,000-350,000 envelope the design
#: brief set; measured at 284,752 parameters (``m10r3-design-audit-001`` section F).
DEFAULT_WIDTH: tuple[int, int, int, int] = (24, 40, 80, 112)

#: Embedding dimension. Equal to the final stage width, so the projection is square.
DEFAULT_EMBED = 112

#: Inverted-residual expansion. 4 rather than GM-Base's 2 — this network is four stages shallower
#: and the width is where its capacity has to come from.
DEFAULT_EXPANSION = 4

#: Softmax temperature for both loss terms. Fixed in the pre-registration; not tuned.
TEMPERATURE = 0.07


class GlyphVerifier(nn.Module):
    """``[B, 2, 64, 40] -> [B, embed]``, L2-normalized.

    Args:
        in_channels: 2 — pixels and the cell mask from :mod:`~glyphmemory.memory.glyph_regions`.
        width: ``(stem, stage1, stage2, stage3)`` channel counts.
        embed: Embedding dimension.
        expansion: Inverted-residual expansion ratio.

    Shape:
        ``[B, 2, H, W] -> [B, embed]`` with unit-norm rows.

    The output is L2-normalized inside :meth:`forward`, so every consumer — the losses, the enrolled
    prototypes, the inference-time score — sees the same geometry. A caller cannot accidentally
    compare a normalized vector against an unnormalized one.
    """

    def __init__(
        self,
        *,
        in_channels: int = 2,
        width: Sequence[int] = DEFAULT_WIDTH,
        embed: int = DEFAULT_EMBED,
        expansion: int = DEFAULT_EXPANSION,
    ) -> None:
        super().__init__()
        width = tuple(int(w) for w in width)
        if len(width) < 2:
            raise ValueError(f"width needs a stem and at least one stage, got {width}")

        self.in_channels = in_channels
        self.width = width
        self.embed = embed
        self.expansion = expansion

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width[0]),
            nn.ReLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        channels = width[0]
        for out_channels in width[1:]:
            blocks.append(
                InvertedResidual2D(channels, out_channels, stride=(2, 2), expansion=expansion)
            )
            blocks.append(
                InvertedResidual2D(out_channels, out_channels, stride=(1, 1), expansion=expansion)
            )
            channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Linear(channels, embed)

    def forward(self, crops: Tensor) -> Tensor:
        """Embed a batch of glyph crops. Always returns unit-norm rows."""
        if crops.ndim != 4 or crops.shape[1] != self.in_channels:
            raise ValueError(
                f"expected [B, {self.in_channels}, H, W], got shape {tuple(crops.shape)}"
            )
        pooled = self.blocks(self.stem(crops)).mean(dim=(2, 3))
        return F.normalize(self.proj(pooled), dim=-1)

    @property
    def parameter_count(self) -> int:
        """Exact parameter count — every model here exposes its own."""
        return parameter_count(self)


# --------------------------------------------------------------------------- enrollment


def compile_character_embeddings(
    verifier: GlyphVerifier,
    crops: Tensor,
    characters: Sequence[str],
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
) -> dict[str, Tensor]:
    """Enroll a writer: crops plus their characters -> one unit-norm embedding per character.

    Forward passes, a group-by and a mean. **No optimizer, no gradients** — this is the whole of
    what permits at deployment, and it is all that happens here.

    Args:
        verifier: The trained encoder. Put in ``eval()`` mode by this function.
        crops: ``[N, 2, H, W]`` support crops.
        characters: ``N`` character labels, aligned with ``crops``.
        batch_size: Forward-pass chunk size.

    Returns:
        ``{character: [embed]}``, each row unit-norm. Characters observed zero times are absent —
        no zero vector is invented for them, because a zero vector has cosine 0 against
        everything and would silently read as "moderately similar".
    """
    if len(characters) != len(crops):
        raise ValueError(f"{len(crops)} crops but {len(characters)} characters")
    if len(crops) == 0:
        return {}

    resolved = device if isinstance(device, torch.device) else torch.device(device)
    verifier.eval().to(resolved)

    embeddings: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            embeddings.append(verifier(crops[start : start + batch_size].to(resolved)).cpu())
    stacked = torch.cat(embeddings)

    grouped: dict[str, list[Tensor]] = {}
    for character, embedding in zip(characters, stacked, strict=True):
        grouped.setdefault(character, []).append(embedding)
    return {
        character: F.normalize(torch.stack(rows).mean(0), dim=-1)
        for character, rows in grouped.items()
    }


def score_candidates(
    query: Tensor,
    profile: Mapping[str, Tensor],
    candidates: Sequence[str],
) -> dict[str, float]:
    """Cosine of one query embedding against the writer's stored characters.

    Candidates absent from the profile are omitted rather than scored against a sentinel.
    """
    return {c: float(query @ profile[c]) for c in candidates if c in profile}


# --------------------------------------------------------------------------- losses


def char_loss(
    queries: Tensor,
    prototypes: Tensor,
    targets: Tensor,
    candidate_mask: Tensor,
    *,
    temperature: float = TEMPERATURE,
) -> Tensor:
    """Within-writer character loss — the decision actually made at inference.

    Args:
        queries: ``[N, D]`` unit-norm query embeddings.
        prototypes: ``[N, C, D]`` unit-norm per-writer character prototypes, one row block per query
            (writers differ across the batch, so this is not shared).
        targets: ``[N]`` index into ``C`` of the true character.
        candidate_mask: ``[N, C]`` bool — the true character plus its mined hard negatives that are
            present in this writer's profile. Positions outside the set are masked to -inf.

    Returns:
        Scalar mean cross-entropy. Rows whose target is masked out contribute nothing and are
        excluded from the mean rather than silently scored.
    """
    logits = torch.einsum("nd,ncd->nc", queries, prototypes) / temperature
    logits = logits.masked_fill(~candidate_mask, float("-inf"))
    valid = candidate_mask.gather(1, targets.unsqueeze(1)).squeeze(1)
    if not bool(valid.any()):
        return queries.new_zeros(())
    return F.cross_entropy(logits[valid], targets[valid])


def writer_loss(
    queries: Tensor,
    writer_prototypes: Tensor,
    targets: Tensor,
    writer_mask: Tensor,
    *,
    temperature: float = TEMPERATURE,
) -> Tensor:
    """Across-writer loss at fixed character — the mandatory Type-A term.

    Args:
        queries: ``[N, D]`` unit-norm query embeddings.
        writer_prototypes: ``[N, W, D]`` — the same character's prototype from each of the ``W``
            writers in the batch.
        targets: ``[N]`` index of the query's own writer.
        writer_mask: ``[N, W]`` bool — writers whose profile actually contains this character.
    """
    logits = torch.einsum("nd,nwd->nw", queries, writer_prototypes) / temperature
    logits = logits.masked_fill(~writer_mask, float("-inf"))
    valid = writer_mask.gather(1, targets.unsqueeze(1)).squeeze(1) & (writer_mask.sum(1) > 1)
    if not bool(valid.any()):
        return queries.new_zeros(())
    return F.cross_entropy(logits[valid], targets[valid])
