"""Offline metric training for `GlyphProjection`.

Two triplet-margin losses, not one, over the relationships the spec names:

    character loss  -- pulls together occurrences of the SAME character (any writer),
                        pushes apart DIFFERENT characters (confusion pairs weighted harder)
    writer loss      -- restricted to occurrences of the SAME character: pulls together the
                        SAME writer's occurrences, pushes apart DIFFERENT writers'

Cosine distance throughout, since both the projection's output and every embedding this project
compares are L2-normalized (`probes/geometry.py`'s convention, reused rather than reimplemented).

Training reads only pre-extracted `CharacterOccurrence`s (`probes/occurrences.py`) -- the base model
is never called again once occurrences are cached, so every step here is cheap tensor arithmetic on
frozen features, not a forward pass through `gm-base-v0`.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from glyphmemory.memory.projection import GlyphProjection
from glyphmemory.probes.occurrences import CharacterOccurrence

ConfusionPairs = dict[str, frozenset[str]]


def confusion_pairs_from_top_confusions(
    top_confusions: Sequence[dict[str, Any]],
) -> ConfusionPairs:
    """Build a symmetric character -> {confusable characters} map from an already-recorded
    `ErrorTaxonomy.top_confusions`-style list (-> `gm_base_h64_gru192x2_iam_v001` -- evaluation).

    Symmetric because the *pair* is what matters for hard-negative sampling -- if `r` is commonly
    misread as `v`, pushing `v` away from `r`'s occurrences is exactly as useful as pushing `r` away
    from `v`'s.
    """
    pairs: dict[str, set[str]] = defaultdict(set)
    for entry in top_confusions:
        reference, hypothesis = entry["reference"], entry["hypothesis"]
        if reference == hypothesis:
            continue
        pairs[reference].add(hypothesis)
        pairs[hypothesis].add(reference)
    return {char: frozenset(others) for char, others in pairs.items()}


@dataclass(frozen=True, slots=True)
class OccurrenceIndex:
    """A stacked feature tensor plus the indices needed to sample triplets from it quickly."""

    features: Tensor  # [N, D], raw (unprojected) sequence features
    characters: tuple[str, ...]  # length N
    writers: tuple[str, ...]  # length N
    by_character: dict[str, tuple[int, ...]]
    by_writer_character: dict[tuple[str, str], tuple[int, ...]]

    @classmethod
    def build(cls, occurrences: Sequence[CharacterOccurrence]) -> OccurrenceIndex:
        if not occurrences:
            raise ValueError("OccurrenceIndex.build requires at least one occurrence, got zero.")
        features = torch.stack([o.sequence_feature for o in occurrences])
        characters = tuple(o.character for o in occurrences)
        writers = tuple(o.writer_id for o in occurrences)

        by_character: dict[str, list[int]] = defaultdict(list)
        by_writer_character: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, (writer, character) in enumerate(zip(writers, characters, strict=True)):
            by_character[character].append(i)
            by_writer_character[(writer, character)].append(i)

        return cls(
            features=features,
            characters=characters,
            writers=writers,
            by_character={c: tuple(idx) for c, idx in by_character.items()},
            by_writer_character={k: tuple(idx) for k, idx in by_writer_character.items()},
        )

    def __len__(self) -> int:
        return self.features.shape[0]


@dataclass(frozen=True, slots=True)
class TripletBatch:
    """Anchor/positive/negative index triples for the character and writer losses, sampled
    independently -- an anchor missing a valid triplet for one loss can still contribute to the
    other.
    """

    char_anchor: tuple[int, ...]
    char_positive: tuple[int, ...]
    char_negative: tuple[int, ...]
    writer_anchor: tuple[int, ...]
    writer_positive: tuple[int, ...]
    writer_negative: tuple[int, ...]
    char_skipped: int
    writer_skipped: int


def sample_triplet_batch(
    index: OccurrenceIndex,
    batch_size: int,
    seed: int,
    *,
    confusion_pairs: ConfusionPairs | None = None,
) -> TripletBatch:
    """Sample up to `batch_size` anchors and, for each, a same/different-character pair for the
    character loss and a same/different-writer pair (restricted to the anchor's own character) for
    the writer loss. An anchor with no eligible partner for a given loss is skipped for that loss
    only, and counted -- never silently dropped from the batch entirely.
    """
    rng = random.Random(seed)
    n = len(index)
    anchors = [rng.randrange(n) for _ in range(batch_size)]
    confusion_pairs = confusion_pairs or {}

    char_a, char_p, char_n = [], [], []
    writer_a, writer_p, writer_n = [], [], []
    char_skipped = writer_skipped = 0

    all_characters = list(index.by_character)

    for anchor in anchors:
        character = index.characters[anchor]
        writer = index.writers[anchor]

        same_char = [i for i in index.by_character[character] if i != anchor]
        if same_char:
            char_a.append(anchor)
            char_p.append(rng.choice(same_char))

            confusable = sorted(confusion_pairs.get(character, frozenset()) & set(all_characters))
            if confusable:
                negative_character = rng.choice(confusable)
            else:
                other_characters = [c for c in all_characters if c != character]
                negative_character = rng.choice(other_characters) if other_characters else None
            if negative_character is not None:
                char_n.append(rng.choice(index.by_character[negative_character]))
            else:
                char_a.pop()
                char_p.pop()
                char_skipped += 1
        else:
            char_skipped += 1

        same_writer_char = [
            i for i in index.by_writer_character.get((writer, character), ()) if i != anchor
        ]
        other_writer_pool = [
            i for i in index.by_character[character] if index.writers[i] != writer
        ]
        if same_writer_char and other_writer_pool:
            writer_a.append(anchor)
            writer_p.append(rng.choice(same_writer_char))
            writer_n.append(rng.choice(other_writer_pool))
        else:
            writer_skipped += 1

    return TripletBatch(
        char_anchor=tuple(char_a),
        char_positive=tuple(char_p),
        char_negative=tuple(char_n),
        writer_anchor=tuple(writer_a),
        writer_positive=tuple(writer_p),
        writer_negative=tuple(writer_n),
        char_skipped=char_skipped,
        writer_skipped=writer_skipped,
    )


def cosine_triplet_loss(
    anchor: Tensor, positive: Tensor, negative: Tensor, margin: float
) -> Tensor:
    """`mean(relu(d(a,p) - d(a,n) + margin))`, cosine distance (`1 - cosine_similarity`).

    Assumes `anchor`/`positive`/`negative` are already L2-normalized (`GlyphProjection.forward`
    guarantees this), so cosine similarity is a plain dot product -- no renormalization here.
    """
    d_pos = 1.0 - (anchor * positive).sum(dim=-1)
    d_neg = 1.0 - (anchor * negative).sum(dim=-1)
    return F.relu(d_pos - d_neg + margin).mean()


@dataclass(frozen=True, slots=True)
class TrainingLog:
    """One training run's record: loss history and how often each loss term had to skip an anchor
    for lack of an eligible partner.
    """

    char_losses: tuple[float, ...]
    writer_losses: tuple[float, ...]
    total_char_skipped: int
    total_writer_skipped: int
    steps: int

    @property
    def final_char_loss(self) -> float | None:
        return self.char_losses[-1] if self.char_losses else None

    @property
    def final_writer_loss(self) -> float | None:
        return self.writer_losses[-1] if self.writer_losses else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "char_losses": list(self.char_losses),
            "writer_losses": list(self.writer_losses),
            "total_char_skipped": self.total_char_skipped,
            "total_writer_skipped": self.total_writer_skipped,
            "steps": self.steps,
            "final_char_loss": self.final_char_loss,
            "final_writer_loss": self.final_writer_loss,
        }


def train_projection(
    occurrences: Sequence[CharacterOccurrence],
    *,
    steps: int,
    batch_size: int = 256,
    lr: float = 1e-3,
    char_margin: float = 0.2,
    writer_margin: float = 0.2,
    char_loss_weight: float = 1.0,
    writer_loss_weight: float = 1.0,
    confusion_pairs: ConfusionPairs | None = None,
    seed: int = 1337,
    projection: GlyphProjection | None = None,
) -> tuple[GlyphProjection, TrainingLog]:
    """Train (or continue training) a `GlyphProjection` on cached occurrences.

    `occurrences` are never re-extracted from the base model here -- this function only ever reads
    their `.sequence_feature`/`.writer_id`/`.character`, all already-detached tensors
    (`extract_occurrences` runs under `torch.no_grad()`), so nothing in this loop can reach back
    into `gm-base-v0`'s own parameters even by accident.
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")

    index = OccurrenceIndex.build(occurrences)
    if projection is None:
        # A fresh nn.Module's initial weights come from torch's global RNG, not from `seed` --
        # scoped with fork_rng so the same seed reproduces the same initialization without leaking a
        # manual_seed() call into whatever the caller's program does afterward.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = GlyphProjection()
    else:
        model = projection
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    char_losses: list[float] = []
    writer_losses: list[float] = []
    total_char_skipped = 0
    total_writer_skipped = 0

    for step in range(steps):
        batch = sample_triplet_batch(
            index, batch_size, seed=seed + step, confusion_pairs=confusion_pairs
        )
        total_char_skipped += batch.char_skipped
        total_writer_skipped += batch.writer_skipped

        optimizer.zero_grad()
        loss_terms = []

        if batch.char_anchor:
            projected = model(
                index.features[list(batch.char_anchor + batch.char_positive + batch.char_negative)]
            )
            n = len(batch.char_anchor)
            char_loss = cosine_triplet_loss(
                projected[:n], projected[n : 2 * n], projected[2 * n :], char_margin
            )
            loss_terms.append(char_loss_weight * char_loss)
            char_losses.append(float(char_loss.detach()))

        if batch.writer_anchor:
            projected = model(
                index.features[
                    list(batch.writer_anchor + batch.writer_positive + batch.writer_negative)
                ]
            )
            n = len(batch.writer_anchor)
            writer_loss = cosine_triplet_loss(
                projected[:n], projected[n : 2 * n], projected[2 * n :], writer_margin
            )
            loss_terms.append(writer_loss_weight * writer_loss)
            writer_losses.append(float(writer_loss.detach()))

        if loss_terms:
            total = torch.stack(loss_terms).sum()
            total.backward()
            optimizer.step()

    model.eval()
    log = TrainingLog(
        char_losses=tuple(char_losses),
        writer_losses=tuple(writer_losses),
        total_char_skipped=total_char_skipped,
        total_writer_skipped=total_writer_skipped,
        steps=steps,
    )
    return model, log
