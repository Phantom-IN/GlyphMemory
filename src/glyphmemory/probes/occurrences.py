"""Per-character, per-frame feature occurrences — the raw material every probe measures."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from glyphmemory.alignment import forced_align
from glyphmemory.ctc.normalization import NFC_V1, normalize
from glyphmemory.ctc.tokenizer import Charset
from glyphmemory.model.htr import GMBase


@dataclass(frozen=True, slots=True)
class CharacterOccurrence:
    """One frame, one character, one writer — the unit every probe in this package consumes."""

    writer_id: str
    sample_id: str
    character: str
    frame_index: int
    visual_feature: Tensor
    sequence_feature: Tensor
    base_head_prediction: str
    alignment_score: float


def extract_occurrences(
    model: GMBase,
    loader: DataLoader,
    charset: Charset,
    *,
    device: torch.device | str = "cpu",
) -> list[CharacterOccurrence]:
    """Run ``model`` over every line in ``loader``, align each against its ground truth, and return
    one :class:`CharacterOccurrence` per frame of every recovered span.
    """
    resolved_device = device if isinstance(device, torch.device) else torch.device(device)
    model.eval()
    occurrences: list[CharacterOccurrence] = []

    with torch.no_grad():
        for batch in loader:
            if batch.is_empty:
                continue
            batch = batch.to(resolved_device)
            output = model(batch.images, batch.input_lengths)

            writer_id = batch.writer_ids[0]
            sample_id = batch.sample_ids[0]
            reference = normalize(batch.texts[0], NFC_V1)
            length = int(batch.input_lengths[0])

            visual = output.visual_features[0, :length]
            sequence = output.sequence_features[0, :length]
            logits = output.logits[0, :length]
            log_probs = torch.log_softmax(logits, dim=-1)

            try:
                alignment = forced_align(log_probs, reference, charset)
            except Exception:
                # An unalignable line (e.g. infeasible width) simply contributes no occurrences.
                continue

            predicted_indices = logits.argmax(dim=-1)

            for span in alignment.spans:
                for t in range(span.start_t, span.end_t):
                    predicted_index = int(predicted_indices[t])
                    predicted_char = (
                        "<blank>"
                        if predicted_index == charset.blank
                        else charset.char_at(predicted_index)
                    )
                    occurrences.append(
                        CharacterOccurrence(
                            writer_id=writer_id,
                            sample_id=sample_id,
                            character=span.token,
                            frame_index=t,
                            visual_feature=visual[t].clone(),
                            sequence_feature=sequence[t].clone(),
                            base_head_prediction=predicted_char,
                            alignment_score=span.score,
                        )
                    )

    return occurrences


def base_head_frame_accuracy(occurrences: Iterable[CharacterOccurrence]) -> tuple[float, int]:
    """The trained head's own frame-level accuracy on exactly the frames the probes score — the
    fixed reference decision rule compares NCM against. Only meaningful for sequence-feature frames,
    since the head consumes that layer; computed once, not per feature layer.
    """
    occurrences = list(occurrences)
    if not occurrences:
        return 0.0, 0
    correct = sum(o.base_head_prediction == o.character for o in occurrences)
    return correct / len(occurrences), len(occurrences)
