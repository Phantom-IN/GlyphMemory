"""Greedy CTC decoding.

```text
argmax per frame  ->  collapse consecutive duplicates  ->  drop blanks  ->  map to characters
```

Three properties of this file matter more than its length.

**The order is collapse-then-strip, and reversing it is silently wrong.** The blank is what
separates a genuine double letter from one letter emitted across two frames. ``l l`` collapses to
``l``; ``l <blank> l`` collapses to ``l <blank> l`` and then strips to ``ll``. Remove blanks first
and every doubled letter in the language disappears — "hello" becomes "helo" and nothing raises.

**Only the first ``input_lengths[i]`` frames of a sample are decoded.** The rest of the row is
padding belonging to a wider batch neighbour. Padding usually argmaxes to blank, so decoding it is
*usually* harmless — which makes the failure intermittent, width-dependent and extremely hard to
attribute later.

**There is no language model here, and there will not be one.** No lexicon, no dictionary, no spell
correction, no beam search over a word list. This is the file where such a thing would be most
tempting and most damaging: GlyphMemory reports what is visually written, not what is linguistically
likely.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from glyphmemory.ctc.tokenizer import BLANK_INDEX, Tokenizer


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    """How a reported number was decoded.

    Greedy no-LM results are not comparable with another paper's LM-assisted results, and the only
    defence against that comparison being made by accident is for the configuration to travel with
    the metric.
    """

    kind: str = "greedy"
    beam_width: int | None = None
    blank_index: int = BLANK_INDEX

    def __post_init__(self) -> None:
        if self.kind != "greedy":
            raise ValueError(
                f"Only greedy decoding exists (got {self.kind!r}). Prefix beam search is an "
                "optional benchmark decoder for later; greedy "
                "remains the deployment baseline."
            )
        if self.beam_width is not None:
            raise ValueError("beam_width is meaningless for greedy decoding.")

    @property
    def label(self) -> str:
        """Short string printed beside every metric, e.g. ``greedy, no LM``."""
        return f"{self.kind}, no LM"

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "beam_width": self.beam_width,
            "blank_index": self.blank_index,
            "language_model": None,
            "lexicon": None,
        }


DEFAULT_DECODER = DecoderConfig()


def collapse_repeats(indices: Sequence[int]) -> list[int]:
    """Collapse runs of the same class into one occurrence. Blanks are left in place."""
    collapsed: list[int] = []
    for index in indices:
        if not collapsed or index != collapsed[-1]:
            collapsed.append(index)
    return collapsed


def strip_blanks(indices: Sequence[int], *, blank: int = BLANK_INDEX) -> list[int]:
    """Remove the blank class. Call **after** :func:`collapse_repeats`, never before."""
    return [index for index in indices if index != blank]


def ctc_collapse(indices: Sequence[int], *, blank: int = BLANK_INDEX) -> list[int]:
    """The CTC collapse rule: collapse consecutive duplicates, then drop blanks."""
    return strip_blanks(collapse_repeats(indices), blank=blank)


def greedy_decode_ids(
    logits: Tensor, length: int | None = None, *, blank: int = BLANK_INDEX
) -> list[int]:
    """Decode one sequence of per-frame scores to class indices.

    Args:
        logits: ``[T, C]``. **Logits or log-probabilities both work** — ``argmax`` is invariant to a
            monotone per-frame transform, so ``log_softmax`` is neither required nor harmful. Do not
            pass ``[C, T]``.
        length: Valid frames. Frames at or beyond it are padding and are not read. ``None`` means
            the whole tensor is valid, which is true only for an unpadded sequence.
        blank: Blank class index.

    Returns:
        Class indices with repeats collapsed and blanks removed.
    """
    if logits.dim() != 2:
        raise ValueError(f"Expected [T, C], got shape {tuple(logits.shape)}")

    time_steps = logits.shape[0]
    if length is None:
        length = time_steps
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    if length > time_steps:
        raise ValueError(
            f"length {length} exceeds T={time_steps}; a sample cannot have more valid frames "
            "than the tensor holds."
        )
    if length == 0:
        return []

    best = logits[:length].argmax(dim=-1).tolist()
    return ctc_collapse(best, blank=blank)


def greedy_decode(
    logits: Tensor,
    tokenizer: Tokenizer,
    length: int | None = None,
    *,
    blank: int = BLANK_INDEX,
) -> str:
    """Decode one sequence ``[T, C]`` to text."""
    return tokenizer.decode(greedy_decode_ids(logits, length, blank=blank))


def greedy_decode_batch(
    logits: Tensor,
    tokenizer: Tokenizer,
    input_lengths: Tensor | Sequence[int] | None = None,
    *,
    blank: int = BLANK_INDEX,
) -> list[str]:
    """Decode a batch ``[B, T, C]`` to one string per sample.

    Args:
        input_lengths: Valid frames per sample. **Omitting this decodes padding** and is only
            correct when every sample really occupies all ``T`` frames.
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected [B, T, C], got shape {tuple(logits.shape)}")

    batch_size, time_steps, _ = logits.shape
    if input_lengths is None:
        lengths = [time_steps] * batch_size
    else:
        lengths = [int(value) for value in input_lengths]
        if len(lengths) != batch_size:
            raise ValueError(
                f"input_lengths has {len(lengths)} entries for a batch of {batch_size}"
            )

    return [
        greedy_decode(logits[index], tokenizer, lengths[index], blank=blank)
        for index in range(batch_size)
    ]


def decode_output(output: Any, tokenizer: Tokenizer, *, blank: int = BLANK_INDEX) -> list[str]:
    """Decode an :class:`~glyphmemory.model.htr.HTROutput`.

    Takes ``input_lengths`` from the output itself, so padding cannot be decoded by forgetting to
    pass them — which is the whole reason lengths travel inside ``HTROutput``.
    """
    return greedy_decode_batch(output.logits, tokenizer, output.input_lengths, blank=blank)


def one_hot_logits(frames: Sequence[int], vocab_size: int, *, confidence: float = 20.0) -> Tensor:
    """Build ``[T, C]`` logits that argmax to ``frames``. For tests and alignment fixtures."""
    logits = torch.full((len(frames), vocab_size), -confidence)
    for step, label in enumerate(frames):
        if not 0 <= label < vocab_size:
            raise ValueError(f"Frame {step} has label {label} outside vocabulary {vocab_size}")
        logits[step, label] = confidence
    return logits
