"""GM-Base: the assembled recognizer, and the output contract everything downstream reads.

``[B, 1, 64, W] -> HTROutput`` with logits, both feature layers, and the lengths that describe them.

Two contracts are established here, and both are load-bearing far beyond this file.

**Intermediate representations are public research infrastructure, not debug hooks**. If they were
reachable only through a forward hook, the memory work would end up rewriting the model rather than
using it, and the "same frozen features" premise of the V0 experiment would quietly stop holding.

**Lengths travel with the logits.** collator exists because ``input_lengths`` derived from a padded
width trains a model that decodes truncated text without ever raising. That guarantee lives in the
collator and stops at its boundary; carrying the lengths inside :class:`HTROutput` is what keeps it
in scope one layer up, where a caller could otherwise reach for ``logits.shape[1]`` and be wrong for
every sample except the widest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from glyphmemory.config.schema import ModelConfig
from glyphmemory.model.encoder import VisualEncoder
from glyphmemory.model.head import CharacterHead
from glyphmemory.model.model_info import ParameterReport, parameter_count, parameter_report
from glyphmemory.model.sequence import SequenceEncoder


@dataclass(frozen=True, slots=True)
class HTROutput:
    """One forward pass, with every tensor the research needs.

    Attributes:
        logits: ``[B, T, vocab_size]``. **Unnormalized** — no softmax has been applied.
        sequence_features: ``[B, T, 384]``, the BiGRU's contextual output, taken *before* the head's
            LayerNorm so it is the representation the head consumes rather than a re-scaled view of
            it.
        visual_features: ``[B, T, 192]``, the visual encoder's output — **after** the height reducer
            and its BatchNorm, which is to say exactly ``VisualEncoder(images)``.
        input_lengths: ``[B]`` valid frames per sample, derived from true unpadded widths.
    """

    logits: Tensor
    sequence_features: Tensor
    visual_features: Tensor
    input_lengths: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.logits.shape[0])

    @property
    def time_steps(self) -> int:
        """``T`` — the padded frame count, **not** any individual sample's valid length."""
        return int(self.logits.shape[1])

    @property
    def vocab_size(self) -> int:
        return int(self.logits.shape[-1])

    def validate(self) -> None:
        """Raise if the tensors do not describe one another.

        Cheap enough to call in tests and at the boundaries of new code. Every condition here has a
        silent failure behind it: mismatched batch or time dimensions mean features and logits came
        from different passes, and ``input_lengths > T`` is the CTC bug this milestone is built to
        prevent.
        """
        if self.logits.dim() != 3:
            raise ValueError(f"logits must be [B, T, C], got {tuple(self.logits.shape)}")
        for name, tensor in (
            ("sequence_features", self.sequence_features),
            ("visual_features", self.visual_features),
        ):
            if tensor.shape[:2] != self.logits.shape[:2]:
                raise ValueError(
                    f"{name} has shape {tuple(tensor.shape)}, which disagrees with logits "
                    f"{tuple(self.logits.shape)} on [B, T]"
                )
        if self.input_lengths.shape != (self.batch_size,):
            raise ValueError(
                f"input_lengths must be [B]={self.batch_size}, got "
                f"{tuple(self.input_lengths.shape)}"
            )
        if self.batch_size and int(self.input_lengths.max()) > self.time_steps:
            raise ValueError(
                f"input_lengths max {int(self.input_lengths.max())} exceeds T="
                f"{self.time_steps}. CTC cannot align more frames than exist."
            )

    def to(self, device: torch.device | str) -> HTROutput:
        return HTROutput(
            logits=self.logits.to(device),
            sequence_features=self.sequence_features.to(device),
            visual_features=self.visual_features.to(device),
            input_lengths=self.input_lengths.to(device),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "time_steps": self.time_steps,
            "vocab_size": self.vocab_size,
            "sequence_feature_dim": int(self.sequence_features.shape[-1]),
            "visual_feature_dim": int(self.visual_features.shape[-1]),
            "input_lengths_min": int(self.input_lengths.min()) if self.batch_size else 0,
            "input_lengths_max": int(self.input_lengths.max()) if self.batch_size else 0,
            "logits_are_normalized": False,
        }


class GMBase(nn.Module):
    """The GM-Base recognizer: visual encoder, BiGRU, character head.

    Args:
        vocab_size: Character count including the CTC blank at index 0.
        config: Architecture configuration. Defaults to :class:`ModelConfig`'s frozen values.

    Shape:
        ``[B, 1, 64, W] -> HTROutput`` with ``T == temporal_length(W)``.

    The base recognizer's job is to be small, reliable and fast enough to make writer-memory
    research possible; novelty belongs in the memory, not here.
    """

    def __init__(self, *, vocab_size: int, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.vocab_size = vocab_size

        self.encoder = VisualEncoder.from_config(self.config)
        self.sequence = SequenceEncoder.from_config(self.config)
        self.head = CharacterHead.from_config(self.config, vocab_size)

        # Defensive, and currently unreachable: both dimensions are derived from
        # ``config.visual_dim``, so they cannot disagree while this constructor owns the assembly.
        # It stays because the failure it describes is a silent shape error at the first forward
        # pass, and it would become reachable the moment anything constructs the submodules
        # independently — a pre-built encoder, a swapped sequence model in an ablation, a checkpoint
        # loaded piecewise.
        if self.encoder.feature_dim != self.sequence.input_size:  # pragma: no cover
            raise ValueError(
                f"Encoder emits {self.encoder.feature_dim} features but the sequence encoder "
                f"expects {self.sequence.input_size}."
            )

    @classmethod
    def from_config(cls, config: ModelConfig, vocab_size: int) -> GMBase:
        """Build from config, so architecture never comes from literals in a trainer."""
        return cls(vocab_size=vocab_size, config=config)

    def forward(self, images: Tensor, input_lengths: Tensor | None = None) -> HTROutput:
        """Run the full stack.

        Args:
            images: ``[B, 1, H, W]``, preprocessed and right-padded.
            input_lengths: Valid frames per sample, from :class:`~glyphmemory.data.collate.Batch`.
                **Pass it whenever the batch holds more than one width** — it is what keeps the
                BiGRU's backward direction out of the padding. When omitted, every sample is treated
                as ``T`` frames long, which is only true for a single-sample or equal-width batch.
        """
        visual = self.encoder(images)  # [B, T, visual_dim]
        time_steps = visual.shape[1]

        if input_lengths is None:
            lengths = torch.full(
                (visual.shape[0],), time_steps, dtype=torch.long, device=visual.device
            )
        else:
            lengths = input_lengths

        sequence = self.sequence(visual, lengths)  # [B, T, 2*hidden]
        logits = self.head(sequence)  # [B, T, vocab]

        return HTROutput(
            logits=logits,
            sequence_features=sequence,
            visual_features=visual,
            input_lengths=lengths,
        )

    # ------------------------------------------------------------------ introspection

    def output_length(self, width: int) -> int:
        """``T`` for an input width, delegated to the encoder and thence to preprocessing."""
        return self.encoder.output_length(width)

    def parameter_report(self) -> ParameterReport:
        """Exact counts, printed at construction."""
        return parameter_report(self)

    def describe(self) -> dict[str, Any]:
        """Architecture provenance for the run record."""
        return {
            "name": self.config.name,
            "vocab_size": self.vocab_size,
            "parameters": parameter_count(self),
            "max_parameters": self.config.max_parameters,
            "encoder": self.encoder.describe(),
            "sequence": self.sequence.describe(),
            "head": self.head.describe(),
        }
