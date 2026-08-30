"""Bidirectional GRU sequence encoder.

GRU rather than LSTM: one fewer gate, fewer parameters, lower compute, still bidirectional and
context-aware.

**The packing is the point of this module.**

A batch is a rectangle; the lines in it are not. Sample *i* occupies ``input_lengths[i]`` frames and
the rest is padding. Run a bidirectional RNN over the raw rectangle and the backward direction
starts at frame ``T-1`` — deep inside another sample's padding — and integrates that padding into
its hidden state *before* it reaches any real ink. Sample *i*'s representation then depends on how
wide its batch neighbours happened to be.

That failure has three properties that make it worth this much care:

- it never raises;
- it is invisible at batch size 1, which is where a developer debugs.

:func:`torch.nn.utils.rnn.pack_padded_sequence` removes it: each sequence is run over its own length
only. :meth:`SequenceEncoder.forward` is verified against a per-sample reference in
``tests/test_sequence.py`` — the padded and unpadded runs of the same line must agree.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from glyphmemory.config.schema import ModelConfig

#: Directions in a bidirectional RNN. Named because ``2`` appears in the output-width arithmetic and
#: ``hidden * 2`` should not read as a magic number.
DIRECTIONS = 2


class SequenceEncoder(nn.Module):
    """Two-layer bidirectional GRU over encoder frames.

    Args:
        input_size: Feature dimension arriving from the visual encoder.
        hidden_size: Hidden units **per direction**.
        num_layers: Stacked GRU layers.
        dropout: Applied between layers. ``nn.GRU`` ignores it when ``num_layers == 1``, so it is
            zeroed explicitly there rather than left to emit a warning.

    Shape:
        ``[B, T, input_size] -> [B, T, hidden_size * 2]``
    """

    def __init__(
        self,
        *,
        input_size: int = 192,
        hidden_size: int = 192,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1:
            raise ValueError(
                f"input_size and hidden_size must be positive, got {input_size}, {hidden_size}"
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0.0
        self.output_size = hidden_size * DIRECTIONS

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout,
        )

    @classmethod
    def from_config(cls, config: ModelConfig) -> SequenceEncoder:
        return cls(
            input_size=config.visual_dim,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            dropout=config.gru_dropout,
        )

    def forward(self, features: Tensor, input_lengths: Tensor | None = None) -> Tensor:
        """``[B, T, input_size] -> [B, T, output_size]``.

        Args:
            features: Encoder frames, right-padded within the batch.
            input_lengths: Valid frame count per sample. **Omitting this is only correct when every
                sample in the batch really is ``T`` frames long** — otherwise the backward direction
                reads padding, for the reasons in the module docstring. It is optional rather than
                required so that single-sequence probes and unit tests stay ergonomic, not because
                it is safe to skip in a training loop.

        Returns:
            Contextual features, padded back to exactly ``T``.
        """
        if features.dim() != 3:
            raise ValueError(f"Expected [B, T, C], got shape {tuple(features.shape)}")

        total_length = features.shape[1]

        if input_lengths is None:
            output, _ = self.gru(features)
            return output

        if input_lengths.shape[0] != features.shape[0]:
            raise ValueError(
                f"input_lengths has {input_lengths.shape[0]} entries for a batch of "
                f"{features.shape[0]}"
            )
        if int(input_lengths.min()) < 1:
            raise ValueError("input_lengths must all be at least 1")
        if int(input_lengths.max()) > total_length:
            raise ValueError(
                f"input_lengths max {int(input_lengths.max())} exceeds T={total_length}; "
                "lengths must describe the tensor they accompany."
            )

        # Lengths must live on the CPU for packing regardless of the tensor's device.
        packed = pack_padded_sequence(
            features,
            input_lengths.detach().to("cpu", dtype=int).clamp(min=1),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        # total_length is explicit: pad_packed_sequence otherwise pads to the longest *sequence*,
        # which is <= T. Silently returning a shorter tensor would desynchronize the logits from the
        # input_lengths that describe them.
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)
        return output

    def describe(self) -> dict[str, Any]:
        return {
            "name": "sequence_encoder",
            "kind": "bigru",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": True,
            "output_size": self.output_size,
        }

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, dropout={self.dropout}, "
            f"output_size={self.output_size}"
        )
