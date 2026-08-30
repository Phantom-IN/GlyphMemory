"""Character head.

``[B, T, 384] -> LayerNorm -> Dropout -> Linear -> [B, T, vocab_size]``

**No softmax here.** ``log_softmax`` is applied in the loss path and, at inference, greedy decoding
takes an ``argmax`` which is invariant to it. Applying softmax in the module would mean applying it
twice during training — ``log_softmax`` over an already-normalized distribution flattens it, the
loss stops discriminating, and nothing raises.

Keeping it a plain ``Linear`` over ``sequence_features`` is what makes that framing honest.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from glyphmemory.config.schema import ModelConfig


class CharacterHead(nn.Module):
    """Projects contextual features to per-frame character logits.

    Args:
        input_size: Feature dimension from the sequence encoder.
        vocab_size: Character count **including the CTC blank at index 0**.
        dropout: Applied before the projection.

    Shape:
        ``[B, T, input_size] -> [B, T, vocab_size]``
    """

    def __init__(self, *, input_size: int = 384, vocab_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        if input_size < 1:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if vocab_size < 2:
            raise ValueError(
                f"vocab_size must be at least 2 (blank plus one character), got {vocab_size}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_size = input_size
        self.vocab_size = vocab_size
        self.dropout_probability = dropout

        self.norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(input_size, vocab_size)

    @classmethod
    def from_config(cls, config: ModelConfig, vocab_size: int) -> CharacterHead:
        return cls(
            input_size=config.gru_hidden * 2,
            vocab_size=vocab_size,
            dropout=config.head_dropout,
        )

    def forward(self, features: Tensor) -> Tensor:
        """``[B, T, input_size] -> [B, T, vocab_size]``. Raw logits, never normalized."""
        if features.dim() != 3:
            raise ValueError(f"Expected [B, T, C], got shape {tuple(features.shape)}")
        if features.shape[-1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {features.shape[-1]}")
        return self.projection(self.dropout(self.norm(features)))

    def describe(self) -> dict[str, Any]:
        return {
            "name": "character_head",
            "input_size": self.input_size,
            "vocab_size": self.vocab_size,
            "dropout": self.dropout_probability,
            "normalization": "layer_norm",
            "applies_softmax": False,
        }

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, vocab_size={self.vocab_size}, "
            f"dropout={self.dropout_probability}"
        )
