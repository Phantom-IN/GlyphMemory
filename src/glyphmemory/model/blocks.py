"""Building blocks for the GM-Base visual encoder.

    1x1 pointwise expansion  -> BatchNorm -> ReLU
    3x3 depthwise conv       -> BatchNorm -> ReLU
    1x1 pointwise projection -> BatchNorm
    optional residual        (only when shape and stride permit)

**Built from standard PyTorch layers, randomly initialized.** No pretrained MobileNet is imported
and none may be; the inverted-residual *pattern* is public architecture knowledge, the weights are
ours.

Two details carry more weight than they look.

**There is no activation after the projection.** That is what makes this a *linear bottleneck*: the
narrow tensor that leaves the block and travels to the next one is not clipped at zero, so the block
can represent a full signed feature space at low width. A ReLU there would discard half of it.

**Convolutions carry no bias.** Every one is followed immediately by BatchNorm, whose shift
parameter subsumes a bias exactly. Keeping both would add parameters that cannot change the
function.
"""

from __future__ import annotations

from torch import Tensor, nn

#: Expansion ratio for the inverted-residual blocks.
DEFAULT_EXPANSION = 2


class InvertedResidual2D(nn.Module):
    """Inverted-residual block with a linear bottleneck.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        stride: ``(height_stride, width_stride)``. The encoder strides height and width separately —
            see :mod:`glyphmemory.model.encoder`.
        expansion: Hidden width multiplier applied to ``in_channels``.

    Shape:
        ``[B, in_channels, H, W] -> [B, out_channels, ceil(H/sh), ceil(W/sw)]``

    The residual connection is present **only** when ``stride == (1, 1)`` and ``in_channels ==
    out_channels``. That condition is a correctness matter rather than a tuning choice: adding a
    residual across a stride or a channel change is a shape error at best, and at worst broadcasts
    silently into a tensor that trains without complaint. The condition is therefore computed once,
    stored, and exposed as :attr:`use_residual` so a test can assert it rather than infer it from
    behaviour.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: tuple[int, int] = (1, 1),
        expansion: int = DEFAULT_EXPANSION,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError(
                f"Channel counts must be positive, got in={in_channels} out={out_channels}"
            )
        if len(stride) != 2 or any(s < 1 for s in stride):
            raise ValueError(f"stride must be two positive integers, got {stride!r}")
        if expansion < 1:
            raise ValueError(f"expansion must be at least 1, got {expansion}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = (int(stride[0]), int(stride[1]))
        self.expansion = expansion
        self.hidden_channels = in_channels * expansion

        self.use_residual = self.stride == (1, 1) and in_channels == out_channels

        hidden = self.hidden_channels
        self.expand = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False)
        self.expand_norm = nn.BatchNorm2d(hidden)

        self.depthwise = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=hidden,
            bias=False,
        )
        self.depthwise_norm = nn.BatchNorm2d(hidden)

        self.project = nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False)
        self.project_norm = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """``[B, C_in, H, W] -> [B, C_out, H', W']``."""
        out = self.activation(self.expand_norm(self.expand(x)))
        out = self.activation(self.depthwise_norm(self.depthwise(out)))
        out = self.project_norm(self.project(out))  # linear bottleneck — no activation
        if self.use_residual:
            out = out + x
        return out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"stride={self.stride}, expansion={self.expansion}, "
            f"residual={self.use_residual}"
        )
