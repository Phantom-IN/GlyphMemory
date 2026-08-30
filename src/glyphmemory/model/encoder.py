"""GM-Base visual encoder.

``[B, 1, 64, W] -> [B, T, 192]`` where ``T == temporal_length(W)``.

    Stem      Conv3x3, 1 -> 32, stride (2,2)        [B,  32, 32, W/2]
    Stage 1   IR  32 ->  48, stride (2,2)           [B,  48, 16, W/4]
              IR  48 ->  48, stride (1,1)
    Stage 2   IR  48 ->  80, stride (2,1)           [B,  80,  8, W/4]
              IR  80 ->  80, stride (1,1)
    Stage 3   IR  80 -> 128, stride (2,1)           [B, 128,  4, W/4]
              IR 128 -> 128, stride (1,1)
    Stage 4   IR 128 -> 192, stride (2,1)           [B, 192,  2, W/4]
              IR 192 -> 192, stride (1,1)
    Height reducer: depthwise conv, kernel (2,1)    [B, 192,  1, W/4]
    squeeze + transpose                             [B, W/4, 192]

**Read the stride column.** Only the stem and stage 1 stride the width axis; stages 2-4 stride
height only. That is what produces exactly 4x horizontal downsampling while height collapses 64 → 32
→ 16 → 8 → 4 → 2 → 1.

The 4x is deliberate and load-bearing. Downsampling harder would be faster and would leave CTC too
little temporal resolution for long cursive, and would leave writer memory too few frames per glyph
to pool a prototype from.

**``T`` is not computed here.** :func:`~glyphmemory.data.preprocessing.temporal_length` is the
single source of truth, and this module is *tested against* it rather than agreeing with it by
construction.
"""

from __future__ import annotations

import math
from typing import Any

from torch import Tensor, nn

from glyphmemory.config.schema import ModelConfig
from glyphmemory.data.preprocessing import (
    DEFAULT_HEIGHT,
    HORIZONTAL_DOWNSAMPLE,
    temporal_length,
)
from glyphmemory.model.blocks import DEFAULT_EXPANSION, InvertedResidual2D
from glyphmemory.model.model_info import parameter_count, parameter_count_by_module

#: Stem output channels.
STEM_CHANNELS = 32

#: Output channels of stages 1-4. The last entry is the encoder's feature dimension and is
#: overridden by ``ModelConfig.visual_dim`` when building from config.
STAGE_CHANNELS: tuple[int, int, int, int] = (48, 80, 128, 192)

#: Blocks per stage: the first strides, the second refines at stride 1.
BLOCKS_PER_STAGE = 2


def _stage_strides(index: int) -> tuple[int, int]:
    """Stride of the first block of stage ``index`` (0-based).

    Stage 1 strides both axes; stages 2-4 stride height only. This single function is where the 4x
    horizontal downsample lives — changing it changes ``T`` and therefore breaks the contract with
    :func:`temporal_length`, which is exactly why it is one line and not scattered across the stage
    builder.
    """
    return (2, 2) if index == 0 else (2, 1)


class VisualEncoder(nn.Module):
    """Lightweight inverted-residual CNN over a normalized grayscale line.

    Args:
        in_channels: Input channels. 1 — the pipeline delivers grayscale.
        stem_channels: Stem output width.
        stage_channels: Output channels per stage; the final entry is the feature dimension.
        expansion: Inverted-residual expansion ratio.
        input_height: Height the encoder is built for. Used to size the height reducer, and
            validated at construction rather than discovered as a shape error at run time.

    Shape:
        ``[B, 1, H, W] -> [B, T, feature_dim]`` with ``T == ceil(W / 4)``.

    The module holds no width-dependent parameters — no ``Linear`` over the width axis, no adaptive
    pooling on the time axis, no flattening of ``W`` — so a single instance handles every width the
    corpus produces.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        stem_channels: int = STEM_CHANNELS,
        stage_channels: tuple[int, ...] = STAGE_CHANNELS,
        expansion: int = DEFAULT_EXPANSION,
        input_height: int = DEFAULT_HEIGHT,
    ) -> None:
        super().__init__()
        if not stage_channels:
            raise ValueError("stage_channels must not be empty")
        if input_height < 1:
            raise ValueError(f"input_height must be positive, got {input_height}")

        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.stage_channels = tuple(stage_channels)
        self.expansion = expansion
        self.input_height = input_height

        # Stem: the only place the width axis is strided besides stage 1.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        channels = stem_channels
        for index, out_channels in enumerate(self.stage_channels):
            blocks: list[nn.Module] = [
                InvertedResidual2D(
                    channels,
                    out_channels,
                    stride=_stage_strides(index),
                    expansion=expansion,
                )
            ]
            blocks.extend(
                InvertedResidual2D(out_channels, out_channels, stride=(1, 1), expansion=expansion)
                for _ in range(BLOCKS_PER_STAGE - 1)
            )
            stages.append(nn.Sequential(*blocks))
            channels = out_channels
        self.stages = nn.Sequential(*stages)

        self.feature_dim = channels
        self.encoded_height = self._encoded_height(input_height, len(self.stage_channels))

        # Height reducer. Learnable rather than pooled: collapsing ascender and descender rows is a
        # weighted decision, and a mean would fix those weights at equal.
        self.height_reducer = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(self.encoded_height, 1),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )
        # No activation here: the encoder's output is a linear bottleneck all the way to the BiGRU,
        # for the reason given in glyphmemory.model.blocks.

        self._initialize_weights()

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_config(cls, config: ModelConfig) -> VisualEncoder:
        """Build from :class:`~glyphmemory.config.schema.ModelConfig`.

        ``visual_dim`` replaces the final stage's channel count, so the configured feature dimension
        and the architecture cannot disagree.
        """
        channels = (*STAGE_CHANNELS[:-1], config.visual_dim)
        return cls(stage_channels=channels, input_height=config.input_height)

    @staticmethod
    def _encoded_height(input_height: int, n_stages: int) -> int:
        """Height after the stem and every stage.

        Each strided 3x3 convolution with padding 1 outputs ``ceil(in / 2)``, so the height
        halves once in the stem and once per stage. At the specified height of 64 over four
        stages this is 64 -> 32 -> 16 -> 8 -> 4 -> 2.
        """
        height = math.ceil(input_height / 2)  # stem
        for _ in range(n_stages):
            height = math.ceil(height / 2)
        if height < 1:
            raise ValueError(
                f"input_height={input_height} collapses below one row across {n_stages} "
                "stages plus the stem."
            )
        return height

    def _initialize_weights(self) -> None:
        """Kaiming-normal convolutions, unit BatchNorm.

        Stated explicitly rather than left to PyTorch's defaults, which use
        ``kaiming_uniform_(a=sqrt(5))`` — a legacy setting tuned for ``Linear``, not for deep ReLU
        convolution stacks.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------ forward

    def forward(self, images: Tensor) -> Tensor:
        """``[B, 1, H, W] -> [B, T, feature_dim]``.

        Raises:
            ValueError: ``images`` is not 4-D, or its height differs from ``input_height``. Height
                is checked because the height reducer's kernel is sized for it, and a mismatch would
                otherwise surface as an opaque convolution shape error several layers later.
        """
        if images.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got shape {tuple(images.shape)}")
        if images.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channel(s), got {images.shape[1]}")
        if images.shape[2] != self.input_height:
            raise ValueError(
                f"Expected height {self.input_height}, got {images.shape[2]}. The encoder is "
                "built for one input height; resize in preprocessing, never here."
            )

        out = self.stem(images)
        out = self.stages(out)
        out = self.height_reducer(out)  # [B, C, 1, T]
        out = out.squeeze(2)  # [B, C, T]
        # transpose + contiguous, never .view(): a view over a transposed tensor reads the
        # underlying strides and would interleave channels with time without raising.
        return out.transpose(1, 2).contiguous()  # [B, T, C]

    # ------------------------------------------------------------------ introspection

    def output_length(self, width: int) -> int:
        """``T`` for an input of ``width`` pixels.

        Delegates to :func:`~glyphmemory.data.preprocessing.temporal_length` rather than
        recomputing, so there is exactly one definition of ``T`` in the codebase.
        """
        return temporal_length(width, HORIZONTAL_DOWNSAMPLE)

    def describe(self) -> dict[str, Any]:
        """Architecture provenance for the run record."""
        return {
            "name": "visual_encoder",
            "in_channels": self.in_channels,
            "input_height": self.input_height,
            "stem_channels": self.stem_channels,
            "stage_channels": list(self.stage_channels),
            "blocks_per_stage": BLOCKS_PER_STAGE,
            "expansion": self.expansion,
            "feature_dim": self.feature_dim,
            "encoded_height": self.encoded_height,
            "horizontal_downsample": HORIZONTAL_DOWNSAMPLE,
            "parameters": parameter_count(self),
            "parameters_by_module": parameter_count_by_module(self, depth=2),
        }


def stage_output_shapes(
    input_height: int = DEFAULT_HEIGHT, width: int = 512
) -> list[tuple[str, tuple[int, int, int]]]:
    """``(name, (channels, height, width))`` after the stem and each stage.

    A debugging aid for reading the stage plan against reality without instrumenting a forward pass.
    Verified against an actual forward pass in the tests, because a shape table that drifts from the
    network is worse than none.
    """
    shapes: list[tuple[str, tuple[int, int, int]]] = []
    height = math.ceil(input_height / 2)
    time = math.ceil(width / 2)
    shapes.append(("stem", (STEM_CHANNELS, height, time)))
    for index, channels in enumerate(STAGE_CHANNELS):
        stride_h, stride_w = _stage_strides(index)
        height = math.ceil(height / stride_h)
        time = math.ceil(time / stride_w)
        shapes.append((f"stage{index + 1}", (channels, height, time)))
    shapes.append(("height_reducer", (STAGE_CHANNELS[-1], 1, time)))
    return shapes
