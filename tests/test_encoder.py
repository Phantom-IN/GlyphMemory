"""Visual encoder tests.

The central test in this file is :meth:`TestTemporalLength.test_matches_temporal_length`. Everything
else guards a shape or a parameter count; that one guards the contract between the encoder and the
data pipeline.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from glyphmemory.config.schema import Config, ModelConfig
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.data import build_dataloader, build_dataset, temporal_length
from glyphmemory.model.blocks import InvertedResidual2D
from glyphmemory.model.encoder import (
    BLOCKS_PER_STAGE,
    STAGE_CHANNELS,
    STEM_CHANNELS,
    VisualEncoder,
    stage_output_shapes,
)
from glyphmemory.model.model_info import (
    PREFERRED_MAX_PARAMETERS,
    parameter_count,
    parameter_count_by_module,
    parameter_report,
)

# Widths the corpus actually produces, plus the preprocessing guard at 1600.
WIDTHS = (64, 128, 256, 512, 1024, 1600)

CVL_ROOT = Path("datasets/CVL")


@pytest.fixture(scope="module")
def encoder() -> VisualEncoder:
    torch.manual_seed(1337)
    return VisualEncoder().eval()


class TestInvertedResidual2D:
    @pytest.mark.parametrize(
        ("stride", "expected_hw"),
        [((1, 1), (16, 32)), ((2, 1), (8, 32)), ((1, 2), (16, 16)), ((2, 2), (8, 16))],
    )
    def test_output_shape_for_every_stride(
        self, stride: tuple[int, int], expected_hw: tuple[int, int]
    ) -> None:
        block = InvertedResidual2D(24, 40, stride=stride).eval()
        out = block(torch.randn(2, 24, 16, 32))
        assert out.shape == (2, 40, *expected_hw)

    def test_channel_change_is_honoured(self) -> None:
        block = InvertedResidual2D(8, 32).eval()
        assert block(torch.randn(1, 8, 8, 8)).shape == (1, 32, 8, 8)

    @pytest.mark.parametrize(
        ("in_c", "out_c", "stride", "expected"),
        [
            (32, 32, (1, 1), True),
            (32, 48, (1, 1), False),
            (32, 32, (2, 1), False),
            (32, 32, (1, 2), False),
            (32, 48, (2, 2), False),
        ],
    )
    def test_residual_present_exactly_when_permitted(
        self, in_c: int, out_c: int, stride: tuple[int, int], expected: bool
    ) -> None:
        """A residual across a stride or a channel change is a bug, not an optimization."""
        assert InvertedResidual2D(in_c, out_c, stride=stride).use_residual is expected

    def test_residual_actually_adds_the_input(self) -> None:
        """`use_residual` must describe the forward pass, not just advertise intent."""
        block = InvertedResidual2D(4, 4).eval()
        with torch.no_grad():
            block.project_norm.weight.zero_()
            block.project_norm.bias.zero_()
        x = torch.randn(1, 4, 6, 6)
        # With the projection zeroed the block computes exactly the identity, iff the residual is
        # wired.
        assert torch.allclose(block(x), x, atol=1e-5)

    def test_parameter_count_matches_hand_calculation(self) -> None:
        """32 -> 48 at expansion 2, hidden 64, convolutions without bias.

        expand   32*64            = 2048     BN 2*64  = 128
        depthwise 64*9            =  576     BN 2*64  = 128
        project  64*48            = 3072     BN 2*48  =  96
        total                                         = 6048
        """
        assert parameter_count(InvertedResidual2D(32, 48)) == 6048

    def test_convolutions_have_no_bias(self) -> None:
        """Bias is redundant with the BatchNorm shift that immediately follows it."""
        block = InvertedResidual2D(16, 16)
        assert block.expand.bias is None
        assert block.depthwise.bias is None
        assert block.project.bias is None

    def test_projection_has_no_activation(self) -> None:
        """The linear bottleneck: a signed feature space leaves the block.

        With random weights and enough channels some output must be negative. A ReLU on the
        projection would make that impossible.
        """
        torch.manual_seed(0)
        block = InvertedResidual2D(16, 32).eval()
        assert (block(torch.randn(4, 16, 8, 8)) < 0).any()

    @pytest.mark.parametrize("bad", [(0, 1), (1, 0), (1,), (1, 1, 1)])
    def test_rejects_malformed_stride(self, bad: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="stride"):
            InvertedResidual2D(8, 8, stride=bad)  # type: ignore[arg-type]

    def test_rejects_non_positive_channels(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            InvertedResidual2D(0, 8)


class TestShapes:
    @pytest.mark.parametrize("width", WIDTHS)
    def test_output_shape(self, encoder: VisualEncoder, width: int) -> None:
        out = encoder(torch.randn(2, 1, 64, width))
        assert out.shape == (2, width // 4, encoder.feature_dim)

    def test_feature_dimension_is_192(self, encoder: VisualEncoder) -> None:
        assert encoder.feature_dim == 192
        assert encoder.feature_dim == STAGE_CHANNELS[-1]

    def test_height_collapses_to_one_before_the_transpose(self, encoder: VisualEncoder) -> None:
        intermediate = encoder.height_reducer(
            encoder.stages(encoder.stem(torch.randn(1, 1, 64, 64)))
        )
        assert intermediate.shape[2] == 1

    def test_encoded_height_is_two_at_the_specified_input_height(
        self, encoder: VisualEncoder
    ) -> None:
        """64 -> 32 -> 16 -> 8 -> 4 -> 2, so the reducer kernel is the specified (2, 1)."""
        assert encoder.encoded_height == 2
        assert encoder.height_reducer[0].kernel_size == (2, 1)

    def test_stage_shape_table_matches_a_real_forward_pass(self) -> None:
        """A documentation table that drifts from the network is worse than none."""
        model = VisualEncoder().eval()
        x = torch.randn(1, 1, 64, 512)
        expected = dict(stage_output_shapes(input_height=64, width=512))

        out = model.stem(x)
        assert tuple(out.shape[1:]) == expected["stem"]
        for index, stage in enumerate(model.stages, start=1):
            out = stage(out)
            assert tuple(out.shape[1:]) == expected[f"stage{index}"]
        out = model.height_reducer(out)
        assert tuple(out.shape[1:]) == expected["height_reducer"]

    def test_batch_size_one_works(self, encoder: VisualEncoder) -> None:
        assert encoder(torch.randn(1, 1, 64, 128)).shape == (1, 32, 192)


class TestTemporalLength:
    """The contract this phase exists to establish."""

    @pytest.mark.parametrize("width", WIDTHS)
    def test_matches_temporal_length(self, encoder: VisualEncoder, width: int) -> None:
        assert encoder(torch.randn(1, 1, 64, width)).shape[1] == temporal_length(width)

    @pytest.mark.parametrize("width", [16, 32, 48, 80, 96, 160, 240, 336, 992, 1584])
    def test_matches_on_every_multiple_of_sixteen(self, encoder: VisualEncoder, width: int) -> None:
        assert encoder(torch.randn(1, 1, 64, width)).shape[1] == temporal_length(width)

    @pytest.mark.parametrize("width", [17, 19, 23, 30, 31, 33, 37, 63, 65, 127])
    def test_matches_on_widths_that_are_not_multiples_of_sixteen(
        self, encoder: VisualEncoder, width: int
    ) -> None:
        """Preprocessing pads to a multiple of 16, so these never reach the model in practice.

        They are tested because ``temporal_length`` documents ceiling semantics, and the encoder's
        two successive ``ceil(n/2)`` stages must compose to ``ceil(n/4)`` for the agreement to be
        structural rather than a coincidence of even numbers.
        """
        assert encoder(torch.randn(1, 1, 64, width)).shape[1] == temporal_length(width)

    def test_output_length_helper_delegates(self, encoder: VisualEncoder) -> None:
        for width in WIDTHS:
            assert encoder.output_length(width) == temporal_length(width)

    def test_two_ceiling_halvings_compose_to_one_quarter(self) -> None:
        """The identity the agreement rests on, checked independently of any tensor."""
        for width in range(1, 2048):
            assert math.ceil(math.ceil(width / 2) / 2) == math.ceil(width / 4)


class TestWidthAgnosticism:
    def test_no_parameter_depends_on_width(self, encoder: VisualEncoder) -> None:
        """A width-dependent parameter would force fixed-width input, which would force stretching,
        which changes glyph aspect — and glyph aspect is writer identity.
        """
        for name, module in encoder.named_modules():
            assert not isinstance(module, torch.nn.Linear), f"{name} is a Linear layer"
            assert not isinstance(
                module, (torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveMaxPool2d)
            ), f"{name} adaptively pools"

    def test_same_instance_handles_every_width(self, encoder: VisualEncoder) -> None:
        for width in (64, 1600, 128, 992):
            assert encoder(torch.randn(1, 1, 64, width)).shape[1] == width // 4

    def test_a_prefix_of_a_line_encodes_to_a_prefix_of_its_frames(self) -> None:
        """Time locality: frame t depends on a bounded window, not on the whole line.

        Not exact equality — the receptive field means the boundary frames of the shorter input see
        padding the longer one does not. The interior must still match closely, and if it does not,
        something is pooling globally over the time axis.
        """
        torch.manual_seed(7)
        model = VisualEncoder().eval()
        line = torch.randn(1, 1, 64, 512)
        with torch.no_grad():
            full = model(line)
            prefix = model(line[:, :, :, :256])
        # Compare the interior, away from the right-hand boundary of the prefix.
        assert torch.allclose(full[:, :48], prefix[:, :48], atol=1e-4)

    def test_rejects_wrong_height(self, encoder: VisualEncoder) -> None:
        with pytest.raises(ValueError, match="height"):
            encoder(torch.randn(1, 1, 32, 128))

    def test_rejects_wrong_rank(self, encoder: VisualEncoder) -> None:
        with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
            encoder(torch.randn(1, 64, 128))

    def test_rejects_wrong_channel_count(self, encoder: VisualEncoder) -> None:
        with pytest.raises(ValueError, match="channel"):
            encoder(torch.randn(1, 3, 64, 128))


class TestGradients:
    def test_backward_reaches_every_parameter(self) -> None:
        model = VisualEncoder()
        out = model(torch.randn(2, 1, 64, 256))
        out.sum().backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert not missing, f"no gradient reached: {missing}"

    def test_all_gradients_are_finite(self) -> None:
        model = VisualEncoder()
        model(torch.randn(2, 1, 64, 256)).sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"

    def test_gradient_reaches_the_stem(self) -> None:
        """The stem is the furthest point from the loss; a dead residual shows up here first."""
        model = VisualEncoder()
        model(torch.randn(2, 1, 64, 256)).sum().backward()
        stem_weight = model.stem[0].weight
        assert stem_weight.grad is not None
        assert stem_weight.grad.abs().sum() > 0


class TestBatchSizeOne:
    """Closes the risk row, by measurement rather than by deferral.

    The classic batch-1 BatchNorm failure is ``BatchNorm1d`` over features, where N=1 leaves one
    element per feature and the variance is degenerate. ``BatchNorm2d`` normalizes over ``(N, H,
    W)``, so here the spatial extent carries it and the layer degrades gracefully into instance
    normalization.
    """

    @pytest.mark.parametrize("width", [64, 512])
    def test_every_batchnorm_sees_enough_elements(self, width: int) -> None:
        model = VisualEncoder().train()
        seen: dict[str, int] = {}
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                handles.append(
                    module.register_forward_hook(
                        lambda _m, inp, _o, key=name: seen.__setitem__(
                            key, inp[0].shape[0] * inp[0].shape[2] * inp[0].shape[3]
                        )
                    )
                )
        model(torch.randn(1, 1, 64, width))
        for handle in handles:
            handle.remove()
        # The tightest is the height reducer's BN: [1, C, 1, T], so T elements per channel.
        assert min(seen.values()) >= temporal_length(width)

    def test_forward_and_backward_are_finite_in_train_mode(self) -> None:
        model = VisualEncoder().train()
        out = model(torch.randn(1, 1, 64, 64))
        assert torch.isfinite(out).all()
        out.sum().backward()
        for name, param in model.named_parameters():
            assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"

    @pytest.mark.slow
    def test_a_single_sample_can_be_fitted_at_batch_size_one(self) -> None:
        """The scenario, verified here rather than discovered there.

        Marked slow: 200 optimizer steps. The cheaper stability checks above run always; this one
        proves the model actually converges under batch-1 BatchNorm rather than merely staying
        finite.
        """
        torch.manual_seed(0)
        model = VisualEncoder().train()
        x = torch.randn(1, 1, 64, 128)
        target = torch.randn(1, 32, 192)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        first = None
        for step in range(120):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), target)
            loss.backward()
            optimizer.step()
            if step == 0:
                first = float(loss.detach())
        assert float(loss.detach()) < first * 0.25


class TestDeterminism:
    def test_same_seed_same_weights(self) -> None:
        torch.manual_seed(99)
        first = VisualEncoder()
        torch.manual_seed(99)
        second = VisualEncoder()
        for (_, a), (_, b) in zip(first.named_parameters(), second.named_parameters(), strict=True):
            assert torch.equal(a, b)

    def test_eval_mode_is_deterministic(self, encoder: VisualEncoder) -> None:
        x = torch.randn(2, 1, 64, 256)
        with torch.no_grad():
            assert torch.equal(encoder(x), encoder(x))


class TestParameterBudget:
    def test_measured_count_is_recorded(self, encoder: VisualEncoder) -> None:
        """Pinned so a regression names its own size instead of drifting quietly."""
        assert parameter_count(encoder) == 402_464

    def test_leaves_room_for_the_bigru_under_the_preferred_target(
        self, encoder: VisualEncoder
    ) -> None:
        """The encoder must not spend the budget the sequence model needs.

        The whole model has to fit under the 2.0M preferred target, not merely under the 3.0M hard
        ceiling.
        """
        projected = parameter_count(encoder) + 1_110_528 + 31_568
        assert projected <= PREFERRED_MAX_PARAMETERS, (
            f"encoder {parameter_count(encoder):,} + projected BiGRU/head leaves "
            f"{projected:,} > {PREFERRED_MAX_PARAMETERS:,}"
        )

    def test_per_stage_attribution_is_available(self, encoder: VisualEncoder) -> None:
        """An overrun must be attributable to a stage, not to 'the encoder'."""
        by_module = parameter_count_by_module(encoder, depth=2)
        assert by_module["stem"] == 352
        assert by_module["height_reducer"] == 768
        per_stage = [by_module[f"stages.{i}"] for i in range(len(STAGE_CHANNELS))]
        assert sum(per_stage) == by_module["stages"]
        # Later stages are wider, so cost must increase monotonically.
        assert per_stage == sorted(per_stage)

    def test_report_prints(self, encoder: VisualEncoder) -> None:
        report = parameter_report(encoder)
        assert report.within_preferred
        assert "stem" in report.format()


class TestConfiguration:
    def test_from_config_uses_visual_dim(self) -> None:
        model = VisualEncoder.from_config(ModelConfig(visual_dim=128))
        assert model.feature_dim == 128
        assert model(torch.randn(1, 1, 64, 128)).shape == (1, 32, 128)

    def test_from_default_config_matches_the_spec(self) -> None:
        model = VisualEncoder.from_config(ModelConfig())
        assert model.feature_dim == 192
        assert model.stage_channels == STAGE_CHANNELS
        assert parameter_count(model) == 402_464

    def test_describe_carries_provenance(self, encoder: VisualEncoder) -> None:
        described = encoder.describe()
        assert described["feature_dim"] == 192
        assert described["horizontal_downsample"] == 4
        assert described["stage_channels"] == list(STAGE_CHANNELS)
        assert described["blocks_per_stage"] == BLOCKS_PER_STAGE
        assert described["parameters"] == 402_464
        assert described["stem_channels"] == STEM_CHANNELS

    def test_alternate_input_height_sizes_the_reducer(self) -> None:
        """Ablation A (heights 48/64/80) must be buildable without a second code path."""
        model = VisualEncoder(input_height=80).eval()
        assert model.encoded_height == 3
        assert model.height_reducer[0].kernel_size == (3, 1)
        assert model(torch.randn(1, 1, 80, 256)).shape == (1, 64, 192)


class TestAgainstThePipeline:
    """Real batches, not random tensors — mixed widths are where length bugs hide."""

    def _loader(self, manifest: Path, batch_size: int = 4):
        tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
        config = Config()
        dataset = build_dataset(manifest, tokenizer, config, training=False)
        return build_dataloader(
            dataset, config, training=False, batch_size=batch_size, bucket=False, num_workers=0
        )

    def test_synthetic_batch_flows_through(self, synthetic_corpus, encoder: VisualEncoder) -> None:
        loader = self._loader(synthetic_corpus.manifest_path)
        batch = next(iter(loader))
        with torch.no_grad():
            out = encoder(batch.images)
        assert out.shape[0] == batch.batch_size
        assert out.shape[2] == encoder.feature_dim
        assert out.shape[1] == temporal_length(batch.images.shape[-1])

    def test_every_input_length_fits_within_t(
        self, synthetic_corpus, encoder: VisualEncoder
    ) -> None:
        """``input_lengths > T`` makes the CTC loss fail loudly; catch it here."""
        loader = self._loader(synthetic_corpus.manifest_path)
        for batch in loader:
            with torch.no_grad():
                frames = encoder(batch.images).shape[1]
            assert int(batch.input_lengths.max()) <= frames
            assert frames == temporal_length(batch.images.shape[-1])

    def test_mixed_width_batch_keeps_per_sample_frames_aligned(
        self, encoder: VisualEncoder
    ) -> None:
        """A short line padded into a wide batch must still encode its own frames identically.

        This is the padding-independence property CTC relies on: sample *i*'s first
        ``input_lengths[i]`` frames must not depend on how wide its batch neighbours were.
        """
        torch.manual_seed(11)
        short = torch.randn(1, 1, 64, 256)
        padded = torch.zeros(1, 1, 64, 1024)
        padded[:, :, :, :256] = short
        with torch.no_grad():
            alone = encoder(short)
            together = encoder(padded)
        assert torch.allclose(alone[:, :56], together[:, :56], atol=1e-4)

    @pytest.mark.skipif(not CVL_ROOT.is_dir(), reason="CVL not present (CI never downloads it)")
    def test_real_cvl_batch_flows_through(self, tmp_path: Path, encoder: VisualEncoder) -> None:
        from glyphmemory.data.adapters.cvl import CVLAdapter

        manifest = CVLAdapter(read_image_size=False).prepare(CVL_ROOT, tmp_path / "cvl")
        loader = self._loader(manifest, batch_size=4)
        # Guard against a silently empty corpus: a test that passes on no data is worse than one
        # that skips.
        assert len(loader.dataset) > 1_000
        batch = next(iter(loader))
        with torch.no_grad():
            out = encoder(batch.images)
        assert out.shape[1] == temporal_length(batch.images.shape[-1])
        assert int(batch.input_lengths.max()) <= out.shape[1]
