"""GM-Base assembly, `HTROutput` contract and the parameter ceiling."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from glyphmemory.config.schema import Config, ModelConfig
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.data import build_dataloader, build_dataset, temporal_length
from glyphmemory.model.htr import GMBase, HTROutput
from glyphmemory.model.loss import ctc_loss_for
from glyphmemory.model.model_info import (
    HARD_MAX_PARAMETERS,
    PREFERRED_MAX_PARAMETERS,
    assert_within_budget,
    parameter_count,
    parameter_count_by_module,
)

VOCAB = 80
CVL_ROOT = Path("datasets/CVL")


@pytest.fixture(scope="module")
def model() -> GMBase:
    torch.manual_seed(1337)
    return GMBase(vocab_size=VOCAB).eval()


class TestForward:
    @pytest.mark.parametrize("width", [64, 256, 512, 1024, 1600])
    def test_shapes_at_every_width(self, model: GMBase, width: int) -> None:
        out = model(torch.randn(2, 1, 64, width))
        expected_t = temporal_length(width)
        assert out.logits.shape == (2, expected_t, VOCAB)
        assert out.sequence_features.shape == (2, expected_t, 384)
        assert out.visual_features.shape == (2, expected_t, 192)
        assert out.input_lengths.shape == (2,)

    def test_output_validates(self, model: GMBase) -> None:
        out = model(torch.randn(3, 1, 64, 512), torch.tensor([128, 90, 40]))
        out.validate()

    def test_time_steps_match_temporal_length(self, model: GMBase) -> None:
        for width in (64, 512, 1600):
            assert model(torch.randn(1, 1, 64, width)).time_steps == temporal_length(width)
        assert model.output_length(512) == temporal_length(512)

    def test_omitting_lengths_assumes_full_width(self, model: GMBase) -> None:
        out = model(torch.randn(2, 1, 64, 256))
        assert out.input_lengths.tolist() == [64, 64]

    def test_lengths_are_carried_through_unchanged(self, model: GMBase) -> None:
        """The reason lengths live inside `HTROutput` at all."""
        lengths = torch.tensor([128, 33])
        out = model(torch.randn(2, 1, 64, 512), lengths)
        assert torch.equal(out.input_lengths, lengths)

    def test_mixed_width_batch_uses_packing(self, model: GMBase) -> None:
        """A short sample in a wide batch must not have its features polluted by padding."""
        torch.manual_seed(4)
        line = torch.randn(1, 1, 64, 256)
        padded = torch.zeros(1, 1, 64, 1024)
        padded[:, :, :, :256] = line
        with torch.no_grad():
            alone = model(line, torch.tensor([64]))
            in_batch = model(padded, torch.tensor([64]))
        assert torch.allclose(
            alone.sequence_features[:, :56], in_batch.sequence_features[:, :56], atol=1e-4
        )


class TestFeatureContract:
    def test_visual_features_are_exactly_the_encoder_output(self, model: GMBase) -> None:
        """Pinned: `visual_features` is `VisualEncoder(images)` — post height-reducer, post-BN."""
        images = torch.randn(2, 1, 64, 256)
        with torch.no_grad():
            out = model(images)
            direct = model.encoder(images)
        assert torch.equal(out.visual_features, direct)

    def test_sequence_features_are_the_head_input(self, model: GMBase) -> None:
        """Taken before the head's LayerNorm, so it is what the head consumes."""
        images = torch.randn(2, 1, 64, 256)
        with torch.no_grad():
            out = model(images)
            direct = model.sequence(model.encoder(images), out.input_lengths)
        assert torch.equal(out.sequence_features, direct)

    def test_both_feature_layers_are_always_populated(self, model: GMBase) -> None:
        out = model(torch.randn(1, 1, 64, 128))
        assert out.visual_features.numel() > 0
        assert out.sequence_features.numel() > 0

    def test_feature_dimensions_are_documented_values(self, model: GMBase) -> None:
        out = model(torch.randn(1, 1, 64, 128))
        assert out.describe()["visual_feature_dim"] == 192
        assert out.describe()["sequence_feature_dim"] == 384


class TestHTROutputValidation:
    def _output(self, **overrides) -> HTROutput:
        base = {
            "logits": torch.randn(2, 10, VOCAB),
            "sequence_features": torch.randn(2, 10, 384),
            "visual_features": torch.randn(2, 10, 192),
            "input_lengths": torch.tensor([10, 6]),
        }
        return HTROutput(**{**base, **overrides})

    def test_valid_output_passes(self) -> None:
        self._output().validate()

    def test_rejects_input_lengths_beyond_t(self) -> None:
        with pytest.raises(ValueError, match="exceeds T"):
            self._output(input_lengths=torch.tensor([11, 6])).validate()

    def test_rejects_feature_time_mismatch(self) -> None:
        with pytest.raises(ValueError, match="disagrees with logits"):
            self._output(sequence_features=torch.randn(2, 9, 384)).validate()

    def test_rejects_feature_batch_mismatch(self) -> None:
        with pytest.raises(ValueError, match="disagrees with logits"):
            self._output(visual_features=torch.randn(3, 10, 192)).validate()

    def test_rejects_wrong_length_vector_shape(self) -> None:
        with pytest.raises(ValueError, match=r"input_lengths must be \[B\]"):
            self._output(input_lengths=torch.tensor([10])).validate()

    def test_properties(self) -> None:
        out = self._output()
        assert (out.batch_size, out.time_steps, out.vocab_size) == (2, 10, VOCAB)

    def test_to_device_moves_every_tensor(self) -> None:
        moved = self._output().to("cpu")
        assert moved.logits.device.type == "cpu"
        assert moved.input_lengths.device.type == "cpu"

    def test_describe_states_logits_are_unnormalized(self) -> None:
        assert self._output().describe()["logits_are_normalized"] is False


class TestParameterBudget:
    def test_exact_total_is_pinned(self, model: GMBase) -> None:
        """Internal helper."""
        assert parameter_count(model) == 1_544_560

    def test_within_the_hard_ceiling(self, model: GMBase) -> None:
        """Internal helper."""
        total = assert_within_budget(model, ceiling=HARD_MAX_PARAMETERS)
        assert total <= HARD_MAX_PARAMETERS

    def test_within_the_preferred_target(self, model: GMBase) -> None:
        total = parameter_count(model)
        assert total <= PREFERRED_MAX_PARAMETERS, (
            f"{total:,} exceeds the 1.5-2.0M target; the BiGRU is the lever "
            "(ablation B), and only after."
        )

    def test_config_ceiling_is_respected(self) -> None:
        config = ModelConfig()
        model = GMBase.from_config(config, VOCAB)
        assert_within_budget(model, ceiling=config.max_parameters)

    def test_per_module_attribution(self, model: GMBase) -> None:
        by_module = parameter_count_by_module(model)
        assert by_module["encoder"] == 402_464
        assert by_module["sequence"] == 1_110_528
        assert by_module["head"] == 31_568
        assert sum(by_module.values()) == parameter_count(model)

    def test_budget_failure_names_the_size(self) -> None:
        with pytest.raises(ValueError, match="exceeding the ceiling"):
            assert_within_budget(GMBase(vocab_size=VOCAB), ceiling=1000)

    def test_report_prints(self, model: GMBase) -> None:
        report = model.parameter_report()
        assert report.within_hard_ceiling and report.within_preferred
        assert "encoder" in report.format()


class TestConfiguration:
    def test_from_config(self) -> None:
        model = GMBase.from_config(ModelConfig(gru_hidden=128), VOCAB)
        assert model.sequence.hidden_size == 128
        assert model.head.input_size == 256
        assert model(torch.randn(1, 1, 64, 128)).logits.shape == (1, 32, VOCAB)

    def test_vocab_agreement_with_the_tokenizer(self) -> None:
        tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
        model = GMBase(vocab_size=tokenizer.vocab_size)
        assert model(torch.randn(1, 1, 64, 128)).vocab_size == tokenizer.vocab_size
        assert tokenizer.blank_index == 0

    @pytest.mark.parametrize("visual_dim", [64, 128, 192, 256])
    def test_encoder_and_sequence_dimensions_stay_in_sync(self, visual_dim: int) -> None:
        """Both derive from ``visual_dim``, so they cannot disagree — checked, not assumed.

        ``GMBase`` also carries a defensive guard for the case where they do. It is unreachable
        through this constructor by design; the invariant tested here is what makes it unreachable.
        """
        model = GMBase(vocab_size=VOCAB, config=ModelConfig(visual_dim=visual_dim))
        assert model.encoder.feature_dim == model.sequence.input_size == visual_dim
        assert model(torch.randn(1, 1, 64, 128)).logits.shape == (1, 32, VOCAB)

    def test_describe_carries_the_full_architecture(self, model: GMBase) -> None:
        described = model.describe()
        assert described["parameters"] == 1_544_560
        assert described["encoder"]["feature_dim"] == 192
        assert described["sequence"]["kind"] == "bigru"
        assert described["head"]["applies_softmax"] is False


class TestGradients:
    def test_backward_reaches_every_parameter(self) -> None:
        model = GMBase(vocab_size=VOCAB)
        model(torch.randn(2, 1, 64, 256)).logits.sum().backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert not missing, f"no gradient reached: {missing}"

    def test_gradient_reaches_the_stem_through_the_loss(self) -> None:
        """The full stack, end to end — not merely 'some parameter has a grad'."""
        torch.manual_seed(0)
        model = GMBase(vocab_size=VOCAB)
        out = model(torch.randn(2, 1, 64, 256), torch.tensor([64, 40]))
        targets = torch.tensor([1, 2, 3, 4, 5, 6])
        loss, diag = ctc_loss_for(out, targets, torch.tensor([3, 3]))
        loss.backward()
        assert diag.loss_is_finite
        stem = model.encoder.stem[0].weight
        assert stem.grad is not None
        assert stem.grad.abs().sum() > 0

    def test_all_gradients_finite(self) -> None:
        torch.manual_seed(0)
        model = GMBase(vocab_size=VOCAB)
        out = model(torch.randn(2, 1, 64, 256), torch.tensor([64, 64]))
        loss, _ = ctc_loss_for(out, torch.tensor([1, 2, 3, 4]), torch.tensor([2, 2]))
        loss.backward()
        for name, param in model.named_parameters():
            assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"


class TestLearningSmokeTest:
    @pytest.mark.slow
    def test_loss_decreases_on_a_single_repeated_sample(self) -> None:
        """Not tiny-overfit — a check that the graph is connected.

        20 steps on one repeated line. If the sign conventions, the CTC layout or the transpose were
        wrong, the loss would wander rather than fall, and this is the cheapest place to find that
        out.
        """
        torch.manual_seed(0)
        model = GMBase(vocab_size=VOCAB).train()
        images = torch.randn(2, 1, 64, 128)
        lengths = torch.tensor([32, 32])
        targets = torch.tensor([5, 9, 14, 5, 9, 14])
        target_lengths = torch.tensor([3, 3])

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss, _ = ctc_loss_for(model(images, lengths), targets, target_lengths)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        assert all(torch.isfinite(torch.tensor(losses)))
        assert losses[-1] < losses[0], f"loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}"


class TestDeterminism:
    def test_same_seed_same_weights(self) -> None:
        torch.manual_seed(7)
        first = GMBase(vocab_size=VOCAB)
        torch.manual_seed(7)
        second = GMBase(vocab_size=VOCAB)
        for (_, a), (_, b) in zip(first.named_parameters(), second.named_parameters(), strict=True):
            assert torch.equal(a, b)

    def test_eval_disables_dropout(self, model: GMBase) -> None:
        images = torch.randn(2, 1, 64, 256)
        with torch.no_grad():
            assert torch.equal(model(images).logits, model(images).logits)

    def test_train_mode_applies_dropout(self) -> None:
        torch.manual_seed(0)
        model = GMBase(vocab_size=VOCAB).train()
        images = torch.randn(2, 1, 64, 256)
        assert not torch.allclose(model(images).logits, model(images).logits)


class TestGateOne:
    """Gate 1 — preprocessor -> model -> loss, at multiple widths."""

    def _loader(self, manifest: Path, batch_size: int = 4):
        tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
        config = Config()
        dataset = build_dataset(manifest, tokenizer, config, training=False)
        return build_dataloader(
            dataset, config, training=False, batch_size=batch_size, bucket=False, num_workers=0
        )

    def test_synthetic_batch_produces_a_finite_loss(self, synthetic_corpus, model: GMBase) -> None:
        loader = self._loader(synthetic_corpus.manifest_path)
        for batch in loader:
            with torch.no_grad():
                out = model(batch.images, batch.input_lengths)
            out.validate()
            loss, diag = ctc_loss_for(out, batch.targets, batch.target_lengths)
            assert torch.isfinite(loss)
            assert diag.infeasible == 0, "should have rejected these upstream"

    @pytest.mark.parametrize("width", [128, 512, 1024])
    def test_random_tensors_flow_through_at_multiple_widths(
        self, model: GMBase, width: int
    ) -> None:
        out = model(torch.randn(2, 1, 64, width))
        loss, _ = ctc_loss_for(out, torch.tensor([1, 2, 3, 4]), torch.tensor([2, 2]))
        assert torch.isfinite(loss)

    def test_short_and_long_in_one_batch(self, model: GMBase) -> None:
        """Gate 1 names this explicitly as where length bugs hide."""
        images = torch.zeros(2, 1, 64, 1024)
        images[0] = torch.randn(1, 64, 1024)
        images[1, :, :, :128] = torch.randn(1, 64, 128)
        out = model(images, torch.tensor([256, 32]))
        out.validate()
        loss, diag = ctc_loss_for(out, torch.tensor([1, 2, 3, 4, 5]), torch.tensor([3, 2]))
        assert torch.isfinite(loss)
        assert diag.infeasible == 0

    @pytest.mark.skipif(not CVL_ROOT.is_dir(), reason="CVL not present (CI never downloads it)")
    def test_real_cvl_batch_produces_a_finite_loss(self, tmp_path: Path, model: GMBase) -> None:
        from glyphmemory.data.adapters.cvl import CVLAdapter

        manifest = CVLAdapter(read_image_size=False).prepare(CVL_ROOT, tmp_path / "cvl")
        loader = self._loader(manifest, batch_size=4)
        assert len(loader.dataset) > 1_000
        batch = next(iter(loader))
        with torch.no_grad():
            out = model(batch.images, batch.input_lengths)
        out.validate()
        loss, diag = ctc_loss_for(out, batch.targets, batch.target_lengths)
        assert torch.isfinite(loss)
        assert diag.infeasible == 0
