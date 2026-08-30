"""Trainer and run-record tests.

Everything here runs on the small synthetic corpus so the file stays fast. What it defends:
selection by CER rather than loss, the clip-rate counter, the non-finite-loss guard, the
augmentation assertion, and the completeness of ``run.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from glyphmemory.config.schema import Config, TrainingConfig
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.data import build_dataloader, build_dataset
from glyphmemory.metrics import EditCounts, MetricResult
from glyphmemory.model import GMBase
from glyphmemory.runtime.device import resolve_device
from glyphmemory.runtime.experiment import ExperimentDir
from glyphmemory.training.checkpoint import (
    BEST_FILENAME,
    LAST_FILENAME,
    SELECTION_METRIC,
    load_checkpoint,
)
from glyphmemory.training.run_record import (
    REQUIRED_FIELDS,
    build_run_record,
    manifest_fingerprints,
    missing_fields,
)
from glyphmemory.training.schedule import build_scheduler
from glyphmemory.training.trainer import EpochStats, Trainer, ValidationStats


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(DEFAULT_CHARSET_PATH)


def make_trainer(
    corpus,
    tokenizer,
    *,
    config: Config | None = None,
    experiment=None,
    with_validation: bool = True,
    limit: int | None = None,
) -> Trainer:
    config = config or Config(training=TrainingConfig(batch_size=4, epochs=1))
    torch.manual_seed(0)

    train_dataset = build_dataset(corpus.manifest_path, tokenizer, config, training=True)
    val_dataset = build_dataset(corpus.manifest_path, tokenizer, config, training=False)
    if limit is not None:
        train_dataset = train_dataset.take(limit)
        val_dataset = val_dataset.take(limit)

    train_loader = build_dataloader(
        train_dataset, config, training=True, batch_size=4, num_workers=0
    )
    val_loader = (
        build_dataloader(
            val_dataset, config, training=False, batch_size=4, bucket=False, num_workers=0
        )
        if with_validation
        else None
    )

    model = GMBase(vocab_size=tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    scheduler, _ = build_scheduler(
        optimizer, config.training, total_steps=max(len(train_loader), 1)
    )
    return Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        experiment=experiment,
    )


class TestTrainingStep:
    def test_one_epoch_updates_parameters(self, synthetic_corpus, tokenizer) -> None:
        trainer = make_trainer(synthetic_corpus, tokenizer)
        before = trainer.model.encoder.stem[0].weight.detach().clone()
        stats = trainer.train_epoch(1)
        after = trainer.model.encoder.stem[0].weight.detach()
        assert stats.steps > 0
        assert not torch.equal(before, after)

    @pytest.mark.slow
    def test_loss_falls_over_a_few_epochs(self, synthetic_corpus, tokenizer) -> None:
        trainer = make_trainer(synthetic_corpus, tokenizer, limit=4)
        first = trainer.train_epoch(1).loss
        for epoch in range(2, 6):
            last = trainer.train_epoch(epoch).loss
        assert last < first

    def test_step_counter_advances(self, synthetic_corpus, tokenizer) -> None:
        trainer = make_trainer(synthetic_corpus, tokenizer)
        stats = trainer.train_epoch(1)
        assert trainer.step == stats.steps

    def test_throughput_is_recorded(self, synthetic_corpus, tokenizer) -> None:
        stats = make_trainer(synthetic_corpus, tokenizer).train_epoch(1)
        assert stats.samples > 0
        assert stats.seconds > 0
        assert stats.samples_per_second > 0
        assert "samples_per_second" in stats.as_dict()


class TestGradientClipping:
    def test_clip_rate_is_counted(self, synthetic_corpus, tokenizer) -> None:
        """An absurdly low `max_norm` must make the counter fire on every step."""
        config = Config(training=TrainingConfig(batch_size=4, grad_clip_norm=1e-9))
        stats = make_trainer(synthetic_corpus, tokenizer, config=config).train_epoch(1)
        assert stats.clipped_steps == stats.steps
        assert stats.clip_rate == 1.0

    def test_a_generous_norm_clips_less(self, synthetic_corpus, tokenizer) -> None:
        config = Config(training=TrainingConfig(batch_size=4, grad_clip_norm=1e9))
        stats = make_trainer(synthetic_corpus, tokenizer, config=config).train_epoch(1)
        assert stats.clipped_steps == 0

    def test_clip_rate_is_in_the_metrics_record(self) -> None:
        stats = EpochStats(1, 10, 1.0, 1e-4, 7, 0, 40, 2.0)
        assert stats.as_dict()["clip_rate"] == pytest.approx(0.7)


class TestNonFiniteLoss:
    def test_skipped_and_counted_without_corrupting_weights(
        self, synthetic_corpus, tokenizer, monkeypatch
    ) -> None:
        """One `inf` must not poison the weights, and must not be invisible either."""
        trainer = make_trainer(synthetic_corpus, tokenizer)
        import glyphmemory.training.trainer as module

        def nan_loss(output, targets, target_lengths, **kwargs):
            return torch.tensor(float("nan"), requires_grad=True), None

        monkeypatch.setattr(module, "ctc_loss_for", nan_loss)
        before = trainer.model.encoder.stem[0].weight.detach().clone()
        stats = trainer.train_epoch(1)

        assert stats.steps == 0
        assert stats.skipped_steps > 0
        assert torch.equal(before, trainer.model.encoder.stem[0].weight.detach())


class TestValidation:
    def test_reports_loss_cer_and_wer(self, synthetic_corpus, tokenizer) -> None:
        stats = make_trainer(synthetic_corpus, tokenizer).validate(1)
        assert stats is not None
        assert stats.samples > 0
        assert stats.cer.name == "cer"
        assert stats.wer.name == "wer"
        assert stats.as_dict()[SELECTION_METRIC] == stats.cer.value

    def test_metric_carries_normalization_and_decoder(self, synthetic_corpus, tokenizer) -> None:
        stats = make_trainer(synthetic_corpus, tokenizer).validate(1)
        assert stats.cer.normalization == tokenizer.policy.name
        assert stats.cer.decoder.label == "greedy, no LM"

    def test_previews_are_printed_beside_ground_truth(self, synthetic_corpus, tokenizer) -> None:
        """Internal helper."""
        stats = make_trainer(synthetic_corpus, tokenizer).validate(1)
        assert stats.previews
        text = stats.format()
        assert "truth" in text and "pred" in text

    def test_previews_are_the_same_samples_every_epoch(self, synthetic_corpus, tokenizer) -> None:
        """Fixed samples, so successive epochs are comparable rather than a fresh draw."""
        trainer = make_trainer(synthetic_corpus, tokenizer)
        first = trainer.validate(1)
        second = trainer.validate(2)
        assert [p[0] for p in first.previews] == [p[0] for p in second.previews]

    def test_augmented_validation_is_refused(self, synthetic_corpus, tokenizer) -> None:
        """Checked against what the pipeline does, not the flag that built it."""
        config = Config()
        dataset = build_dataset(synthetic_corpus.manifest_path, tokenizer, config, training=True)
        loader = build_dataloader(dataset, config, training=False, batch_size=2, num_workers=0)
        model = GMBase(vocab_size=tokenizer.vocab_size)
        with pytest.raises(ValueError, match="never augmented"):
            Trainer(
                model=model,
                tokenizer=tokenizer,
                optimizer=torch.optim.AdamW(model.parameters()),
                scheduler=None,
                train_loader=loader,
                val_loader=loader,
                config=config,
            )

    def test_an_identity_pipeline_is_accepted(self, synthetic_corpus, tokenizer) -> None:
        """`training=False` yields Identity, not None — a None check would be wrong."""
        trainer = make_trainer(synthetic_corpus, tokenizer)
        assert trainer.val_loader.dataset.augmentation.is_identity

    def test_no_validation_loader_means_no_stats(self, synthetic_corpus, tokenizer) -> None:
        trainer = make_trainer(synthetic_corpus, tokenizer, with_validation=False)
        assert trainer.validate(1) is None


class TestFitAndCheckpointing:
    @pytest.mark.slow
    def test_writes_metrics_and_checkpoints(
        self, synthetic_corpus, tokenizer, tmp_path: Path
    ) -> None:
        experiment = ExperimentDir.create(tmp_path, "gm_base_test")
        trainer = make_trainer(synthetic_corpus, tokenizer, experiment=experiment)
        history = trainer.fit(2)

        assert len(history) == 2
        assert experiment.metrics_stream_path.is_file()
        assert (experiment.checkpoints_dir / LAST_FILENAME).is_file()
        assert (experiment.checkpoints_dir / BEST_FILENAME).is_file()

        lines = experiment.metrics_stream_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["epoch"] == 1

    def test_best_is_selected_by_cer_not_loss(
        self, synthetic_corpus, tokenizer, tmp_path: Path
    ) -> None:
        """The failure this guards: a run whose loss improves while its CER degrades.

        Selection is driven through `_checkpoint` with scripted validation results, because making a
        real model regress on cue is not something a test should try to arrange.
        """
        experiment = ExperimentDir.create(tmp_path, "gm_base_test")
        trainer = make_trainer(synthetic_corpus, tokenizer, experiment=experiment)

        def validation(epoch: int, loss: float, cer_value: float) -> ValidationStats:
            counts = EditCounts(substitutions=int(cer_value * 100), reference_length=100)
            metric = MetricResult(name="cer", counts=counts, normalization="nfc_v1")
            assert metric.value == pytest.approx(cer_value)
            return ValidationStats(
                epoch=epoch, loss=loss, cer=metric, wer=metric, samples=100, seconds=0.0
            )

        assert trainer._checkpoint(1, validation(1, 9.0, 0.40)) is True
        assert trainer.best_value == pytest.approx(0.40)

        # Loss improves 9.0 -> 1.0 but CER worsens 0.40 -> 0.90. Selecting on loss would
        # overwrite best.pt here; selecting on CER must not.
        assert trainer._checkpoint(2, validation(2, 1.0, 0.90)) is False
        assert trainer.best_value == pytest.approx(0.40)
        assert load_checkpoint(experiment.checkpoints_dir / BEST_FILENAME).meta.epoch == 1
        assert load_checkpoint(experiment.checkpoints_dir / LAST_FILENAME).meta.epoch == 2

    def test_a_validation_less_epoch_cannot_displace_the_best(
        self, synthetic_corpus, tokenizer, tmp_path: Path
    ) -> None:
        experiment = ExperimentDir.create(tmp_path, "gm_base_test")
        trainer = make_trainer(synthetic_corpus, tokenizer, experiment=experiment)
        trainer.best_value = 0.4
        assert trainer._checkpoint(2, None) is False
        assert trainer.best_value == pytest.approx(0.4)

    @pytest.mark.slow
    def test_last_checkpoint_is_written_every_epoch(
        self, synthetic_corpus, tokenizer, tmp_path: Path
    ) -> None:
        experiment = ExperimentDir.create(tmp_path, "gm_base_test")
        trainer = make_trainer(synthetic_corpus, tokenizer, experiment=experiment)
        trainer.fit(2)
        meta = load_checkpoint(experiment.checkpoints_dir / LAST_FILENAME).meta
        assert meta.epoch == 2

    @pytest.mark.slow
    def test_checkpoints_carry_the_charset_fingerprint(
        self, synthetic_corpus, tokenizer, tmp_path: Path
    ) -> None:
        experiment = ExperimentDir.create(tmp_path, "gm_base_test")
        trainer = make_trainer(synthetic_corpus, tokenizer, experiment=experiment)
        trainer.fit(1)
        loaded = load_checkpoint(
            experiment.checkpoints_dir / LAST_FILENAME,
            charset_fingerprint=tokenizer.charset.fingerprint(),
        )
        assert loaded.meta.tokenizer_fingerprint == tokenizer.fingerprint()

    @pytest.mark.slow
    def test_early_stopping_fires_on_patience(
        self, synthetic_corpus, tokenizer, tmp_path: Path
    ) -> None:
        experiment = ExperimentDir.create(tmp_path, "gm_base_test")
        config = Config(training=TrainingConfig(batch_size=4, patience=1))
        trainer = make_trainer(synthetic_corpus, tokenizer, config=config, experiment=experiment)
        history = trainer.fit(10)
        assert len(history) < 10

    def test_without_an_experiment_nothing_is_written(self, synthetic_corpus, tokenizer) -> None:
        trainer = make_trainer(synthetic_corpus, tokenizer)
        assert trainer.fit(1)
        assert trainer.best_value is None


class TestRunRecord:
    def _record(self, tokenizer, tmp_path: Path) -> dict:
        model = GMBase(vocab_size=tokenizer.vocab_size)
        return build_run_record(
            run_id="gm_base__20260818T000000Z",
            config=Config(),
            tokenizer=tokenizer,
            device=resolve_device("cpu", log=False),
            model=model,
            seed=1337,
            manifests={"train": "abc"},
        )

    def test_contains_every_required_field(self, tokenizer, tmp_path: Path) -> None:
        """Checked by name, so a dropped field fails the build rather than being noticed in six
        months — by which time the runs that lack it are already the evidence.
        """
        assert missing_fields(self._record(tokenizer, tmp_path)) == []

    def test_required_field_list_matches_the_training_spec(self) -> None:
        for name in ("run_id", "git_commit", "config", "seed", "torch_version", "device"):
            assert name in REQUIRED_FIELDS

    def test_fingerprints_are_real(self, tokenizer, tmp_path: Path) -> None:
        record = self._record(tokenizer, tmp_path)
        assert record["charset_fingerprint"] == tokenizer.charset.fingerprint()
        assert record["parameter_count"] == 1_544_560

    def test_is_json_serializable(self, tokenizer, tmp_path: Path) -> None:
        json.dumps(self._record(tokenizer, tmp_path))

    def test_manifest_fingerprints_skip_missing_paths(
        self, synthetic_corpus, tmp_path: Path
    ) -> None:
        fingerprints = manifest_fingerprints(
            train=synthetic_corpus.manifest_path, val=None, test=tmp_path / "nope.jsonl"
        )
        assert set(fingerprints) == {"train"}
        assert len(fingerprints["train"]) == 64
