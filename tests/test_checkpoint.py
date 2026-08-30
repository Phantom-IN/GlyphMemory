"""Checkpoint tests.

The charset-mismatch refusal is the one that matters. Everything else here guards against losing
evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.model import GMBase
from glyphmemory.training.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    SELECTION_METRIC,
    CheckpointCompatibilityError,
    CheckpointMeta,
    is_better,
    load_checkpoint,
    restore_model,
    save_checkpoint,
)

VOCAB = 80


@pytest.fixture
def tokenizer():
    return load_tokenizer(DEFAULT_CHARSET_PATH)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GMBase(vocab_size=VOCAB)


def meta_for(tokenizer, **overrides) -> CheckpointMeta:
    base = {
        "epoch": 3,
        "step": 42,
        "metrics": {SELECTION_METRIC: 0.25},
        "charset_fingerprint": tokenizer.charset.fingerprint(),
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "manifest_fingerprints": {"train": "abc123"},
        "config": {"model": {"name": "gm_base"}},
        "parameter_count": 1_544_560,
        "git_commit": "deadbeef",
        "seed": 1337,
        "run_id": "gm_base__20260818T000000Z",
    }
    return CheckpointMeta(**{**base, **overrides})


class TestRoundTrip:
    def test_weights_are_restored_exactly(self, tmp_path: Path, model, tokenizer) -> None:
        path = save_checkpoint(tmp_path / "c.pt", model=model, meta=meta_for(tokenizer))

        restored = GMBase(vocab_size=VOCAB)
        restore_model(restored, load_checkpoint(path))

        model.eval()
        restored.eval()
        images = torch.randn(2, 1, 64, 256)
        with torch.no_grad():
            assert torch.equal(model(images).logits, restored(images).logits)

    def test_metadata_survives(self, tmp_path: Path, model, tokenizer) -> None:
        path = save_checkpoint(tmp_path / "c.pt", model=model, meta=meta_for(tokenizer))
        loaded = load_checkpoint(path)
        assert loaded.meta.epoch == 3
        assert loaded.meta.step == 42
        assert loaded.meta.git_commit == "deadbeef"
        assert loaded.meta.seed == 1337
        assert loaded.meta.parameter_count == 1_544_560
        assert loaded.meta.manifest_fingerprints == {"train": "abc123"}
        assert loaded.meta.selection_value == pytest.approx(0.25)

    def test_optimizer_and_scheduler_state_are_stored(
        self, tmp_path: Path, model, tokenizer
    ) -> None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        path = save_checkpoint(
            tmp_path / "c.pt",
            model=model,
            meta=meta_for(tokenizer),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        loaded = load_checkpoint(path)
        assert loaded.optimizer_state is not None
        assert loaded.scheduler_state is not None

    def test_the_real_scheduler_is_serializable(self, tmp_path: Path, model, tokenizer) -> None:
        """``WarmupCosine`` is a callable object, and ``LambdaLR`` serializes those via ``__dict__``
        — which a slotted dataclass does not have. Pinned because it fails only at save time,
        several epochs into a run.
        """
        from glyphmemory.config.schema import TrainingConfig
        from glyphmemory.training.schedule import build_scheduler

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler, _ = build_scheduler(optimizer, TrainingConfig(), total_steps=10)
        path = save_checkpoint(
            tmp_path / "c.pt", model=model, meta=meta_for(tokenizer), scheduler=scheduler
        )
        assert load_checkpoint(path).scheduler_state is not None


class TestCharsetCompatibility:
    def test_matching_fingerprint_loads(self, tmp_path: Path, model, tokenizer) -> None:
        path = save_checkpoint(tmp_path / "c.pt", model=model, meta=meta_for(tokenizer))
        loaded = load_checkpoint(path, charset_fingerprint=tokenizer.charset.fingerprint())
        assert loaded.meta.epoch == 3

    def test_mismatched_fingerprint_refuses(self, tmp_path: Path, model, tokenizer) -> None:
        """Silent catastrophe otherwise: the same logits decode into a different alphabet."""
        path = save_checkpoint(tmp_path / "c.pt", model=model, meta=meta_for(tokenizer))
        with pytest.raises(CheckpointCompatibilityError, match="trained against charset"):
            load_checkpoint(path, charset_fingerprint="a" * 64)

    def test_mismatch_can_be_inspected_deliberately(self, tmp_path: Path, model, tokenizer) -> None:
        path = save_checkpoint(tmp_path / "c.pt", model=model, meta=meta_for(tokenizer))
        loaded = load_checkpoint(path, charset_fingerprint="a" * 64, strict_charset=False)
        assert loaded.meta.charset_fingerprint == tokenizer.charset.fingerprint()

    def test_no_fingerprint_given_skips_the_check(self, tmp_path: Path, model, tokenizer) -> None:
        path = save_checkpoint(tmp_path / "c.pt", model=model, meta=meta_for(tokenizer))
        assert load_checkpoint(path).meta.epoch == 3

    def test_schema_version_mismatch_refuses(self, tmp_path: Path, model, tokenizer) -> None:
        path = save_checkpoint(
            tmp_path / "c.pt", model=model, meta=meta_for(tokenizer, schema_version="0")
        )
        with pytest.raises(CheckpointCompatibilityError, match="checkpoint schema"):
            load_checkpoint(path)

    def test_current_schema_version_is_recorded(self, tokenizer) -> None:
        assert meta_for(tokenizer).schema_version == CHECKPOINT_SCHEMA_VERSION

    def test_rejects_a_file_that_is_not_a_checkpoint(self, tmp_path: Path) -> None:
        path = tmp_path / "notacheckpoint.pt"
        torch.save({"weights": torch.zeros(3)}, path)
        with pytest.raises(CheckpointCompatibilityError, match="does not look like"):
            load_checkpoint(path)


class TestAtomicity:
    def test_no_temporary_file_survives_a_successful_save(
        self, tmp_path: Path, model, tokenizer
    ) -> None:
        save_checkpoint(tmp_path / "c.pt", model=model, meta=meta_for(tokenizer))
        assert [p.name for p in tmp_path.iterdir()] == ["c.pt"]

    def test_a_failed_save_leaves_no_partial_file(
        self, tmp_path: Path, model, tokenizer, monkeypatch
    ) -> None:
        """An interrupted save must not leave a truncated file with a plausible name."""
        import glyphmemory.training.checkpoint as module

        def explode(*args, **kwargs):
            raise KeyboardInterrupt("simulated interruption mid-write")

        monkeypatch.setattr(module.torch, "save", explode)
        with pytest.raises(KeyboardInterrupt):
            save_checkpoint(tmp_path / "c.pt", model=model, meta=meta_for(tokenizer))
        assert list(tmp_path.iterdir()) == []

    def test_an_existing_checkpoint_is_replaced_not_corrupted(
        self, tmp_path: Path, model, tokenizer
    ) -> None:
        path = tmp_path / "c.pt"
        save_checkpoint(path, model=model, meta=meta_for(tokenizer, epoch=1))
        save_checkpoint(path, model=model, meta=meta_for(tokenizer, epoch=2))
        assert load_checkpoint(path).meta.epoch == 2


class TestSelection:
    def test_lower_is_better(self) -> None:
        assert is_better(0.1, 0.2)
        assert not is_better(0.3, 0.2)
        assert not is_better(0.2, 0.2)

    def test_first_value_always_wins(self) -> None:
        assert is_better(0.9, None)

    def test_none_never_wins(self) -> None:
        """A validation that produced no value cannot displace a real result."""
        assert not is_better(None, 0.5)
        assert not is_better(None, None)

    def test_selection_metric_is_cer_not_loss(self) -> None:
        """Named in one place so 'best' cannot drift into meaning 'lowest loss'."""
        assert SELECTION_METRIC == "val_cer"
