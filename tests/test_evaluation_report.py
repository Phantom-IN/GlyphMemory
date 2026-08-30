"""``evaluate_checkpoint`` end to end, on the synthetic corpus and a freshly saved checkpoint.

Correctness of the *recognition* is not the point here — the model is randomly initialized, so its
CER will be near total. The point is that the plumbing (checkpoint load, dataset/loader, inference,
CER/WER, per-writer distribution, taxonomy, gate) all runs and produces a self-consistent,
reproducible report.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from glyphmemory.config.schema import Config
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.evaluation import CER_GATE_THRESHOLD, TAIL_GATE_THRESHOLD, evaluate_checkpoint
from glyphmemory.model import GMBase
from glyphmemory.training.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointMeta,
    save_checkpoint,
)

VOCAB_CHARSET = DEFAULT_CHARSET_PATH


def _save_checkpoint(tmp_path: Path, tokenizer, vocab_size: int) -> Path:
    torch.manual_seed(0)
    model = GMBase(vocab_size=vocab_size)
    meta = CheckpointMeta(
        epoch=1,
        step=1,
        metrics={"val_cer": 1.0},
        charset_fingerprint=tokenizer.charset.fingerprint(),
        tokenizer_fingerprint=tokenizer.fingerprint(),
        manifest_fingerprints={},
        config={},
        parameter_count=sum(p.numel() for p in model.parameters()),
        git_commit=None,
        seed=0,
        run_id="test_run",
    )
    return save_checkpoint(tmp_path / "checkpoint.pt", model=model, meta=meta)


def test_evaluate_checkpoint_produces_a_self_consistent_report(tmp_path, synthetic_corpus):
    tokenizer = load_tokenizer(VOCAB_CHARSET)
    checkpoint = _save_checkpoint(tmp_path, tokenizer, tokenizer.vocab_size)

    report = evaluate_checkpoint(
        checkpoint,
        synthetic_corpus.manifest_path,
        config=Config(),
        tokenizer=tokenizer,
        split="train",
        device="cpu",
        batch_size=4,
        num_workers=0,
    )

    assert report.cer.n_samples == len(synthetic_corpus.records)
    assert report.per_writer.n_writers == len(synthetic_corpus.writers)
    assert report.taxonomy.lines == len(synthetic_corpus.records)
    assert report.gate.cer_threshold == CER_GATE_THRESHOLD
    assert report.gate.tail_threshold == TAIL_GATE_THRESHOLD
    # Condition 2 always starts unreviewed — a human, not this function, decides it.
    assert report.gate.pipeline_bug_reviewed is None
    assert report.gate.passed is None


def test_report_round_trips_through_as_dict(tmp_path, synthetic_corpus):
    tokenizer = load_tokenizer(VOCAB_CHARSET)
    checkpoint = _save_checkpoint(tmp_path, tokenizer, tokenizer.vocab_size)
    report = evaluate_checkpoint(
        checkpoint,
        synthetic_corpus.manifest_path,
        config=Config(),
        tokenizer=tokenizer,
        split="train",
        device="cpu",
        batch_size=4,
        num_workers=0,
    )
    payload = report.as_dict()
    assert payload["cer"]["value"] == report.cer.value
    assert payload["per_writer"]["n_writers"] == report.per_writer.n_writers
    assert "single_glyph_confusion_definition" in payload["taxonomy"]
    assert payload["gate"]["overall_passed"] is None


def test_gate_reviewed_records_the_human_verdict(tmp_path, synthetic_corpus):
    tokenizer = load_tokenizer(VOCAB_CHARSET)
    checkpoint = _save_checkpoint(tmp_path, tokenizer, tokenizer.vocab_size)
    report = evaluate_checkpoint(
        checkpoint,
        synthetic_corpus.manifest_path,
        config=Config(),
        tokenizer=tokenizer,
        split="train",
        device="cpu",
        batch_size=4,
        num_workers=0,
    )
    reviewed = report.gate.reviewed(no_pipeline_bug=True, notes="errors look like glyph confusion")
    assert reviewed.pipeline_bug_reviewed is True
    # Overall verdict now resolves to a concrete bool (cer_pass is almost certainly False here,
    # since the model is untrained, so the whole gate should read as failed, not None).
    assert reviewed.passed is False


def test_evaluate_checkpoint_refuses_a_charset_mismatch(tmp_path, synthetic_corpus):
    tokenizer = load_tokenizer(VOCAB_CHARSET)
    checkpoint = _save_checkpoint(tmp_path, tokenizer, tokenizer.vocab_size)

    from glyphmemory.ctc.tokenizer import Charset, Tokenizer

    other = Tokenizer(charset=Charset.from_texts(["abc"], name="tiny"))
    with pytest.raises(CheckpointCompatibilityError):
        evaluate_checkpoint(
            checkpoint,
            synthetic_corpus.manifest_path,
            config=Config(),
            tokenizer=other,
            split="train",
            device="cpu",
        )


def test_evaluate_checkpoint_rejects_a_split_with_no_records(tmp_path, synthetic_corpus):
    tokenizer = load_tokenizer(VOCAB_CHARSET)
    checkpoint = _save_checkpoint(tmp_path, tokenizer, tokenizer.vocab_size)
    with pytest.raises(ValueError, match="no records"):
        evaluate_checkpoint(
            checkpoint,
            synthetic_corpus.manifest_path,
            config=Config(),
            tokenizer=tokenizer,
            split="test",
            device="cpu",
        )
