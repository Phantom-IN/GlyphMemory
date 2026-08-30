"""An unalignable **support** line must not kill an episodic training run (Log)."""

from __future__ import annotations

import pytest
import torch

from glyphmemory.alignment import AlignmentInfeasibleError
from glyphmemory.config.schema import MemoryConfig
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.data.episodes import EpisodeSampler
from glyphmemory.model import GMBase
from glyphmemory.training import episodic as episodic_module
from glyphmemory.training.episodic import train_episodic_v0

MODEL_FINGERPRINT = "deadbeefcafef00d"


def _model_and_tokenizer():
    torch.manual_seed(0)
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    return GMBase(vocab_size=tokenizer.vocab_size), tokenizer


def _sampler_and_records(synthetic_corpus):
    sampler = EpisodeSampler(
        list(synthetic_corpus.records), query_size=1, support_sizes=(1,), seed=1337
    )
    return sampler, {r.sample_id: r for r in synthetic_corpus.records}


def _memory_config() -> MemoryConfig:
    return MemoryConfig(
        enabled=True,
        feature_layer="sequence",
        pooling="posterior_weighted",
        prototype_strategy="mean",
        alpha=0.5,
    )


def _failing_every_other_step(monkeypatch):
    """Make `episodic_step` raise `AlignmentInfeasibleError` on every other call."""
    real = episodic_module.episodic_step
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            raise AlignmentInfeasibleError(
                "support line sample_id='synthetic/w000-l000' path='x.png': target of 56 tokens "
                "needs 113 lattice states but only 92 frames are available"
            )
        return real(*args, **kwargs)

    monkeypatch.setattr(episodic_module, "episodic_step", flaky)
    return calls


def test_an_unalignable_support_line_is_skipped_and_counted_not_fatal(
    synthetic_corpus, monkeypatch, caplog
):
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    _failing_every_other_step(monkeypatch)

    with caplog.at_level("WARNING"):
        log = train_episodic_v0(
            model,
            tokenizer,
            sampler,
            records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            n_steps=3,
            memory_config=_memory_config(),
        )

    assert log.n_steps == 3  # the run still delivers the steps it was asked for
    assert log.skipped_alignment == 3  # and does not hide what it had to drop
    assert log.skipped_steps == 3
    assert log.as_dict()["skipped_alignment"] == 3
    # Invariant 7: traceable to a sample, with a reason — not a bare count.
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "alignment_infeasible" in logged
    assert "support_ids" in logged
    assert "sample_id='synthetic/w000-l000'" in logged


def test_a_systematically_infeasible_configuration_fails_loudly(synthetic_corpus, monkeypatch):
    """The counterpart to skipping: never quietly spin forever, or train on a fraction of the data
    the caller asked for without saying so.
    """
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)

    def always_fails(*args, **kwargs):
        raise AlignmentInfeasibleError("every support line is unalignable")

    monkeypatch.setattr(episodic_module, "episodic_step", always_fails)

    with pytest.raises(RuntimeError, match="consecutive unusable draws"):
        train_episodic_v0(
            model,
            tokenizer,
            sampler,
            records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            n_steps=1,
            memory_config=_memory_config(),
        )


def test_a_clean_run_reports_no_alignment_skips(synthetic_corpus):
    """Internal helper."""
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)

    log = train_episodic_v0(
        model,
        tokenizer,
        sampler,
        records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        n_steps=3,
        memory_config=_memory_config(),
    )

    assert log.skipped_alignment == 0
    assert log.skipped_steps == 0
