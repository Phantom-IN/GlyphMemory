"""The periodic generic-recognition probe for episodic training."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from glyphmemory.config.schema import Config, MemoryConfig
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.data.episodes import EpisodeSampler
from glyphmemory.data.splits import SplitLeakError
from glyphmemory.model import GMBase
from glyphmemory.training.episodic import train_episodic_v0
from glyphmemory.training.episodic_validation import (
    HARNESS_WRITERS,
    ValidationProbe,
    select_probe_records,
)

MODEL_FINGERPRINT = "deadbeefcafef00d"


def _val_records(synthetic_corpus):
    """The synthetic corpus relabelled as a validation split — the probe only ever reads a held-out
    split, and the shared fixture is all-train by construction.
    """
    return tuple(replace(r, split="val") for r in synthetic_corpus.records)


def _model_and_tokenizer():
    torch.manual_seed(0)
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    return GMBase(vocab_size=tokenizer.vocab_size), tokenizer


def _memory_config() -> MemoryConfig:
    return MemoryConfig(
        enabled=True,
        feature_layer="sequence",
        pooling="posterior_weighted",
        prototype_strategy="mean",
        alpha=0.5,
    )


# ------------------------------------------------------------------ select_probe_records


def test_selects_the_requested_shape(synthetic_corpus):
    records = _val_records(synthetic_corpus)

    selected = select_probe_records(records, n_writers=3, lines_per_writer=2, exclude_writers=())

    assert len(selected) == 6
    chosen = {r.writer_id for r in selected}
    per_writer = {w: sum(1 for r in selected if r.writer_id == w) for w in chosen}
    assert len(per_writer) == 3
    assert set(per_writer.values()) == {2}  # every writer contributes equally, not one dominating


def test_excludes_the_harness_writers(synthetic_corpus):
    """The probe must never see `iam/025`/`iam/058`/`iam/061` — selects checkpoints on this signal,
    and those 3 writers are what the final adaptation-gain verdict is measured on.
    """
    base = _val_records(synthetic_corpus)
    writers = sorted({r.writer_id for r in base})
    harness = sorted(HARNESS_WRITERS)[0]
    # Relabel one synthetic writer with a real harness ID: it is now the *only* writer that must not
    # be selected, and every remaining writer is asked for, so a leak cannot hide in slack.
    records = tuple(
        replace(r, writer_id=harness if r.writer_id == writers[0] else r.writer_id) for r in base
    )
    assert harness in {r.writer_id for r in records}

    selected = select_probe_records(
        records, n_writers=len(writers) - 1, lines_per_writer=1, seed=7
    )

    assert not ({r.writer_id for r in selected} & HARNESS_WRITERS)


def test_only_reads_the_requested_split(synthetic_corpus):
    records = tuple(_val_records(synthetic_corpus)) + tuple(synthetic_corpus.records)

    selected = select_probe_records(records, n_writers=2, lines_per_writer=2, exclude_writers=())

    assert {r.split for r in selected} == {"val"}


def test_is_deterministic(synthetic_corpus):
    records = _val_records(synthetic_corpus)
    kwargs = {"n_writers": 3, "lines_per_writer": 2, "exclude_writers": (), "seed": 99}

    first = select_probe_records(records, **kwargs)
    second = select_probe_records(records, **kwargs)

    assert [r.sample_id for r in first] == [r.sample_id for r in second]


def test_refuses_a_smaller_probe_than_asked_for(synthetic_corpus):
    records = _val_records(synthetic_corpus)
    n_writers = len({r.writer_id for r in records})

    with pytest.raises(ValueError, match="asked for"):
        select_probe_records(
            records, n_writers=n_writers + 1, lines_per_writer=1, exclude_writers=()
        )


def test_rejects_non_positive_sizes(synthetic_corpus):
    with pytest.raises(ValueError, match="must both be positive"):
        select_probe_records(_val_records(synthetic_corpus), n_writers=0, exclude_writers=())


# ------------------------------------------------------------------ ValidationProbe


def test_probe_scores_the_subset_and_restores_training_mode(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    records = select_probe_records(
        _val_records(synthetic_corpus), n_writers=2, lines_per_writer=2, exclude_writers=()
    )
    probe = ValidationProbe(records, tokenizer, Config())
    model.train()

    check = probe.evaluate(model, step=17)

    assert check.step == 17
    assert check.n_lines == 4
    assert check.cer is not None and check.cer >= 0.0
    assert check.wer is not None and check.wer >= 0.0
    assert check.seconds > 0.0
    assert model.training  # the loop that called this is still a training loop afterwards


def test_probe_is_deterministic_across_calls(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    records = select_probe_records(
        _val_records(synthetic_corpus), n_writers=2, lines_per_writer=2, exclude_writers=()
    )
    probe = ValidationProbe(records, tokenizer, Config())

    first = probe.evaluate(model, step=0)
    second = probe.evaluate(model, step=0)

    assert first.cer == second.cer
    assert first.wer == second.wer


def test_probe_reports_its_own_writers_and_refuses_an_overlap(synthetic_corpus):
    _model, tokenizer = _model_and_tokenizer()
    records = select_probe_records(
        _val_records(synthetic_corpus), n_writers=2, lines_per_writer=2, exclude_writers=()
    )
    probe = ValidationProbe(records, tokenizer, Config())

    probe.assert_disjoint_from(HARNESS_WRITERS)  # synthetic writers are not IAM writers
    with pytest.raises(SplitLeakError, match="overlaps"):
        probe.assert_disjoint_from(probe.writers)


def test_probe_rejects_an_empty_subset(synthetic_corpus):
    _model, tokenizer = _model_and_tokenizer()

    with pytest.raises(ValueError, match="at least one record"):
        ValidationProbe((), tokenizer, Config())


# ------------------------------------------------------------------ loop instrumentation


def _sampler_and_records(synthetic_corpus):
    sampler = EpisodeSampler(
        list(synthetic_corpus.records), query_size=1, support_sizes=(1,), seed=1337
    )
    return sampler, {r.sample_id: r for r in synthetic_corpus.records}


def _probe(synthetic_corpus, tokenizer) -> ValidationProbe:
    records = select_probe_records(
        _val_records(synthetic_corpus), n_writers=2, lines_per_writer=1, exclude_writers=()
    )
    return ValidationProbe(records, tokenizer, Config())


def test_probing_does_not_change_what_the_run_trains_into(synthetic_corpus):
    """The whole diagnostic rests on this: an instrumented run must be the *same* run.

    The probe takes no optimizer step, consumes no RNG (it runs in ``eval()``, where the head's
    dropout — the training path's one global-RNG consumer — is a no-op, and its batches are
    preprocessed once at construction and never augmented) and restores the model's training mode,
    so the weights after N steps must be bit-identical with and without it.

    Both runs re-seed the global RNG first, for the same reason
    `test_train_episodic_v0_is_deterministic_given_the_same_seed` does: training-mode dropout
    advances the global generator, so two back-to-back runs in one process do not start from the
    same state on their own. That is a property of the loop, not of the probe.
    """
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)

    plain_model, tokenizer = _model_and_tokenizer()
    probed_model = copy.deepcopy(plain_model)
    probe = _probe(synthetic_corpus, tokenizer)

    shared = {
        "model_fingerprint": MODEL_FINGERPRINT,
        "n_steps": 4,
        "memory_config": _memory_config(),
        "seed": 1337,
    }
    torch.manual_seed(0)
    train_episodic_v0(plain_model, tokenizer, sampler, records_by_id, **shared)
    torch.manual_seed(0)
    probed_log = train_episodic_v0(
        probed_model,
        tokenizer,
        sampler,
        records_by_id,
        probe=probe,
        probe_every=2,
        **shared,
    )

    assert probed_log.probe_checks  # the probe really did run
    for name, plain_param in plain_model.state_dict().items():
        assert torch.equal(plain_param, probed_model.state_dict()[name]), name


def test_probe_checks_cover_step_zero_and_the_final_step(synthetic_corpus):
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    model, tokenizer = _model_and_tokenizer()

    log = train_episodic_v0(
        model,
        tokenizer,
        sampler,
        records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        n_steps=5,
        memory_config=_memory_config(),
        probe=_probe(synthetic_corpus, tokenizer),
        probe_every=2,
    )

    # step 0 (before any update) then every 2nd step, plus the final step even though 5 % 2 != 0
    assert [c.step for c in log.probe_checks] == [0, 2, 4, 5]


def test_probe_time_is_reported_separately_from_training_time(synthetic_corpus):
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    model, tokenizer = _model_and_tokenizer()

    log = train_episodic_v0(
        model,
        tokenizer,
        sampler,
        records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        n_steps=2,
        memory_config=_memory_config(),
        probe=_probe(synthetic_corpus, tokenizer),
        probe_every=1,
    )

    assert log.probe_seconds > 0.0
    assert log.training_seconds == pytest.approx(log.seconds - log.probe_seconds)
    assert log.as_dict()["probe_checks"][0]["step"] == 0


def test_an_uninstrumented_run_records_no_probe_checks(synthetic_corpus):
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    model, tokenizer = _model_and_tokenizer()

    log = train_episodic_v0(
        model,
        tokenizer,
        sampler,
        records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        n_steps=2,
        memory_config=_memory_config(),
    )

    assert log.probe_checks == ()
    assert log.probe_seconds == 0.0
    assert log.training_seconds == log.seconds


def test_probe_every_must_be_positive_when_a_probe_is_given(synthetic_corpus):
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    model, tokenizer = _model_and_tokenizer()

    with pytest.raises(ValueError, match="probe_every must be positive"):
        train_episodic_v0(
            model,
            tokenizer,
            sampler,
            records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            n_steps=1,
            memory_config=_memory_config(),
            probe=_probe(synthetic_corpus, tokenizer),
            probe_every=0,
        )
