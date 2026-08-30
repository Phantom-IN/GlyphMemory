"""Learning-rate schedule wiring for episodic training.

A schedule is a different intervention: it changes what the *early* steps do.
"""

from __future__ import annotations

import pytest
import torch

from glyphmemory.config.schema import MemoryConfig
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.data.episodes import EpisodeSampler
from glyphmemory.model import GMBase
from glyphmemory.training.episodic import DEFAULT_LEARNING_RATE, train_episodic_v0
from glyphmemory.training.schedule import WarmupCosine

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


def _train(synthetic_corpus, *, n_steps, schedule=None, learning_rate=DEFAULT_LEARNING_RATE):
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    model, tokenizer = _model_and_tokenizer()
    return train_episodic_v0(
        model,
        tokenizer,
        sampler,
        records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        n_steps=n_steps,
        memory_config=_memory_config(),
        learning_rate=learning_rate,
        schedule=schedule,
    )


def test_warmup_ramps_the_rate_the_optimizer_actually_uses(synthetic_corpus):
    """The ramp must reach the optimizer, not merely be described alongside it."""
    schedule = WarmupCosine(total_steps=4, warmup_steps=4, min_factor=1.0)

    log = _train(synthetic_corpus, n_steps=4, schedule=schedule, learning_rate=1e-4)

    rates = [s.learning_rate for s in log.steps]
    assert rates == pytest.approx([0.25e-4, 0.5e-4, 0.75e-4, 1e-4])


def test_warmup_only_schedule_holds_the_base_rate_after_the_ramp(synthetic_corpus):
    """`min_factor=1.0` makes `WarmupCosine` a warmup-*only* schedule: its post-warmup factor is
    exactly 1.0.
    """
    schedule = WarmupCosine(total_steps=6, warmup_steps=2, min_factor=1.0)

    log = _train(synthetic_corpus, n_steps=6, schedule=schedule, learning_rate=1e-4)

    rates = [s.learning_rate for s in log.steps]
    assert rates[:2] == pytest.approx([0.5e-4, 1e-4])
    assert rates[2:] == pytest.approx([1e-4] * 4)  # no decay


def test_cosine_decay_is_available_and_actually_decays(synthetic_corpus):
    schedule = WarmupCosine(total_steps=6, warmup_steps=2, min_factor=0.01)

    log = _train(synthetic_corpus, n_steps=6, schedule=schedule, learning_rate=1e-4)

    rates = [s.learning_rate for s in log.steps]
    assert rates[1] == pytest.approx(1e-4)  # end of warmup
    assert rates[-1] < rates[2]  # and then genuinely decays


def test_an_unscheduled_run_holds_a_flat_rate(synthetic_corpus):
    """Internal helper."""
    log = _train(synthetic_corpus, n_steps=3, learning_rate=1e-4)

    assert log.schedule is None
    assert [s.learning_rate for s in log.steps] == pytest.approx([1e-4] * 3)


def test_a_schedule_with_the_wrong_horizon_is_refused(synthetic_corpus):
    """A cosine computed against the wrong horizon decays to the wrong place without ever erroring
    (`training/schedule.py`'s own module docstring) — refused, not silently mistrained.
    """
    schedule = WarmupCosine(total_steps=999, warmup_steps=10, min_factor=1.0)

    with pytest.raises(ValueError, match="must equal n_steps"):
        _train(synthetic_corpus, n_steps=3, schedule=schedule)


def test_the_run_log_records_the_schedule_it_used(synthetic_corpus):
    schedule = WarmupCosine(total_steps=3, warmup_steps=1, min_factor=1.0)

    log = _train(synthetic_corpus, n_steps=3, schedule=schedule)

    assert log.schedule is schedule
    described = log.as_dict()["schedule"]
    assert described["warmup_steps"] == 1
    assert described["total_steps"] == 3


def test_warmup_changes_the_weights_a_run_reaches(synthetic_corpus):
    """A sanity check that the schedule is a real intervention on this loop, not a no-op: the same
    seed and steps with a warmup must not land on the same weights as without one.
    """
    schedule = WarmupCosine(total_steps=3, warmup_steps=3, min_factor=1.0)

    torch.manual_seed(0)
    plain = _train(synthetic_corpus, n_steps=3)
    torch.manual_seed(0)
    warmed = _train(synthetic_corpus, n_steps=3, schedule=schedule)

    assert [s.learning_rate for s in plain.steps] != [s.learning_rate for s in warmed.steps]
    assert [s.loss for s in plain.steps] != pytest.approx([s.loss for s in warmed.steps])
