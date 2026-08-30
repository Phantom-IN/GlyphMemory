"""Gradient accumulation for episodic training (Configuration B).

The property that decides whether this is the intervention it claims to be: accumulating `k`
episodes must produce the group's **mean** gradient, not its sum. That is asserted directly here
against a hand-computed reference, not taken on trust from the division in the loop.
"""

from __future__ import annotations

import copy

import pytest
import torch

from glyphmemory.config.schema import MemoryConfig
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.data.episodes import EpisodeSampler, iter_writer_cycle
from glyphmemory.model import GMBase
from glyphmemory.training.episodic import episodic_step, train_episodic_v0

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


def test_accumulation_produces_the_mean_gradient_not_the_sum(synthetic_corpus):
    """The whole point of Configuration B: a larger effective batch, at an unchanged rate.

    Computed against a hand-rolled reference over the same episodes the loop would draw, so the
    assertion is about the gradient that actually reaches the parameters, not about the presence of
    a division in the source.
    """
    k = 3
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    model, tokenizer = _model_and_tokenizer()
    reference_model = copy.deepcopy(model)

    # Reproduce the loop's own draw order for one k-way group.
    cycle = iter_writer_cycle(sorted(sampler.writers), seed=1337)
    episodes = [sampler.sample(next(cycle), draw) for draw in range(k)]

    losses = []
    reference_model.train()
    reference_model.zero_grad(set_to_none=True)
    # Both paths run the same number of dropout-bearing forward passes in the same order, so seeding
    # identically before each makes them bit-comparable. Without this the two differ by dropout
    # noise alone and the assertion below would be about the wrong thing.
    torch.manual_seed(11)
    for episode in episodes:
        loss, _diagnostics, _profile = episodic_step(
            reference_model,
            tokenizer,
            episode,
            records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            memory_config=_memory_config(),
            device=torch.device("cpu"),
        )
        losses.append(loss)
    mean_gradients = torch.autograd.grad(
        sum(losses) / k, [p for p in reference_model.parameters() if p.requires_grad]
    )
    expected_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g) for g in mean_gradients])
    )

    torch.manual_seed(11)
    log = train_episodic_v0(
        model,
        tokenizer,
        sampler,
        records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        n_steps=1,
        memory_config=_memory_config(),
        accumulation_steps=k,
    )

    assert log.steps[0].grad_norm == pytest.approx(float(expected_norm), rel=1e-4)
    # And emphatically not the sum, which would be k times larger.
    assert log.steps[0].grad_norm != pytest.approx(k * float(expected_norm), rel=1e-2)


def test_n_steps_counts_optimizer_steps_and_n_episodes_counts_data(synthetic_corpus):
    """`n_steps` alone would silently overstate how little data a k-way run touched (or understate
    it) — the confound this phase's own risk table names.
    """
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    model, tokenizer = _model_and_tokenizer()

    log = train_episodic_v0(
        model,
        tokenizer,
        sampler,
        records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        n_steps=3,
        memory_config=_memory_config(),
        accumulation_steps=4,
    )

    assert log.n_steps == 3
    assert log.n_episodes == 12
    assert all(step.episodes == 4 for step in log.steps)
    assert log.as_dict()["n_episodes"] == 12


def test_accumulated_step_records_the_group_total_and_mean(synthetic_corpus):
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    model, tokenizer = _model_and_tokenizer()

    log = train_episodic_v0(
        model,
        tokenizer,
        sampler,
        records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        n_steps=1,
        memory_config=_memory_config(),
        accumulation_steps=2,
    )

    step = log.steps[0]
    assert step.query_lines == 2  # total across the group (query_size=1 each), not one episode's
    assert step.episodes == 2


def test_accumulation_of_one_is_the_original_loop(synthetic_corpus):
    """Internal helper."""
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)

    plain_model, tokenizer = _model_and_tokenizer()
    explicit_model = copy.deepcopy(plain_model)
    shared = {
        "model_fingerprint": MODEL_FINGERPRINT,
        "n_steps": 3,
        "memory_config": _memory_config(),
    }

    torch.manual_seed(0)
    plain = train_episodic_v0(plain_model, tokenizer, sampler, records_by_id, **shared)
    torch.manual_seed(0)
    explicit = train_episodic_v0(
        explicit_model, tokenizer, sampler, records_by_id, accumulation_steps=1, **shared
    )

    assert [s.loss for s in plain.steps] == pytest.approx([s.loss for s in explicit.steps])
    for name, tensor in plain_model.state_dict().items():
        assert torch.equal(tensor, explicit_model.state_dict()[name]), name


def test_accumulation_must_be_at_least_one(synthetic_corpus):
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    model, tokenizer = _model_and_tokenizer()

    with pytest.raises(ValueError, match="accumulation_steps must be at least 1"):
        train_episodic_v0(
            model,
            tokenizer,
            sampler,
            records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            n_steps=1,
            memory_config=_memory_config(),
            accumulation_steps=0,
        )
