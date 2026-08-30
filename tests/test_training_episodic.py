"""V0 episodic training: the detachment boundary (support detached via `compile_profile`'s own
`no_grad`, query/fusion gradient-attached) proven directly, plus the training loop's basic
correctness on the synthetic corpus fixture.
"""

from __future__ import annotations

import copy

import pytest
import torch

from glyphmemory.config.schema import MemoryConfig
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.data.episodes import EpisodeSampler
from glyphmemory.model import GMBase
from glyphmemory.training.episodic import episodic_step, train_episodic_v0

MODEL_FINGERPRINT = "deadbeefcafef00d"


def _model_and_tokenizer():
    torch.manual_seed(0)
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    model = GMBase(vocab_size=tokenizer.vocab_size)
    return model, tokenizer


def _sampler_and_records(synthetic_corpus):
    # SYNTHETIC_LINES=4 per writer: query_size=1, support_sizes=(1,) always fits (need=2).
    sampler = EpisodeSampler(
        list(synthetic_corpus.records), query_size=1, support_sizes=(1,), seed=1337
    )
    records_by_id = {r.sample_id: r for r in synthetic_corpus.records}
    return sampler, records_by_id


def _memory_config() -> MemoryConfig:
    return MemoryConfig(
        enabled=True, feature_layer="sequence", pooling="posterior_weighted",
        prototype_strategy="mean", alpha=0.5,
    )


# ------------------------------------------------------------------ episodic_step


def test_episodic_step_requires_memory_config_enabled(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    episode = sampler.sample(synthetic_corpus.writers[0], draw_index=0)

    with pytest.raises(ValueError, match="enabled=True"):
        episodic_step(
            model, tokenizer, episode, records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            memory_config=MemoryConfig(enabled=False),
            device=torch.device("cpu"),
        )


def test_episodic_step_returns_a_finite_loss_and_a_profile(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    episode = sampler.sample(synthetic_corpus.writers[0], draw_index=0)

    loss, diagnostics, profile = episodic_step(
        model, tokenizer, episode, records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        memory_config=_memory_config(),
        device=torch.device("cpu"),
    )

    assert torch.isfinite(loss)
    assert diagnostics.batch_size == len(episode.query_ids)
    assert profile.characters  # at least some characters aligned from the support line(s)


def test_episodic_step_rejects_an_episode_with_no_query_lines(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    episode = sampler.sample(synthetic_corpus.writers[0], draw_index=0)
    from dataclasses import replace

    empty_query_episode = replace(episode, query_ids=())

    with pytest.raises(ValueError, match="no CTC-feasible query line"):
        episodic_step(
            model, tokenizer, empty_query_episode, records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            memory_config=_memory_config(),
            device=torch.device("cpu"),
        )


# ------------------------------------------------------------------ the detachment boundary


def test_support_profile_prototypes_carry_no_grad_tracking_inside_a_training_step(
    synthetic_corpus,
):
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    episode = sampler.sample(synthetic_corpus.writers[0], draw_index=0)

    with torch.enable_grad():
        loss, _diagnostics, profile = episodic_step(
            model, tokenizer, episode, records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            memory_config=_memory_config(),
            device=torch.device("cpu"),
        )

    # The support path's own detachment (compile_profile's internal no_grad) must hold even though
    # this whole step ran inside a gradient-enabled context -- the mirror-image half of the boundary
    # proof from the query-path test below.
    for glyph in profile.glyphs.values():
        assert glyph.prototype.requires_grad is False
        assert glyph.prototype.grad_fn is None

    assert loss.requires_grad  # the loss itself is still attached, via the query path


def test_query_path_gradient_reaches_the_model_parameters(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    episode = sampler.sample(synthetic_corpus.writers[0], draw_index=0)

    with torch.enable_grad():
        loss, _diagnostics, _profile = episodic_step(
            model, tokenizer, episode, records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            memory_config=_memory_config(),
            device=torch.device("cpu"),
        )
        loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no model parameter received a gradient -- the query path is not attached"
    assert any(torch.any(g != 0) for g in grads), "every gradient was exactly zero"


def test_episodic_step_restores_the_models_training_mode_after_the_support_pass(
    synthetic_corpus,
):
    # A training step must therefore find the model back in train() mode for its own query forward
    # pass -- otherwise BatchNorm/dropout would run in eval mode throughout an episodic "training"
    # step, silently.
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    episode = sampler.sample(synthetic_corpus.writers[0], draw_index=0)

    model.train()
    with torch.enable_grad():
        episodic_step(
            model, tokenizer, episode, records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            memory_config=_memory_config(),
            device=torch.device("cpu"),
        )
    assert model.training is True


# ------------------------------------------------------------------ train_episodic_v0


def test_train_episodic_v0_runs_the_requested_number_of_steps(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)

    log = train_episodic_v0(
        model, tokenizer, sampler, records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        n_steps=5,
        memory_config=_memory_config(),
        seed=1337,
        device="cpu",
    )

    assert log.n_steps == 5
    assert all(torch.isfinite(torch.tensor(s.loss)) for s in log.steps)


def test_train_episodic_v0_actually_changes_the_models_weights(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    before = copy.deepcopy(model.state_dict())

    train_episodic_v0(
        model, tokenizer, sampler, records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        n_steps=5,
        memory_config=_memory_config(),
        learning_rate=1e-2,
        seed=1337,
        device="cpu",
    )

    after = model.state_dict()
    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed, "no parameter changed after 5 episodic training steps"


def test_train_episodic_v0_is_deterministic_given_the_same_seed(synthetic_corpus):
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)

    model_a, tokenizer = _model_and_tokenizer()
    log_a = train_episodic_v0(
        model_a, tokenizer, sampler, records_by_id,
        model_fingerprint=MODEL_FINGERPRINT, n_steps=3,
        memory_config=_memory_config(), seed=1337, device="cpu",
    )

    model_b, tokenizer_b = _model_and_tokenizer()
    log_b = train_episodic_v0(
        model_b, tokenizer_b, sampler, records_by_id,
        model_fingerprint=MODEL_FINGERPRINT, n_steps=3,
        memory_config=_memory_config(), seed=1337, device="cpu",
    )

    assert [s.writer_id for s in log_a.steps] == [s.writer_id for s in log_b.steps]
    assert [s.loss for s in log_a.steps] == pytest.approx([s.loss for s in log_b.steps])


def test_train_episodic_v0_never_calls_torch_save(synthetic_corpus, monkeypatch):
    # A pure in-memory training function -- checkpoint saving is the caller's job (the real run
    # script). Monkeypatching torch.save to raise proves this behaviorally rather than by reading
    # the source: if this function ever gains a stray save call (e.g. reusing
    # `training/checkpoint.py`'s save path without redirecting it -- this phase's own risk table),
    # this test fails immediately rather than only being caught by a real-run's manual file-hash
    # check.
    def _forbidden_save(*args, **kwargs):
        raise AssertionError("train_episodic_v0 must never call torch.save")

    monkeypatch.setattr(torch, "save", _forbidden_save)

    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    train_episodic_v0(
        model, tokenizer, sampler, records_by_id,
        model_fingerprint=MODEL_FINGERPRINT, n_steps=3,
        memory_config=_memory_config(), seed=1337, device="cpu",
    )
