"""V1 episodic training: the mirror-image detachment boundary from V0 -- gradient must reach the
support path's prototypes too, not just query/fusion -- proven directly, plus the training loop's
basic correctness.
"""

from __future__ import annotations

import copy

import pytest
import torch

from glyphmemory.config.schema import MemoryConfig
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.data.episodes import EpisodeSampler
from glyphmemory.model import GMBase
from glyphmemory.training.episodic import episodic_step_v1, train_episodic_v1

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


# ------------------------------------------------------------------ episodic_step_v1


def test_episodic_step_v1_requires_memory_config_enabled(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    episode = sampler.sample(synthetic_corpus.writers[0], draw_index=0)

    with pytest.raises(ValueError, match="enabled=True"):
        episodic_step_v1(
            model, tokenizer, episode, records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            memory_config=MemoryConfig(enabled=False),
            device=torch.device("cpu"),
        )


def test_episodic_step_v1_rejects_a_non_mean_prototype_strategy(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    episode = sampler.sample(synthetic_corpus.writers[0], draw_index=0)

    with pytest.raises(ValueError, match="prototype_strategy"):
        episodic_step_v1(
            model, tokenizer, episode, records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            memory_config=MemoryConfig(enabled=True, prototype_strategy="top_k"),
            device=torch.device("cpu"),
        )


def test_episodic_step_v1_rejects_an_unknown_feature_layer(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    episode = sampler.sample(synthetic_corpus.writers[0], draw_index=0)

    with pytest.raises(ValueError, match="feature_layer"):
        episodic_step_v1(
            model, tokenizer, episode, records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            memory_config=MemoryConfig(enabled=True, feature_layer="learned_projection"),
            device=torch.device("cpu"),
        )


def test_episodic_step_v1_returns_a_finite_loss_and_a_profile(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    episode = sampler.sample(synthetic_corpus.writers[0], draw_index=0)

    loss, diagnostics, profile = episodic_step_v1(
        model, tokenizer, episode, records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        memory_config=_memory_config(),
        device=torch.device("cpu"),
    )

    assert torch.isfinite(loss)
    assert diagnostics.batch_size == len(episode.query_ids)
    assert profile.characters


# ------------------------------------------------------------------ the (mirror-image) boundary


def test_v1_support_prototypes_carry_live_gradient_tracking(synthetic_corpus):
    # The opposite of V0's own boundary test: V1's whole point is that the support path is NOT
    # detached, so its prototypes must carry real gradient tracking back to the encoder.
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    episode = sampler.sample(synthetic_corpus.writers[0], draw_index=0)

    with torch.enable_grad():
        loss, _diagnostics, profile = episodic_step_v1(
            model, tokenizer, episode, records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            memory_config=_memory_config(),
            device=torch.device("cpu"),
        )

    for glyph in profile.glyphs.values():
        assert glyph.prototype.requires_grad is True
        assert glyph.prototype.grad_fn is not None

    assert loss.requires_grad


def test_v1_gradient_from_the_support_path_reaches_model_parameters(synthetic_corpus):
    # Stronger than V0's query-only check: prove gradient flows via the SUPPORT path specifically,
    # by comparing the encoder's gradient with vs. without the support-side contribution to the
    # loss. Summing just the profile's prototypes (not the CTC loss) and backpropagating isolates
    # the support path exactly.
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    episode = sampler.sample(synthetic_corpus.writers[0], draw_index=0)

    with torch.enable_grad():
        _loss, _diagnostics, profile = episodic_step_v1(
            model, tokenizer, episode, records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            memory_config=_memory_config(),
            device=torch.device("cpu"),
        )
        prototype_sum = sum(g.prototype.sum() for g in profile.glyphs.values())
        prototype_sum.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no model parameter received a gradient from the support path"
    assert any(torch.any(g != 0) for g in grads), "every support-path gradient was exactly zero"


def test_v1_query_path_gradient_still_reaches_the_model_parameters(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    episode = sampler.sample(synthetic_corpus.writers[0], draw_index=0)

    with torch.enable_grad():
        loss, _diagnostics, _profile = episodic_step_v1(
            model, tokenizer, episode, records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            memory_config=_memory_config(),
            device=torch.device("cpu"),
        )
        loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert any(torch.any(g != 0) for g in grads)


def test_v1_restores_the_models_training_mode(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    episode = sampler.sample(synthetic_corpus.writers[0], draw_index=0)

    model.train()
    with torch.enable_grad():
        episodic_step_v1(
            model, tokenizer, episode, records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            memory_config=_memory_config(),
            device=torch.device("cpu"),
        )
    assert model.training is True


# ------------------------------------------------------------------ train_episodic_v1


def test_train_episodic_v1_runs_the_requested_number_of_steps(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)

    log = train_episodic_v1(
        model, tokenizer, sampler, records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        n_steps=5,
        memory_config=_memory_config(),
        seed=1337,
        device="cpu",
    )

    assert log.n_steps == 5
    assert all(torch.isfinite(torch.tensor(s.loss)) for s in log.steps)


def test_train_episodic_v1_actually_changes_the_models_weights(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    before = copy.deepcopy(model.state_dict())

    train_episodic_v1(
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


def test_train_episodic_v1_is_deterministic_given_the_same_seed(synthetic_corpus):
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)

    model_a, tokenizer = _model_and_tokenizer()
    log_a = train_episodic_v1(
        model_a, tokenizer, sampler, records_by_id,
        model_fingerprint=MODEL_FINGERPRINT, n_steps=3,
        memory_config=_memory_config(), seed=1337, device="cpu",
    )

    model_b, tokenizer_b = _model_and_tokenizer()
    log_b = train_episodic_v1(
        model_b, tokenizer_b, sampler, records_by_id,
        model_fingerprint=MODEL_FINGERPRINT, n_steps=3,
        memory_config=_memory_config(), seed=1337, device="cpu",
    )

    assert [s.writer_id for s in log_a.steps] == [s.writer_id for s in log_b.steps]
    assert [s.loss for s in log_a.steps] == pytest.approx([s.loss for s in log_b.steps])


def test_train_episodic_v1_never_calls_torch_save(synthetic_corpus, monkeypatch):
    def _forbidden_save(*args, **kwargs):
        raise AssertionError("train_episodic_v1 must never call torch.save")

    monkeypatch.setattr(torch, "save", _forbidden_save)

    model, tokenizer = _model_and_tokenizer()
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    train_episodic_v1(
        model, tokenizer, sampler, records_by_id,
        model_fingerprint=MODEL_FINGERPRINT, n_steps=3,
        memory_config=_memory_config(), seed=1337, device="cpu",
    )
