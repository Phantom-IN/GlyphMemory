"""Checkpoint selection for episodic training."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from glyphmemory.config.schema import Config, MemoryConfig
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.data.episodes import EpisodeSampler
from glyphmemory.model import GMBase
from glyphmemory.training.episodic import train_episodic_v0
from glyphmemory.training.episodic_validation import (
    ProbeCheck,
    ValidationProbe,
    select_probe_records,
)

MODEL_FINGERPRINT = "deadbeefcafef00d"


def _val_records(synthetic_corpus):
    """The shared corpus relabelled as a validation split — the probe only reads a held-out split,
    and the fixture is all-train by construction.
    """
    return tuple(replace(r, split="val") for r in synthetic_corpus.records)


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


def _probe(synthetic_corpus, tokenizer) -> ValidationProbe:
    records = select_probe_records(
        _val_records(synthetic_corpus), n_writers=2, lines_per_writer=1, exclude_writers=()
    )
    return ValidationProbe(records, tokenizer, Config())


class _ScriptedProbe(ValidationProbe):
    """A probe whose CER follows a fixed script, so a test can decide which step is best.

    Real probe values on a 4-step synthetic run are noise; scripting them is what makes the
    selection *rule* testable rather than the model's behavior.
    """

    def __init__(self, inner: ValidationProbe, cers: list[float]) -> None:
        self._inner = inner
        self._cers = cers
        self._calls = 0

    def evaluate(self, model, *, step: int) -> ProbeCheck:  # type: ignore[override]
        cer = self._cers[self._calls]
        self._calls += 1
        return ProbeCheck(step=step, cer=cer, wer=cer, n_lines=1, seconds=0.001)


def _train(synthetic_corpus, probe, *, n_steps=4, probe_every=1, select_best=True):
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    model, tokenizer = _model_and_tokenizer()
    log = train_episodic_v0(
        model,
        tokenizer,
        sampler,
        records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        n_steps=n_steps,
        memory_config=_memory_config(),
        probe=probe,
        probe_every=probe_every,
        select_best=select_best,
    )
    return model, log


def test_selects_the_best_scoring_step_not_the_last(synthetic_corpus):
    _model, tokenizer = _model_and_tokenizer()
    # checks land at steps 0,1,2,3,4 -> step 2 is the best *trained* point
    probe = _ScriptedProbe(_probe(synthetic_corpus, tokenizer), [0.50, 0.40, 0.20, 0.30, 0.35])

    _model, log = _train(synthetic_corpus, probe)

    assert log.selected_step == 2
    assert log.selected_cer == pytest.approx(0.20)
    assert log.as_dict()["selected_step"] == 2


def test_the_untrained_step_zero_state_is_never_selected(synthetic_corpus):
    """The decision that keeps this "checkpoint selection" rather than "do not train".

    Episodic training only ever *hurts* generic recognition, so on a generic-recognition probe step
    0 is the best point of the whole run. Admitting it would discard the personalization the run
    exists to produce along with the regression.
    """
    _model, tokenizer = _model_and_tokenizer()
    # step 0 scores best by a wide margin; every trained step is worse
    probe = _ScriptedProbe(_probe(synthetic_corpus, tokenizer), [0.01, 0.40, 0.30, 0.50, 0.45])

    _model, log = _train(synthetic_corpus, probe)

    assert log.selected_step == 2  # the best *trained* step, not step 0
    assert log.selected_cer == pytest.approx(0.30)
    assert log.probe_checks[0].cer == pytest.approx(0.01)  # still recorded as the reference


def test_the_model_really_ends_on_the_selected_weights(synthetic_corpus):
    """Reporting the best step but returning the last one would be worse than not selecting."""
    _model, tokenizer = _model_and_tokenizer()
    best_first = _ScriptedProbe(_probe(synthetic_corpus, tokenizer), [0.9, 0.1, 0.8, 0.8, 0.8])
    best_last = _ScriptedProbe(_probe(synthetic_corpus, tokenizer), [0.9, 0.8, 0.8, 0.8, 0.1])

    early_model, early_log = _train(synthetic_corpus, best_first)
    late_model, late_log = _train(synthetic_corpus, best_last)

    assert early_log.selected_step == 1
    assert late_log.selected_step == 4
    # Same seed, same data, same steps: the two runs differ only in which state they kept.
    differs = any(
        not torch.equal(tensor, late_model.state_dict()[name])
        for name, tensor in early_model.state_dict().items()
    )
    assert differs, "selection did not change the weights the run ended on"


def test_selection_off_is_the_original_behavior(synthetic_corpus):
    _model, tokenizer = _model_and_tokenizer()
    probe = _ScriptedProbe(_probe(synthetic_corpus, tokenizer), [0.9, 0.1, 0.8, 0.8, 0.8])

    _model, log = _train(synthetic_corpus, probe, select_best=False)

    assert log.selected_step is None
    assert log.selected_cer is None


def test_selection_requires_a_probe(synthetic_corpus):
    """Without a probe there is no validation CER, and falling back to loss is precisely what
    forbids.
    """
    sampler, records_by_id = _sampler_and_records(synthetic_corpus)
    model, tokenizer = _model_and_tokenizer()

    with pytest.raises(ValueError, match="select_best requires a probe"):
        train_episodic_v0(
            model,
            tokenizer,
            sampler,
            records_by_id,
            model_fingerprint=MODEL_FINGERPRINT,
            n_steps=1,
            memory_config=_memory_config(),
            select_best=True,
        )


def test_a_probe_check_that_produced_no_value_cannot_win(synthetic_corpus):
    """`is_better`'s own rule, inherited rather than re-implemented: `None` never displaces a real
    result.
    """
    _model, tokenizer = _model_and_tokenizer()
    probe = _ScriptedProbe(_probe(synthetic_corpus, tokenizer), [0.9, 0.4, None, None, None])

    _model, log = _train(synthetic_corpus, probe)

    assert log.selected_step == 1
    assert log.selected_cer == pytest.approx(0.4)
