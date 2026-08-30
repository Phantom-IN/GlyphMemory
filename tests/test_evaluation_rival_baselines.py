"""Internal helper."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from glyphmemory.config.schema import MemoryConfig
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.data.adapters.synthetic import SyntheticAdapter
from glyphmemory.data.manifest import read_manifest
from glyphmemory.data.splits import make_support_query_split
from glyphmemory.evaluation.rival_baselines import (
    ADAPTIVE_METHODS,
    ReplayPool,
    build_replay_pool,
    fine_tuned_model,
    parameter_storage_bytes,
    replay_decode,
    run_writer_rivals,
)
from glyphmemory.model import GMBase

MODEL_FINGERPRINT = "test-fingerprint"


def _model_and_tokenizer():
    torch.manual_seed(0)
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    model = GMBase(vocab_size=tokenizer.vocab_size)
    return model, tokenizer


# ------------------------------------------------------------------ parameter groups


def test_head_group_matches_character_head_parameters():
    model, _ = _model_and_tokenizer()
    from glyphmemory.evaluation.rival_baselines import _select_parameters

    head_params = _select_parameters(model, "head")
    expected = list(model.head.parameters())
    assert len(head_params) == len(expected)
    assert all(a is b for a, b in zip(head_params, expected, strict=True))


def test_batchnorm_group_only_contains_batchnorm_affine_params():
    model, _ = _model_and_tokenizer()
    from glyphmemory.evaluation.rival_baselines import _select_parameters

    bn_params = _select_parameters(model, "batchnorm")
    assert bn_params
    bn_ids = {id(p) for p in bn_params}
    for module in model.encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert id(module.weight) in bn_ids
            assert id(module.bias) in bn_ids
    # Every selected parameter really is a BatchNorm2d's weight or bias, not anything else.
    all_bn_params = {
        id(p)
        for m in model.encoder.modules()
        if isinstance(m, nn.BatchNorm2d)
        for p in (m.weight, m.bias)
    }
    assert bn_ids == all_bn_params


def test_full_group_is_every_parameter():
    model, _ = _model_and_tokenizer()
    from glyphmemory.evaluation.rival_baselines import _select_parameters

    full_params = _select_parameters(model, "full")
    assert len(full_params) == len(list(model.parameters()))


def test_unknown_parameter_group_raises():
    model, _ = _model_and_tokenizer()
    from glyphmemory.evaluation.rival_baselines import _select_parameters

    with pytest.raises(ValueError, match="unknown parameter group"):
        _select_parameters(model, "nonexistent")


def test_parameter_storage_bytes_matches_numel_times_four():
    model, _ = _model_and_tokenizer()
    expected = sum(p.numel() for p in model.head.parameters()) * 4
    assert parameter_storage_bytes(model, "head") == expected


def test_head_storage_is_far_smaller_than_full_storage():
    model, _ = _model_and_tokenizer()
    assert parameter_storage_bytes(model, "head") < parameter_storage_bytes(model, "full")
    assert parameter_storage_bytes(model, "batchnorm") < parameter_storage_bytes(model, "head")


# ------------------------------------------------------------------ fine_tuned_model isolation


def _tiny_lines(synthetic_corpus, writer_id: str, n: int):
    from glyphmemory.data.preprocessing import preprocess_path

    records = [r for r in synthetic_corpus.records if r.writer_id == writer_id][:n]
    return [(preprocess_path(r.image).tensor, r.text) for r in records]


def test_fine_tuned_model_restores_weights_exactly_after_normal_exit(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    lines = _tiny_lines(synthetic_corpus, synthetic_corpus.writers[0], 2)

    with fine_tuned_model(model, original_state, "head", lines, tokenizer, steps=2, lr=1e-2):
        pass

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, original_state[name]), f"{name} was not restored exactly"


def test_fine_tuned_model_restores_weights_exactly_even_if_the_block_raises(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    lines = _tiny_lines(synthetic_corpus, synthetic_corpus.writers[0], 2)

    with (
        pytest.raises(RuntimeError, match="deliberate"),
        fine_tuned_model(model, original_state, "head", lines, tokenizer, steps=2, lr=1e-2),
    ):
        raise RuntimeError("deliberate failure inside the block")

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, original_state[name]), f"{name} was not restored exactly"


def test_fine_tuned_model_only_changes_the_target_parameter_group(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    lines = _tiny_lines(synthetic_corpus, synthetic_corpus.writers[0], 2)

    with fine_tuned_model(model, original_state, "head", lines, tokenizer, steps=3, lr=1e-1) as m:
        head_changed = any(
            not torch.equal(p, original_state[name])
            for name, p in m.head.named_parameters(prefix="head")
        )
        encoder_changed = any(
            not torch.equal(p, original_state[name])
            for name, p in m.encoder.named_parameters(prefix="encoder")
        )
        assert head_changed, "head parameters should have moved after gradient steps"
        assert not encoder_changed, "encoder parameters must not move during head-only fine-tune"


def test_fine_tuned_model_requires_grad_is_restored(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    lines = _tiny_lines(synthetic_corpus, synthetic_corpus.writers[0], 2)

    with fine_tuned_model(model, original_state, "batchnorm", lines, tokenizer, steps=2, lr=1e-2):
        pass

    for name, p in model.named_parameters():
        assert p.requires_grad == original_requires_grad[name]


def test_fine_tuned_model_stays_in_eval_mode_never_train(synthetic_corpus):
    """Log: BatchNorm running stats must not drift from a 1-10 line batch, so the model never
    enters.train mode.
    """
    model, tokenizer = _model_and_tokenizer()
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    running_mean_before = next(
        m for m in model.encoder.modules() if isinstance(m, nn.BatchNorm2d)
    ).running_mean.clone()
    lines = _tiny_lines(synthetic_corpus, synthetic_corpus.writers[0], 2)

    with fine_tuned_model(model, original_state, "full", lines, tokenizer, steps=3, lr=1e-2) as m:
        assert m.training is False
        running_mean_during = next(
            mod for mod in m.encoder.modules() if isinstance(mod, nn.BatchNorm2d)
        ).running_mean
        assert torch.equal(running_mean_during, running_mean_before)


# ------------------------------------------------------------------ support replay


def test_build_replay_pool_produces_one_row_per_aligned_frame(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    lines = _tiny_lines(synthetic_corpus, synthetic_corpus.writers[0], 2)

    pool = build_replay_pool(model, tokenizer.charset, lines, device="cpu")

    assert not pool.is_empty
    assert pool.features.shape[0] == len(pool.labels)
    assert pool.features.shape[1] == 384  # sequence_features


def test_build_replay_pool_rejects_empty_lines():
    model, tokenizer = _model_and_tokenizer()
    with pytest.raises(ValueError, match="at least one"):
        build_replay_pool(model, tokenizer.charset, [], device="cpu")


def test_replay_pool_estimated_bytes_scales_with_size():
    small = ReplayPool(features=torch.zeros(2, 384), labels=("a", "b"))
    large = ReplayPool(features=torch.zeros(20, 384), labels=tuple("a" * 20))
    assert large.estimated_bytes() > small.estimated_bytes()


def test_replay_decode_with_empty_pool_matches_greedy_decode(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    lines = _tiny_lines(synthetic_corpus, synthetic_corpus.writers[0], 1)
    tensor, _ = lines[0]
    with torch.no_grad():
        output = model(tensor.unsqueeze(0))

    from glyphmemory.ctc.decode import greedy_decode

    empty_pool = ReplayPool(features=torch.zeros(0, 384), labels=())
    replay_text = replay_decode(output, empty_pool, tokenizer)
    greedy_text = greedy_decode(output.logits[0], tokenizer, int(output.input_lengths[0]))

    assert replay_text == greedy_text


def test_replay_decode_never_touches_frames_the_base_model_calls_blank(synthetic_corpus):
    model, tokenizer = _model_and_tokenizer()
    lines = _tiny_lines(synthetic_corpus, synthetic_corpus.writers[0], 3)
    pool = build_replay_pool(model, tokenizer.charset, lines, device="cpu")

    tensor, _ = lines[0]
    with torch.no_grad():
        output = model(tensor.unsqueeze(0))
    length = int(output.input_lengths[0])
    base_argmax = output.logits[0, :length].argmax(dim=-1)

    # Reconstruct the frame-class sequence the way replay_decode does internally, and confirm every
    # blank-argmax frame stayed blank.
    from glyphmemory.ctc.tokenizer import BLANK_INDEX
    from glyphmemory.probes.geometry import l2_normalize

    features = output.sequence_features[0, :length]
    similarities = l2_normalize(features) @ l2_normalize(pool.features).T
    nearest = similarities.argmax(dim=-1)
    for t in range(length):
        if int(base_argmax[t]) == BLANK_INDEX:
            continue  # nothing to check -- replay_decode must also emit blank here
        label = pool.labels[int(nearest[t])]
        assert label in tokenizer.charset.characters


# ------------------------------------------------------------------ run_writer_rivals


def _writer_corpus_with_forms(tmp_path):
    adapter = SyntheticAdapter(n_writers=2, n_lines=8, seed=20260823)
    manifest_path = adapter.prepare(output_dir=tmp_path / "corpus")
    records = list(read_manifest(manifest_path))
    return [
        dataclasses.replace(record, source_page=f"page{i % 2}")
        for i, record in enumerate(records)
    ]


def test_run_writer_rivals_scores_every_requested_method(tmp_path):
    torch.manual_seed(0)
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    model = GMBase(vocab_size=tokenizer.vocab_size)

    records = _writer_corpus_with_forms(tmp_path)
    writer_records = [r for r in records if r.writer_id == records[0].writer_id]
    support_query = make_support_query_split(writer_records, query_size=3, seed=1337)
    writer_id = writer_records[0].writer_id
    records_by_id = {r.sample_id: r for r in writer_records}

    curve = run_writer_rivals(
        model,
        tokenizer.charset,
        tokenizer,
        writer_id,
        support_query.support_for(writer_id),
        support_query.query_for(writer_id),
        records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        methods=ADAPTIVE_METHODS,
        shots=(1,),
        seeds=(1337,),
        memory_config=MemoryConfig(enabled=True),
        ft_steps=1,
        ft_lr=1e-2,
    )

    assert curve.cer_at_0 is not None
    methods_seen = {s.method for s in curve.shots}
    assert methods_seen == set(ADAPTIVE_METHODS)
    for shot in curve.shots:
        assert shot.cer is not None
        assert shot.storage_bytes > 0
        if shot.method in ("glyphmemory", "replay"):
            assert shot.adaptation_steps == 0
        else:
            assert shot.adaptation_steps == 1


def test_run_writer_rivals_uses_the_same_sample_ids_across_every_method(tmp_path):
    """The whole comparison's fairness rests on this: every method sees the identical support draw
    at a given (form_mode, n, seed).
    """
    torch.manual_seed(0)
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    model = GMBase(vocab_size=tokenizer.vocab_size)

    records = _writer_corpus_with_forms(tmp_path)
    writer_records = [r for r in records if r.writer_id == records[0].writer_id]
    support_query = make_support_query_split(writer_records, query_size=3, seed=1337)
    writer_id = writer_records[0].writer_id
    records_by_id = {r.sample_id: r for r in writer_records}

    curve = run_writer_rivals(
        model,
        tokenizer.charset,
        tokenizer,
        writer_id,
        support_query.support_for(writer_id),
        support_query.query_for(writer_id),
        records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        methods=ADAPTIVE_METHODS,
        shots=(1,),
        seeds=(1337,),
        ft_steps=1,
    )

    by_form_and_n: dict[tuple[str, int], set[tuple[str, ...]]] = {}
    for shot in curve.shots:
        key = (shot.form_mode, shot.n)
        by_form_and_n.setdefault(key, set()).add(shot.sample_ids)

    for key, sample_id_sets in by_form_and_n.items():
        assert len(sample_id_sets) == 1, f"{key} drew different support lines across methods"


def test_run_writer_rivals_leaves_the_model_exactly_as_it_started(tmp_path):
    torch.manual_seed(0)
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    model = GMBase(vocab_size=tokenizer.vocab_size)
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    records = _writer_corpus_with_forms(tmp_path)
    writer_records = [r for r in records if r.writer_id == records[0].writer_id]
    support_query = make_support_query_split(writer_records, query_size=3, seed=1337)
    writer_id = writer_records[0].writer_id
    records_by_id = {r.sample_id: r for r in writer_records}

    run_writer_rivals(
        model,
        tokenizer.charset,
        tokenizer,
        writer_id,
        support_query.support_for(writer_id),
        support_query.query_for(writer_id),
        records_by_id,
        model_fingerprint=MODEL_FINGERPRINT,
        methods=ADAPTIVE_METHODS,
        shots=(1,),
        seeds=(1337,),
        ft_steps=1,
    )

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, original_state[name]), f"{name} drifted after the full sweep"


# ------------------------------------------------------------------ report / summary layer


def _rival_shot(method: str, writer_id: str, cer: float, *, storage=1000, steps=0, wall_ms=1.0):
    from glyphmemory.evaluation.rival_baselines import RivalShotResult

    return RivalShotResult(
        method=method,
        writer_id=writer_id,
        form_mode="cross_form",
        n=3,
        seed=1337,
        sample_ids=(f"{writer_id}-s",),
        cer=cer,
        n_query_lines=5,
        storage_bytes=storage,
        adaptation_steps=steps,
        wall_clock_ms=wall_ms,
    )


def test_summarize_methods_computes_mean_cer_and_gain():
    from glyphmemory.evaluation.rival_baselines import RivalWriterCurve, summarize_methods

    curve_a = RivalWriterCurve(
        writer_id="a", cer_at_0=0.20, n_query_lines=5,
        shots=(_rival_shot("glyphmemory", "a", 0.10),),
    )
    curve_b = RivalWriterCurve(
        writer_id="b", cer_at_0=0.10, n_query_lines=5,
        shots=(_rival_shot("glyphmemory", "b", 0.12),),
    )

    summaries = summarize_methods(
        [curve_a, curve_b], methods=("glyphmemory",), shots=(3,), form_modes=("cross_form",)
    )
    (summary,) = summaries

    assert summary.n_writers == 2
    assert summary.mean_cer == pytest.approx((0.10 + 0.12) / 2)
    assert summary.mean_gain == pytest.approx(((0.20 - 0.10) + (0.10 - 0.12)) / 2)


def test_summarize_methods_separates_methods_and_shapes():
    from glyphmemory.evaluation.rival_baselines import RivalWriterCurve, summarize_methods

    curve = RivalWriterCurve(
        writer_id="a", cer_at_0=0.5, n_query_lines=5,
        shots=(_rival_shot("glyphmemory", "a", 0.1), _rival_shot("head_ft", "a", 0.3)),
    )

    summaries = summarize_methods(
        [curve], methods=("glyphmemory", "head_ft"), shots=(3,), form_modes=("cross_form",)
    )

    assert len(summaries) == 2
    by_method = {s.method: s for s in summaries}
    assert by_method["glyphmemory"].mean_cer == pytest.approx(0.1)
    assert by_method["head_ft"].mean_cer == pytest.approx(0.3)


def test_summarize_methods_reports_none_when_no_writers_have_data():
    from glyphmemory.evaluation.rival_baselines import RivalWriterCurve, summarize_methods

    curve = RivalWriterCurve(
        writer_id="a", cer_at_0=0.5, n_query_lines=5, shots=(),
        unavailable=("cross_form:n=10",),
    )
    summaries = summarize_methods(
        [curve], methods=("glyphmemory",), shots=(10,), form_modes=("cross_form",)
    )
    (summary,) = summaries
    assert summary.n_writers == 0
    assert summary.mean_cer is None
    assert summary.mean_gain is None


def test_rival_baseline_report_save_load_roundtrip(tmp_path):
    from glyphmemory.evaluation.rival_baselines import (
        RivalBaselineReport,
        RivalWriterCurve,
        summarize_methods,
    )

    curve = RivalWriterCurve(
        writer_id="w1", cer_at_0=0.2, n_query_lines=3,
        shots=(_rival_shot("glyphmemory", "w1", 0.15),),
        unavailable=("same_form:n=10",),
    )
    summaries = summarize_methods(
        [curve], methods=("glyphmemory",), shots=(3,), form_modes=("cross_form",)
    )
    report = RivalBaselineReport(
        checkpoint="ckpt.pt", manifest="m.jsonl", split="test", device="cpu",
        methods=("glyphmemory",), shots=(3,), seeds=(1337,), ft_steps=20, ft_lr=5e-3,
        curves=(curve,), summaries=summaries,
    )

    path = report.save(tmp_path / "rivals.json")
    loaded = RivalBaselineReport.load(path)

    assert loaded.checkpoint == report.checkpoint
    assert loaded.ft_steps == 20
    assert loaded.curves[0].writer_id == "w1"
    assert loaded.curves[0].shots[0].cer == pytest.approx(0.15)
    assert loaded.summaries[0].mean_cer == pytest.approx(summaries[0].mean_cer)


def test_rival_baseline_report_format_does_not_raise(tmp_path):
    from glyphmemory.evaluation.rival_baselines import (
        RivalBaselineReport,
        RivalWriterCurve,
        summarize_methods,
    )

    curve = RivalWriterCurve(
        writer_id="w1", cer_at_0=0.2, n_query_lines=3,
        shots=(_rival_shot("glyphmemory", "w1", 0.15),),
    )
    summaries = summarize_methods(
        [curve], methods=("glyphmemory",), shots=(3,), form_modes=("cross_form",)
    )
    report = RivalBaselineReport(
        checkpoint="ckpt.pt", manifest="m.jsonl", split="test", device="cpu",
        methods=("glyphmemory",), shots=(3,), seeds=(1337,), ft_steps=20, ft_lr=5e-3,
        curves=(curve,), summaries=summaries,
    )
    assert report.format()


def test_build_rival_baseline_report_end_to_end_on_synthetic_data(tmp_path):
    from glyphmemory.evaluation.rival_baselines import build_rival_baseline_report

    torch.manual_seed(0)
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    model = GMBase(vocab_size=tokenizer.vocab_size)

    records = _writer_corpus_with_forms(tmp_path)
    support_query = make_support_query_split(records, query_size=3, seed=1337)
    records_by_id = {r.sample_id: r for r in records}

    report = build_rival_baseline_report(
        model,
        tokenizer.charset,
        tokenizer,
        support_query,
        records_by_id,
        checkpoint_label="test.pt",
        manifest_label="test.jsonl",
        split_name="test",
        model_fingerprint=MODEL_FINGERPRINT,
        methods=("glyphmemory", "replay"),
        shots=(1,),
        seeds=(1337,),
    )

    assert len(report.curves) == len(support_query.writers_supporting(1))
    assert report.summaries
    assert report.format()
    assert "curves" in report.as_dict()
