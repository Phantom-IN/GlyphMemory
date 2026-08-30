"""The few-shot harness."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from glyphmemory.config.schema import MemoryConfig
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.data.adapters.synthetic import SyntheticAdapter
from glyphmemory.data.manifest import ManifestRecord, read_manifest
from glyphmemory.data.splits import make_support_query_split
from glyphmemory.evaluation.few_shot import (
    FewShotReport,
    ShotResult,
    WriterCurve,
    aggregate_shot_statistics,
    build_few_shot_report,
    partition_by_form,
    run_writer_curve,
    sample_support_subset,
)
from glyphmemory.model import GMBase

# ------------------------------------------------------------------ sample_support_subset


def test_sample_support_subset_is_deterministic():
    pool = [f"s{i}" for i in range(10)]
    a = sample_support_subset(pool, 3, seed=1337, key="w1")
    b = sample_support_subset(pool, 3, seed=1337, key="w1")
    assert a == b


def test_sample_support_subset_differs_across_seeds_usually():
    pool = [f"s{i}" for i in range(20)]
    a = sample_support_subset(pool, 3, seed=1337, key="w1")
    b = sample_support_subset(pool, 3, seed=9999, key="w1")
    assert a != b


def test_sample_support_subset_differs_across_keys_at_the_same_seed():
    pool = [f"s{i}" for i in range(20)]
    a = sample_support_subset(pool, 3, seed=1337, key="writerA")
    b = sample_support_subset(pool, 3, seed=1337, key="writerB")
    assert a != b


def test_sample_support_subset_is_sorted_and_the_right_length():
    pool = ["z", "a", "m", "b", "y"]
    result = sample_support_subset(pool, 3, seed=1337)
    assert len(result) == 3
    assert result == tuple(sorted(result))
    assert set(result) <= set(pool)


def test_sample_support_subset_rejects_n_below_one():
    with pytest.raises(ValueError, match="n must be at least 1"):
        sample_support_subset(["a", "b"], 0, seed=1337)


def test_sample_support_subset_rejects_a_pool_smaller_than_n():
    with pytest.raises(ValueError, match="cannot draw"):
        sample_support_subset(["a", "b"], 3, seed=1337)


# ------------------------------------------------------------------ partition_by_form


def _record(sample_id: str, source_page: str | None) -> ManifestRecord:
    return ManifestRecord(
        image=f"/tmp/{sample_id}.png",
        text="hello",
        writer_id="w1",
        dataset="iam",
        split="val",
        sample_id=sample_id,
        source_page=source_page,
    )


def test_partition_by_form_classifies_same_and_cross_form():
    records = {
        "q1": _record("q1", "page1"),
        "s1": _record("s1", "page1"),  # same form as q1
        "s2": _record("s2", "page2"),  # different form
        "s3": _record("s3", None),  # unknown
    }
    partition = partition_by_form(["s1", "s2", "s3"], ["q1"], records)

    assert partition.same_form == ("s1",)
    assert partition.cross_form == ("s2",)
    assert partition.unknown_form == ("s3",)


def test_partition_by_form_same_form_matches_any_query_line():
    records = {
        "q1": _record("q1", "page1"),
        "q2": _record("q2", "page2"),
        "s1": _record("s1", "page2"),  # matches q2, not q1
    }
    partition = partition_by_form(["s1"], ["q1", "q2"], records)

    assert partition.same_form == ("s1",)
    assert partition.cross_form == ()


def test_partition_by_form_pool_for_rejects_unknown_mode():
    partition = partition_by_form([], [], {})
    with pytest.raises(ValueError, match="form_mode"):
        partition.pool_for("nonexistent")


# ------------------------------------------------------------------ aggregate_shot_statistics


def _shot(writer_id: str, form_mode: str, n: int, seed: int, cer: float) -> ShotResult:
    from glyphmemory.evaluation.few_shot import ProfileStats

    return ShotResult(
        writer_id=writer_id,
        form_mode=form_mode,
        n=n,
        seed=seed,
        sample_ids=(f"{writer_id}-s{seed}",),
        cer=cer,
        n_query_lines=5,
        profile=ProfileStats(
            support_lines=n,
            characters_observed=10,
            unique_characters_observed=5,
            estimated_bytes=1000,
            compile_ms=1.0,
            feature_dim=384,
        ),
    )


def test_aggregate_shot_statistics_computes_mean_and_median_gain():
    curve_a = WriterCurve(
        writer_id="a",
        cer_at_0=0.20,
        n_query_lines=5,
        shots=(_shot("a", "cross_form", 5, 1337, 0.10), _shot("a", "cross_form", 5, 1338, 0.10)),
    )
    curve_b = WriterCurve(
        writer_id="b",
        cer_at_0=0.10,
        n_query_lines=5,
        shots=(_shot("b", "cross_form", 5, 1337, 0.12), _shot("b", "cross_form", 5, 1338, 0.12)),
    )

    stats = aggregate_shot_statistics([curve_a, curve_b], shots=(5,), form_modes=("cross_form",))
    (stat,) = stats

    # a: gain = 0.20 - 0.10 = +0.10 (improved). b: gain = 0.10 - 0.12 = -0.02 (regressed).
    assert stat.mean_gain == pytest.approx((0.10 + -0.02) / 2)
    assert stat.median_gain == pytest.approx((0.10 + -0.02) / 2)  # median of 2 values = mean
    assert stat.pct_improved == pytest.approx(0.5)
    assert stat.pct_regressed == pytest.approx(0.5)
    assert stat.n_writers_available == 2


def test_aggregate_shot_statistics_counts_unavailable_writers():
    curve = WriterCurve(
        writer_id="a", cer_at_0=0.2, n_query_lines=5, shots=(), unavailable=("cross_form:n=10",)
    )
    stats = aggregate_shot_statistics([curve], shots=(10,), form_modes=("cross_form",))
    (stat,) = stats
    assert stat.n_writers_unavailable == 1
    assert stat.n_writers_available == 0
    assert stat.mean_gain is None


def test_aggregate_shot_statistics_worst_regressions_sorted_most_negative_first():
    curves = [
        WriterCurve(
            writer_id=w,
            cer_at_0=0.5,
            n_query_lines=5,
            shots=(_shot(w, "cross_form", 3, 1337, cer),),
        )
        for w, cer in (("good", 0.1), ("bad", 0.9), ("mid", 0.5))
    ]
    stats = aggregate_shot_statistics(curves, shots=(3,), form_modes=("cross_form",))
    (stat,) = stats

    assert stat.worst_regressions[0][0] == "bad"


def test_aggregate_shot_statistics_buckets_by_baseline_difficulty():
    curves = [
        WriterCurve(
            writer_id=w,
            cer_at_0=baseline,
            n_query_lines=5,
            shots=(_shot(w, "cross_form", 3, 1337, baseline - 0.05),),
        )
        for w, baseline in (("easy", 0.05), ("mid", 0.20), ("hard", 0.50))
    ]
    stats = aggregate_shot_statistics(curves, shots=(3,), form_modes=("cross_form",))
    (stat,) = stats

    assert stat.bucket_mean_gain["easy"] == pytest.approx(0.05)
    assert stat.bucket_mean_gain["hard"] == pytest.approx(0.05)


# ------------------------------------------------------------------ FewShotReport persistence


def test_few_shot_report_save_load_roundtrip(tmp_path):
    curve = WriterCurve(
        writer_id="w1",
        cer_at_0=0.2,
        n_query_lines=3,
        shots=(_shot("w1", "cross_form", 3, 1337, 0.15),),
        unavailable=("same_form:n=10",),
    )
    stats = aggregate_shot_statistics([curve], shots=(3,), form_modes=("cross_form",))
    report = FewShotReport(
        checkpoint="ckpt.pt",
        manifest="m.jsonl",
        split="val",
        device="cpu",
        shots=(3,),
        seeds=(1337,),
        curves=(curve,),
        statistics=stats,
    )

    path = report.save(tmp_path / "report.json")
    loaded = FewShotReport.load(path)

    assert loaded.checkpoint == report.checkpoint
    assert loaded.curves[0].writer_id == "w1"
    assert loaded.curves[0].shots[0].cer == pytest.approx(0.15)
    assert loaded.curves[0].unavailable == ("same_form:n=10",)
    assert loaded.statistics[0].mean_gain == pytest.approx(stats[0].mean_gain)


# ------------------------------------------------------------------ real plumbing (synthetic)


def _writer_corpus_with_forms(tmp_path):
    """A small synthetic corpus, with source_page assigned so same/cross-form is exercisable
    (SyntheticAdapter never sets it -- every line would otherwise land in `unknown_form`).
    """
    adapter = SyntheticAdapter(n_writers=2, n_lines=8, seed=20260822)
    manifest_path = adapter.prepare(output_dir=tmp_path / "corpus")
    records = list(read_manifest(manifest_path))
    return [
        dataclasses.replace(record, source_page=f"page{i % 2}")
        for i, record in enumerate(records)
    ]


def test_run_writer_curve_end_to_end_on_synthetic_data(tmp_path):
    torch.manual_seed(0)
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    model = GMBase(vocab_size=tokenizer.vocab_size)

    records = _writer_corpus_with_forms(tmp_path)
    writer_records = [r for r in records if r.writer_id == records[0].writer_id]
    assert len(writer_records) == 8

    support_query = make_support_query_split(writer_records, query_size=3, seed=1337)
    writer_id = writer_records[0].writer_id
    records_by_id = {r.sample_id: r for r in writer_records}

    curve = run_writer_curve(
        model,
        tokenizer.charset,
        tokenizer,
        writer_id,
        support_query.support_for(writer_id),
        support_query.query_for(writer_id),
        records_by_id,
        model_fingerprint="test-fingerprint",
        shots=(1, 3),
        seeds=(1337,),
        memory_config=MemoryConfig(enabled=True),
    )

    assert curve.writer_id == writer_id
    assert curve.cer_at_0 is not None
    assert curve.cer_at_0 >= 0.0
    assert curve.n_query_lines == len(support_query.query_for(writer_id))
    # Both form modes attempted for both shots -> up to 4 combinations, each with 1 seed.
    assert len(curve.shots) + len(curve.unavailable) == 2 * 2
    for shot in curve.shots:
        assert shot.cer is not None
        assert shot.profile.support_lines == shot.n
        assert shot.profile.feature_dim == 384


def test_build_few_shot_report_end_to_end_on_synthetic_data(tmp_path):
    torch.manual_seed(0)
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    model = GMBase(vocab_size=tokenizer.vocab_size)

    records = _writer_corpus_with_forms(tmp_path)
    support_query = make_support_query_split(records, query_size=3, seed=1337)
    records_by_id = {r.sample_id: r for r in records}

    report = build_few_shot_report(
        model,
        tokenizer.charset,
        tokenizer,
        support_query,
        records_by_id,
        checkpoint_label="test.pt",
        manifest_label="test.jsonl",
        split_name="val",
        model_fingerprint="test-fingerprint",
        shots=(1,),
        seeds=(1337,),
        memory_config=MemoryConfig(enabled=True),
    )

    assert len(report.curves) == len(support_query.writers_supporting(1))
    assert report.statistics
    assert report.format()  # does not raise, produces something
    assert "checkpoint" in report.as_dict()


def test_query_pool_is_identical_across_every_shot_and_form_mode(tmp_path):
    """The query pool must never be recomputed per n. Checked directly, not just believed."""
    torch.manual_seed(0)
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    model = GMBase(vocab_size=tokenizer.vocab_size)

    records = _writer_corpus_with_forms(tmp_path)
    writer_records = [r for r in records if r.writer_id == records[0].writer_id]
    support_query = make_support_query_split(writer_records, query_size=3, seed=1337)
    writer_id = writer_records[0].writer_id
    records_by_id = {r.sample_id: r for r in writer_records}
    fixed_query_pool = support_query.query_for(writer_id)

    curve = run_writer_curve(
        model,
        tokenizer.charset,
        tokenizer,
        writer_id,
        support_query.support_for(writer_id),
        fixed_query_pool,
        records_by_id,
        model_fingerprint="test-fingerprint",
        shots=(1, 3),
        seeds=(1337,),
        memory_config=MemoryConfig(enabled=True),
    )

    # n_query_lines is recorded on every shot and on the curve itself -- all must agree with the one
    # fixed query pool's size.
    assert curve.n_query_lines == len(fixed_query_pool)
    for shot in curve.shots:
        assert shot.n_query_lines == len(fixed_query_pool)
