"""``run_benchmark`` end to end: the grid, the round-trip cost, and the AMP verdict, on a freshly
saved checkpoint. Kept tiny (small widths/batches, 1 warmup, 2 measured iterations) — this is a
plumbing test, not a real measurement.
"""

from __future__ import annotations

from pathlib import Path

import torch

from glyphmemory.benchmark import run_benchmark
from glyphmemory.benchmark.context import missing_fields
from glyphmemory.config.schema import Config
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.model import GMBase
from glyphmemory.training.checkpoint import CheckpointMeta, save_checkpoint


def _save_checkpoint(tmp_path: Path, tokenizer) -> Path:
    torch.manual_seed(0)
    model = GMBase(vocab_size=tokenizer.vocab_size)
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
        run_id="bench_test",
    )
    return save_checkpoint(tmp_path / "checkpoint.pt", model=model, meta=meta)


def test_run_benchmark_produces_a_grid_with_full_context(tmp_path):
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    checkpoint = _save_checkpoint(tmp_path, tokenizer)

    report = run_benchmark(
        checkpoint,
        config=Config(),
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        widths=(64, 128),
        batch_sizes=(1, 2),
        warmup_iterations=1,
        measurement_iterations=2,
        threads=1,
    )

    assert len(report.grid) == 4  # 2 widths x 2 batch sizes
    seen = {(p.measurement.input_width, p.measurement.batch_size) for p in report.grid}
    assert seen == {(64, 1), (64, 2), (128, 1), (128, 2)}

    for point in report.grid:
        assert missing_fields(point.as_dict()) == []


def test_round_trip_and_amp_are_measured_on_the_requested_device(tmp_path):
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    checkpoint = _save_checkpoint(tmp_path, tokenizer)

    report = run_benchmark(
        checkpoint,
        config=Config(),
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        widths=(64,),
        batch_sizes=(1,),
        warmup_iterations=1,
        measurement_iterations=2,
        threads=1,
    )

    assert report.roundtrip_on_device.device == "cpu"
    # On a CPU-only run there is nothing to compare against, so the CPU baseline is the same
    # measurement, not a second one silently fabricated.
    assert report.roundtrip_on_cpu.device == "cpu"
    assert report.amp.device == "cpu"


def test_as_dict_round_trips_every_section(tmp_path):
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    checkpoint = _save_checkpoint(tmp_path, tokenizer)

    report = run_benchmark(
        checkpoint,
        config=Config(),
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        widths=(64,),
        batch_sizes=(1,),
        warmup_iterations=1,
        measurement_iterations=2,
        threads=1,
    )
    payload = report.as_dict()
    assert "grid" in payload
    assert "roundtrip_on_device" in payload
    assert "amp" in payload
    assert payload["grid"][0]["p50_ms"] >= 0


def test_threads_argument_is_applied(tmp_path):
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    checkpoint = _save_checkpoint(tmp_path, tokenizer)
    previous = torch.get_num_threads()
    try:
        report = run_benchmark(
            checkpoint,
            config=Config(),
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            widths=(64,),
            batch_sizes=(1,),
            warmup_iterations=1,
            measurement_iterations=2,
            threads=2,
        )
        assert torch.get_num_threads() == 2
        assert report.grid[0].context.num_threads == 2
    finally:
        torch.set_num_threads(previous)
