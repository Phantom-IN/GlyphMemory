"""BenchmarkContext: the required-fields contract."""

from __future__ import annotations

from pathlib import Path

from glyphmemory.benchmark.context import (
    REQUIRED_FIELDS,
    BenchmarkContext,
    checkpoint_fingerprint,
    missing_fields,
)


def _write_dummy_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "dummy.pt"
    path.write_bytes(b"not a real checkpoint, just bytes to hash and stat")
    return path


def test_capture_produces_every_required_field(tmp_path):
    checkpoint = _write_dummy_checkpoint(tmp_path)
    context = BenchmarkContext.capture(
        device="cpu",
        input_height=64,
        input_width=512,
        batch_size=8,
        warmup_iterations=5,
        measurement_iterations=20,
        checkpoint_path=checkpoint,
        parameter_count=1_544_560,
    )
    assert missing_fields(context.as_dict()) == []


def test_required_fields_covers_the_docs_list():
    """Internal helper."""
    expected = {
        "cpu_model",
        "os",
        "python_version",
        "runtime",
        "runtime_version",
        "device",
        "num_threads",
        "model_format",
        "input_height",
        "input_width",
        "batch_size",
        "warmup_iterations",
        "measurement_iterations",
        "model_fingerprint",
        "parameter_count",
        "file_size_bytes",
    }
    assert set(REQUIRED_FIELDS) == expected


def test_missing_fields_reports_what_is_absent():
    payload = {"cpu_model": "Apple"}
    missing = missing_fields(payload)
    assert "cpu_model" not in missing
    assert "os" in missing
    assert len(missing) == len(REQUIRED_FIELDS) - 1


def test_file_size_is_measured_from_disk(tmp_path):
    checkpoint = _write_dummy_checkpoint(tmp_path)
    expected_size = checkpoint.stat().st_size
    context = BenchmarkContext.capture(
        device="cpu",
        input_height=64,
        input_width=256,
        batch_size=1,
        warmup_iterations=1,
        measurement_iterations=1,
        checkpoint_path=checkpoint,
        parameter_count=100,
    )
    assert context.file_size_bytes == expected_size


def test_checkpoint_fingerprint_is_deterministic_and_content_based(tmp_path):
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    a.write_bytes(b"same content")
    b.write_bytes(b"same content")
    assert checkpoint_fingerprint(a) == checkpoint_fingerprint(b)

    c = tmp_path / "c.pt"
    c.write_bytes(b"different content")
    assert checkpoint_fingerprint(a) != checkpoint_fingerprint(c)


def test_num_threads_is_the_configured_torch_value(tmp_path):
    import torch

    checkpoint = _write_dummy_checkpoint(tmp_path)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        context = BenchmarkContext.capture(
            device="cpu",
            input_height=64,
            input_width=256,
            batch_size=1,
            warmup_iterations=1,
            measurement_iterations=1,
            checkpoint_path=checkpoint,
            parameter_count=100,
        )
        assert context.num_threads == 2
    finally:
        torch.set_num_threads(previous)
