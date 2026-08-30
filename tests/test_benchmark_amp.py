"""AMP verification: never auto-declared safe, always measured against FP32."""

from __future__ import annotations

import torch

from glyphmemory.benchmark.amp_check import verify_amp
from glyphmemory.model import GMBase


def test_verify_amp_on_cpu_runs_and_reports_both_timings():
    torch.manual_seed(0)
    model = GMBase(vocab_size=20)
    verdict = verify_amp(
        model,
        device=torch.device("cpu"),
        input_height=64,
        input_width=64,
        batch_size=2,
        warmup_iterations=1,
        iterations=2,
    )
    assert verdict.device == "cpu"
    assert verdict.fp32_ms > 0
    # CPU's default dtype is bfloat16, which is expected to run without raising.
    assert verdict.error is None
    assert verdict.dtype == str(torch.bfloat16)


def test_default_dtype_matches_device_kind():
    torch.manual_seed(0)
    model = GMBase(vocab_size=20)
    cpu_verdict = verify_amp(
        model,
        device=torch.device("cpu"),
        input_height=64,
        input_width=64,
        batch_size=1,
        warmup_iterations=1,
        iterations=2,
    )
    assert cpu_verdict.dtype == str(torch.bfloat16)


def test_supported_requires_a_meaningful_speedup_not_just_a_faster_run():
    """A 1.001x speedup is noise, not a recommendation — the bar is stated and used."""
    torch.manual_seed(0)
    model = GMBase(vocab_size=20)
    verdict = verify_amp(
        model,
        device=torch.device("cpu"),
        input_height=64,
        input_width=64,
        batch_size=2,
        warmup_iterations=1,
        iterations=2,
    )
    if verdict.speedup is not None:
        assert verdict.supported == (verdict.speedup >= 1.02)


def test_verdict_is_never_silently_declared_supported_when_output_is_non_finite(monkeypatch):
    """A speedup that comes with garbage output must never read as a recommendation.

    Forces the finiteness check to fail regardless of the actual (finite) logits, to prove the
    rejection path is reachable and wins over a positive speedup rather than being dead code.
    """
    import glyphmemory.benchmark.amp_check as amp_module

    torch.manual_seed(0)
    model = GMBase(vocab_size=20)

    monkeypatch.setattr(
        amp_module.torch, "isfinite", lambda x: torch.zeros_like(x, dtype=torch.bool)
    )

    verdict = verify_amp(
        model,
        device=torch.device("cpu"),
        input_height=64,
        input_width=64,
        batch_size=1,
        warmup_iterations=1,
        iterations=2,
    )
    assert verdict.supported is False
    assert "non-finite" in verdict.notes
