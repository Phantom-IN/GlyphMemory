"""The MPS CTC round-trip cost measurement, exercised on CPU (MPS-specific behaviour is covered by
own tests; this checks the measurement plumbing, which must work identically regardless of which
device happens to be available in CI).
"""

from __future__ import annotations

import torch

from glyphmemory.benchmark.roundtrip import measure_ctc_roundtrip
from glyphmemory.model import GMBase


def test_roundtrip_measures_both_variants():
    torch.manual_seed(0)
    model = GMBase(vocab_size=20)
    result = measure_ctc_roundtrip(
        model,
        device=torch.device("cpu"),
        input_height=64,
        input_width=64,
        batch_size=2,
        vocab_size=20,
        warmup_iterations=1,
        iterations=2,
    )
    assert result.forward_only_ms > 0
    assert result.forward_plus_loss_ms > 0
    assert result.device == "cpu"
    assert result.input_width == 64
    assert result.batch_size == 2


def test_overhead_properties_are_consistent_with_the_two_timings():
    torch.manual_seed(0)
    model = GMBase(vocab_size=20)
    result = measure_ctc_roundtrip(
        model,
        device=torch.device("cpu"),
        input_height=64,
        input_width=64,
        batch_size=2,
        vocab_size=20,
        warmup_iterations=1,
        iterations=2,
    )
    assert result.loss_overhead_ms == result.forward_plus_loss_ms - result.forward_only_ms
    expected_fraction = result.loss_overhead_ms / result.forward_plus_loss_ms
    assert result.loss_overhead_fraction == expected_fraction


def test_as_dict_includes_derived_fields():
    torch.manual_seed(0)
    model = GMBase(vocab_size=20)
    result = measure_ctc_roundtrip(
        model,
        device=torch.device("cpu"),
        input_height=64,
        input_width=64,
        batch_size=2,
        vocab_size=20,
        warmup_iterations=1,
        iterations=2,
    )
    payload = result.as_dict()
    assert "loss_overhead_ms" in payload
    assert "loss_overhead_fraction" in payload
