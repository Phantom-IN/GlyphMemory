"""Forward-latency measurement: warmup discard, percentile ordering, and real-model plumbing."""

from __future__ import annotations

import torch
from torch import nn

from glyphmemory.benchmark.latency import measure_forward_latency
from glyphmemory.model import GMBase


class _CountingModel(nn.Module):
    """Records every call so warmup-vs-measured can be checked precisely, without paying for a real
    forward pass.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, images: torch.Tensor, lengths: torch.Tensor):
        self.calls += 1
        return self.dummy


def test_warmup_and_measurement_counts_are_exact():
    model = _CountingModel()
    measure_forward_latency(
        model,
        device=torch.device("cpu"),
        input_height=8,
        input_width=16,
        batch_size=2,
        warmup_iterations=3,
        measurement_iterations=7,
    )
    assert model.calls == 3 + 7


def test_result_echoes_the_requested_shape():
    model = _CountingModel()
    result = measure_forward_latency(
        model,
        device=torch.device("cpu"),
        input_height=8,
        input_width=32,
        batch_size=4,
        warmup_iterations=1,
        measurement_iterations=5,
    )
    assert result.input_width == 32
    assert result.batch_size == 4
    assert result.warmup_iterations == 1
    assert result.measurement_iterations == 5


def test_p95_is_never_below_p50():
    model = _CountingModel()
    result = measure_forward_latency(
        model,
        device=torch.device("cpu"),
        input_height=8,
        input_width=16,
        batch_size=1,
        warmup_iterations=1,
        measurement_iterations=10,
    )
    assert result.p95_ms >= result.p50_ms


def test_lines_per_second_scales_with_batch_size():
    """Throughput, not latency, should roughly track batch size for a batch-insensitive fake model —
    a real check that batch_size feeds the throughput formula, not just the shape.
    """
    model = _CountingModel()
    small = measure_forward_latency(
        model,
        device=torch.device("cpu"),
        input_height=8,
        input_width=16,
        batch_size=1,
        warmup_iterations=1,
        measurement_iterations=5,
    )
    large = measure_forward_latency(
        model,
        device=torch.device("cpu"),
        input_height=8,
        input_width=16,
        batch_size=8,
        warmup_iterations=1,
        measurement_iterations=5,
    )
    assert large.lines_per_second > small.lines_per_second


def test_measures_a_real_gm_base_forward_pass():
    torch.manual_seed(0)
    model = GMBase(vocab_size=20)
    result = measure_forward_latency(
        model,
        device=torch.device("cpu"),
        input_height=64,
        input_width=64,
        batch_size=1,
        warmup_iterations=1,
        measurement_iterations=2,
    )
    assert result.p50_ms > 0
    assert result.lines_per_second > 0
