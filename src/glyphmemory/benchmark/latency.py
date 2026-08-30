"""Forward-pass latency: p50/p95, not a mean.

GPU-backed devices (MPS, CUDA) queue work asynchronously, so a naive ``perf_counter()`` bracket
around the Python call measures dispatch time, not compute time. Each device kind gets its own
synchronize call before starting and after finishing the clock, the same discipline
`perf-mps-audit-001` used per-segment.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from glyphmemory.data.preprocessing import temporal_length


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    # CPU is synchronous by construction — nothing to wait for.


@dataclass(frozen=True, slots=True)
class LatencyMeasurement:
    """p50/p95 latency and throughput for one (width, batch_size) point."""

    input_width: int
    batch_size: int
    warmup_iterations: int
    measurement_iterations: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    lines_per_second: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self) -> str:
        return (
            f"  W{self.input_width:>5} B{self.batch_size:<3}   "
            f"p50 {self.p50_ms:>8.2f} ms   p95 {self.p95_ms:>8.2f} ms   "
            f"{self.lines_per_second:>7.1f} lines/s"
        )


@torch.no_grad()
def measure_forward_latency(
    model: nn.Module,
    *,
    device: torch.device,
    input_height: int,
    input_width: int,
    batch_size: int,
    warmup_iterations: int = 5,
    measurement_iterations: int = 20,
) -> LatencyMeasurement:
    """Time ``measurement_iterations`` forward passes at one (width, batch) point.

    Inputs are synthetic — random ``[0, 1]`` pixels at a fixed width, so every sample in the batch
    is unpadded and ``input_lengths`` is the same for all of them. Latency depends on tensor shape,
    not pixel content, so this is representative without needing real images.
    """
    model.eval()
    images = torch.rand(batch_size, 1, input_height, input_width, device=device)
    lengths = torch.full(
        (batch_size,), temporal_length(input_width), dtype=torch.long, device=device
    )

    for _ in range(warmup_iterations):
        model(images, lengths)
    _synchronize(device)

    samples_ms: list[float] = []
    for _ in range(measurement_iterations):
        _synchronize(device)
        started = time.perf_counter()
        model(images, lengths)
        _synchronize(device)
        samples_ms.append((time.perf_counter() - started) * 1000.0)

    samples_ms.sort()
    mean_ms = statistics.mean(samples_ms)
    lines_per_second = (batch_size * 1000.0 / mean_ms) if mean_ms > 0 else float("inf")

    return LatencyMeasurement(
        input_width=input_width,
        batch_size=batch_size,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        p50_ms=statistics.median(samples_ms),
        p95_ms=_percentile(samples_ms, 0.95),
        mean_ms=mean_ms,
        lines_per_second=lines_per_second,
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile over already-sorted values. No interpolation, no dependency."""
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, max(0, round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]
