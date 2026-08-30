"""The MPS CTC round-trip cost, priced."""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from glyphmemory.data.preprocessing import temporal_length
from glyphmemory.model.loss import ctc_loss_for


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@dataclass(frozen=True, slots=True)
class RoundtripResult:
    """Forward-only vs forward-plus-CTC-loss latency, same device, same batch."""

    device: str
    input_width: int
    batch_size: int
    iterations: int
    forward_only_ms: float
    forward_plus_loss_ms: float

    @property
    def loss_overhead_ms(self) -> float:
        return self.forward_plus_loss_ms - self.forward_only_ms

    @property
    def loss_overhead_fraction(self) -> float:
        if self.forward_plus_loss_ms == 0:
            return 0.0
        return self.loss_overhead_ms / self.forward_plus_loss_ms

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["loss_overhead_ms"] = self.loss_overhead_ms
        payload["loss_overhead_fraction"] = self.loss_overhead_fraction
        return payload

    def format(self) -> str:
        return (
            f"  {self.device:<5} W{self.input_width} B{self.batch_size}   "
            f"forward-only {self.forward_only_ms:>7.2f} ms   "
            f"forward+loss {self.forward_plus_loss_ms:>7.2f} ms   "
            f"overhead {self.loss_overhead_ms:>6.2f} ms "
            f"({self.loss_overhead_fraction:.1%})"
        )


def measure_ctc_roundtrip(
    model: nn.Module,
    *,
    device: torch.device,
    input_height: int,
    input_width: int,
    batch_size: int,
    vocab_size: int,
    warmup_iterations: int = 5,
    iterations: int = 20,
) -> RoundtripResult:
    """Time forward-only and forward+CTC-loss at one (width, batch) point on ``device``.

    Targets are synthetic random labels, short enough relative to ``T`` to always be CTC-alignable —
    this measures cost, not correctness, so any valid target works.
    """
    model.eval()
    t = temporal_length(input_width)
    target_length = max(1, min(8, t // 4))

    images = torch.rand(batch_size, 1, input_height, input_width, device=device)
    input_lengths = torch.full((batch_size,), t, dtype=torch.long, device=device)
    target_lengths = torch.full((batch_size,), target_length, dtype=torch.long, device=device)
    targets = torch.randint(1, vocab_size, (batch_size * target_length,), device=device)

    def forward_only() -> None:
        model(images, input_lengths)

    def forward_plus_loss() -> None:
        output = model(images, input_lengths)
        ctc_loss_for(output, targets, target_lengths)

    for _ in range(warmup_iterations):
        forward_only()
        forward_plus_loss()
    _synchronize(device)

    forward_only_ms = _time_it(forward_only, device, iterations)
    forward_plus_loss_ms = _time_it(forward_plus_loss, device, iterations)

    return RoundtripResult(
        device=device.type,
        input_width=input_width,
        batch_size=batch_size,
        iterations=iterations,
        forward_only_ms=forward_only_ms,
        forward_plus_loss_ms=forward_plus_loss_ms,
    )


@torch.no_grad()
def _time_it(fn, device: torch.device, iterations: int) -> float:
    samples_ms: list[float] = []
    for _ in range(iterations):
        _synchronize(device)
        started = time.perf_counter()
        fn()
        _synchronize(device)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples_ms)
