"""AMP, verified rather than assumed.

Two things are checked together, because a speed gain that also corrupts the output is not a gain:
numerical parity against FP32 (max absolute logit difference) and the actual wall-clock speedup.
Neither is invented — both come out of the same measured pass.
"""

from __future__ import annotations

import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from glyphmemory.data.preprocessing import temporal_length

#: The default lower-precision dtype per device kind. CPU autocast does not support float16
#: meaningfully; bfloat16 is the supported combination there.
_DEFAULT_DTYPE = {
    "mps": torch.float16,
    "cuda": torch.float16,
    "cpu": torch.bfloat16,
}

#: Below this, a speedup is noise, not a finding worth recommending.
_MEANINGFUL_SPEEDUP = 1.02


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@dataclass(frozen=True, slots=True)
class AMPVerdict:
    """Whether autocast is numerically safe and actually faster, on one device."""

    device: str
    dtype: str
    input_width: int
    batch_size: int
    iterations: int
    fp32_ms: float
    amp_ms: float | None
    speedup: float | None
    max_abs_logit_diff: float | None
    error: str | None
    supported: bool
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self) -> str:
        verdict = "SUPPORTED" if self.supported else "NOT RECOMMENDED"
        amp_ms = "n/a" if self.amp_ms is None else f"{self.amp_ms:.2f} ms"
        speedup = "n/a" if self.speedup is None else f"{self.speedup:.2f}x"
        diff = "n/a" if self.max_abs_logit_diff is None else f"{self.max_abs_logit_diff:.2e}"
        return (
            f"AMP  {self.device}/{self.dtype}   fp32 {self.fp32_ms:.2f} ms   amp {amp_ms}   "
            f"speedup {speedup}   max|diff| {diff}   {verdict}\n"
            f"     {self.notes}"
        )


@torch.no_grad()
def verify_amp(
    model: nn.Module,
    *,
    device: torch.device,
    input_height: int,
    input_width: int,
    batch_size: int,
    dtype: torch.dtype | None = None,
    warmup_iterations: int = 5,
    iterations: int = 20,
) -> AMPVerdict:
    """Measure autocast correctness and speed against FP32, on ``device``."""
    resolved_dtype = dtype or _DEFAULT_DTYPE.get(device.type, torch.bfloat16)
    model.eval()

    images = torch.rand(batch_size, 1, input_height, input_width, device=device)
    lengths = torch.full(
        (batch_size,), temporal_length(input_width), dtype=torch.long, device=device
    )

    fp32_logits = model(images, lengths).logits
    fp32_ms = _time_forward(model, images, lengths, device, warmup_iterations, iterations)

    try:
        with torch.autocast(device_type=device.type, dtype=resolved_dtype):
            amp_logits = model(images, lengths).logits
        amp_ms = _time_forward(
            model,
            images,
            lengths,
            device,
            warmup_iterations,
            iterations,
            autocast_dtype=resolved_dtype,
        )
    except (RuntimeError, NotImplementedError) as exc:
        return AMPVerdict(
            device=device.type,
            dtype=str(resolved_dtype),
            input_width=input_width,
            batch_size=batch_size,
            iterations=iterations,
            fp32_ms=fp32_ms,
            amp_ms=None,
            speedup=None,
            max_abs_logit_diff=None,
            error=str(exc),
            supported=False,
            notes=f"autocast raised on {device.type}: {exc}",
        )

    if not torch.isfinite(amp_logits).all():
        return AMPVerdict(
            device=device.type,
            dtype=str(resolved_dtype),
            input_width=input_width,
            batch_size=batch_size,
            iterations=iterations,
            fp32_ms=fp32_ms,
            amp_ms=amp_ms,
            speedup=fp32_ms / amp_ms if amp_ms else None,
            max_abs_logit_diff=None,
            error=None,
            supported=False,
            notes="autocast produced non-finite logits — rejected regardless of speed",
        )

    max_abs_diff = float((fp32_logits.float() - amp_logits.float()).abs().max())
    speedup = fp32_ms / amp_ms if amp_ms else None
    supported = speedup is not None and speedup >= _MEANINGFUL_SPEEDUP
    notes = (
        f"speedup {speedup:.2f}x {'>=' if supported else '<'} the "
        f"{_MEANINGFUL_SPEEDUP}x meaningful-gain bar"
        if speedup is not None
        else "no timing available"
    )

    return AMPVerdict(
        device=device.type,
        dtype=str(resolved_dtype),
        input_width=input_width,
        batch_size=batch_size,
        iterations=iterations,
        fp32_ms=fp32_ms,
        amp_ms=amp_ms,
        speedup=speedup,
        max_abs_logit_diff=max_abs_diff,
        error=None,
        supported=supported,
        notes=notes,
    )


def _time_forward(
    model: nn.Module,
    images: torch.Tensor,
    lengths: torch.Tensor,
    device: torch.device,
    warmup_iterations: int,
    iterations: int,
    *,
    autocast_dtype: torch.dtype | None = None,
) -> float:
    context = (
        nullcontext()
        if autocast_dtype is None
        else torch.autocast(device_type=device.type, dtype=autocast_dtype)
    )
    with context:
        for _ in range(warmup_iterations):
            model(images, lengths)
        _synchronize(device)

        samples_ms: list[float] = []
        for _ in range(iterations):
            _synchronize(device)
            started = time.perf_counter()
            model(images, lengths)
            _synchronize(device)
            samples_ms.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples_ms)
