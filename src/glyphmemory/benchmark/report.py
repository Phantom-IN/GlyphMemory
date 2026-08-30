"""Assembles one `glyphmemory benchmark` run: the latency grid, the MPS CTC round-trip cost, and the
AMP verdict — each point self-contained with the context requires, so no number in the output can be
quoted without its hardware and shape.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from glyphmemory.benchmark.amp_check import AMPVerdict, verify_amp
from glyphmemory.benchmark.context import BenchmarkContext
from glyphmemory.benchmark.latency import LatencyMeasurement, measure_forward_latency
from glyphmemory.benchmark.memory import peak_rss_bytes
from glyphmemory.benchmark.roundtrip import RoundtripResult, measure_ctc_roundtrip
from glyphmemory.config.schema import Config
from glyphmemory.ctc.tokenizer import Tokenizer
from glyphmemory.model.htr import GMBase
from glyphmemory.model.model_info import parameter_count
from glyphmemory.training.checkpoint import load_checkpoint

DEFAULT_WIDTHS: tuple[int, ...] = (256, 512, 1024)
DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 8, 16)


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One (width, batch) measurement, self-contained with its full context."""

    context: BenchmarkContext
    measurement: LatencyMeasurement

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.context.as_dict(),
            "p50_ms": self.measurement.p50_ms,
            "p95_ms": self.measurement.p95_ms,
            "mean_ms": self.measurement.mean_ms,
            "lines_per_second": self.measurement.lines_per_second,
        }

    def format(self) -> str:
        return self.measurement.format()


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    checkpoint: str
    device: str
    grid: tuple[GridPoint, ...]
    peak_rss_bytes: int | None
    roundtrip_on_device: RoundtripResult
    roundtrip_on_cpu: RoundtripResult
    amp: AMPVerdict

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "device": self.device,
            "grid": [point.as_dict() for point in self.grid],
            "peak_rss_bytes": self.peak_rss_bytes,
            "roundtrip_on_device": self.roundtrip_on_device.as_dict(),
            "roundtrip_on_cpu": self.roundtrip_on_cpu.as_dict(),
            "amp": self.amp.as_dict(),
        }

    def format(self) -> str:
        lines = [
            f"checkpoint   {self.checkpoint}",
            f"device       {self.device}",
        ]
        if self.grid:
            ctx = self.grid[0].context
            lines.append(
                f"context      {ctx.cpu_model}   {ctx.os}   python {ctx.python_version}   "
                f"{ctx.runtime} {ctx.runtime_version}   threads {ctx.num_threads}"
            )
            lines.append(
                f"model        {ctx.parameter_count:,} params   "
                f"{ctx.file_size_bytes / 1e6:.2f} MB on disk   "
                f"fingerprint {ctx.model_fingerprint}"
            )
        rss = "n/a (platform does not report it)" if self.peak_rss_bytes is None else (
            f"{self.peak_rss_bytes / 1e6:.1f} MB"
        )
        lines.append(f"peak RSS     {rss}")
        lines.append("")
        lines.append("latency (forward pass only, eval mode, no grad):")
        for point in self.grid:
            lines.append(point.format())
        lines.append("")
        lines.append("MPS CTC round-trip cost (forward-only vs forward+loss):")
        lines.append(self.roundtrip_on_device.format())
        if self.roundtrip_on_cpu.device != self.roundtrip_on_device.device:
            lines.append(self.roundtrip_on_cpu.format())
        lines.append("")
        lines.append(self.amp.format())
        return "\n".join(lines)


def run_benchmark(
    checkpoint_path: str | Path,
    *,
    config: Config,
    tokenizer: Tokenizer,
    device: torch.device,
    widths: tuple[int, ...] = DEFAULT_WIDTHS,
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
    input_height: int = 64,
    warmup_iterations: int = 5,
    measurement_iterations: int = 20,
    threads: int | None = None,
) -> BenchmarkReport:
    """Run the full benchmark: latency grid, CTC round-trip cost, AMP verdict."""
    if threads is not None:
        torch.set_num_threads(threads)

    loaded = load_checkpoint(checkpoint_path, charset_fingerprint=tokenizer.charset.fingerprint())
    model = GMBase.from_config(config.model, tokenizer.vocab_size)
    model.load_state_dict(loaded.model_state)
    model.to(device)
    model.eval()

    total_params = parameter_count(model)

    grid: list[GridPoint] = []
    for width in widths:
        for batch_size in batch_sizes:
            measurement = measure_forward_latency(
                model,
                device=device,
                input_height=input_height,
                input_width=width,
                batch_size=batch_size,
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
            )
            context = BenchmarkContext.capture(
                device=device.type,
                input_height=input_height,
                input_width=width,
                batch_size=batch_size,
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
                checkpoint_path=checkpoint_path,
                parameter_count=total_params,
            )
            grid.append(GridPoint(context=context, measurement=measurement))

    rss = peak_rss_bytes()

    roundtrip_width = _closest(widths, 512)
    roundtrip_batch = _closest(batch_sizes, 8)
    roundtrip_on_device = measure_ctc_roundtrip(
        model,
        device=device,
        input_height=input_height,
        input_width=roundtrip_width,
        batch_size=roundtrip_batch,
        vocab_size=tokenizer.vocab_size,
        warmup_iterations=warmup_iterations,
        iterations=measurement_iterations,
    )
    if device.type != "cpu":
        cpu_model = copy.deepcopy(model).to(torch.device("cpu"))
        roundtrip_on_cpu = measure_ctc_roundtrip(
            cpu_model,
            device=torch.device("cpu"),
            input_height=input_height,
            input_width=roundtrip_width,
            batch_size=roundtrip_batch,
            vocab_size=tokenizer.vocab_size,
            warmup_iterations=warmup_iterations,
            iterations=measurement_iterations,
        )
    else:
        roundtrip_on_cpu = roundtrip_on_device

    amp = verify_amp(
        model,
        device=device,
        input_height=input_height,
        input_width=roundtrip_width,
        batch_size=roundtrip_batch,
        warmup_iterations=warmup_iterations,
        iterations=measurement_iterations,
    )

    return BenchmarkReport(
        checkpoint=str(checkpoint_path),
        device=device.type,
        grid=tuple(grid),
        peak_rss_bytes=rss,
        roundtrip_on_device=roundtrip_on_device,
        roundtrip_on_cpu=roundtrip_on_cpu,
        amp=amp,
    )


def _closest(values: tuple[int, ...], target: int) -> int:
    return min(values, key=lambda v: abs(v - target))
