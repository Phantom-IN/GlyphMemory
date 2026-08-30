"""The context block every latency number must carry.

``REQUIRED_FIELDS`` and :func:`missing_fields` exist so a field dropped in a refactor fails a test
rather than being noticed months later when a claim can no longer be traced back to how it was
produced — the same pattern ``training.checkpoint``'s ``run.json`` uses for the same reason.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from glyphmemory.runtime.fingerprint import checkpoint_fingerprint

REQUIRED_FIELDS: tuple[str, ...] = (
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
)


def _cpu_model() -> str:
    """Best-effort CPU model name. ``platform.processor()`` returns ``''`` on Apple silicon."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()




@dataclass(frozen=True, slots=True)
class BenchmarkContext:
    """Everything requires, for one measurement."""

    cpu_model: str
    os: str
    python_version: str
    runtime: str
    runtime_version: str
    device: str
    num_threads: int
    model_format: str
    input_height: int
    input_width: int
    batch_size: int
    warmup_iterations: int
    measurement_iterations: int
    model_fingerprint: str
    parameter_count: int
    file_size_bytes: int

    @classmethod
    def capture(
        cls,
        *,
        device: str,
        input_height: int,
        input_width: int,
        batch_size: int,
        warmup_iterations: int,
        measurement_iterations: int,
        checkpoint_path: str | Path,
        parameter_count: int,
        model_format: str = "pytorch_fp32",
    ) -> BenchmarkContext:
        path = Path(checkpoint_path)
        return cls(
            cpu_model=_cpu_model(),
            os=platform.platform(),
            python_version=sys.version.split()[0],
            runtime="pytorch",
            runtime_version=torch.__version__,
            device=device,
            num_threads=torch.get_num_threads(),
            model_format=model_format,
            input_height=input_height,
            input_width=input_width,
            batch_size=batch_size,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            model_fingerprint=checkpoint_fingerprint(path),
            parameter_count=parameter_count,
            file_size_bytes=path.stat().st_size,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def missing_fields(payload: dict[str, Any]) -> list[str]:
    """Required keys absent from ``payload``. Empty means the record is complete."""
    return [name for name in REQUIRED_FIELDS if name not in payload]
