"""Device resolution for GlyphMemory.

One abstraction over ``mps`` / ``cuda`` / ``cpu``.

Two policies are enforced here rather than left to callers:

1. ``auto`` prefers MPS, then CUDA, then CPU.
2. An explicitly requested but unavailable device is an **error**, never a silent fallback to CPU. A
   run that quietly drops to CPU and takes twenty times longer is worse than one that refuses to
   start.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass

import torch

VALID_REQUESTS: tuple[str, ...] = ("auto", "cpu", "mps", "cuda")
ACCELERATORS: tuple[str, ...] = ("mps", "cuda")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedDevice:
    """The outcome of device resolution.

    Attributes:
        kind: Concrete backend actually selected — one of ``cpu``/``mps``/``cuda``.
        requested: What the caller asked for, including ``auto``.
        reason: Human-readable explanation, logged and stored in the run record.
        backend_version: Backend build string where one exists, else ``None``.
    """

    kind: str
    requested: str
    reason: str
    backend_version: str | None = None

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.kind)

    @property
    def is_accelerator(self) -> bool:
        return self.kind in ACCELERATORS

    def as_dict(self) -> dict[str, str | None]:
        return {
            "kind": self.kind,
            "requested": self.requested,
            "reason": self.reason,
            "backend_version": self.backend_version,
        }

    def __str__(self) -> str:
        return f"{self.kind} (requested={self.requested}; {self.reason})"


def mps_available() -> bool:
    """Whether a usable MPS backend is present.

    ``is_built()`` alone is not enough: a torch built with MPS still reports unavailable on
    non-Apple-silicon hardware or an unsupported macOS.
    """
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    return bool(backend.is_built() and backend.is_available())


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def _backend_version(kind: str) -> str | None:
    if kind == "cuda":
        return torch.version.cuda
    if kind == "mps":
        # MPS has no separate version string; the macOS build is the meaningful identifier.
        return f"macos-{platform.mac_ver()[0]}" if platform.system() == "Darwin" else None
    return None


def available_devices() -> list[str]:
    """Every backend usable on this machine, most preferred first."""
    found = ["mps"] if mps_available() else []
    if cuda_available():
        found.append("cuda")
    found.append("cpu")
    return found


def resolve_device(requested: str = "auto", *, log: bool = True) -> ResolvedDevice:
    """Resolve a device request into a concrete backend.

    Args:
        requested: One of ``auto``, ``cpu``, ``mps``, ``cuda``.
        log: Emit an INFO line recording the resolution. Defaults to ``True`` because the device
            must never be chosen silently.

    Raises:
        ValueError: ``requested`` is not a recognised device name.
        RuntimeError: A specific accelerator was requested but is unavailable.
    """
    normalized = requested.strip().lower()
    if normalized not in VALID_REQUESTS:
        raise ValueError(
            f"Unknown device {requested!r}. Expected one of: {', '.join(VALID_REQUESTS)}."
        )

    if normalized == "auto":
        if mps_available():
            resolved = ResolvedDevice(
                "mps", normalized, "auto: MPS available", _backend_version("mps")
            )
        elif cuda_available():
            resolved = ResolvedDevice(
                "cuda", normalized, "auto: CUDA available, MPS not", _backend_version("cuda")
            )
        else:
            resolved = ResolvedDevice("cpu", normalized, "auto: no accelerator available")
    elif normalized == "mps":
        if not mps_available():
            raise RuntimeError(
                "Device 'mps' was requested but is not available. "
                "Refusing to fall back to CPU silently — pass --device cpu to run on CPU, "
                "or --device auto to select the best available backend."
            )
        resolved = ResolvedDevice(
            "mps", normalized, "explicitly requested", _backend_version("mps")
        )
    elif normalized == "cuda":
        if not cuda_available():
            raise RuntimeError(
                "Device 'cuda' was requested but is not available. "
                "Refusing to fall back to CPU silently — pass --device cpu to run on CPU, "
                "or --device auto to select the best available backend."
            )
        resolved = ResolvedDevice(
            "cuda", normalized, "explicitly requested", _backend_version("cuda")
        )
    else:
        resolved = ResolvedDevice("cpu", normalized, "explicitly requested")

    if log:
        logger.info("Resolved device: %s", resolved)
    return resolved
