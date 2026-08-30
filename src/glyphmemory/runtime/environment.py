"""Environment capture for reproducibility.

Every experiment stores an ``environment.json`` so a result can be traced without terminal history.

Git state is captured opportunistically: the repository may not be initialised yet, and that must
degrade to ``None`` rather than crash a training run.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import torch

from glyphmemory.runtime.device import available_devices


def _git(*args: str, cwd: Path | None = None) -> str | None:
    """Run a git command, returning ``None`` if git or the repository is absent."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


@dataclass(frozen=True)
class GitState:
    commit: str | None = None
    branch: str | None = None
    dirty: bool | None = None

    @classmethod
    def capture(cls, cwd: Path | None = None) -> GitState:
        commit = _git("rev-parse", "HEAD", cwd=cwd)
        if commit is None:
            # Not a repository, or no commit yet. Both are legitimate early states.
            return cls()
        branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)
        status = _git("status", "--porcelain", cwd=cwd)
        return cls(commit=commit, branch=branch, dirty=bool(status))


@dataclass(frozen=True)
class Environment:
    """Everything needed to reproduce, or at least explain, a run."""

    timestamp_utc: str
    python_version: str
    platform: str
    machine: str
    processor: str
    torch_version: str
    glyphmemory_version: str
    devices_available: list[str] = field(default_factory=list)
    git: GitState = field(default_factory=GitState)

    @classmethod
    def capture(cls, cwd: Path | None = None) -> Environment:
        from glyphmemory import __version__

        return cls(
            timestamp_utc=datetime.now(UTC).isoformat(),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            machine=platform.machine(),
            processor=platform.processor() or platform.machine(),
            torch_version=torch.__version__,
            glyphmemory_version=__version__,
            devices_available=available_devices(),
            git=GitState.capture(cwd),
        )

    def as_dict(self) -> dict:
        return asdict(self)
