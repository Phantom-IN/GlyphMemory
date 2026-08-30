"""Content-based identity for on-disk artifacts.

A checkpoint's *path* says nothing about what it contains — a filename can be reused, copied, or
point at a rebuilt file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def checkpoint_fingerprint(path: str | Path) -> str:
    """A short, deterministic identity for a checkpoint file — its content hash, not its path."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]
