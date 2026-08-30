"""The run record — what makes a training run reproducible.

The field list is not free-form. :data:`REQUIRED_FIELDS` names it, and a test asserts every one is
present — a field quietly dropped during a refactor would otherwise be noticed months later, by
which time the runs that lack it are already the evidence.

Nothing here is computed for the first time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from glyphmemory.config.schema import Config, to_dict
from glyphmemory.ctc.tokenizer import Tokenizer
from glyphmemory.data.manifest import manifest_fingerprint
from glyphmemory.model.model_info import parameter_count
from glyphmemory.runtime.device import ResolvedDevice
from glyphmemory.runtime.environment import Environment

#: Every key ``run.json`` must carry. Asserted by test.
REQUIRED_FIELDS: tuple[str, ...] = (
    "run_id",
    "git_commit",
    "config",
    "seed",
    "torch_version",
    "python_version",
    "device",
    "device_backend_version",
    "manifest_fingerprints",
    "charset_fingerprint",
    "tokenizer_fingerprint",
    "parameter_count",
    "model",
    "environment",
    "started_utc",
)


def manifest_fingerprints(**manifests: str | Path | None) -> dict[str, str]:
    """Fingerprint each named manifest. Missing paths are skipped, not faked.

    Example::

        manifest_fingerprints(train="runs/x/train.jsonl", val=None) # -> {"train": "9f2c..."}
    """
    fingerprints: dict[str, str] = {}
    for name, path in manifests.items():
        if path is None:
            continue
        candidate = Path(path)
        if candidate.is_file():
            fingerprints[name] = manifest_fingerprint(candidate)
    return fingerprints


def build_run_record(
    *,
    run_id: str,
    config: Config,
    tokenizer: Tokenizer,
    device: ResolvedDevice,
    model: Any,
    seed: int,
    manifests: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the record written to ``run.json``."""
    environment = Environment.capture(cwd=Path.cwd())

    record: dict[str, Any] = {
        "run_id": run_id,
        "git_commit": environment.git.commit,
        "git_branch": environment.git.branch,
        "git_dirty": environment.git.dirty,
        "config": to_dict(config),
        "seed": seed,
        "torch_version": environment.torch_version,
        "python_version": environment.python_version,
        "device": device.kind,
        "device_backend_version": device.as_dict().get("backend_version"),
        "device_reason": device.reason,
        "manifest_fingerprints": dict(manifests or {}),
        "charset_fingerprint": tokenizer.charset.fingerprint(),
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "parameter_count": parameter_count(model),
        "model": model.describe(),
        "environment": environment.as_dict(),
        "started_utc": environment.timestamp_utc,
    }
    if extra:
        record.update(extra)
    return record


def missing_fields(record: dict[str, Any]) -> list[str]:
    """Required keys absent from ``record``. Empty means the record is complete."""
    return [name for name in REQUIRED_FIELDS if name not in record]
