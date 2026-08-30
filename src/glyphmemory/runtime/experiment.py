"""Experiment directory conventions.

    artifacts/runs/<run_id>/
        config.yaml         the exact config used
        environment.json    python/torch/platform/git/device capture
        metrics.json        final metrics
        metrics.jsonl       per-step or per-epoch metrics, appended
        predictions.jsonl   prediction-level output
        checkpoints/

Run IDs embed the experiment name and a UTC timestamp so directories sort chronologically and remain
traceable to their configuration::

    gm_base_h64_gru192x2_iam_v001__20260816T214500Z
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_ID_TIME_FORMAT = "%Y%m%dT%H%M%SZ"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

# Names that make a checkpoint untraceable.
FORBIDDEN_NAMES = frozenset({"final", "final2", "best_new", "latest_final", "latest", "new"})


def sanitize_name(name: str) -> str:
    """Reduce an experiment name to a filesystem-safe token."""
    cleaned = _SAFE_NAME.sub("_", name.strip()).strip("_")
    if not cleaned:
        raise ValueError("Experiment name must contain at least one alphanumeric character.")
    if cleaned.lower() in FORBIDDEN_NAMES:
        raise ValueError(
            f"Experiment name {name!r} is not traceable. Use a descriptive ID such as "
            "'gm_base_h64_gru192x2_iam_v001'."
        )
    return cleaned


def new_run_id(name: str, *, now: datetime | None = None) -> str:
    """Build a run ID from an experiment name and a UTC timestamp."""
    stamp = (now or datetime.now(UTC)).strftime(RUN_ID_TIME_FORMAT)
    return f"{sanitize_name(name)}__{stamp}"


@dataclass(frozen=True)
class ExperimentDir:
    """A created run directory and the paths it guarantees."""

    root: Path
    run_id: str

    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"

    @property
    def environment_path(self) -> Path:
        return self.root / "environment.json"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.json"

    @property
    def metrics_stream_path(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def predictions_path(self) -> Path:
        return self.root / "predictions.jsonl"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @classmethod
    def create(cls, base: Path, name: str, *, now: datetime | None = None) -> ExperimentDir:
        """Create ``base/<run_id>/`` and its checkpoints subdirectory.

        Raises:
            FileExistsError: The run directory already exists. Runs are never silently overwritten —
                that would destroy the evidence the directory holds.
        """
        run_id = new_run_id(name, now=now)
        root = Path(base) / run_id
        if root.exists():
            raise FileExistsError(f"Run directory already exists: {root}")
        (root / "checkpoints").mkdir(parents=True)
        return cls(root=root, run_id=run_id)

    def write_json(self, path: Path, payload: dict[str, Any]) -> Path:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def append_jsonl(self, path: Path, record: dict[str, Any]) -> Path:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return path
