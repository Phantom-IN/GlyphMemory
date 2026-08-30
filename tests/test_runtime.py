"""Environment capture, experiment directories, seeding and logging."""

from __future__ import annotations

import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from glyphmemory.runtime import (
    Environment,
    ExperimentDir,
    new_run_id,
    seed_everything,
    setup_logging,
)
from glyphmemory.runtime.environment import GitState
from glyphmemory.runtime.experiment import sanitize_name

# --------------------------------------------------------------------------- environment


def test_environment_capture_is_complete():
    env = Environment.capture()
    assert env.python_version
    assert env.torch_version
    assert env.glyphmemory_version
    assert "cpu" in env.devices_available
    assert env.timestamp_utc.endswith("+00:00")


def test_environment_serialises_to_json():
    payload = Environment.capture().as_dict()
    assert json.loads(json.dumps(payload))["git"].keys() >= {"commit", "branch", "dirty"}


def test_git_state_degrades_gracefully_outside_a_repository(tmp_path):
    """A missing repository must not crash a run — it is a legitimate early state."""
    state = GitState.capture(cwd=tmp_path)
    assert state.commit is None
    assert state.dirty is None


# --------------------------------------------------------------------------- run ids


def test_run_id_embeds_name_and_timestamp():
    stamp = datetime(2026, 8, 16, 21, 45, 0, tzinfo=UTC)
    assert new_run_id("gm_base_v001", now=stamp) == "gm_base_v001__20260816T214500Z"


def test_run_id_sanitises_unsafe_characters():
    assert new_run_id("gm base/v1", now=datetime(2026, 1, 1, tzinfo=UTC)).startswith("gm_base_v1__")


@pytest.mark.parametrize("name", ["final", "final2", "best_new", "latest_final", "LATEST"])
def test_untraceable_checkpoint_names_rejected(name):
    """Internal helper."""
    with pytest.raises(ValueError, match="not traceable"):
        sanitize_name(name)


def test_empty_name_rejected():
    with pytest.raises(ValueError, match="at least one alphanumeric"):
        sanitize_name("///")


# --------------------------------------------------------------------------- experiment dirs


def test_experiment_dir_creates_expected_layout(tmp_path):
    exp = ExperimentDir.create(tmp_path, "gm_base_v001")
    assert exp.root.is_dir()
    assert exp.checkpoints_dir.is_dir()
    assert exp.root.name == exp.run_id


def test_experiment_dir_refuses_to_overwrite(tmp_path):
    """Runs are never silently overwritten — that destroys the evidence they hold."""
    stamp = datetime(2026, 8, 16, 21, 45, 0, tzinfo=UTC)
    ExperimentDir.create(tmp_path, "gm_base_v001", now=stamp)
    with pytest.raises(FileExistsError):
        ExperimentDir.create(tmp_path, "gm_base_v001", now=stamp)


def test_experiment_dir_writes_json_and_jsonl(tmp_path):
    exp = ExperimentDir.create(tmp_path, "gm_base_v001")
    exp.write_json(exp.metrics_path, {"cer": 0.5})
    exp.append_jsonl(exp.metrics_stream_path, {"step": 1})
    exp.append_jsonl(exp.metrics_stream_path, {"step": 2})

    assert json.loads(exp.metrics_path.read_text())["cer"] == 0.5
    lines = exp.metrics_stream_path.read_text().strip().splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2]


# --------------------------------------------------------------------------- seeding


def test_seed_everything_makes_all_rngs_reproducible():
    seed_everything(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(123)
    second = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == second


def test_different_seeds_diverge():
    seed_everything(1)
    a = torch.rand(4)
    seed_everything(2)
    b = torch.rand(4)
    assert not torch.equal(a, b)


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_out_of_range_seed_rejected(seed):
    with pytest.raises(ValueError, match="Seed must be in"):
        seed_everything(seed)


# --------------------------------------------------------------------------- logging


def test_setup_logging_is_idempotent():
    """Repeated setup must not duplicate handlers and therefore duplicate log lines."""
    first = setup_logging(logging.INFO)
    count = len(first.handlers)
    second = setup_logging(logging.INFO)
    assert len(second.handlers) == count


def test_setup_logging_writes_to_file(tmp_path: Path):
    log_file = tmp_path / "nested" / "run.log"
    logger = setup_logging(logging.INFO, log_file=log_file)
    logger.info("hello from the test")
    for handler in logger.handlers:
        handler.flush()
    assert "hello from the test" in log_file.read_text(encoding="utf-8")
