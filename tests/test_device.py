"""Device resolution tests.

Availability is monkeypatched so the policy is verified identically on any machine — CI runners have
neither MPS nor CUDA, and the policy must still be tested.
"""

from __future__ import annotations

import pytest

from glyphmemory.runtime import device as device_mod
from glyphmemory.runtime.device import ResolvedDevice, available_devices, resolve_device


@pytest.fixture
def no_accelerators(monkeypatch):
    monkeypatch.setattr(device_mod, "mps_available", lambda: False)
    monkeypatch.setattr(device_mod, "cuda_available", lambda: False)


@pytest.fixture
def only_mps(monkeypatch):
    monkeypatch.setattr(device_mod, "mps_available", lambda: True)
    monkeypatch.setattr(device_mod, "cuda_available", lambda: False)


@pytest.fixture
def only_cuda(monkeypatch):
    monkeypatch.setattr(device_mod, "mps_available", lambda: False)
    monkeypatch.setattr(device_mod, "cuda_available", lambda: True)


@pytest.fixture
def both_accelerators(monkeypatch):
    monkeypatch.setattr(device_mod, "mps_available", lambda: True)
    monkeypatch.setattr(device_mod, "cuda_available", lambda: True)


def test_auto_prefers_mps(only_mps):
    assert resolve_device("auto").kind == "mps"


def test_auto_prefers_mps_over_cuda(both_accelerators):
    """Internal helper."""
    assert resolve_device("auto").kind == "mps"


def test_auto_falls_back_to_cuda(only_cuda):
    assert resolve_device("auto").kind == "cuda"


def test_auto_falls_back_to_cpu(no_accelerators):
    resolved = resolve_device("auto")
    assert resolved.kind == "cpu"
    assert resolved.requested == "auto"


def test_cpu_always_resolves(no_accelerators):
    assert resolve_device("cpu").kind == "cpu"


@pytest.mark.parametrize("requested", ["mps", "cuda"])
def test_unavailable_accelerator_raises_rather_than_falling_back(no_accelerators, requested):
    """Explicit request for a missing accelerator is an error, never a silent CPU run."""
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device(requested)


def test_unknown_device_rejected():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")


def test_request_is_case_and_space_insensitive(only_mps):
    assert resolve_device("  MPS ").kind == "mps"


def test_available_devices_always_includes_cpu_last(no_accelerators):
    assert available_devices() == ["cpu"]


def test_available_devices_orders_by_preference(both_accelerators):
    assert available_devices() == ["mps", "cuda", "cpu"]


def test_resolved_device_exposes_torch_device(no_accelerators):
    resolved = resolve_device("cpu")
    assert resolved.torch_device.type == "cpu"
    assert resolved.is_accelerator is False


def test_resolved_device_serialises():
    resolved = ResolvedDevice("cpu", "auto", "test")
    assert resolved.as_dict()["kind"] == "cpu"
    assert set(resolved.as_dict()) == {"kind", "requested", "reason", "backend_version"}


def test_resolution_is_logged(no_accelerators, caplog):
    """The device must never be chosen silently."""
    with caplog.at_level("INFO", logger="glyphmemory.runtime.device"):
        resolve_device("auto")
    assert any("Resolved device" in record.message for record in caplog.records)
