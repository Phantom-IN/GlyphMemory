"""Parameter accounting and budget enforcement."""

from __future__ import annotations

import pytest
from torch import nn

from glyphmemory.model import (
    HARD_MAX_PARAMETERS,
    PREFERRED_MAX_PARAMETERS,
    assert_within_budget,
    parameter_count,
    parameter_count_by_module,
    parameter_report,
)


class TinyModel(nn.Module):
    """Known parameter count: Linear(10,4) = 44, Linear(4,2) = 10, total 54."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(10, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x):  # No cover - not exercised in
        return self.head(self.encoder(x))


def test_parameter_count_is_exact():
    assert parameter_count(TinyModel()) == 54


def test_trainable_only_excludes_frozen_parameters():
    model = TinyModel()
    for param in model.encoder.parameters():
        param.requires_grad = False
    assert parameter_count(model) == 54
    assert parameter_count(model, trainable_only=True) == 10


def test_counts_by_module_sum_to_total():
    model = TinyModel()
    by_module = parameter_count_by_module(model)
    assert by_module == {"encoder": 44, "head": 10}
    assert sum(by_module.values()) == parameter_count(model)


def test_nested_depth_reporting():
    model = nn.Sequential(TinyModel())
    shallow = parameter_count_by_module(model, depth=1)
    deep = parameter_count_by_module(model, depth=2)
    assert set(shallow) == {"0"}
    assert "0.encoder" in deep


def test_report_flags_budget_compliance():
    report = parameter_report(TinyModel())
    assert report.total == 54
    assert report.within_preferred is True
    assert report.within_hard_ceiling is True
    assert report.fp32_megabytes == pytest.approx(54 * 4 / (1024 * 1024))
    assert "total parameters" in report.format()


def test_assert_within_budget_passes_and_returns_count():
    assert assert_within_budget(TinyModel()) == 54


def test_assert_within_budget_raises_when_exceeded():
    """The ceiling is enforced by a failing test, not by good intentions."""
    model = TinyModel()
    with pytest.raises(ValueError, match="exceeding the ceiling"):
        assert_within_budget(model, ceiling=10)


def test_budget_constants_match_documented_policy():
    """Internal helper."""
    assert PREFERRED_MAX_PARAMETERS == 2_000_000
    assert HARD_MAX_PARAMETERS == 3_000_000


def test_oversized_model_would_be_rejected_at_the_documented_ceiling():
    """A model above 3M must fail the default budget check once GM-Base lands in."""
    oversized = nn.Linear(2000, 2000)  # 4,002,000 parameters
    assert parameter_count(oversized) > HARD_MAX_PARAMETERS
    with pytest.raises(ValueError, match="exceeding the ceiling"):
        assert_within_budget(oversized)
