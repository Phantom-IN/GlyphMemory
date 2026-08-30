"""LR schedule tests.

The schedule is driven by *total optimizer steps*. A cosine computed against the wrong horizon
decays to the wrong place and never errors, so the shape is checked at the three points that pin it:
the first step, the end of warmup, and the last step.
"""

from __future__ import annotations

import pytest
import torch

from glyphmemory.config.schema import TrainingConfig
from glyphmemory.training.schedule import WarmupCosine, build_scheduler, warmup_steps_for


class TestWarmupCosineShape:
    def test_ramps_linearly_through_warmup(self) -> None:
        schedule = WarmupCosine(total_steps=100, warmup_steps=10, min_factor=0.0)
        assert schedule(0) == pytest.approx(0.1)
        assert schedule(4) == pytest.approx(0.5)
        assert schedule(9) == pytest.approx(1.0)

    def test_peaks_at_exactly_one_at_the_end_of_warmup(self) -> None:
        schedule = WarmupCosine(total_steps=1000, warmup_steps=50, min_factor=0.0)
        assert schedule(49) == pytest.approx(1.0)
        assert schedule(50) == pytest.approx(1.0)

    def test_first_step_is_not_zero(self) -> None:
        """A zero-rate step is a wasted forward and backward pass."""
        assert WarmupCosine(total_steps=100, warmup_steps=10, min_factor=0.0)(0) > 0

    def test_decays_to_min_factor_at_the_last_step(self) -> None:
        schedule = WarmupCosine(total_steps=100, warmup_steps=10, min_factor=0.01)
        assert schedule(99) == pytest.approx(0.01, abs=1e-3)

    def test_never_negative_and_never_above_one(self) -> None:
        schedule = WarmupCosine(total_steps=500, warmup_steps=25, min_factor=0.0)
        for step in range(600):
            assert 0.0 <= schedule(step) <= 1.0

    def test_monotone_decreasing_after_warmup(self) -> None:
        schedule = WarmupCosine(total_steps=200, warmup_steps=20, min_factor=0.0)
        values = [schedule(step) for step in range(20, 200)]
        assert values == sorted(values, reverse=True)

    def test_clamps_past_the_horizon(self) -> None:
        """A run that overshoots its declared step count must not decay below the floor."""
        schedule = WarmupCosine(total_steps=100, warmup_steps=10, min_factor=0.05)
        assert schedule(500) == pytest.approx(0.05)

    def test_no_warmup(self) -> None:
        schedule = WarmupCosine(total_steps=100, warmup_steps=0, min_factor=0.0)
        assert schedule(0) == pytest.approx(1.0)

    def test_degenerate_total_below_warmup(self) -> None:
        """``total_steps < warmup_steps`` must not divide by zero or go negative."""
        schedule = WarmupCosine(total_steps=5, warmup_steps=10, min_factor=0.1)
        for step in range(20):
            assert 0.0 <= schedule(step) <= 1.0

    def test_single_step_run(self) -> None:
        schedule = WarmupCosine(total_steps=1, warmup_steps=0, min_factor=0.1)
        assert 0.0 <= schedule(0) <= 1.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"total_steps": 0, "warmup_steps": 0, "min_factor": 0.0}, "total_steps"),
            ({"total_steps": 10, "warmup_steps": -1, "min_factor": 0.0}, "warmup_steps"),
            ({"total_steps": 10, "warmup_steps": 1, "min_factor": 2.0}, "min_factor"),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            WarmupCosine(**kwargs)

    def test_rejects_negative_step(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            WarmupCosine(total_steps=10, warmup_steps=1, min_factor=0.0)(-1)


class TestWarmupSteps:
    def test_ratio_is_applied(self) -> None:
        assert warmup_steps_for(1000, 0.05) == 50

    def test_never_rounds_a_positive_ratio_to_zero(self) -> None:
        """Short runs are exactly where instability bites; silently dropping warmup is worst."""
        assert warmup_steps_for(10, 0.05) == 1

    def test_zero_ratio_means_zero(self) -> None:
        assert warmup_steps_for(1000, 0.0) == 0

    def test_capped_at_total(self) -> None:
        assert warmup_steps_for(10, 1.0) == 10

    def test_single_step_run_gets_no_warmup(self) -> None:
        assert warmup_steps_for(1, 0.5) == 0


class TestBuildScheduler:
    def _optimizer(self, lr: float = 3e-4):
        return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=lr)

    def test_drives_the_optimizer(self) -> None:
        optimizer = self._optimizer()
        scheduler, schedule = build_scheduler(optimizer, TrainingConfig(), total_steps=100)
        assert schedule.warmup_steps == 5
        rates = []
        for _ in range(100):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        assert rates[4] == pytest.approx(3e-4)
        assert rates[-1] < rates[4]
        assert min(rates) > 0

    def test_min_factor_comes_from_the_config_ratio(self) -> None:
        _, schedule = build_scheduler(
            self._optimizer(),
            TrainingConfig(learning_rate=1e-3, min_learning_rate=1e-6),
            total_steps=10,
        )
        assert schedule.min_factor == pytest.approx(1e-3)

    def test_describe_is_recorded_shaped(self) -> None:
        _, schedule = build_scheduler(self._optimizer(), TrainingConfig(), total_steps=50)
        described = schedule.describe()
        assert described["kind"] == "linear_warmup_cosine_decay"
        assert described["total_steps"] == 50

    def test_rejects_min_above_base(self) -> None:
        with pytest.raises(ValueError, match="decay upward"):
            build_scheduler(
                self._optimizer(),
                TrainingConfig(learning_rate=1e-6, min_learning_rate=1e-3),
                total_steps=10,
            )

    def test_rejects_non_positive_learning_rate(self) -> None:
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            build_scheduler(self._optimizer(), TrainingConfig(learning_rate=0.0), total_steps=10)
