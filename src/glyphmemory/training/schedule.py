"""Learning-rate schedule: linear warmup, then cosine decay.

    linear warmup over the first ~5% of steps, then cosine decay, min lr ~1e-6

**The schedule is driven by total optimizer steps, not by epochs.** It has to be told the step count
up front. Deriving it from ``epochs * len(loader)`` inside the scheduler looks equivalent and is
not: a loader that drops a partial batch, a ``--max-steps`` cap, or a run stopped early all change
the real step count, and a cosine curve computed against the wrong horizon decays to the wrong place
without ever erroring. Passing the number in makes the horizon an explicit input that the run record
can carry.

Warmup exists because CTC is unstable in its first few hundred steps: the model emits blank
everywhere, the loss sits near ``-log(1/C)``, and a full learning rate at that point drives it into
an all-blank optimum it does not leave.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from glyphmemory.config.schema import TrainingConfig


# Deliberately not ``slots=True``: ``LambdaLR.state_dict()`` serializes a non-function callable by
# copying its ``__dict__``, which a slotted class does not have. Adding slots here would save
# nothing and break checkpointing with an ``AttributeError`` at save time.
@dataclass(frozen=True)
class WarmupCosine:
    """Multiplier on the base learning rate, as a function of step index.

    Returns a *factor* in ``[min_factor, 1.0]`` rather than an absolute rate, so it composes with
    :class:`torch.optim.lr_scheduler.LambdaLR` and with per-group base rates.

    Attributes:
        total_steps: Optimizer steps the run will take. Must be known in advance.
        warmup_steps: Steps spent ramping linearly from ``0`` to the base rate.
        min_factor: Floor the cosine decays to, as a fraction of the base rate.
    """

    total_steps: int
    warmup_steps: int
    min_factor: float

    def __post_init__(self) -> None:
        if self.total_steps < 1:
            raise ValueError(f"total_steps must be at least 1, got {self.total_steps}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if not 0.0 <= self.min_factor <= 1.0:
            raise ValueError(f"min_factor must be in [0, 1], got {self.min_factor}")

    def __call__(self, step: int) -> float:
        """Factor for ``step`` (0-based, as ``LambdaLR`` counts)."""
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")

        if self.warmup_steps and step < self.warmup_steps:
            # (step + 1) so the first step is not exactly zero — a zero-rate step is a wasted
            # forward and backward pass.
            return (step + 1) / self.warmup_steps

        decay_steps = max(self.total_steps - self.warmup_steps, 1)
        progress = min((step - self.warmup_steps) / decay_steps, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_factor + (1.0 - self.min_factor) * cosine

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "linear_warmup_cosine_decay",
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_factor": self.min_factor,
        }


def warmup_steps_for(total_steps: int, warmup_ratio: float) -> int:
    """Warmup length from a ratio.

    Clamped to ``[0, total_steps]``. At least one step whenever the ratio is positive and the run
    has more than one step — a warmup rounded down to zero silently removes the stabiliser on the
    exact short runs where instability is most likely.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be at least 1, got {total_steps}")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(f"warmup_ratio must be in [0, 1], got {warmup_ratio}")
    steps = int(total_steps * warmup_ratio)
    if warmup_ratio > 0 and steps == 0 and total_steps > 1:
        steps = 1
    return min(steps, total_steps)


def build_scheduler(
    optimizer: Optimizer, config: TrainingConfig, *, total_steps: int
) -> tuple[LambdaLR, WarmupCosine]:
    """Build the schedule for a run of exactly ``total_steps`` optimizer steps.

    Returns both the ``LambdaLR`` and the underlying :class:`WarmupCosine`, because the run record
    stores the schedule's parameters and reconstructing them from the scheduler object afterwards is
    guesswork.
    """
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.learning_rate}")
    if config.min_learning_rate < 0:
        raise ValueError(f"min_learning_rate must be non-negative, got {config.min_learning_rate}")
    if config.min_learning_rate > config.learning_rate:
        raise ValueError(
            f"min_learning_rate ({config.min_learning_rate}) exceeds learning_rate "
            f"({config.learning_rate}); the cosine would decay upward."
        )

    schedule = WarmupCosine(
        total_steps=total_steps,
        warmup_steps=warmup_steps_for(total_steps, config.warmup_ratio),
        min_factor=config.min_learning_rate / config.learning_rate,
    )
    return LambdaLR(optimizer, lr_lambda=schedule), schedule
