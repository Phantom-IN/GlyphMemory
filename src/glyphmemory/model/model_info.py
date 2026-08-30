"""Parameter accounting and budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

PREFERRED_MAX_PARAMETERS = 2_000_000
HARD_MAX_PARAMETERS = 3_000_000

BYTES_PER_FP32 = 4


def parameter_count(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Total number of parameters in ``model``."""
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def parameter_count_by_module(model: nn.Module, *, depth: int = 1) -> dict[str, int]:
    """Parameter counts for immediate submodules, for budget attribution.

    Args:
        depth: How many levels of the module tree to report.
    """
    counts: dict[str, int] = {}
    for name, module in model.named_children():
        counts[name] = parameter_count(module)
        if depth > 1:
            for child, value in parameter_count_by_module(module, depth=depth - 1).items():
                counts[f"{name}.{child}"] = value
    return counts


def fp32_size_bytes(model: nn.Module) -> int:
    """Serialized FP32 size of the parameters, excluding buffers and framework overhead."""
    return parameter_count(model) * BYTES_PER_FP32


@dataclass(frozen=True)
class ParameterReport:
    total: int
    trainable: int
    fp32_megabytes: float
    within_preferred: bool
    within_hard_ceiling: bool
    by_module: dict[str, int]

    def format(self) -> str:
        preferred = "ok" if self.within_preferred else "EXCEEDED"
        ceiling = "ok" if self.within_hard_ceiling else "EXCEEDED"
        lines = [
            f"total parameters   {self.total:>12,}",
            f"trainable          {self.trainable:>12,}",
            f"FP32 size          {self.fp32_megabytes:>11.2f} MB",
            f"preferred <= {PREFERRED_MAX_PARAMETERS:,}   {preferred}",
            f"ceiling   <= {HARD_MAX_PARAMETERS:,}   {ceiling}",
        ]
        if self.by_module:
            lines.append("by module:")
            width = max(len(name) for name in self.by_module)
            for name, count in self.by_module.items():
                lines.append(f"  {name:<{width}}  {count:>12,}")
        return "\n".join(lines)


def parameter_report(model: nn.Module) -> ParameterReport:
    """Build a full parameter report for logging at model construction."""
    total = parameter_count(model)
    return ParameterReport(
        total=total,
        trainable=parameter_count(model, trainable_only=True),
        fp32_megabytes=fp32_size_bytes(model) / (1024 * 1024),
        within_preferred=total <= PREFERRED_MAX_PARAMETERS,
        within_hard_ceiling=total <= HARD_MAX_PARAMETERS,
        by_module=parameter_count_by_module(model),
    )


def assert_within_budget(model: nn.Module, *, ceiling: int = HARD_MAX_PARAMETERS) -> int:
    """Raise if ``model`` exceeds the parameter ceiling. Returns the count when it passes."""
    total = parameter_count(model)
    if total > ceiling:
        raise ValueError(
            f"Model has {total:,} parameters, exceeding the ceiling of {ceiling:,}. "
            "Crossing the ceiling requires an explicit experiment and a recorded decision."
        )
    return total
