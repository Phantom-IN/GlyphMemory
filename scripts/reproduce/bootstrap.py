"""Writer-level bootstrap — recovered historical implementation.

RECOVERED VERBATIM from the session scripts that produced the published intervals. See
``publication/repro/BOOTSTRAP-PROVENANCE.md`` for the file-by-file provenance and for the exact
parameters each published interval used.

Two facts here are load-bearing and must not be tidied:

1. The resampling unit is the **writer**, always, and never the character. Characters within a
   writer are correlated; resampling them would understate the variance.
2. Two distinct upper-percentile index conventions were used historically. ``percentile_ci`` takes
   ``int(0.975 * draws)``; ``percentile_ci_legacy_upper`` takes ``int(0.975 * draws) - 1``. The
   second was used by the gate-C2, IAM-confirmation and CVL runs. Unifying them would silently
   change published numbers by one order statistic.

The implementation deliberately uses ``random.Random`` and ``statistics.fmean`` rather than numpy,
because that is what the historical scripts used and the draw sequence depends on it.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence


def _resampled_means(values: Sequence[float], seed: int, draws: int) -> list[float]:
    """The historical draw loop. One resample of size ``len(values)`` with replacement, per draw."""
    rng = random.Random(seed)
    return sorted(
        statistics.fmean([values[rng.randrange(len(values))] for _ in range(len(values))])
        for _ in range(draws)
    )


def percentile_ci(values: Sequence[float], seed: int = 0, draws: int = 4000) -> tuple[float, float]:
    """Writer-level percentile bootstrap, ``int(0.975 * draws)`` upper index."""
    if not values:
        return (float("nan"), float("nan"))
    means = _resampled_means(values, seed, draws)
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def percentile_ci_legacy_upper(
    values: Sequence[float], seed: int = 0, draws: int = 4000
) -> tuple[float, float]:
    """Writer-level percentile bootstrap, ``int(0.975 * draws) - 1`` upper index.

    Used by: gate C2 (``m10r2_gate_c2.py``, seed 0, **2,000 draws**, indices 50 / 1949), the IAM
    confirmation run (``m10r5a_confirm.py``, seed 0, 4,000 draws, indices 100 / 3899) and the CVL
    external run (``m10r5a_cvl.py``, seed 0, 4,000 draws, indices 100 / 3899).

    The one-index difference from :func:`percentile_ci` is historical and is preserved, not fixed.
    """
    if not values:
        return (float("nan"), float("nan"))
    means = _resampled_means(values, seed, draws)
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def paired_percentile_ci(
    pairs: Sequence[tuple[float, float]], seed: int = 1337, draws: int = 4000
) -> tuple[float, float]:
    """CI of the per-writer difference ``a - b``. Resamples writers, not characters.

    Recovered from ``m11t0_common.paired_bootstrap_ci``. The difference is formed per writer first
    and the bootstrap then runs over those differences, which is what makes the comparison paired.
    """
    return percentile_ci([a - b for a, b in pairs], seed=seed, draws=draws)


def paired_statistic_ci(
    units: Sequence[str],
    statistic,
    seed: int = 1337,
    draws: int = 4000,
) -> tuple[float, float]:
    """Bootstrap of a statistic that must be **recomputed inside each resample**.

    Recovered from ``m10r4_gate_b.py``. Gate B's endpoint is a difference of areas under a curve,
    which is not a mean of per-writer values, so the statistic is recomputed on each resampled
    writer set. ``statistic`` receives a list of writer ids and returns a float, or ``None`` when
    the resample is degenerate (all one class); degenerate draws are dropped, so the percentile
    index is taken over the surviving draws rather than over ``draws``.
    """
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(draws):
        sample = [units[rng.randrange(len(units))] for _ in units]
        value = statistic(sample)
        if value is not None:
            out.append(value)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]
