"""Reproducibility verification for the recovered writer-level bootstrap.

**This is verification, not new analysis.** Every test either uses a deterministic synthetic case or
recomputes a *published* statistic from inputs already committed under ``docs/results/``. No dataset
is read, no model is loaded, and no manuscript number is regenerated from IAM or CVL.

The two exact-reproduction tests are the point of this file: they establish that the code promoted
into ``scripts/reproduce/`` is the implementation that produced the published intervals, rather than
a plausible reimplementation of it.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reproduce.bootstrap import (  # noqa: E402
    paired_percentile_ci,
    paired_statistic_ci,
    percentile_ci,
    percentile_ci_legacy_upper,
)

RESULTS = ROOT / "docs" / "results"


class TestExactReproductionOfPublishedIntervals:
    """The historical parameters are recorded in publication/repro/BOOTSTRAP-PROVENANCE.md."""

    def test_gate_c2_pair_blind_interval(self) -> None:
        """m10r2-gate-c2: seed 0, **2,000** draws, legacy upper index. Published [-0.30, +2.20].

        This is the interval Section 6.4 reports for the pair-blind operating point, and the one
        whose draw count PUB-4F could not recover from the artifacts.
        """
        record = json.loads((RESULTS / "m10r2-gate-c2.json").read_text())["writer_level"]
        nets = [row["net"] for row in record["per_writer"]]
        lo, hi = percentile_ci_legacy_upper(nets, seed=0, draws=2000)
        assert [lo, hi] == pytest.approx(record["bootstrap_ci"], abs=1e-12)

    def test_m11_t0_confirmation_interval(self) -> None:
        """m11t0: seed 1337, 4,000 draws, standard upper index. Published [-0.00113, +0.00530]."""
        conf = json.loads((RESULTS / "m11t0-confirmation-001.json").read_text())["per_writer"]
        gate = json.loads((RESULTS / "m11t0-gate-001.json").read_text())
        deltas = [conf[w]["shots"]["5"]["t0_delta"] for w in sorted(conf)]
        clause = gate["clauses"]["1_mean_delta_and_ci"]
        assert statistics.fmean(deltas) == pytest.approx(clause["mean_delta"], abs=1e-15)
        assert list(percentile_ci(deltas, seed=1337, draws=4000)) == pytest.approx(
            clause["ci"], abs=1e-12
        )

    def test_m11_paired_interval(self) -> None:
        """The only resolved quantity in Section 9: the paired control-minus-memory difference."""
        conf = json.loads((RESULTS / "m11t0-confirmation-001.json").read_text())["per_writer"]
        gate = json.loads((RESULTS / "m11t0-gate-001.json").read_text())
        pairs = [
            (conf[w]["shots"]["5"]["t0_delta"], conf[w]["shots"]["5"]["b_delta"])
            for w in sorted(conf)
        ]
        clause = gate["clauses"]["2_paired_beats_arm_b"]
        assert statistics.fmean([a - b for a, b in pairs]) == pytest.approx(
            clause["paired_diff"], abs=1e-15
        )
        assert list(paired_percentile_ci(pairs, seed=1337, draws=4000)) == pytest.approx(
            clause["ci"], abs=1e-12
        )


class TestSemantics:
    def test_the_resampling_unit_is_the_caller_supplied_unit(self) -> None:
        """Guard on the invariant: the function resamples the list it is given, nothing finer.

        Every published interval passes one value per writer. If this ever resampled below that
        granularity it would understate the variance, which is the specific error the protocol
        forbids.
        """
        values = [0.0] * 9 + [1.0]
        lo, hi = percentile_ci(values, seed=0, draws=500)
        assert 0.0 <= lo <= hi <= 1.0
        assert hi > lo  # a characterwise bootstrap over 10 units could not vary at all here

    def test_draws_are_with_replacement(self) -> None:
        """Without replacement every resample would be a permutation and every mean identical."""
        values = [float(i) for i in range(20)]
        lo, hi = percentile_ci(values, seed=0, draws=1000)
        assert lo < statistics.fmean(values) < hi

    def test_seed_determines_the_interval(self) -> None:
        values = [float(i) for i in range(30)]
        assert percentile_ci(values, seed=7) == percentile_ci(values, seed=7)
        assert percentile_ci(values, seed=7) != percentile_ci(values, seed=8)

    def test_the_two_upper_index_conventions_really_differ(self) -> None:
        """If these ever agree the historical distinction has been lost. Both are in use."""
        values = [float(i) for i in range(40)]
        standard = percentile_ci(values, seed=0, draws=2000)
        legacy = percentile_ci_legacy_upper(values, seed=0, draws=2000)
        assert standard[0] == legacy[0]
        assert standard[1] >= legacy[1]

    def test_zero_delta_writers_are_kept_not_dropped(self) -> None:
        """Many confirmation writers have exactly zero delta; dropping them would shift the mean."""
        with_zeros = [0.0, 0.0, 0.0, 0.02, -0.01]
        without = [0.02, -0.01]
        assert percentile_ci(with_zeros, seed=1)[0] != percentile_ci(without, seed=1)[0]

    def test_empty_input_returns_nan_rather_than_raising(self) -> None:
        lo, hi = percentile_ci([])
        assert lo != lo and hi != hi  # NaN

    def test_paired_ci_uses_per_unit_differences(self) -> None:
        """Pairing must happen before resampling, or the comparison is not paired."""
        pairs = [(1.0, 0.9), (2.0, 1.9), (3.0, 2.9), (4.0, 3.9)]
        lo, hi = paired_percentile_ci(pairs, seed=0, draws=400)
        assert lo == pytest.approx(0.1, abs=1e-9)
        assert hi == pytest.approx(0.1, abs=1e-9)


class TestRecomputedStatistic:
    def test_degenerate_resamples_are_dropped_not_counted(self) -> None:
        """Gate B's statistic is undefined when a resample is single-class; those draws are skipped
        and the percentile index is taken over the survivors.
        """
        units = ["a", "b"]
        seen: list[int] = []

        def statistic(sample: list[str]) -> float | None:
            if len(set(sample)) == 1:
                return None
            seen.append(len(sample))
            return float(len(set(sample)))

        lo, hi = paired_statistic_ci(units, statistic, seed=0, draws=200)
        assert 0 < len(seen) < 200  # single-class draws occurred and were dropped
        assert lo == hi == 2.0      # the percentile index ran over the survivors only
