"""Structural verification for the recovered Router B implementation.

**Verification, not new analysis.** The row-level input the historical run consumed
(``m10r5a_rows.json``, 4.6 MB of per-slot records) was never committed, so the published operating
point cannot be recomputed inside this test suite. It *was* recomputed once during PUB-5B against
the surviving session data and matched exactly — HELP 78, DAMAGE 33, precision 0.7027027027,
coverage 0.0143175310 — and that verification is recorded in
``publication/repro/ROUTER-B-PROVENANCE.md``.

What these tests hold is the specification: the nine features in their frozen order, the shrinkage
constant, the label convention, the determinism of the fit, and the leave-one-writer-out isolation.
Those are the properties a future edit could break silently.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from reproduce.router_b import (
    FEATURE_NAMES,
    FROZEN_TAU,
    L2,
    PSEUDO,
    STEPS,
    TAUS,
    eligible,
    features,
    fit_logistic,
    fit_pair_prior,
    label,
    leave_one_writer_out,
)


def row(writer="w", base="a", mem="o", base_ok=True, mem_ok=False, in_top2=True, **kw):
    r = {
        "writer": writer, "base": base, "mem": mem,
        "base_ok": base_ok, "mem_ok": mem_ok, "in_top2": in_top2,
        "conf": 0.8, "margin": 2.0, "entropy": 0.3, "mem_sim": 0.5,
        "mem_margin": 0.1, "mem_obs": 4, "rank": 1, "frames": 3,
    }
    r.update(kw)
    return r


class TestFrozenProtocol:
    def test_nine_features_in_the_registered_order(self) -> None:
        """The weights are positional. Reordering silently changes what the model scores."""
        assert FEATURE_NAMES == (
            "base_confidence", "top1_top2_margin", "entropy",
            "memory_similarity", "memory_runner_up_margin",
            "log1p_memory_observations", "candidate_rank",
            "log1p_span_frames", "shrunk_pair_prior",
        )
        assert len(features(row(), {}, 0.5)) == 9

    def test_feature_values_match_their_names(self) -> None:
        f = features(row(conf=0.7, margin=1.5, entropy=0.25, mem_sim=0.6,
                         mem_margin=0.2, mem_obs=9, rank=2, frames=7), {}, 0.42)
        assert f[0] == pytest.approx(0.7)
        assert f[1] == pytest.approx(1.5)
        assert f[2] == pytest.approx(0.25)
        assert f[3] == pytest.approx(0.6)
        assert f[4] == pytest.approx(0.2)
        assert f[5] == pytest.approx(math.log1p(9))
        assert f[6] == pytest.approx(2.0)
        assert f[7] == pytest.approx(math.log1p(7))
        assert f[8] == pytest.approx(0.42)  # unseen pair falls back to the global prior

    def test_registered_constants(self) -> None:
        assert (PSEUDO, L2, STEPS) == (5.0, 1e-2, 600)
        assert FROZEN_TAU == 0.60
        assert FROZEN_TAU in TAUS


class TestEligibilityAndLabels:
    def test_eligibility_requires_disagreement_and_top2_membership(self) -> None:
        assert eligible([row(base="a", mem="a")]) == []
        assert eligible([row(in_top2=False)]) == []
        assert len(eligible([row()])) == 1

    def test_label_convention(self) -> None:
        assert label(row(base_ok=False, mem_ok=True)) == 1    # HELP
        assert label(row(base_ok=True, mem_ok=False)) == 0    # DAMAGE
        assert label(row(base_ok=False, mem_ok=False)) == -1  # NEITHER

    def test_neither_slots_are_excluded_from_fitting(self) -> None:
        """NEITHER costs nothing either way, so it must not pull the decision boundary."""
        rows = [row(base_ok=False, mem_ok=False) for _ in range(50)]
        assert [r for r in eligible(rows) if label(r) >= 0] == []


class TestPairPrior:
    def test_shrinks_towards_the_global_rate(self) -> None:
        """A pair seen once must not be trusted as if it were seen fifty times."""
        rows = [row(base="e", mem="a", base_ok=False, mem_ok=True)]
        rows += [row(base="n", mem="m", base_ok=True, mem_ok=False) for _ in range(49)]
        prior, p0 = fit_pair_prior(rows)
        assert prior[("e", "a")] < 1.0
        assert prior[("e", "a")] == pytest.approx((1 + PSEUDO * p0) / (1 + PSEUDO))

    def test_unseen_pairs_fall_back_to_the_global_prior(self) -> None:
        prior, p0 = fit_pair_prior([row(base_ok=False, mem_ok=True)])
        assert features(row(base="z", mem="q"), prior, p0)[8] == pytest.approx(p0)


class TestFit:
    def test_fit_is_deterministic(self) -> None:
        """Zero init plus LBFGS means no random state. Two fits must agree bit for bit."""
        X = torch.tensor([[float(i), float(i % 3)] for i in range(40)])
        y = torch.tensor([float(i % 2) for i in range(40)])
        a = fit_logistic(X, y)
        b = fit_logistic(X, y)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    def test_standardisation_is_returned_for_reuse_on_held_out_rows(self) -> None:
        """Held-out rows must be standardised with the *training* fold's statistics."""
        X = torch.tensor([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
        y = torch.tensor([0.0, 1.0, 1.0])
        _, _, mu, sd = fit_logistic(X, y)
        assert mu.tolist() == pytest.approx([3.0, 30.0])
        assert (sd > 0).all()

    def test_zero_variance_feature_does_not_divide_by_zero(self) -> None:
        X = torch.tensor([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
        y = torch.tensor([0.0, 1.0, 1.0])
        w, b, _mu, sd = fit_logistic(X, y)
        assert torch.isfinite(w).all() and torch.isfinite(b).all()
        assert sd[1] >= 1e-6


class TestLeaveOneWriterOut:
    def test_the_held_out_writer_never_enters_its_own_model(self) -> None:
        """The protocol's central guarantee. If this breaks, every reported precision is in-fold."""
        rows_by_writer = {
            "w1": [row(writer="w1", base="e", mem="a", base_ok=False, mem_ok=True)] * 6,
            "w2": [row(writer="w2", base="n", mem="m", base_ok=True, mem_ok=False)] * 6,
            "w3": [row(writer="w3", base="o", mem="c", base_ok=False, mem_ok=True)] * 6,
        }
        folds = leave_one_writer_out(rows_by_writer)
        assert set(folds) == {"w1", "w2", "w3"}
        for held, scored in folds.items():
            assert all(r["writer"] == held for r, _ in scored)
            assert all(0.0 <= p <= 1.0 for _, p in scored)

    def test_a_writer_with_no_eligible_slot_yields_an_empty_fold(self) -> None:
        rows_by_writer = {
            "w1": [row(writer="w1", base="a", mem="a")],           # ineligible: no disagreement
            "w2": [row(writer="w2", base_ok=False, mem_ok=True)] * 4,
            "w3": [row(writer="w3", base_ok=True, mem_ok=False)] * 4,
        }
        assert leave_one_writer_out(rows_by_writer)["w1"] == []
