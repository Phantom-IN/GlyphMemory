"""Router B — recovered historical implementation of the selective policy.

RECOVERED VERBATIM from ``m10r5a_router_b.py``, the session script that fitted the policy whose
operating points are reported in Sections 6 and 7 and committed in
``docs/results/m10r5a-router.json``. Provenance in ``publication/repro/ROUTER-B-PROVENANCE.md``.

Semantics preserved exactly, including:

* the pseudo-count of 5.0 in the shrunk pair prior;
* the nine features in their historical order;
* z-score standardisation fitted on the training folds only, with ``sd`` floored at 1e-6;
* LBFGS with ``max_iter=600`` and ``lr=0.1`` from a zero initialisation, which makes the fit
  deterministic with no random state;
* ``NEITHER`` slots excluded from fitting but retained in the coverage count.

Nothing here is refactored for clarity. The only changes from the historical script are that the
data-loading and printing were dropped and the functions given type hints and docstrings.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import torch

# Historical constants, verbatim from the script's line 20.
PSEUDO: float = 5.0
L2: float = 1e-2
STEPS: int = 600
LR: float = 0.1
TAUS: list[float] = [x / 100 for x in range(30, 96, 5)]
FROZEN_TAU: float = 0.60

FEATURE_NAMES: tuple[str, ...] = (
    "base_confidence",
    "top1_top2_margin",
    "entropy",
    "memory_similarity",
    "memory_runner_up_margin",
    "log1p_memory_observations",
    "candidate_rank",
    "log1p_span_frames",
    "shrunk_pair_prior",
)


def eligible(rows: Iterable[Mapping]) -> list[Mapping]:
    """A slot is eligible when memory disagrees with the base and its proposal is in the top-2."""
    return [r for r in rows if r["base"] != r["mem"] and r["in_top2"]]


def label(row: Mapping) -> int:
    """1 = HELP, 0 = DAMAGE, -1 = NEITHER (excluded from fitting; it costs nothing either way)."""
    if not row["base_ok"] and row["mem_ok"]:
        return 1
    if row["base_ok"] and not row["mem_ok"]:
        return 0
    return -1


def fit_pair_prior(train_rows: Iterable[Mapping]) -> tuple[dict[tuple[str, str], float], float]:
    """Shrunk per-confusion-pair HELP rate, ``(h + m*p0) / (h + d + m)`` with ``m = PSEUDO``."""
    el = eligible(train_rows)
    h0 = sum(1 for r in el if label(r) == 1)
    p0 = h0 / len(el) if el else 0.0
    tally: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in el:
        lab = label(r)
        if lab == 1:
            tally[(r["base"], r["mem"])][0] += 1
        elif lab == 0:
            tally[(r["base"], r["mem"])][1] += 1
    return {k: (h + PSEUDO * p0) / (h + dm + PSEUDO) for k, (h, dm) in tally.items()}, p0


def features(row: Mapping, prior: Mapping[tuple[str, str], float], p0: float) -> list[float]:
    """The nine features, in the frozen protocol's order. Do not reorder: weights are positional."""
    return [
        row["conf"],
        row["margin"],
        row["entropy"],
        row["mem_sim"],
        row["mem_margin"],
        math.log1p(row["mem_obs"]),
        float(row["rank"]),
        math.log1p(row["frames"]),
        prior.get((row["base"], row["mem"]), p0),
    ]


def fit_logistic(X: torch.Tensor, y: torch.Tensor):
    """L2-regularised logistic regression by LBFGS. Deterministic: zero init, no sampling.

    Returns ``(w, b, mu, sd)``; ``mu`` and ``sd`` are the standardisation fitted on ``X`` and must
    be reapplied to any held-out design matrix before scoring.
    """
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Xs = (X - mu) / sd
    w = torch.zeros(Xs.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=STEPS, lr=LR)

    def closure():
        opt.zero_grad()
        logit = Xs @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y) + L2 * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach(), mu, sd


def score(rows: Sequence[Mapping], prior, p0: float, w, b, mu, sd) -> list[float]:
    """Sigmoid scores for ``rows`` under a fold's fitted model."""
    if not rows:
        return []
    X = torch.tensor([features(r, prior, p0) for r in rows], dtype=torch.float32)
    return torch.sigmoid(((X - mu) / sd) @ w + b).tolist()


def leave_one_writer_out(rows_by_writer: Mapping[str, Sequence[Mapping]]) -> dict[str, list]:
    """Fit one model per held-out writer. Every fitted quantity comes from the other writers only:
    the pair prior, the standardisation, the weights and the threshold.
    """
    folds: dict[str, list] = {}
    for held in sorted(rows_by_writer):
        train = [r for w, rs in rows_by_writer.items() if w != held for r in rs]
        prior, p0 = fit_pair_prior(train)
        tr = [r for r in eligible(train) if label(r) >= 0]
        X = torch.tensor([features(r, prior, p0) for r in tr], dtype=torch.float32)
        y = torch.tensor([float(label(r)) for r in tr])
        w, b, mu, sd = fit_logistic(X, y)
        held_el = eligible(rows_by_writer[held])
        folds[held] = list(zip(held_el, score(held_el, prior, p0, w, b, mu, sd), strict=True))
    return folds
