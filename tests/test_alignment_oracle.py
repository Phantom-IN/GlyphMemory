"""Reference-oracle agreement: our Viterbi aligner against ``torchaudio.functional.forced_align`` on
random logits.

``torchaudio`` is a **dev-only** dependency (``pyproject.toml``'s ``dev`` group) — nothing in
``src/glyphmemory`` imports it.
"""

from __future__ import annotations

import random

import pytest
import torch

torchaudio = pytest.importorskip("torchaudio")

from glyphmemory.alignment.forced_align import _extended_sequence, viterbi_align  # noqa: E402

TRIALS = 200


def _random_case(rng: random.Random):
    """A random (log_probs, target) pair, feasible by construction, with some width slack."""
    vocab = rng.randint(3, 8)
    target_len = rng.randint(1, 6)
    min_frames = 2 * target_len + 1
    frames = min_frames + rng.randint(0, 15)
    logits = torch.randn(frames, vocab)
    log_probs = torch.log_softmax(logits, dim=-1)
    target = [rng.randint(1, vocab - 1) for _ in range(target_len)]
    return log_probs, target


def test_frame_labels_agree_with_torchaudio_on_random_logits():
    rng = random.Random(20260821)
    mismatches = []
    for trial in range(TRIALS):
        log_probs, target = _random_case(rng)
        ours = viterbi_align(log_probs, target, blank=0)
        extended = _extended_sequence(target, 0)
        our_labels = [extended[s] for s in ours.states]

        ta_labels, _ta_scores = torchaudio.functional.forced_align(
            log_probs.unsqueeze(0), torch.tensor([target]), blank=0
        )
        ta_labels = ta_labels[0].tolist()

        if our_labels != ta_labels:
            mismatches.append((trial, our_labels, ta_labels))

    assert not mismatches, f"{len(mismatches)}/{TRIALS} trials disagreed with the oracle"


def test_path_log_prob_agrees_with_torchaudio_within_floating_tolerance():
    rng = random.Random(999)
    for _ in range(50):
        log_probs, target = _random_case(rng)
        ours = viterbi_align(log_probs, target, blank=0)

        _ta_labels, ta_scores = torchaudio.functional.forced_align(
            log_probs.unsqueeze(0), torch.tensor([target]), blank=0
        )
        # torchaudio returns per-frame scores for its own recovered label path; summing them is the
        # same quantity as our path_log_prob when the two paths agree (checked separately above) —
        # cross-checking the score, not just the label sequence.
        oracle_log_prob = float(ta_scores[0].sum())
        assert ours.log_prob == pytest.approx(oracle_log_prob, abs=1e-3)


def test_tight_boundary_cases_agree_with_the_oracle_too():
    """Random-case generation above always leaves slack; the exact S == T boundary is where a
    Viterbi implementation is most likely to diverge, so it gets its own dedicated check.
    """
    rng = random.Random(2026)
    for _ in range(50):
        vocab = rng.randint(3, 6)
        target_len = rng.randint(1, 5)
        frames = 2 * target_len + 1  # exactly S == T, no slack
        logits = torch.randn(frames, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)
        target = [rng.randint(1, vocab - 1) for _ in range(target_len)]

        ours = viterbi_align(log_probs, target, blank=0)
        extended = _extended_sequence(target, 0)
        our_labels = [extended[s] for s in ours.states]

        ta_labels, _ = torchaudio.functional.forced_align(
            log_probs.unsqueeze(0), torch.tensor([target]), blank=0
        )
        assert our_labels == ta_labels[0].tolist()
