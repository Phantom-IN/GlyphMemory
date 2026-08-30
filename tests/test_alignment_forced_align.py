"""The Viterbi aligner on hand-built logits — the required edge cases from: repeated characters,
spaces, blank-dominated paths, low-confidence alignments, single-character lines, and the tight
``L`` ≈ ``T`` case.

Reference-oracle agreement against ``torchaudio.functional.forced_align`` lives in
``tests/test_alignment_oracle.py``; this file is about the shape of *our* output and the documented
edge cases, hand-verified by construction rather than by an external comparison.
"""

from __future__ import annotations

import math

import pytest
import torch

from glyphmemory.alignment import (
    AlignmentInfeasibleError,
    forced_align,
    viterbi_align,
)
from glyphmemory.ctc.tokenizer import Charset

CHARSET = Charset(symbols=("<blank>", " ", "a", "b", "c"))
BLANK, SPACE, A, B, C = 0, 1, 2, 3, 4


def confident_log_probs(
    class_per_frame: list[int], vocab: int, *, peak: float = 0.94
) -> torch.Tensor:
    """Build ``[T, vocab]`` log-probabilities where frame ``t`` strongly favors
    ``class_per_frame[t]``, the rest of the mass spread over everything else.
    """
    rest = (1.0 - peak) / (vocab - 1)
    rows = []
    for cls in class_per_frame:
        row = [rest] * vocab
        row[cls] = peak
        rows.append(row)
    return torch.tensor(rows).log()


# --------------------------------------------------------------------------- repeated characters


def test_repeated_characters_produce_two_distinct_spans():
    # "aa" needs an explicit blank between the two label states.
    frames = [A, A, BLANK, A, A]
    log_probs = confident_log_probs(frames, vocab=5)
    result = forced_align(log_probs, "aa", CHARSET)
    assert len(result.spans) == 2
    first, second = result.spans
    assert first.token == second.token == "a"
    assert first.end_t <= second.start_t  # non-overlapping and in order
    assert first.start_t < first.end_t
    assert second.start_t < second.end_t


def test_repeated_characters_without_enough_frames_is_infeasible():
    # "aa" needs S = 2*2+1 = 5 lattice states; 4 frames cannot fit it.
    log_probs = confident_log_probs([A, BLANK, A, A], vocab=5)
    with pytest.raises(AlignmentInfeasibleError):
        forced_align(log_probs, "aa", CHARSET)


def test_three_repeated_characters_each_get_their_own_span():
    # "aaa" -> S = 2*3+1 = 7; each state gets exactly one frame at this tight boundary.
    frames = [BLANK, A, BLANK, A, BLANK, A, BLANK]
    log_probs = confident_log_probs(frames, vocab=5)
    result = forced_align(log_probs, "aaa", CHARSET)
    assert [s.token for s in result.spans] == ["a", "a", "a"]
    starts = [s.start_t for s in result.spans]
    assert starts == sorted(starts)
    assert len(set(starts)) == 3  # three genuinely distinct spans, not one collapsed run


# --------------------------------------------------------------------------- spaces


def test_space_is_an_ordinary_non_blank_token():
    # "a b" -> S = 2*3+1 = 7; tight boundary, one frame per lattice state.
    frames = [BLANK, A, BLANK, SPACE, BLANK, B, BLANK]
    log_probs = confident_log_probs(frames, vocab=5)
    result = forced_align(log_probs, "a b", CHARSET)
    tokens = [s.token for s in result.spans]
    assert tokens == ["a", " ", "b"]


# --------------------------------------------------------------------------- blank-dominated paths


def test_blank_dominated_path_still_recovers_correct_spans():
    # Long blank runs before, between and after the two characters.
    frames = [BLANK] * 4 + [A] + [BLANK] * 4 + [B] + [BLANK] * 4
    log_probs = confident_log_probs(frames, vocab=5)
    result = forced_align(log_probs, "ab", CHARSET)
    assert [s.token for s in result.spans] == ["a", "b"]
    a_span, b_span = result.spans
    assert a_span.start_t == 4
    assert b_span.start_t == 9


# --------------------------------------------------------------------------- low-confidence


def test_low_confidence_span_scores_lower_than_a_confident_one():
    confident = confident_log_probs([A, A, BLANK, B, B], vocab=5, peak=0.97)
    uncertain = confident_log_probs([A, A, BLANK, B, B], vocab=5, peak=0.35)

    confident_result = forced_align(confident, "ab", CHARSET)
    uncertain_result = forced_align(uncertain, "ab", CHARSET)

    for c_span, u_span in zip(confident_result.spans, uncertain_result.spans, strict=True):
        assert u_span.score < c_span.score

    for span in confident_result.spans + uncertain_result.spans:
        assert 0.0 <= span.score <= 1.0


# --------------------------------------------------------------------------- single character


def test_single_character_line():
    log_probs = confident_log_probs([BLANK, A, A, BLANK], vocab=5)
    result = forced_align(log_probs, "a", CHARSET)
    assert len(result.spans) == 1
    span = result.spans[0]
    assert span.token == "a"
    assert span.start_t == 1
    assert span.end_t == 3


def test_single_character_at_the_tightest_possible_width():
    # S = 2*1 + 1 = 3, so T = 3 is the minimum feasible width.
    log_probs = confident_log_probs([BLANK, A, BLANK], vocab=5)
    result = forced_align(log_probs, "a", CHARSET)
    assert result.spans[0].start_t == 1
    assert result.spans[0].end_t == 2


# --------------------------------------------------------------------------- L approx T (tight)


def test_target_length_near_frame_count_is_feasible_at_the_exact_boundary():
    # "abc" -> S = 2*3+1 = 7. Exactly 7 frames is the tightest feasible case.
    frames = [BLANK, A, BLANK, B, BLANK, C, BLANK]
    log_probs = confident_log_probs(frames, vocab=5)
    result = forced_align(log_probs, "abc", CHARSET)
    assert [s.token for s in result.spans] == ["a", "b", "c"]
    for span in result.spans:
        assert span.length == 1  # no slack at all at the exact boundary


def test_one_frame_short_of_the_boundary_is_infeasible():
    frames = [BLANK, A, BLANK, B, BLANK, C]  # 6 frames, need 7
    log_probs = confident_log_probs(frames, vocab=5)
    with pytest.raises(AlignmentInfeasibleError, match="infeasible"):
        forced_align(log_probs, "abc", CHARSET)


# --------------------------------------------------------------------------- determinism


def test_output_is_deterministic_given_the_same_input():
    frames = [BLANK, A, A, BLANK, B, BLANK]
    log_probs = confident_log_probs(frames, vocab=5)
    first = forced_align(log_probs, "ab", CHARSET)
    second = forced_align(log_probs, "ab", CHARSET)
    assert first.spans == second.spans
    assert first.log_prob == second.log_prob


def test_exact_ties_prefer_the_transition_that_advances_least():
    """A uniform-probability input makes every transition exactly tied. The documented rule (stay >
    advance-by-1 > advance-by-2) must still produce one specific, repeatable answer.

    Hand-verified for this exact case: with every alpha value tied throughout, "stay" always wins,
    and the label state (index 1) is reachable at t=0 just as cheaply as the leading blank (index 0)
    — so the path jumps straight to the label and never leaves it. Both ends of a length-3 target
    lattice ([blank, label, blank]) tie at the final frame too, and the label (index S-2) wins that
    tie by the documented rule.
    """
    log_probs = torch.full((3, 2), math.log(0.5))
    path = viterbi_align(log_probs, [1], blank=0)
    assert path.states == (1, 1, 1)


# --------------------------------------------------------------------------- shape validation


def test_log_probs_must_be_two_dimensional():
    with pytest.raises(ValueError, match=r"\[T, C\]"):
        viterbi_align(torch.zeros(2, 3, 4), [1])


def test_empty_target_is_a_degenerate_but_valid_case():
    log_probs = confident_log_probs([BLANK, BLANK], vocab=5)
    result = forced_align(log_probs, "", CHARSET)
    assert result.spans == ()
