"""Base-emitted character occurrences and their alignment to truth (-R2 / Gate C2).

These two pieces decide whether every downstream R2 number is meaningful. If span construction
merged or split slots, or if the alignment mis-paired emitted characters with reference ones, the
HELP/DAMAGE counts would be wrong in a way nothing downstream could catch — so the CTC collapsing
rule and the deletion case are asserted directly against hand-worked examples.
"""

from __future__ import annotations

import pytest
import torch

from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.ctc.decode import greedy_decode_ids, one_hot_logits
from glyphmemory.evaluation.emitted_spans import (
    Outcome,
    align_operations,
    emitted_occurrences,
    outcome_counts,
)


@pytest.fixture(scope="module")
def charset():
    return load_tokenizer(DEFAULT_CHARSET_PATH).charset


def _frames(charset, symbols):
    """One-hot logits for a hand-written frame sequence; '' means blank."""
    ids = [0 if s == "" else charset.index_of(s) for s in symbols]
    return one_hot_logits(ids, charset.size)


# ------------------------------------------------------------------ span construction


def test_a_run_of_identical_frames_is_one_character(charset):
    occurrences = emitted_occurrences(_frames(charset, ["a", "a", "a"]), charset)

    assert [o.character for o in occurrences] == ["a"]
    assert (occurrences[0].start, occurrences[0].end, occurrences[0].frames) == (0, 2, 3)


def test_a_blank_splits_a_repeated_character_into_two(charset):
    """The CTC rule that makes 'aa' expressible. Getting this wrong merges two slots into one."""
    occurrences = emitted_occurrences(_frames(charset, ["a", "", "a"]), charset)

    assert [o.character for o in occurrences] == ["a", "a"]
    assert [(o.start, o.end) for o in occurrences] == [(0, 0), (2, 2)]


def test_spans_agree_with_the_projects_own_greedy_decoder(charset):
    symbols = ["", "h", "h", "", "e", "l", "l", "", "l", "o", "o", ""]
    logits = _frames(charset, symbols)

    emitted = "".join(o.character for o in emitted_occurrences(logits, charset))
    decoded = "".join(charset.char_at(i) for i in greedy_decode_ids(logits))

    assert emitted == decoded == "hello"


def test_blanks_only_emit_nothing(charset):
    assert emitted_occurrences(_frames(charset, ["", "", ""]), charset) == []


def test_padding_beyond_length_is_not_read(charset):
    logits = _frames(charset, ["a", "b", "c"])

    assert [o.character for o in emitted_occurrences(logits, charset, length=2)] == ["a", "b"]
    assert emitted_occurrences(logits, charset, length=0) == []


def test_each_occurrence_carries_the_candidate_set_a_reranker_may_choose_among(charset):
    occurrences = emitted_occurrences(_frames(charset, ["a", "b"]), charset, top_k=3)

    for occurrence in occurrences:
        assert len(occurrence.candidates) == 3
        assert occurrence.candidates[0] == occurrence.character  # top-1 is what was emitted
        assert occurrence.confidence > 0.0


def test_degenerate_shapes_and_lengths_are_refused(charset):
    with pytest.raises(ValueError, match=r"Expected \[T, C\]"):
        emitted_occurrences(torch.zeros(3), charset)
    with pytest.raises(ValueError, match="outside"):
        emitted_occurrences(_frames(charset, ["a"]), charset, length=9)
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        emitted_occurrences(_frames(charset, ["a"]), charset, top_k=0)


# ------------------------------------------------------------------ alignment


def test_identical_sequences_are_all_correct():
    positions = align_operations("hello", "hello")

    assert [p.outcome for p in positions] == [Outcome.CORRECT] * 5
    assert outcome_counts(positions)["correct"] == 5


def test_a_substitution_pairs_the_emitted_character_with_the_reference_one():
    positions = align_operations("cat", "cot")

    substitution = next(p for p in positions if p.outcome is Outcome.SUBSTITUTION)
    assert (substitution.emitted, substitution.reference) == ("o", "a")
    assert substitution.emitted_index == 1 and substitution.reference_index == 1


def test_a_deletion_has_no_emitted_slot():
    """The case Gate C2 exists for: a reranker has nothing to act on here."""
    positions = align_operations("cat", "ct")

    deletion = next(p for p in positions if p.outcome is Outcome.DELETION)
    assert deletion.emitted_index is None
    assert deletion.reference == "a"


def test_an_insertion_has_no_reference_slot():
    positions = align_operations("ct", "cat")

    insertion = next(p for p in positions if p.outcome is Outcome.INSERTION)
    assert insertion.reference_index is None
    assert insertion.emitted == "a"


def test_counts_match_the_projects_own_edit_metric():
    from glyphmemory.metrics.edit import edit_counts

    for reference, hypothesis in [("kitten", "sitting"), ("", "abc"), ("abc", ""),
                                  ("handwriting", "handwrlting")]:
        counts = edit_counts(reference, hypothesis)
        tally = outcome_counts(align_operations(reference, hypothesis))
        assert tally["substitution"] == counts.substitutions
        assert tally["insertion"] == counts.insertions
        assert tally["deletion"] == counts.deletions


def test_every_emitted_index_appears_at_most_once():
    positions = align_operations("handwriting", "handwrlting")
    emitted = [p.emitted_index for p in positions if p.emitted_index is not None]

    assert len(emitted) == len(set(emitted)) == 11
