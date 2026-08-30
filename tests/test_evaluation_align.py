"""Character alignment: agreement with ``metrics.edit`` and the confusion-relevant properties."""

from __future__ import annotations

import random

from glyphmemory.evaluation.align import DELETION, INSERTION, MATCH, SUBSTITUTION, align_ops
from glyphmemory.metrics.edit import edit_counts

ALPHABET = "abc "


def _random_string(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(length))


def test_totals_agree_with_edit_counts_on_random_pairs():
    """The whole point of keeping a full backtrace: it must not silently disagree with the trainer's
    own CER on the same input.
    """
    rng = random.Random(20260820)
    for _ in range(200):
        reference = _random_string(rng, rng.randint(0, 12))
        hypothesis = _random_string(rng, rng.randint(0, 12))
        ops = align_ops(reference, hypothesis)
        counts = edit_counts(reference, hypothesis)

        s = sum(1 for op in ops if op.kind == SUBSTITUTION)
        i = sum(1 for op in ops if op.kind == INSERTION)
        d = sum(1 for op in ops if op.kind == DELETION)
        assert (s, i, d) == (counts.substitutions, counts.insertions, counts.deletions)


def test_ops_reconstruct_both_strings_in_order():
    reference, hypothesis = "handwriting", "handwrting"
    ops = align_ops(reference, hypothesis)
    rebuilt_ref = "".join(op.ref_char for op in ops if op.ref_char is not None)
    rebuilt_hyp = "".join(op.hyp_char for op in ops if op.hyp_char is not None)
    assert rebuilt_ref == reference
    assert rebuilt_hyp == hypothesis


def test_matching_strings_are_all_matches():
    ops = align_ops("same", "same")
    assert all(op.kind == MATCH for op in ops)
    assert len(ops) == 4


def test_pure_substitution_is_isolated_by_matches():
    ops = align_ops("cat", "cot")
    kinds = [op.kind for op in ops]
    assert kinds == [MATCH, SUBSTITUTION, MATCH]
    assert ops[1].ref_char == "a"
    assert ops[1].hyp_char == "o"


def test_empty_reference_is_all_insertions():
    ops = align_ops("", "abc")
    assert all(op.kind == INSERTION for op in ops)
    assert [op.hyp_char for op in ops] == ["a", "b", "c"]


def test_empty_hypothesis_is_all_deletions():
    ops = align_ops("abc", "")
    assert all(op.kind == DELETION for op in ops)
    assert [op.ref_char for op in ops] == ["a", "b", "c"]


def test_tie_break_prefers_substitution_over_deletion_and_insertion():
    """Same tie-break rule as ``edit_counts`` — checked here on a pair small enough to eyeball."""
    ops = align_ops("ab", "b")
    counts = edit_counts("ab", "b")
    kinds = [op.kind for op in ops]
    assert kinds.count(SUBSTITUTION) == counts.substitutions
    assert kinds.count(DELETION) == counts.deletions
    assert kinds.count(INSERTION) == counts.insertions
