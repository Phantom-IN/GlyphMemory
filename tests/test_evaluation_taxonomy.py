"""Error taxonomy: S/I/D totals, the single-glyph-confusion definition, and the confusion matrix."""

from __future__ import annotations

from glyphmemory.ctc.normalization import NFC_V1
from glyphmemory.evaluation.align import align_ops
from glyphmemory.evaluation.taxonomy import build_taxonomy, is_single_glyph_confusion
from glyphmemory.metrics.edit import edit_counts


def test_taxonomy_totals_agree_with_edit_counts():
    pairs = [("cat", "cot"), ("dog", "dogs"), ("bird", "bir"), ("same", "same")]
    taxonomy = build_taxonomy(pairs)
    expected_s = expected_i = expected_d = 0
    for ref, hyp in pairs:
        c = edit_counts(ref, hyp)
        expected_s += c.substitutions
        expected_i += c.insertions
        expected_d += c.deletions
    assert (taxonomy.substitutions, taxonomy.insertions, taxonomy.deletions) == (
        expected_s,
        expected_i,
        expected_d,
    )
    assert taxonomy.lines == len(pairs)


def test_isolated_substitution_is_a_single_glyph_confusion():
    # "cat" -> "cot": one substitution, matches on both sides.
    ops = align_ops("cat", "cot")
    assert is_single_glyph_confusion(ops, 1)


def test_substitution_next_to_another_substitution_is_not_isolated():
    # "cats" -> "cops": c(match) a->o(sub) t->p(sub) s(match) — two adjacent substitutions,
    # each disqualified by the other.
    ops = align_ops("cats", "cops")
    subs = [i for i, op in enumerate(ops) if op.kind == "sub"]
    assert len(subs) == 2
    for index in subs:
        assert not is_single_glyph_confusion(ops, index)


def test_substitution_at_the_line_boundary_counts_as_isolated():
    # A substitution with no neighbour on one side (start of line) is still isolated if the other
    # side is a match or the line's other boundary.
    ops = align_ops("x", "y")
    assert is_single_glyph_confusion(ops, 0)


def test_single_glyph_fraction_matches_manual_count():
    pairs = [("cat", "cot"), ("cats", "cops"), ("same", "same")]
    taxonomy = build_taxonomy(pairs)
    # "cat"->"cot": 1 isolated sub (a->o). "cats"->"cops": 2 adjacent subs (a->o, t->p), neither
    # isolated. 3 substitutions total, 1 single-glyph.
    assert taxonomy.substitutions == 3
    assert taxonomy.single_glyph_confusions == 1
    assert taxonomy.single_glyph_fraction == 1 / 3


def test_single_glyph_fraction_is_none_with_no_substitutions():
    taxonomy = build_taxonomy([("same", "same")])
    assert taxonomy.substitutions == 0
    assert taxonomy.single_glyph_fraction is None


def test_confusion_matrix_counts_every_substitution_not_just_isolated_ones():
    pairs = [("cat", "cot"), ("cats", "cops")]
    taxonomy = build_taxonomy(pairs)
    # Both lines substitute 'a' for 'o' — one isolated, one not — both must be counted.
    assert taxonomy.confusion_counts[("a", "o")] == 2
    assert taxonomy.confusion_counts[("t", "p")] == 1


def test_top_confusions_orders_by_frequency():
    pairs = [("cat", "cot")] * 3 + [("bag", "big")]
    taxonomy = build_taxonomy(pairs)
    top = taxonomy.top_confusions(1)
    assert top[0] == (("a", "o"), 3)


def test_as_dict_states_the_definition_alongside_the_number():
    taxonomy = build_taxonomy([("cat", "cot")])
    payload = taxonomy.as_dict()
    assert "single_glyph_confusion_definition" in payload
    assert payload["single_glyph_fraction"] == 1.0
    assert payload["normalization"] == NFC_V1.name
