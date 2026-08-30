"""CER/WER tests.

Three conventions get pinned here, because each one silently changes every number the project will
ever report: micro vs macro aggregation, the empty-reference rule, and whether the normalization
policy is actually applied rather than merely recorded.

The S/I/D breakdown is checked against an independent reference implementation on random string
pairs — a hand-rolled DP that carries per-operation counts is exactly the kind of code that gets the
total right and the split wrong.
"""

from __future__ import annotations

import random
from typing import ClassVar

import pytest

from glyphmemory.ctc.decode import DecoderConfig
from glyphmemory.ctc.normalization import IDENTITY, NFC_V1
from glyphmemory.metrics import (
    MACRO,
    MICRO,
    EditCounts,
    cer,
    character_counts,
    corpus_cer,
    corpus_wer,
    edit_counts,
    edit_distance,
    macro_cer,
    wer,
    word_counts,
)


def reference_distance(a: str, b: str) -> int:
    """Textbook Levenshtein, full matrix, no operation tracking. Independent of ours."""
    rows = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        rows[i][0] = i
    for j in range(len(b) + 1):
        rows[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            rows[i][j] = min(rows[i - 1][j] + 1, rows[i][j - 1] + 1, rows[i - 1][j - 1] + cost)
    return rows[len(a)][len(b)]


class TestEditDistance:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ("kitten", "sitting", 3),
            ("", "", 0),
            ("abc", "abc", 0),
            ("abc", "", 3),
            ("", "abc", 3),
            ("a", "b", 1),
            ("flaw", "lawn", 2),
        ],
    )
    def test_known_distances(self, a: str, b: str, expected: int) -> None:
        assert edit_distance(a, b) == expected

    def test_operation_split_for_a_worked_example(self) -> None:
        """``kitten -> sitting``: k->s, e->i, then append g."""
        counts = edit_counts("kitten", "sitting")
        assert (counts.substitutions, counts.insertions, counts.deletions) == (2, 1, 0)
        assert counts.total == 3

    def test_insertion_is_what_the_model_added(self) -> None:
        counts = edit_counts("cat", "cart")
        assert counts.insertions == 1
        assert counts.deletions == 0

    def test_deletion_is_what_the_model_dropped(self) -> None:
        counts = edit_counts("cart", "cat")
        assert counts.deletions == 1
        assert counts.insertions == 0

    def test_counts_always_sum_to_the_distance(self) -> None:
        rng = random.Random(1337)
        alphabet = "abcde"
        for _ in range(400):
            a = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
            b = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
            counts = edit_counts(a, b)
            assert counts.total == reference_distance(a, b), (a, b)
            assert counts.substitutions + counts.insertions + counts.deletions == counts.total

    def test_works_on_word_lists_not_only_strings(self) -> None:
        assert edit_distance(["the", "cat"], ["the", "dog"]) == 1

    def test_addition_accumulates(self) -> None:
        total = EditCounts(1, 2, 3, 10) + EditCounts(1, 0, 0, 5)
        assert (total.substitutions, total.insertions, total.deletions) == (2, 2, 3)
        assert total.reference_length == 15


class TestSinglePair:
    def test_identical_is_zero(self) -> None:
        assert cer("hello world", "hello world") == 0.0
        assert wer("hello world", "hello world") == 0.0

    def test_single_character_substitution(self) -> None:
        assert cer("abcde", "abXde") == pytest.approx(0.2)

    def test_single_character_reference(self) -> None:
        assert cer("a", "b") == 1.0
        assert cer("a", "a") == 0.0

    def test_word_error_rate_counts_words(self) -> None:
        assert wer("the quick brown fox", "the quick brown dog") == pytest.approx(0.25)

    def test_hypothesis_longer_than_reference_can_exceed_one(self) -> None:
        assert cer("a", "abcdef") == 5.0


class TestNormalizationIsApplied:
    def test_whitespace_only_difference_under_nfc_v1(self) -> None:
        """The test that proves the policy runs rather than merely being recorded."""
        assert cer("hello   world", "hello world", policy=NFC_V1) == 0.0

    def test_same_pair_under_identity(self) -> None:
        assert cer("hello   world", "hello world", policy=IDENTITY) == pytest.approx(2 / 13)

    def test_leading_and_trailing_whitespace_is_stripped(self) -> None:
        assert cer("  hello  ", "hello", policy=NFC_V1) == 0.0

    def test_case_is_never_folded(self) -> None:
        """No policy lowercases."""
        assert cer("Hello", "hello", policy=NFC_V1) == pytest.approx(0.2)

    def test_punctuation_is_never_stripped(self) -> None:
        assert cer("hello.", "hello", policy=NFC_V1) == pytest.approx(1 / 6)

    def test_policy_name_is_recorded(self) -> None:
        assert corpus_cer([("a", "a")], policy=IDENTITY).normalization == "identity"
        assert corpus_cer([("a", "a")]).normalization == "nfc_v1"

    def test_wer_splits_after_normalization(self) -> None:
        """`nfc_v1` collapses the run first, so both sides yield two words."""
        assert wer("two    words", "two words", policy=NFC_V1) == 0.0


class TestEmptyReferences:
    def test_empty_versus_empty_is_zero_edits(self) -> None:
        counts = character_counts("", "")
        assert counts.total == 0
        assert counts.error_rate is None

    def test_empty_reference_with_output_counts_insertions(self) -> None:
        counts = character_counts("", "spurious")
        assert counts.insertions == 8
        assert counts.reference_length == 0

    def test_per_sample_rate_is_none_not_one(self) -> None:
        """A placeholder here would silently distort any macro average taken over it."""
        assert cer("", "spurious") is None
        assert cer("", "") is None

    def test_micro_average_absorbs_them_correctly(self) -> None:
        """The insertions land in the numerator; nothing is divided by zero."""
        result = corpus_cer([("abc", "abc"), ("", "xx")])
        assert result.counts.insertions == 2
        assert result.counts.reference_length == 3
        assert result.value == pytest.approx(2 / 3)

    def test_all_empty_references_give_no_value(self) -> None:
        result = corpus_cer([("", ""), ("", "x")])
        assert result.value is None
        assert "n/a" in result.format()

    def test_macro_excludes_undefined_rates(self) -> None:
        result = macro_cer([("abcd", "abcd"), ("", "xx")])
        assert result.value == 0.0  # only the defined sample counts


class TestAggregation:
    UNBALANCED: ClassVar = [
        ("a", "b"),  # 1 edit over 1 char   -> rate 1.0
        ("x" * 99, "x" * 99),  # 0 edits over 99 chars -> rate 0.0
    ]

    def test_micro_is_the_default(self) -> None:
        assert corpus_cer(self.UNBALANCED).aggregation == MICRO

    def test_micro_and_macro_genuinely_differ(self) -> None:
        """1/100 versus 0.5 — the reason the convention must travel with the number."""
        assert corpus_cer(self.UNBALANCED).value == pytest.approx(0.01)
        assert macro_cer(self.UNBALANCED).value == pytest.approx(0.5)

    def test_macro_is_labelled(self) -> None:
        result = macro_cer(self.UNBALANCED)
        assert result.aggregation == MACRO
        assert "macro-averaged" in result.format()

    def test_micro_equals_total_edits_over_total_length(self) -> None:
        result = corpus_cer([("abcd", "abXd"), ("ef", "ef")])
        assert result.counts.total == 1
        assert result.counts.reference_length == 6
        assert result.value == pytest.approx(1 / 6)

    def test_empty_corpus(self) -> None:
        result = corpus_cer([])
        assert result.n_samples == 0
        assert result.value is None


class TestMetricResult:
    def test_carries_normalization_and_decoder(self) -> None:
        result = corpus_cer([("abc", "abd")])
        assert result.normalization == "nfc_v1"
        assert result.decoder.label == "greedy, no LM"
        assert result.as_dict()["decoder"]["language_model"] is None

    def test_a_non_default_decoder_is_recorded(self) -> None:
        result = corpus_cer([("abc", "abd")], decoder=DecoderConfig(blank_index=0))
        assert result.as_dict()["decoder"]["blank_index"] == 0

    def test_per_sample_records_are_retained(self) -> None:
        """Internal helper."""
        result = corpus_cer([("cat", "cot"), ("dog", "dog")], sample_ids=["s1", "s2"])
        assert [s.sample_id for s in result.samples] == ["s1", "s2"]
        assert result.samples[0].counts.substitutions == 1
        assert result.samples[1].is_exact_match

    def test_exact_match_count(self) -> None:
        result = corpus_cer([("a", "a"), ("b", "b"), ("c", "d")])
        assert result.exact_matches == 2

    def test_worst_lists_the_highest_rates_first(self) -> None:
        result = corpus_cer([("aaaa", "aaaa"), ("bb", "xx"), ("cccc", "cccx")])
        worst = result.worst(2)
        assert worst[0].reference == "bb"
        assert worst[1].reference == "cccc"

    def test_worst_excludes_undefined_rates(self) -> None:
        result = corpus_cer([("", "xxxx"), ("ab", "xb")])
        assert [s.reference for s in result.worst()] == ["ab"]

    def test_as_dict_is_json_shaped(self) -> None:
        payload = corpus_cer([("abc", "abd")]).as_dict()
        assert payload["metric"] == "cer"
        assert payload["aggregation"] == "micro"
        assert payload["substitutions"] == 1
        assert "per_sample" not in payload

    def test_as_dict_can_include_samples(self) -> None:
        payload = corpus_cer([("abc", "abd")]).as_dict(include_samples=True)
        assert payload["per_sample"][0]["hypothesis"] == "abd"

    def test_format_states_everything_needed_to_quote_the_number(self) -> None:
        text = corpus_cer([("abc", "abd")]).format()
        assert "CER" in text
        assert "micro-averaged" in text
        assert "nfc_v1" in text
        assert "greedy, no LM" in text


class TestWordLevel:
    def test_corpus_wer_counts_words(self) -> None:
        result = corpus_wer([("the cat sat", "the dog sat"), ("hi there", "hi there")])
        assert result.counts.reference_length == 5
        assert result.counts.total == 1
        assert result.value == pytest.approx(0.2)

    def test_word_counts_ignores_intra_word_position(self) -> None:
        counts = word_counts("a b c", "a x c")
        assert counts.substitutions == 1
        assert counts.reference_length == 3

    def test_name_is_recorded(self) -> None:
        assert corpus_wer([("a", "a")]).name == "wer"
