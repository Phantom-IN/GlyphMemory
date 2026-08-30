"""Text normalization.

The critical property is what normalization does *not* do.
"""

from __future__ import annotations

import unicodedata

import pytest

from glyphmemory.ctc import IDENTITY, NFC_V1, get_policy, normalize, normalizes_to_empty

# Deliberately hostile: mixed case, punctuation, digits, contractions, quotes.
MIXED = "The Quick BROWN fox; it's #1 (really!) -- jumped 42% ... or 0.5?"


# --------------------------------------------------------------------------- what it must NOT do


def test_case_is_preserved():
    assert normalize("MiXeD CaSe TeXt") == "MiXeD CaSe TeXt"


def test_punctuation_is_preserved():
    assert normalize("Hello, world! (yes) -- it's #1?") == "Hello, world! (yes) -- it's #1?"


def test_spelling_is_never_corrected():
    misspelled = "teh quik brwn fx recieve seperate"
    assert normalize(misspelled) == misspelled


def test_digits_are_preserved():
    assert normalize("Room 101 costs 42.50") == "Room 101 costs 42.50"


def test_mixed_fixture_survives_apart_from_documented_transforms():
    """The success criterion: identity on text with no whitespace or Unicode oddities."""
    assert normalize(MIXED) == MIXED


def test_repeated_characters_are_never_collapsed():
    """Only whitespace collapses. Letter runs are the decoder's problem, not ours."""
    for word in ("hello", "book", "letter", "committee", "aaa"):
        assert normalize(word) == word


def test_single_internal_spaces_untouched():
    assert normalize("a b c") == "a b c"


# --------------------------------------------------------------------------- what it must do


def test_nfc_composes_decomposed_sequences():
    decomposed = "café"  # e + combining acute
    assert len(decomposed) == 5
    result = normalize(decomposed)
    assert result == "café"
    assert len(result) == 4  # character count now matches perception, so CER is meaningful


def test_nfc_is_idempotent():
    assert normalize(normalize("café")) == normalize("café")


def test_already_composed_text_unchanged():
    composed = unicodedata.normalize("NFC", "café naïve")
    assert normalize(composed) == composed


@pytest.mark.parametrize("line_ending", ["\r\n", "\r", "\n"])
def test_line_endings_normalized(line_ending):
    assert normalize(f"first{line_ending}second") == "first second"


def test_whitespace_runs_collapse():
    assert normalize("a    b\t\tc") == "a b c"


def test_leading_and_trailing_whitespace_stripped():
    assert normalize("   padded   ") == "padded"


def test_tabs_and_newlines_become_spaces():
    assert normalize("a\tb\nc") == "a b c"


def test_non_breaking_space_collapses_to_ascii_space():
    """U+00A0 is visually indistinguishable in handwriting; documented in the policy notes."""
    assert normalize("a\u00a0b") == "a b"


def test_empty_and_whitespace_only():
    assert normalize("") == ""
    assert normalize("   \t\n ") == ""


def test_normalizes_to_empty_detects_whitespace_only_transcripts():
    assert normalizes_to_empty("  \u00a0 \t ")
    assert not normalizes_to_empty(" a ")


def test_normalization_is_idempotent_on_the_mixed_fixture():
    once = normalize(MIXED)
    assert normalize(once) == once


# --------------------------------------------------------------------------- policy plumbing


def test_identity_policy_changes_nothing():
    messy = "  a\r\n\u00a0b  "
    assert normalize(messy, IDENTITY) == messy


def test_policy_lookup_by_name():
    assert get_policy("nfc_v1") is NFC_V1


def test_unknown_policy_rejected():
    with pytest.raises(ValueError, match="Unknown normalization policy"):
        get_policy("aggressive_v9")


def test_policy_describes_itself_for_reporting():
    """Internal helper."""
    described = NFC_V1.describe()
    assert described["name"] == "nfc_v1"
    assert len(described["transforms"]) == 4
    assert any("whitespace" in note.lower() for note in described["notes"])
    assert any("comparable" in note.lower() for note in described["notes"])
