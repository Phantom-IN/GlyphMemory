"""Charset, tokenizer and charset coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glyphmemory.ctc import (
    BLANK_INDEX,
    BLANK_TOKEN,
    DEFAULT_CHARSET_PATH,
    NFC_V1,
    Charset,
    Tokenizer,
    UnsupportedCharacterError,
    charset_coverage,
    load_tokenizer,
    normalize,
    normalizes_to_empty,
)
from glyphmemory.data import ManifestRecord, validate_records
from glyphmemory.data.validation import IntegrityCategory

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def charset() -> Charset:
    return Charset.english_v1()


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer.english_v1()


def sample(text: str, sample_id: str = "synthetic/0") -> ManifestRecord:
    return ManifestRecord(
        image=f"/data/{sample_id.replace('/', '_')}.png",
        text=text,
        writer_id="synthetic/w0",
        dataset="synthetic",
        split="train",
        sample_id=sample_id,
    )


# --------------------------------------------------------------------------- blank invariant


def test_blank_is_index_zero(charset):
    """Loss, decoder, alignment and fusion all assume this."""
    assert charset.blank == BLANK_INDEX == 0
    assert charset.symbols[0] == BLANK_TOKEN


def test_charset_without_blank_at_zero_rejected():
    with pytest.raises(ValueError, match="blank must occupy index 0"):
        Charset(symbols=("a", BLANK_TOKEN))


def test_blank_is_not_a_real_character(charset):
    assert BLANK_TOKEN not in "".join(charset.characters)


# --------------------------------------------------------------------------- charset contents


def test_v1_has_expected_inventory(charset):
    characters = "".join(charset.characters)
    assert " " in characters
    for group in ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz", "0123456789"):
        assert all(c in characters for c in group)
    assert charset.size == 80  # 79 symbols + blank


def test_v1_excludes_punctuation_absent_from_the_reference_inventory(charset):
    """Provisional set; a measured successor ships as charset_en_v2 (see module docstring)."""
    for absent in "$%@=_<>[]{}|\\~^`":
        assert absent not in charset


def test_no_unk_token(charset):
    """Decision: an unsupported character is a counted data problem, not a token."""
    assert not any("unk" in symbol.lower() for symbol in charset.symbols)


def test_duplicate_symbols_rejected():
    with pytest.raises(ValueError, match="duplicate symbol"):
        Charset(symbols=(BLANK_TOKEN, "a", "a"))


def test_multi_character_symbols_rejected():
    with pytest.raises(ValueError, match="single characters"):
        Charset(symbols=(BLANK_TOKEN, "ab"))


def test_empty_charset_rejected():
    with pytest.raises(ValueError, match="at least the blank"):
        Charset(symbols=())


# --------------------------------------------------------------------------- encode / decode


def test_roundtrip_over_every_charset_character(tokenizer):
    """The success criterion: decode(encode(t)) == normalize(t)."""
    text = "".join(tokenizer.charset.characters)
    assert tokenizer.decode(tokenizer.encode(text)) == normalize(text)


@pytest.mark.parametrize("word", ["hello", "book", "letter", "committee", "aaa", "sees"])
def test_repeated_characters_roundtrip_without_deduplication(tokenizer, word):
    """Internal helper."""
    encoded = tokenizer.encode(word)
    assert len(encoded) == len(word)
    assert tokenizer.decode(encoded) == word


def test_encode_normalizes_by_default(tokenizer):
    assert tokenizer.decode(tokenizer.encode("  a   b  ")) == "a b"


def test_encode_can_skip_normalization(tokenizer):
    assert tokenizer.decode(tokenizer.encode("a b", apply_normalization=False)) == "a b"


def test_case_survives_roundtrip(tokenizer):
    text = "The Quick BROWN Fox"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_punctuation_survives_roundtrip(tokenizer):
    text = "Hello, world! (yes) -- it's #1?"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_empty_text_encodes_to_empty(tokenizer):
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_blank_never_produced_by_encode(tokenizer):
    text = "".join(tokenizer.charset.characters)
    assert BLANK_INDEX not in tokenizer.encode(text)


# --------------------------------------------------------------------------- unsupported chars


def test_unsupported_character_raises_naming_char_and_codepoint(tokenizer):
    with pytest.raises(UnsupportedCharacterError) as excinfo:
        tokenizer.encode("cost: 50€")
    message = str(excinfo.value)
    assert "€" in message
    assert "U+20AC" in message
    assert "unk" in message.lower()  # explains why it is not mapped away


def test_unsupported_character_reports_position(tokenizer):
    with pytest.raises(UnsupportedCharacterError) as excinfo:
        tokenizer.encode("ab€")
    assert excinfo.value.position == 2
    assert excinfo.value.character == "€"


def test_supports_and_unsupported_characters_helpers(tokenizer):
    assert tokenizer.supports("plain ascii text")
    assert not tokenizer.supports("50€ and 30£")
    assert tokenizer.unsupported_characters("50€ and 30£") == {"€", "£"}


def test_unsupported_character_is_counted_by_validation(tokenizer):
    """Wires IntegrityCategory.UNSUPPORTED_CHARACTER, declared but unraised in."""
    report = validate_records(
        [sample("costs 50€"), sample("plain text", "synthetic/1")],
        check_images=False,
        tokenizer=tokenizer,
    )
    assert report.counters.count_of(IntegrityCategory.UNSUPPORTED_CHARACTER) == 1
    assert report.valid_records == 1
    issue = report.counters.issues[0]
    assert "U+20AC" in issue.reason
    assert issue.sample_id == "synthetic/0"


def test_validation_without_tokenizer_does_not_flag_characters():
    report = validate_records([sample("costs 50€")], check_images=False)
    assert report.counters.count_of(IntegrityCategory.UNSUPPORTED_CHARACTER) == 0


@pytest.mark.parametrize("blank_text", ["   ", "\t\n ", "\u00a0\u00a0", "\r\n"])
def test_whitespace_only_transcripts_counted_as_missing(tokenizer, blank_text):
    report = validate_records([sample(blank_text)], check_images=False, tokenizer=tokenizer)
    assert report.counters.count_of(IntegrityCategory.MISSING_TRANSCRIPT) == 1
    assert "nfc_v1" in report.counters.issues[0].reason


def test_emptiness_check_names_the_policy_that_decided_it(tokenizer):
    """The reason must say *which* policy judged it empty, since that can change."""
    with_tok = validate_records([sample("\u00a0\u00a0")], check_images=False, tokenizer=tokenizer)
    without = validate_records([sample("\u00a0\u00a0")], check_images=False)
    assert "nfc_v1 normalization" in with_tok.counters.issues[0].reason
    assert "whitespace-only" in without.counters.issues[0].reason


def test_nfc_v1_emptiness_currently_coincides_with_str_strip():
    """Documents a real equivalence rather than implying the tokenizer catches more.

    Under ``nfc_v1`` both checks reduce to "is this only Unicode whitespace?", because
    ``str.split()`` and ``str.strip()`` share a whitespace definition and NFC never maps a non-space
    character to a space. The policy-aware check earns its place by naming the policy that decided,
    not by catching extra cases today — a future policy (or an NFKC-based one) could diverge.
    """
    for text in ["   ", "\t\n ", "\u00a0", "\r\n", " a ", "a", ""]:
        assert normalizes_to_empty(text) == (not text.strip())


# --------------------------------------------------------------------------- decode strictness


def test_decode_rejects_blank(tokenizer):
    """decode() is the inverse of encode(), not a CTC decoder."""
    with pytest.raises(ValueError, match="CTC decoder"):
        tokenizer.decode([BLANK_INDEX])


def test_decode_rejects_out_of_range_index(tokenizer):
    with pytest.raises(IndexError, match="out of range"):
        tokenizer.decode([tokenizer.vocab_size])


# --------------------------------------------------------------------------- fingerprint


def test_fingerprint_stable_across_loads(tmp_path, charset):
    path = charset.save(tmp_path / "cs.json")
    assert Charset.load(path).fingerprint() == charset.fingerprint()


def test_fingerprint_changes_when_a_symbol_is_added(charset):
    widened = Charset(symbols=(*charset.symbols, "€"), name=charset.name)
    assert widened.fingerprint() != charset.fingerprint()


def test_fingerprint_changes_when_symbols_are_reordered(charset):
    reordered = Charset(
        symbols=(charset.symbols[0], charset.symbols[2], charset.symbols[1], *charset.symbols[3:]),
        name=charset.name,
    )
    assert reordered.fingerprint() != charset.fingerprint()


def test_fingerprint_ignores_the_name(charset):
    """It answers 'same integer sequences?', so a rename must not invalidate checkpoints."""
    renamed = Charset(symbols=charset.symbols, name="charset_en_renamed")
    assert renamed.fingerprint() == charset.fingerprint()


def test_tokenizer_fingerprint_covers_the_policy(charset):
    from glyphmemory.ctc import IDENTITY

    assert Tokenizer(charset, NFC_V1).fingerprint() != Tokenizer(charset, IDENTITY).fingerprint()


# --------------------------------------------------------------------------- persistence


def test_committed_artifact_exists_and_loads():
    """artifacts/charset_en_v1.json is source of truth, not run output."""
    path = REPO_ROOT / DEFAULT_CHARSET_PATH
    assert path.is_file()
    loaded = Charset.load(path)
    assert loaded.size == 80
    assert loaded.fingerprint() == Charset.english_v1().fingerprint()


def test_load_tokenizer_from_committed_artifact():
    tok = load_tokenizer(REPO_ROOT / DEFAULT_CHARSET_PATH)
    assert tok.vocab_size == 80
    assert tok.blank_index == 0


def test_edited_artifact_is_detected(tmp_path, charset):
    """A hand-edited charset whose fingerprint no longer matches must not load silently."""
    path = charset.save(tmp_path / "cs.json")
    payload = json.loads(path.read_text())
    payload["characters"] = payload["characters"] + "€"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        Charset.load(path)


def test_artifact_declaring_nonzero_blank_rejected(tmp_path, charset):
    path = charset.save(tmp_path / "cs.json")
    payload = json.loads(path.read_text())
    payload["blank_index"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="blank_index"):
        Charset.load(path)


def test_describe_carries_everything_a_run_record_needs(tokenizer):
    described = tokenizer.describe()
    assert described["vocab_size"] == 80
    assert described["blank_index"] == 0
    assert described["normalization"]["name"] == "nfc_v1"
    assert described["charset_fingerprint"]


# --------------------------------------------------------------------------- derivation


def test_charset_derived_from_texts_is_deterministic_and_sorted():
    texts = ["banana", "apple", "Cherry 42!"]
    derived = Charset.from_texts(texts, name="derived_v1")
    assert derived.symbols[0] == BLANK_TOKEN
    assert list(derived.characters) == sorted(set("".join(normalize(t) for t in texts)))
    assert Charset.from_texts(reversed(texts), name="derived_v1") == derived


# --------------------------------------------------------------------------- coverage


def test_coverage_on_fully_supported_corpus(charset):
    report = charset_coverage([sample("hello world"), sample("HELLO", "s/1")], charset)
    assert report.is_covered
    assert report.total_samples == 2
    assert report.affected_samples == 0
    assert report.frequencies["l"] == 3


def test_coverage_reports_missing_characters(charset):
    report = charset_coverage(
        [sample("costs 50€"), sample("plain", "s/1"), sample("30£ and 5€", "s/2")], charset
    )
    assert not report.is_covered
    assert report.missing == {"€": 2, "£": 1}
    assert report.affected_samples == 2
    assert report.affected_fraction == pytest.approx(2 / 3)
    assert "s/2" in report.affected_sample_ids


def test_coverage_identifies_unused_symbols(charset):
    report = charset_coverage([sample("abc")], charset)
    unused = report.unused_symbols(charset)
    assert "z" in unused
    assert "a" not in unused


def test_coverage_report_serialises_and_formats(charset):
    report = charset_coverage([sample("hi 50€")], charset)
    assert json.loads(json.dumps(report.as_dict()))["affected_samples"] == 1
    rendered = report.format(charset=charset)
    assert "U+20AC" in rendered
    assert "unused symbols" in rendered


def test_coverage_of_empty_corpus(charset):
    report = charset_coverage([], charset)
    assert report.total_samples == 0
    assert report.affected_fraction == 0.0
    assert report.is_covered


def test_coverage_records_bounded_ids_but_exact_counts(charset):
    many = [sample("€", f"s/{i}") for i in range(120)]
    report = charset_coverage(many, charset, max_recorded_ids=10)
    assert report.affected_samples == 120
    assert len(report.affected_sample_ids) == 10
