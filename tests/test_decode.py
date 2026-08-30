"""Greedy CTC decoder tests.

The collapse rule is the single most common CTC decoder bug and it is silent: reverse the order of
"collapse repeats" and "strip blanks" and every doubled letter in the language vanishes, while the
loss keeps falling and nothing raises.

The second silent failure is decoding padding. Padding usually argmaxes to blank, so a broken
decoder is *usually* right — which is why the padding tests fill padded frames with a **non-blank**
class. A test that padded with blanks would pass against a broken decoder.
"""

from __future__ import annotations

import pytest
import torch

from glyphmemory.ctc import BLANK_INDEX, DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.ctc.decode import (
    DecoderConfig,
    collapse_repeats,
    ctc_collapse,
    decode_output,
    greedy_decode,
    greedy_decode_batch,
    greedy_decode_ids,
    one_hot_logits,
    strip_blanks,
)


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(DEFAULT_CHARSET_PATH)


def frames_for(tokenizer, text: str) -> list[int]:
    """A minimal valid CTC frame sequence for ``text``: a blank between adjacent repeats.

    This is the alignment a perfectly confident model would emit, and the shortest one that can
    represent the string at all.
    """
    ids = tokenizer.encode(text)
    frames: list[int] = []
    for position, label in enumerate(ids):
        if position and label == ids[position - 1]:
            frames.append(BLANK_INDEX)
        frames.append(label)
    return frames


class TestCollapseRule:
    def test_collapse_leaves_blanks_in_place(self) -> None:
        assert collapse_repeats([5, 5, 0, 0, 5, 5]) == [5, 0, 5]

    def test_strip_removes_only_blanks(self) -> None:
        assert strip_blanks([5, 0, 5, 0]) == [5, 5]

    def test_unseparated_repeat_collapses_to_one(self) -> None:
        """``l l`` is one letter emitted over two frames."""
        assert ctc_collapse([5, 5]) == [5]

    def test_blank_separated_repeat_survives_as_two(self) -> None:
        """``l <blank> l`` is a genuine double letter. This pair *is* the collapse rule."""
        assert ctc_collapse([5, 0, 5]) == [5, 5]

    def test_order_matters(self) -> None:
        """Strip-then-collapse would give the wrong answer; this pins the difference."""
        frames = [5, 0, 5]
        assert ctc_collapse(frames) == [5, 5]
        assert collapse_repeats(strip_blanks(frames)) == [5]

    def test_all_blank_decodes_to_nothing(self) -> None:
        assert ctc_collapse([0, 0, 0, 0]) == []

    def test_empty_input(self) -> None:
        assert ctc_collapse([]) == []


class TestNamedRepeatStrings:
    """Internal helper."""

    @pytest.mark.parametrize("text", ["hello", "book", "letter", "committee"])
    def test_round_trip_through_a_minimal_alignment(self, tokenizer, text: str) -> None:
        frames = frames_for(tokenizer, text)
        logits = one_hot_logits(frames, tokenizer.vocab_size)
        assert greedy_decode(logits, tokenizer) == text

    @pytest.mark.parametrize(
        ("text", "expected_frames"),
        [("hello", 6), ("book", 5), ("letter", 7), ("committee", 12)],
    )
    def test_minimal_alignment_lengths(self, tokenizer, text: str, expected_frames: int) -> None:
        """The frame budget each string needs — ``len(text) + adjacent_repeats``.

        If these disagree the pipeline rejects lines it could have decoded, or accepts lines it
        cannot.
        """
        assert len(frames_for(tokenizer, text)) == expected_frames

    @pytest.mark.parametrize("text", ["hello", "book", "letter", "committee"])
    def test_repeated_emission_still_decodes(self, tokenizer, text: str) -> None:
        """A real model holds each class for several frames; that must not double letters."""
        frames = [frame for frame in frames_for(tokenizer, text) for _ in range(3)]
        logits = one_hot_logits(frames, tokenizer.vocab_size)
        assert greedy_decode(logits, tokenizer) == text


class TestRoundTrip:
    @pytest.mark.parametrize(
        "text",
        [
            "a",
            "The quick brown fox jumps over the lazy dog",
            "aaa bbb ccc",
            "Mr. Gaitskell's 1961 (draft) plan!",
            "committee bookkeeper aardvark",
            "I I I",
            "0123456789",
        ],
    )
    def test_encode_align_decode_is_identity(self, tokenizer, text: str) -> None:
        frames = frames_for(tokenizer, text)
        logits = one_hot_logits(frames, tokenizer.vocab_size)
        assert greedy_decode(logits, tokenizer) == text

    def test_every_charset_symbol_round_trips(self, tokenizer) -> None:
        """Property test over the whole vocabulary.

        Each symbol is embedded as ``x<symbol>y`` rather than tested alone, because a lone space is
        not a representable transcript: ``nfc_v1`` strips leading and trailing whitespace, so ``"
        "`` normalizes to ``""``. That is a property of the normalization policy, not of the
        decoder, and embedding the symbol tests the decoder without pretending otherwise.
        """
        for character in tokenizer.charset.characters:
            text = f"x{character}y"
            frames = frames_for(tokenizer, text)
            logits = one_hot_logits(frames, tokenizer.vocab_size)
            assert greedy_decode(logits, tokenizer) == text

    def test_adjacent_pairs_of_every_symbol_round_trip(self, tokenizer) -> None:
        """Doubled characters are where the collapse rule bites.

        The space is excluded: ``nfc_v1`` collapses whitespace runs, so ``"x  y"`` is not a
        transcript the tokenizer can represent either.
        """
        for character in tokenizer.charset.characters:
            if character == " ":
                continue
            text = f"x{character * 2}y"
            frames = frames_for(tokenizer, text)
            logits = one_hot_logits(frames, tokenizer.vocab_size)
            assert greedy_decode(logits, tokenizer) == text


class TestPadding:
    def test_padded_frames_are_not_decoded(self, tokenizer) -> None:
        """Padding filled with a **non-blank** class, so a broken decoder cannot pass."""
        real = frames_for(tokenizer, "hi")
        noise = [tokenizer.encode("Z")[0]] * 20
        logits = one_hot_logits(real + noise, tokenizer.vocab_size)
        assert greedy_decode(logits, tokenizer, len(real)) == "hi"
        assert greedy_decode(logits, tokenizer) == "hiZ"  # without the length, padding leaks

    def test_batch_honours_per_sample_lengths(self, tokenizer) -> None:
        short = frames_for(tokenizer, "cat")
        long = frames_for(tokenizer, "elephant")
        noise = tokenizer.encode("Q")[0]

        batch = torch.stack(
            [
                one_hot_logits(short + [noise] * (len(long) - len(short)), tokenizer.vocab_size),
                one_hot_logits(long, tokenizer.vocab_size),
            ]
        )
        lengths = torch.tensor([len(short), len(long)])
        assert greedy_decode_batch(batch, tokenizer, lengths) == ["cat", "elephant"]

    def test_batch_without_lengths_decodes_padding(self, tokenizer) -> None:
        """Documented behaviour, pinned: omitting lengths is only safe for equal-length rows."""
        short = frames_for(tokenizer, "cat")
        noise = tokenizer.encode("Q")[0]
        batch = one_hot_logits(short + [noise] * 4, tokenizer.vocab_size).unsqueeze(0)
        assert greedy_decode_batch(batch, tokenizer) == ["catQ"]

    def test_zero_length_decodes_to_empty(self, tokenizer) -> None:
        logits = one_hot_logits(frames_for(tokenizer, "abc"), tokenizer.vocab_size)
        assert greedy_decode(logits, tokenizer, 0) == ""

    def test_rejects_length_beyond_t(self, tokenizer) -> None:
        logits = one_hot_logits([1, 2, 3], tokenizer.vocab_size)
        with pytest.raises(ValueError, match="exceeds T"):
            greedy_decode_ids(logits, 4)

    def test_rejects_negative_length(self, tokenizer) -> None:
        logits = one_hot_logits([1, 2, 3], tokenizer.vocab_size)
        with pytest.raises(ValueError, match="non-negative"):
            greedy_decode_ids(logits, -1)

    def test_rejects_length_count_mismatch(self, tokenizer) -> None:
        batch = one_hot_logits([1, 2, 3], tokenizer.vocab_size).unsqueeze(0)
        with pytest.raises(ValueError, match="entries for a batch"):
            greedy_decode_batch(batch, tokenizer, torch.tensor([3, 3]))


class TestInputForms:
    def test_logits_and_log_probs_agree(self, tokenizer) -> None:
        """``argmax`` is invariant to ``log_softmax``, so both forms are accepted."""
        logits = one_hot_logits(frames_for(tokenizer, "test"), tokenizer.vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)
        assert greedy_decode(logits, tokenizer) == greedy_decode(log_probs, tokenizer)

    def test_rejects_wrong_rank_for_single(self, tokenizer) -> None:
        with pytest.raises(ValueError, match=r"\[T, C\]"):
            greedy_decode_ids(torch.randn(2, 5, tokenizer.vocab_size))

    def test_rejects_wrong_rank_for_batch(self, tokenizer) -> None:
        with pytest.raises(ValueError, match=r"\[B, T, C\]"):
            greedy_decode_batch(torch.randn(5, tokenizer.vocab_size), tokenizer)

    def test_one_hot_logits_rejects_out_of_range_labels(self) -> None:
        with pytest.raises(ValueError, match="outside vocabulary"):
            one_hot_logits([0, 99], 10)


class TestDecoderConfig:
    def test_default_is_greedy_no_lm(self) -> None:
        config = DecoderConfig()
        assert config.kind == "greedy"
        assert config.label == "greedy, no LM"
        assert config.describe()["language_model"] is None
        assert config.describe()["lexicon"] is None

    def test_blank_index_is_recorded(self) -> None:
        assert DecoderConfig().blank_index == BLANK_INDEX == 0

    def test_rejects_a_decoder_that_does_not_exist(self) -> None:
        """A stub is a decoder that appears to exist."""
        with pytest.raises(ValueError, match="Only greedy decoding exists"):
            DecoderConfig(kind="beam")

    def test_rejects_beam_width_on_greedy(self) -> None:
        with pytest.raises(ValueError, match="beam_width is meaningless"):
            DecoderConfig(beam_width=5)


class TestAgainstTheModel:
    def test_untrained_model_output_decodes_without_raising(self, tokenizer) -> None:
        """Integration smoke test, not an accuracy test — the model is random."""
        from glyphmemory.model import GMBase

        torch.manual_seed(0)
        model = GMBase(vocab_size=tokenizer.vocab_size).eval()
        with torch.no_grad():
            output = model(torch.randn(3, 1, 64, 512), torch.tensor([128, 90, 40]))
        texts = decode_output(output, tokenizer)
        assert len(texts) == 3
        assert all(isinstance(text, str) for text in texts)

    def test_decode_output_uses_the_carried_lengths(self, tokenizer) -> None:
        """Padding cannot be decoded by forgetting to pass lengths — they come along."""
        from glyphmemory.model import GMBase

        torch.manual_seed(1)
        model = GMBase(vocab_size=tokenizer.vocab_size).eval()
        with torch.no_grad():
            output = model(torch.randn(2, 1, 64, 512), torch.tensor([128, 20]))
        assert decode_output(output, tokenizer) == greedy_decode_batch(
            output.logits, tokenizer, output.input_lengths
        )
