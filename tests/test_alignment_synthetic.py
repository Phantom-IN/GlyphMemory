"""Oracle tests on the synthetic font-as-writer corpus.

Instead, each synthetic record's **known transcript** is used to build log- probabilities with a
known, constructed per-character timing (uniform frame count per character, an explicit blank
inserted between adjacent identical characters), and the test checks the aligner recovers exactly
that structure. This exercises real, varied transcripts — repeated letters and spaces occurring
naturally, not hand-picked — as a check independent of `test_alignment_forced_align.py`'s
deliberately-constructed edge cases.

**Correctness evidence only, never a performance benchmark**: nothing here is reported as a claim
about real-handwriting alignment quality.
"""

from __future__ import annotations

import torch

from glyphmemory.alignment import forced_align
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer

FRAMES_PER_CHARACTER = 3


def _confident_log_probs_for_text(text: str, charset, *, peak: float = 0.9) -> torch.Tensor:
    """Build ``[T, C]`` log-probs whose most-likely path is exactly ``text``, uniform-width per
    character, with an explicit blank frame forced between adjacent identical characters.
    """
    vocab = charset.size
    frame_classes: list[int] = [charset.blank]
    previous = None
    for character in text:
        index = charset.index_of(character)
        if previous == index:
            frame_classes.append(charset.blank)
        frame_classes.extend([index] * FRAMES_PER_CHARACTER)
        previous = index
    frame_classes.append(charset.blank)

    rest = (1.0 - peak) / (vocab - 1)
    rows = []
    for cls in frame_classes:
        row = [rest] * vocab
        row[cls] = peak
        rows.append(row)
    return torch.tensor(rows).log()


def test_recovers_exact_character_sequence_for_every_synthetic_record(synthetic_corpus):
    charset = load_tokenizer(DEFAULT_CHARSET_PATH).charset
    checked = 0
    for record in synthetic_corpus.records:
        text = record.text
        if not text or any(character not in charset for character in text):
            continue
        log_probs = _confident_log_probs_for_text(text, charset)
        result = forced_align(log_probs, text, charset)
        assert [span.token for span in result.spans] == list(text)
        checked += 1
    assert checked > 0, "no synthetic records were usable — the test fixture may have changed"


def test_repeated_character_spans_are_distinct_nonoverlapping_and_ordered(synthetic_corpus):
    """The random "coverage" text generator does not guarantee an adjacent repeated character in any
    given small sample, so one is constructed by duplicating a character from real sampled text —
    grounded in the corpus's own vocabulary and mixed with real surrounding characters, not a fully
    hand-picked string.
    """
    charset = load_tokenizer(DEFAULT_CHARSET_PATH).charset
    checked_a_repeat = False
    for record in synthetic_corpus.records:
        base_text = record.text
        if not base_text or any(character not in charset for character in base_text):
            continue
        # Duplicate the first character in place, guaranteeing an adjacent repeat.
        text = base_text[0] + base_text
        checked_a_repeat = True

        log_probs = _confident_log_probs_for_text(text, charset)
        result = forced_align(log_probs, text, charset)

        assert len(result.spans) == len(text)
        for earlier, later in zip(result.spans, result.spans[1:], strict=False):
            assert earlier.end_t <= later.start_t

    assert checked_a_repeat, "expected at least one usable synthetic record to duplicate"


def test_span_scores_are_high_confidence_by_construction(synthetic_corpus):
    charset = load_tokenizer(DEFAULT_CHARSET_PATH).charset
    record = next(r for r in synthetic_corpus.records if r.text)
    log_probs = _confident_log_probs_for_text(record.text, charset, peak=0.95)
    result = forced_align(log_probs, record.text, charset)
    for span in result.spans:
        assert span.score > 0.9
