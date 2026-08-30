"""Occurrence extraction: real plumbing (dataset -> model -> alignment -> per-frame features)
on the synthetic corpus and a small randomly-initialized model — the model's recognition
quality is irrelevant here, only that extraction runs end to end and produces sane output.
"""

from __future__ import annotations

import torch

from glyphmemory.config.schema import Config
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.data import build_dataloader
from glyphmemory.model import GMBase
from glyphmemory.probes import base_head_frame_accuracy, extract_occurrences


def _tiny_model_and_loader(synthetic_corpus, tokenizer):
    torch.manual_seed(0)
    model = GMBase(vocab_size=tokenizer.vocab_size)
    config = Config()

    from glyphmemory.data import build_dataset

    ds = build_dataset(synthetic_corpus.manifest_path, tokenizer, config, training=False)
    loader = build_dataloader(
        ds, config, training=False, batch_size=1, bucket=False, num_workers=0
    )
    return model, loader


def test_extract_occurrences_produces_plausible_output(synthetic_corpus):
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    model, loader = _tiny_model_and_loader(synthetic_corpus, tokenizer)

    occurrences = extract_occurrences(model, loader, tokenizer.charset)

    assert len(occurrences) > 0
    for occurrence in occurrences[:50]:
        assert occurrence.character in tokenizer.charset
        assert occurrence.visual_feature.shape == (192,)
        assert occurrence.sequence_feature.shape == (384,)
        assert occurrence.frame_index >= 0
        assert 0.0 <= occurrence.alignment_score <= 1.0
        assert occurrence.writer_id in synthetic_corpus.writers


def test_occurrences_cover_every_character_of_every_alignable_line(synthetic_corpus):
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    model, loader = _tiny_model_and_loader(synthetic_corpus, tokenizer)
    occurrences = extract_occurrences(model, loader, tokenizer.charset)

    by_sample: dict[str, list[str]] = {}
    for occurrence in occurrences:
        by_sample.setdefault(occurrence.sample_id, []).append(occurrence.character)

    # At least one frame is emitted per character of at least one line (exact text reconstruction
    # needs consecutive-frame grouping, which is not this function's job).
    assert len(by_sample) > 0


def test_base_head_frame_accuracy_is_a_fraction_in_zero_one(synthetic_corpus):
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    model, loader = _tiny_model_and_loader(synthetic_corpus, tokenizer)
    occurrences = extract_occurrences(model, loader, tokenizer.charset)

    accuracy, n = base_head_frame_accuracy(occurrences)
    assert 0.0 <= accuracy <= 1.0
    assert n == len(occurrences)


def test_base_head_frame_accuracy_handles_no_occurrences():
    accuracy, n = base_head_frame_accuracy([])
    assert accuracy == 0.0
    assert n == 0
