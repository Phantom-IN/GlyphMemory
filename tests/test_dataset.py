"""LineDataset.

Covers the sample contract and the rejection paths. A dataset that raises inside ``__getitem__``
loses a whole epoch to one bad file, so unusable records come back as typed `RejectedSample`s
instead — counted by the collator, never silently dropped.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

from glyphmemory.config import Config
from glyphmemory.ctc import Tokenizer, normalize
from glyphmemory.data import (
    LineDataset,
    LineSample,
    ManifestRecord,
    RejectedSample,
    build_augmentation,
    build_dataset,
    required_ctc_length,
    temporal_length,
    write_manifest,
)
from glyphmemory.data.validation import IntegrityCategory


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer.english_v1()


@pytest.fixture
def dataset(synthetic_corpus, tokenizer) -> LineDataset:
    return LineDataset(records=synthetic_corpus.records, tokenizer=tokenizer)


# --------------------------------------------------------------------------- required length


@pytest.mark.parametrize(
    ("text", "expected"),
    [("hello", 6), ("book", 5), ("letter", 7), ("committee", 12), ("aaa", 5), ("abc", 3)],
)
def test_required_length_counts_adjacent_repeats(tokenizer, text, expected):
    """CTC must insert a blank between identical consecutive labels."""
    assert required_ctc_length(tokenizer.encode(text)) == expected


def test_required_length_of_empty_is_zero():
    assert required_ctc_length([]) == 0


def test_required_length_never_below_target_length(tokenizer):
    for text in ("hello", "committee", "abcdef", "aaaa"):
        targets = tokenizer.encode(text)
        assert required_ctc_length(targets) >= len(targets)


# --------------------------------------------------------------------------- sample contract


def test_sample_shape_and_dtypes(dataset):
    sample = dataset[0]
    assert isinstance(sample, LineSample)
    assert sample.image.shape[0] == 1
    assert sample.image.shape[1] == 64
    assert sample.image.dtype == torch.float32
    assert sample.targets.dtype == torch.long


def test_sample_input_length_comes_from_true_width(dataset):
    sample = dataset[0]
    assert sample.input_length == temporal_length(sample.true_width)
    assert sample.true_width <= int(sample.image.shape[-1])


def test_sample_text_is_normalized_and_round_trips(dataset, tokenizer):
    for index in range(len(dataset)):
        sample = dataset[index]
        assert tokenizer.decode(sample.targets.tolist()) == sample.text
        assert sample.text == normalize(sample.text)


def test_sample_carries_provenance(dataset, synthetic_corpus):
    sample = dataset[0]
    assert sample.sample_id.startswith("synthetic/")
    assert sample.writer_id.startswith("synthetic/")
    assert Path(sample.image_path).is_file()


def test_dataset_length_matches_records(dataset, synthetic_corpus):
    assert len(dataset) == len(synthetic_corpus.records)


def test_widths_are_exposed_for_bucketing(dataset):
    widths = dataset.widths
    assert len(widths) == len(dataset)
    assert all(width > 0 for width in widths)


def test_pad_value_matches_normalization(dataset):
    assert dataset.pad_value == pytest.approx(0.0)


# --------------------------------------------------------------------------- rejections


def make_manifest(tmp_path: Path, records: list[ManifestRecord]) -> Path:
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, records)
    return path


def test_missing_image_is_rejected_not_raised(tmp_path, tokenizer):
    record = ManifestRecord(
        image=str(tmp_path / "absent.png"),
        text="hello",
        writer_id="synthetic/w0",
        dataset="synthetic",
        split="train",
        sample_id="synthetic/0",
    )
    rejection = LineDataset(records=(record,), tokenizer=tokenizer)[0]
    assert isinstance(rejection, RejectedSample)
    assert rejection.category is IntegrityCategory.UNREADABLE_IMAGE
    assert rejection.sample_id == "synthetic/0"
    assert rejection.reason


def test_unsupported_character_is_rejected(tmp_path, tokenizer, synthetic_corpus):
    record = ManifestRecord(
        image=synthetic_corpus.records[0].image,
        text="costs 50€",
        writer_id="synthetic/w0",
        dataset="synthetic",
        split="train",
        sample_id="synthetic/euro",
    )
    rejection = LineDataset(records=(record,), tokenizer=tokenizer)[0]
    assert isinstance(rejection, RejectedSample)
    assert rejection.category is IntegrityCategory.UNSUPPORTED_CHARACTER
    assert "U+20AC" in rejection.reason


def test_whitespace_only_transcript_is_rejected(tokenizer, synthetic_corpus):
    record = ManifestRecord(
        image=synthetic_corpus.records[0].image,
        text="   ",
        writer_id="synthetic/w0",
        dataset="synthetic",
        split="train",
        sample_id="synthetic/blank",
    )
    rejection = LineDataset(records=(record,), tokenizer=tokenizer)[0]
    assert isinstance(rejection, RejectedSample)
    assert rejection.category is IntegrityCategory.MISSING_TRANSCRIPT


# --------------------------------------------------------------------------- construction


def test_from_manifest_filters_by_split(synthetic_corpus, tokenizer):
    everything = LineDataset.from_manifest(synthetic_corpus.manifest_path, tokenizer)
    train = LineDataset.from_manifest(synthetic_corpus.manifest_path, tokenizer, split="train")
    absent = LineDataset.from_manifest(synthetic_corpus.manifest_path, tokenizer, split="test")
    assert len(train) == len(everything)
    assert len(absent) == 0


def test_build_dataset_applies_config(synthetic_corpus, tokenizer):
    config = Config()
    dataset = build_dataset(synthetic_corpus.manifest_path, tokenizer, config, training=False)
    assert dataset.height == config.data.image_height
    assert dataset.max_width == config.data.max_width


def test_evaluation_dataset_is_unaugmented(synthetic_corpus, tokenizer):
    config = Config()
    evaluation = build_dataset(synthetic_corpus.manifest_path, tokenizer, config, training=False)
    assert evaluation.augmentation.is_identity


def test_training_dataset_is_augmented(synthetic_corpus, tokenizer):
    config = Config()
    training = build_dataset(synthetic_corpus.manifest_path, tokenizer, config, training=True)
    assert not training.augmentation.is_identity


def test_augmentation_changes_samples_but_not_targets(synthetic_corpus, tokenizer):
    plain = LineDataset(records=synthetic_corpus.records, tokenizer=tokenizer)
    augmented = LineDataset(
        records=synthetic_corpus.records,
        tokenizer=tokenizer,
        augmentation=build_augmentation(Config().data.augmentation, training=True),
    )
    torch.manual_seed(0)
    a = augmented[0]
    b = plain[0]
    assert torch.equal(a.targets, b.targets)
    assert a.text == b.text


# --------------------------------------------------------------------------- picklability


def test_dataset_is_picklable(dataset):
    """DataLoader workers use `spawn` on macOS, so the dataset must pickle."""
    restored = pickle.loads(pickle.dumps(dataset))
    assert len(restored) == len(dataset)
    assert restored[0].sample_id == dataset[0].sample_id


def test_augmented_dataset_is_picklable(synthetic_corpus, tokenizer):
    dataset = LineDataset(
        records=synthetic_corpus.records,
        tokenizer=tokenizer,
        augmentation=build_augmentation(Config().data.augmentation, training=True),
    )
    assert len(pickle.loads(pickle.dumps(dataset))) == len(dataset)
