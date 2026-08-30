"""Width-bucket sampling and DataLoader wiring."""

from __future__ import annotations

import random

import pytest

from glyphmemory.config import Config
from glyphmemory.ctc import Tokenizer
from glyphmemory.data import (
    LineDataset,
    ManifestRecord,
    SequentialBatchSampler,
    WidthBucketSampler,
    build_dataloader,
    build_dataset,
    padding_efficiency,
)

TOKENIZER = Tokenizer.english_v1()


def skewed_widths(count: int = 256, seed: int = 0) -> list[int]:
    """A realistically skewed distribution: many short lines, a long tail."""
    rng = random.Random(seed)
    return [int(rng.lognormvariate(6.2, 0.55)) for _ in range(count)]


def mean_efficiency(sampler, widths) -> float:
    scores = []
    for batch in sampler:
        chosen = [widths[index] for index in batch]
        scores.append(padding_efficiency(chosen))
    return sum(scores) / len(scores)


# --------------------------------------------------------------------------- efficiency metric


def test_padding_efficiency_is_one_when_widths_match():
    assert padding_efficiency([100, 100, 100]) == pytest.approx(1.0)


def test_padding_efficiency_halves_for_a_doubled_maximum():
    assert padding_efficiency([50, 100]) == pytest.approx(0.75)


def test_padding_efficiency_of_empty_is_zero():
    assert padding_efficiency([]) == 0.0


def test_padding_efficiency_honours_an_explicit_maximum():
    assert padding_efficiency([100, 100], max_width=200) == pytest.approx(0.5)


# --------------------------------------------------------------------------- bucketing


def test_bucketing_beats_sequential_on_a_skewed_distribution():
    """The reason bucketing exists at all.

    Measured on a lognormal width distribution (137-1903 px): sequential batching wastes over half
    the batch tensor on padding, bucketing recovers most of it.
    """
    widths = skewed_widths()
    bucketed = mean_efficiency(WidthBucketSampler(widths, 16, seed=1), widths)
    sequential = mean_efficiency(SequentialBatchSampler(len(widths), 16), widths)
    assert sequential < 0.6
    assert bucketed > 0.8
    assert bucketed - sequential > 0.25


def test_larger_pools_group_more_tightly():
    """Pool size trades randomness for width homogeneity; multiplier=1 is a no-op."""
    widths = skewed_widths()
    scores = [
        mean_efficiency(
            WidthBucketSampler(widths, 16, seed=1, bucket_multiplier=multiplier), widths
        )
        for multiplier in (1, 4, 16)
    ]
    assert scores[0] < scores[1] < scores[2]

    sequential = mean_efficiency(SequentialBatchSampler(len(widths), 16), widths)
    assert scores[0] == pytest.approx(sequential, abs=0.05)


def test_bucketing_covers_every_index_exactly_once():
    widths = skewed_widths(100)
    seen = [index for batch in WidthBucketSampler(widths, 8, seed=2) for index in batch]
    assert sorted(seen) == list(range(100))


def test_batch_sizes_are_correct():
    widths = skewed_widths(100)
    batches = list(WidthBucketSampler(widths, 8, seed=3))
    assert all(len(batch) <= 8 for batch in batches)
    assert sum(len(batch) for batch in batches) == 100


def test_drop_last_removes_the_short_batch():
    widths = skewed_widths(100)
    batches = list(WidthBucketSampler(widths, 8, seed=3, drop_last=True))
    assert all(len(batch) == 8 for batch in batches)
    assert len(batches) == 12


def test_len_matches_the_number_of_batches():
    widths = skewed_widths(100)
    for drop_last in (False, True):
        sampler = WidthBucketSampler(widths, 8, seed=4, drop_last=drop_last)
        assert len(sampler) == len(list(sampler))


def test_batches_are_width_homogeneous():
    widths = skewed_widths()
    spreads = []
    for batch in WidthBucketSampler(widths, 16, seed=5):
        chosen = [widths[index] for index in batch]
        spreads.append(max(chosen) - min(chosen))
    assert sum(spreads) / len(spreads) < (max(widths) - min(widths)) / 4


# --------------------------------------------------------------------------- determinism


def test_sampler_is_deterministic_under_seed():
    widths = skewed_widths(64)
    first = list(WidthBucketSampler(widths, 8, seed=7))
    second = list(WidthBucketSampler(widths, 8, seed=7))
    assert first == second


def test_different_seeds_give_different_batches():
    widths = skewed_widths(64)
    assert list(WidthBucketSampler(widths, 8, seed=1)) != list(
        WidthBucketSampler(widths, 8, seed=2)
    )


def test_set_epoch_changes_the_grouping():
    """Without this, every epoch sees identical batches and content correlates with width."""
    widths = skewed_widths(64)
    sampler = WidthBucketSampler(widths, 8, seed=9)
    first = list(sampler)
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_shuffle_off_is_stable_across_epochs():
    widths = skewed_widths(64)
    sampler = WidthBucketSampler(widths, 8, seed=9, shuffle=False)
    first = list(sampler)
    sampler.set_epoch(3)
    assert list(sampler) == first


def test_empty_dataset_yields_no_batches():
    assert list(WidthBucketSampler([], 8)) == []


@pytest.mark.parametrize(("batch_size", "multiplier"), [(0, 8), (8, 0)])
def test_invalid_sampler_parameters_rejected(batch_size, multiplier):
    with pytest.raises(ValueError, match="at least 1"):
        WidthBucketSampler([10, 20], batch_size, bucket_multiplier=multiplier)


def test_sequential_sampler_preserves_order():
    assert list(SequentialBatchSampler(10, 4)) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_sequential_sampler_rejects_bad_batch_size():
    with pytest.raises(ValueError, match="at least 1"):
        SequentialBatchSampler(10, 0)


# --------------------------------------------------------------------------- dataloader


def test_dataloader_yields_valid_batches(synthetic_corpus):
    config = Config()
    dataset = build_dataset(synthetic_corpus.manifest_path, TOKENIZER, config, training=False)
    loader = build_dataloader(dataset, config, training=False, batch_size=4, num_workers=0)

    total = 0
    for batch in loader:
        total += batch.batch_size
        assert batch.images.shape[1] == 1
        assert batch.images.shape[2] == config.data.image_height
        assert int(batch.target_lengths.sum()) == int(batch.targets.numel())
    assert total == len(dataset)


@pytest.mark.slow
def test_dataloader_with_workers_is_deterministic_and_complete(synthetic_corpus):
    """Exercises the `spawn` path DataLoader workers use on macOS.

    Kept despite costing real wall-clock: this is what caught a policy comparison done with `is`,
    which silently broke every multi-worker run once the dataset crossed a process boundary. One
    test covers both properties so only one set of workers is spawned.
    """
    config = Config()
    dataset = build_dataset(synthetic_corpus.manifest_path, TOKENIZER, config, training=False)

    def run(workers):
        loader = build_dataloader(
            dataset, config, training=False, batch_size=3, num_workers=workers
        )
        return [batch.sample_ids for batch in loader]

    with_workers = run(1)
    assert run(1) == with_workers, "worker batches are not reproducible"

    flat_workers = sorted(sid for batch in with_workers for sid in batch)
    flat_single = sorted(sid for batch in run(0) for sid in batch)
    assert flat_workers == flat_single, "workers lost or duplicated samples"


def test_evaluation_loader_does_not_shuffle(synthetic_corpus):
    config = Config()
    dataset = build_dataset(synthetic_corpus.manifest_path, TOKENIZER, config, training=False)
    loader = build_dataloader(dataset, config, training=False, batch_size=4, num_workers=0)
    assert [b.sample_ids for b in loader] == [b.sample_ids for b in loader]


def test_bucketing_can_be_disabled_without_losing_samples(synthetic_corpus):
    config = Config()
    dataset = build_dataset(synthetic_corpus.manifest_path, TOKENIZER, config, training=False)

    def ids(bucket):
        loader = build_dataloader(
            dataset, config, training=False, batch_size=3, bucket=bucket, num_workers=0
        )
        return sorted(sid for batch in loader for sid in batch.sample_ids)

    assert ids(True) == ids(False)


def test_bucketing_improves_efficiency_on_real_fixtures(synthetic_corpus):
    config = Config()
    dataset = LineDataset(records=synthetic_corpus.records, tokenizer=TOKENIZER)

    def efficiency(bucket):
        loader = build_dataloader(
            dataset, config, training=False, batch_size=3, bucket=bucket, num_workers=0
        )
        scores = [batch.padding_efficiency for batch in loader]
        return sum(scores) / len(scores)

    assert efficiency(True) >= efficiency(False)


# --------------------------------------------------------------- height-varying sources
# The original tests could not have caught the defect below: synthetic
# lines render at a constant height, so the manifest width and the post-preprocessing width
# are exactly proportional and bucketing on either gives the same batches. It takes sources
# with *varying* heights — real handwriting — for the two orderings to diverge.


def _record(sample_id: str, width: int, height: int) -> ManifestRecord:
    return ManifestRecord(
        image=f"/tmp/{sample_id}.png",
        text="text",
        writer_id="w",
        dataset="test",
        split="train",
        sample_id=sample_id,
        width=width,
        height=height,
    )


def _dataset(records):
    from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer

    return LineDataset(records=tuple(records), tokenizer=load_tokenizer(DEFAULT_CHARSET_PATH))


def test_widths_predict_the_post_preprocessing_width_not_the_manifest_width():
    """The defect found: the collator pads to the height-normalized width."""
    # Same source width, very different heights -> very different padded widths.
    dataset = _dataset([_record("a", 1000, 50), _record("b", 1000, 200)])
    assert dataset.widths == [1280, 320]


def test_widths_are_unchanged_when_the_source_is_already_at_target_height():
    dataset = _dataset([_record("a", 400, 64), _record("b", 900, 64)])
    assert dataset.widths == [400, 900]


def test_widths_fall_back_to_the_manifest_width_without_a_height():
    dataset = _dataset(
        [
            ManifestRecord(
                image="/tmp/x.png",
                text="t",
                writer_id="w",
                dataset="d",
                split="train",
                sample_id="x",
                width=512,
            )
        ]
    )
    assert dataset.widths == [512]


def test_widths_fall_back_to_zero_without_either():
    dataset = _dataset(
        [
            ManifestRecord(
                image="/tmp/x.png",
                text="t",
                writer_id="w",
                dataset="d",
                split="train",
                sample_id="x",
            )
        ]
    )
    assert dataset.widths == [0]


def test_bucketing_groups_by_padded_width_on_height_varying_sources():
    """The regression guard.

    Sources whose manifest widths are all identical but whose heights vary widely: bucketing on the
    manifest width can do nothing at all here, because every source width is equal. Bucketing on the
    padded width must still group them tightly.
    """
    records = [
        _record(f"s{i}", 1000, height) for i, height in enumerate([40, 45, 50, 160, 170, 180])
    ]
    dataset = _dataset(records)
    widths = dataset.widths
    assert max(widths) / min(widths) > 3, "fixture must actually vary the padded width"

    sampler = WidthBucketSampler(widths, batch_size=3, shuffle=False, bucket_multiplier=8)
    batches = list(sampler)
    for batch in batches:
        grouped = [widths[i] for i in batch]
        # Every batch is drawn from one end of the range, never mixed across it.
        assert max(grouped) / min(grouped) < 1.5, grouped


def test_bucketing_beats_sequential_on_height_varying_sources():
    import random

    rng = random.Random(7)
    records = [_record(f"s{i}", rng.randint(300, 2000), rng.randint(44, 176)) for i in range(64)]
    widths = _dataset(records).widths

    sequential = [
        padding_efficiency([widths[i] for i in b])
        for b in SequentialBatchSampler(len(widths), batch_size=8)
    ]
    bucketed = [
        padding_efficiency([widths[i] for i in b])
        for b in WidthBucketSampler(widths, batch_size=8, shuffle=False)
    ]
    sequential_mean = sum(sequential) / len(sequential)
    bucketed_mean = sum(bucketed) / len(bucketed)
    assert bucketed_mean > sequential_mean + 0.15, (
        f"bucketing {bucketed_mean:.3f} vs sequential {sequential_mean:.3f} — the whole point "
        "of bucketing is that this gap is large"
    )
