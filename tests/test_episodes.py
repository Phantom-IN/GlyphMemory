"""Writer episodes for episodic training: many `(support, query)` draws per writer, support size
varying, strictly train-split writers only — the different access pattern `SupportQuerySplit` (one
fixed reservation per writer) does not serve.
"""

from __future__ import annotations

import itertools

import pytest

from glyphmemory.data import ManifestRecord, SplitLeakError
from glyphmemory.data.episodes import (
    DEFAULT_QUERY_SIZE,
    DEFAULT_SUPPORT_SIZES,
    Episode,
    EpisodeSampler,
    iter_writer_cycle,
)


def record(
    writer: str,
    index: int,
    *,
    passage: str | None = None,
    split: str = "train",
    dataset: str = "synthetic",
) -> ManifestRecord:
    return ManifestRecord(
        image=f"/data/{writer}-{index}.png",
        text=f"line {index} by {writer}",
        writer_id=writer,
        dataset=dataset,
        split=split,
        sample_id=f"{dataset}/{writer}-{index}",
        passage_id=passage,
    )


def writer_lines(writer: str, n: int, *, passages_of: int | None = None) -> list[ManifestRecord]:
    """``n`` train-split lines for ``writer``. ``passages_of`` groups them into passages of that
    size (e.g. ``passages_of=3`` -> lines 0-2 share a passage, 3-5 share the next, ...); ``None``
    gives every line its own passage (fully disjoint, like `test_splits.py`'s default corpus).
    """
    if passages_of is None:
        return [record(writer, i, passage=f"{writer}-p{i}") for i in range(n)]
    return [record(writer, i, passage=f"{writer}-p{i // passages_of}") for i in range(n)]


# ------------------------------------------------------------------ construction / disjointness


def test_rejects_a_non_train_split_record():
    records = [*writer_lines("w0", 10), record("w1", 0, split="val")]
    with pytest.raises(SplitLeakError, match="train-split"):
        EpisodeSampler(records)


def test_rejects_a_test_split_record_too():
    records = [*writer_lines("w0", 10), record("w1", 0, split="test")]
    with pytest.raises(SplitLeakError):
        EpisodeSampler(records)


def test_accepts_pure_train_split_records():
    sampler = EpisodeSampler(writer_lines("w0", 10))
    assert sampler.writers == frozenset({"w0"})


def test_rejects_zero_query_size():
    with pytest.raises(ValueError, match="query_size"):
        EpisodeSampler(writer_lines("w0", 10), query_size=0)


def test_rejects_empty_support_sizes():
    with pytest.raises(ValueError, match="support_sizes"):
        EpisodeSampler(writer_lines("w0", 10), support_sizes=())


def test_rejects_non_positive_support_size():
    with pytest.raises(ValueError, match="support_sizes"):
        EpisodeSampler(writer_lines("w0", 10), support_sizes=(1, 0, 5))


def test_rejects_a_record_with_no_sample_id():
    bad = ManifestRecord(
        image="/data/x.png", text="hi", writer_id="w0", dataset="synthetic", split="train"
    )
    with pytest.raises(ValueError, match="sample_id"):
        EpisodeSampler([bad])


# ------------------------------------------------------------------ feasible_support_sizes


def test_feasible_support_sizes_filters_by_real_line_count():
    # query_size default 2; writer has 6 lines -> feasible n: 1 (needs>=3), 3 (needs>=5) but not 5
    # (needs>=7) or 10 (needs>=12).
    sampler = EpisodeSampler(writer_lines("w0", 6))
    assert sampler.feasible_support_sizes("w0") == (1, 3)


def test_feasible_support_sizes_empty_for_an_unknown_writer():
    sampler = EpisodeSampler(writer_lines("w0", 100))
    assert sampler.feasible_support_sizes("nonexistent") == ()


def test_feasible_support_sizes_all_shots_for_a_large_writer():
    sampler = EpisodeSampler(writer_lines("w0", 200))
    assert sampler.feasible_support_sizes("w0") == DEFAULT_SUPPORT_SIZES


# ------------------------------------------------------------------ sample() — happy path


def test_sample_returns_an_episode_with_requested_writer():
    sampler = EpisodeSampler(writer_lines("w0", 50), group_of=None)
    episode = sampler.sample("w0", draw_index=0)
    assert isinstance(episode, Episode)
    assert episode.writer_id == "w0"


def test_sample_query_length_is_exactly_query_size():
    sampler = EpisodeSampler(writer_lines("w0", 50), query_size=3, group_of=None)
    episode = sampler.sample("w0", draw_index=0)
    assert len(episode.query_ids) == 3


def test_sample_support_size_is_one_of_the_feasible_sizes():
    sampler = EpisodeSampler(writer_lines("w0", 50), group_of=None)
    episode = sampler.sample("w0", draw_index=0)
    assert episode.support_size in sampler.feasible_support_sizes("w0")


def test_sample_support_and_query_never_overlap():
    sampler = EpisodeSampler(writer_lines("w0", 50), group_of=None)
    for draw_index in range(20):
        episode = sampler.sample("w0", draw_index)
        assert not (set(episode.support_ids) & set(episode.query_ids))


def test_sample_rejects_an_unknown_writer():
    sampler = EpisodeSampler(writer_lines("w0", 50))
    with pytest.raises(ValueError, match="not a train-split writer"):
        sampler.sample("nonexistent", draw_index=0)


def test_sample_rejects_a_writer_with_too_few_lines_for_any_support_size():
    # 2 lines, query_size=2 -> even support_size=1 needs 3 lines. Infeasible.
    sampler = EpisodeSampler(writer_lines("w0", 2), group_of=None)
    with pytest.raises(ValueError, match="too few"):
        sampler.sample("w0", draw_index=0)


# ------------------------------------------------------------------ determinism


def test_sample_is_deterministic_given_the_same_seed_writer_and_draw_index():
    a = EpisodeSampler(writer_lines("w0", 50), seed=1337, group_of=None).sample("w0", 3)
    b = EpisodeSampler(writer_lines("w0", 50), seed=1337, group_of=None).sample("w0", 3)
    assert a == b


def test_sample_differs_across_draw_indices():
    sampler = EpisodeSampler(writer_lines("w0", 200), group_of=None)
    episodes = [sampler.sample("w0", i) for i in range(10)]
    # Not a single fixed reservation: at least one draw differs from the first (support size or the
    # specific IDs drawn) -- the exact property `SupportQuerySplit` does not have.
    assert any(e != episodes[0] for e in episodes[1:])


def test_sample_differs_across_seeds_usually():
    a = EpisodeSampler(writer_lines("w0", 200), seed=1, group_of=None).sample("w0", 0)
    b = EpisodeSampler(writer_lines("w0", 200), seed=2, group_of=None).sample("w0", 0)
    assert a != b


# ------------------------------------------------------------------ passage-disjoint grouping


def test_opting_into_passage_grouping_keeps_support_and_query_passage_disjoint():
    # 10 passages of 3 lines each = 30 lines, plenty for every support size at query_size=2.
    lines = writer_lines("w0", 30, passages_of=3)
    sampler = EpisodeSampler(lines, group_of=lambda r: r.passage_id)
    by_id = {r.sample_id: r for r in lines}
    for draw_index in range(10):
        episode = sampler.sample("w0", draw_index)
        support_passages = {by_id[i].passage_id for i in episode.support_ids}
        query_passages = {by_id[i].passage_id for i in episode.query_ids}
        assert not (support_passages & query_passages)


def test_grouping_raises_when_the_support_pool_is_left_too_small():
    # One giant passage (20 lines) and one tiny passage (1 line) -- each group must go entirely to
    # one side. Whichever order the two groups get shuffled into, the resulting support pool is
    # either 0 or 1 records, never the requested support_size=10 -- this must raise regardless of
    # which of the two possible group orderings the seeded RNG picks.
    lines = [record("w0", i, passage="big") for i in range(20)] + [
        record("w0", 20, passage="small")
    ]
    sampler = EpisodeSampler(
        lines, query_size=2, support_sizes=(10,), group_of=lambda r: r.passage_id
    )
    with pytest.raises(ValueError, match="support candidate"):
        sampler.sample("w0", draw_index=0)


def test_default_ungrouped_mode_does_not_enforce_passage_disjointness():
    # All 30 lines share one passage -- impossible to satisfy passage-disjoint grouping at all
    # (every draw would raise), but the default (group_of=None) must still work fine: real IAM
    # measurement showed 53.7% of train writers have exactly one passage, which is why ungrouped is
    # the default rather than opt-out (module docstring).
    lines = writer_lines("w0", 30, passages_of=30)
    sampler = EpisodeSampler(lines)
    episode = sampler.sample("w0", draw_index=0)
    assert episode.support_size in DEFAULT_SUPPORT_SIZES
    assert len(episode.query_ids) == DEFAULT_QUERY_SIZE


# ------------------------------------------------------------------ iter_writer_cycle


def test_iter_writer_cycle_yields_every_writer_exactly_once_per_pass():
    writers = [f"w{i:02d}" for i in range(20)]
    cycle = iter_writer_cycle(writers, seed=1337)
    first_pass = list(itertools.islice(cycle, 20))
    assert sorted(first_pass) == sorted(writers)


def test_iter_writer_cycle_is_deterministic_given_the_same_seed():
    writers = [f"w{i:02d}" for i in range(20)]
    a = list(itertools.islice(iter_writer_cycle(writers, seed=1337), 40))
    b = list(itertools.islice(iter_writer_cycle(writers, seed=1337), 40))
    assert a == b


def test_iter_writer_cycle_reshuffles_across_passes():
    writers = [f"w{i:02d}" for i in range(20)]
    values = list(itertools.islice(iter_writer_cycle(writers, seed=1337), 40))
    first_pass, second_pass = values[:20], values[20:]
    assert sorted(first_pass) == sorted(second_pass) == sorted(writers)
    assert first_pass != second_pass


def test_iter_writer_cycle_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        next(iter_writer_cycle([], seed=1337))
