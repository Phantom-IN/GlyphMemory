"""Writer-disjoint splits and support/query pools.

A leak here would invalidate every personalization result the project produces, so the assertions
are tested as carefully as the happy path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from glyphmemory.data import (
    ManifestRecord,
    SplitLeakError,
    SupportQuerySplit,
    WriterSplit,
    apply_writer_split,
    assert_writer_disjoint,
    make_support_query_split,
    make_writer_disjoint_split,
    split_statistics,
    writers_in,
)


def record(writer: str, index: int, *, passage: str | None = None, dataset: str = "synthetic"):
    return ManifestRecord(
        image=f"/data/{writer}-{index}.png",
        text=f"line {index} by {writer}",
        writer_id=writer,
        dataset=dataset,
        split="train",
        sample_id=f"{dataset}/{writer}-{index}",
        passage_id=passage,
    )


def corpus(n_writers: int = 10, lines_each: int = 5) -> list[ManifestRecord]:
    return [record(f"synthetic/w{w:02d}", i) for w in range(n_writers) for i in range(lines_each)]


# --------------------------------------------------------------------------- disjointness


def test_disjoint_split_passes():
    assert_writer_disjoint(WriterSplit(frozenset({"a"}), frozenset({"b"}), frozenset({"c"})))


@pytest.mark.parametrize(
    ("train", "val", "test", "leaked"),
    [
        ({"a", "b"}, {"b"}, {"c"}, "b"),
        ({"a"}, {"b"}, {"a"}, "a"),
        ({"a"}, {"b", "c"}, {"c"}, "c"),
    ],
)
def test_leak_raises_and_names_the_writer(train, val, test, leaked):
    split = WriterSplit(frozenset(train), frozenset(val), frozenset(test))
    with pytest.raises(SplitLeakError) as excinfo:
        assert_writer_disjoint(split)
    assert leaked in str(excinfo.value)
    assert "invalid" in str(excinfo.value).lower()


def test_leak_message_truncates_but_says_so():
    shared = {f"w{i}" for i in range(30)}
    with pytest.raises(SplitLeakError, match=r"\+10 more"):
        assert_writer_disjoint(WriterSplit(frozenset(shared), frozenset(shared), frozenset()))


# --------------------------------------------------------------------------- split building


def test_split_is_deterministic_under_seed():
    records = corpus()
    assert make_writer_disjoint_split(records, seed=42) == make_writer_disjoint_split(
        records, seed=42
    )


def test_different_seeds_give_different_splits():
    records = corpus(20)
    assert make_writer_disjoint_split(records, seed=1) != make_writer_disjoint_split(
        records, seed=2
    )


def test_split_independent_of_manifest_order():
    """Writers are sorted before shuffling, so manifest order must not matter."""
    records = corpus()
    assert make_writer_disjoint_split(records, seed=7) == make_writer_disjoint_split(
        list(reversed(records)), seed=7
    )


def test_split_never_divides_a_writer():
    records = corpus(12)
    split = make_writer_disjoint_split(records, seed=3)
    for writer in writers_in(records):
        memberships = [s for s in ("train", "val", "test") if writer in split.writers_for(s)]
        assert len(memberships) == 1


def test_every_writer_is_assigned():
    records = corpus(12)
    split = make_writer_disjoint_split(records, seed=3)
    assert split.all_writers == frozenset(writers_in(records))


def test_result_is_asserted_disjoint_on_construction():
    assert_writer_disjoint(make_writer_disjoint_split(corpus(15), seed=5))


def test_ratios_are_respected_approximately():
    split = make_writer_disjoint_split(corpus(100), ratios=(0.8, 0.1, 0.1), seed=11)
    sizes = split.sizes()
    assert sizes["train"] == 80
    assert sizes["val"] == 10
    assert sizes["test"] == 10


def test_small_writer_count_still_fills_every_split():
    """Three writers, three splits — each must get exactly one."""
    split = make_writer_disjoint_split(corpus(3), seed=1)
    assert sorted(split.sizes().values()) == [1, 1, 1]


def test_too_few_writers_raises_rather_than_reusing():
    with pytest.raises(ValueError, match="not a split"):
        make_writer_disjoint_split(corpus(2), seed=1)


@pytest.mark.parametrize("ratios", [(0.5, 0.5, 0.5), (0.5, 0.5), (-0.1, 0.6, 0.5)])
def test_invalid_ratios_rejected(ratios):
    with pytest.raises(ValueError):
        make_writer_disjoint_split(corpus(), ratios=ratios, seed=1)


def test_two_way_split_allowed():
    split = make_writer_disjoint_split(corpus(10), ratios=(0.8, 0.0, 0.2), seed=1)
    assert split.sizes()["val"] == 0
    assert split.sizes()["train"] and split.sizes()["test"]


# --------------------------------------------------------------------------- serialization


def test_writer_split_roundtrips_through_disk(tmp_path: Path):
    split = make_writer_disjoint_split(corpus(10), seed=9)
    assert WriterSplit.load(split.save(tmp_path / "split.json")) == split


def test_split_for_returns_none_for_unknown_writer():
    assert make_writer_disjoint_split(corpus(5), seed=1).split_for("nobody") is None


def test_writers_for_rejects_bad_split_name():
    with pytest.raises(ValueError, match="Unknown split"):
        make_writer_disjoint_split(corpus(5), seed=1).writers_for("validation")


# --------------------------------------------------------------------------- applying splits


def test_apply_writer_split_assigns_split_field():
    records = corpus(6)
    split = make_writer_disjoint_split(records, seed=2)
    for assigned in apply_writer_split(records, split):
        assert assigned.split == split.split_for(assigned.writer_id)


def test_unassigned_writers_are_excluded_loudly(caplog):
    records = corpus(6)
    split = make_writer_disjoint_split(records, seed=2)
    extra = [*records, record("synthetic/stranger", 0)]
    with caplog.at_level("WARNING", logger="glyphmemory.data.splits"):
        assigned = apply_writer_split(extra, split)
    assert len(assigned) == len(records)
    assert any("absent from the split" in r.message for r in caplog.records)


def test_apply_writer_split_preserves_optional_fields():
    """Records are rebuilt through to_dict(); only `split` may change."""
    original = ManifestRecord(
        image="/i.png",
        text="t",
        writer_id="w",
        dataset="cvl",
        split="train",
        sample_id="cvl/1",
        passage_id="p3",
        source_page="pg7",
        language="en",
        height=64,
        width=512,
    )
    split = WriterSplit(frozenset(), frozenset(), frozenset({"w"}))
    assigned = apply_writer_split([original], split)[0]

    assert assigned.split == "test"
    changed = {k: v for k, v in assigned.to_dict().items() if original.to_dict().get(k) != v}
    assert changed == {"split": "test"}


def test_split_statistics_reports_writers_and_lines():
    records = corpus(10, lines_each=4)
    stats = split_statistics(records, make_writer_disjoint_split(records, seed=4))
    assert sum(s["lines"] for s in stats.values()) == 40
    assert sum(s["writers"] for s in stats.values()) == 10


# --------------------------------------------------------------------------- support / query


def test_query_pool_is_reserved_and_support_is_the_remainder():
    records = corpus(4, lines_each=10)
    split = make_support_query_split(records, query_size=3, seed=1)
    for writer in split.writers:
        assert len(split.query_for(writer)) == 3
        assert len(split.support_for(writer)) == 7


def test_support_and_query_are_disjoint():
    split = make_support_query_split(corpus(5, lines_each=8), query_size=3, seed=1)
    split.assert_disjoint()
    for writer in split.writers:
        assert not set(split.support_for(writer)) & set(split.query_for(writer))


def test_overlapping_pools_raise():
    bad = SupportQuerySplit(support={"w": ("a", "b")}, query={"w": ("b",)})
    with pytest.raises(SplitLeakError) as excinfo:
        bad.assert_disjoint()
    assert "'b'" in str(excinfo.value)


def test_support_query_split_is_deterministic():
    records = corpus(5, lines_each=8)
    assert make_support_query_split(records, query_size=3, seed=1).as_dict() == (
        make_support_query_split(records, query_size=3, seed=1).as_dict()
    )


def test_query_pool_is_stable_when_support_size_changes():
    """CER@0 and CER@n must share one query set."""
    records = corpus(5, lines_each=12)
    a = make_support_query_split(records, query_size=4, seed=1)
    b = make_support_query_split(records, query_size=4, seed=1)
    assert a.query == b.query


def test_writers_with_too_few_lines_are_skipped_loudly(caplog):
    records = [*corpus(3, lines_each=6), record("synthetic/tiny", 0)]
    with caplog.at_level("WARNING", logger="glyphmemory.data.splits"):
        split = make_support_query_split(records, query_size=3, seed=1)
    assert "synthetic/tiny" not in split.writers
    assert any("too few lines" in r.message for r in caplog.records)


def test_writers_supporting_reports_usable_writer_count():
    """How many writers a reported CER@n actually rests on."""
    records = [record(f"synthetic/w{w}", i) for w in range(3) for i in range(w + 5)]
    split = make_support_query_split(records, query_size=2, seed=1)
    assert len(split.writers_supporting(1)) == 3
    assert len(split.writers_supporting(10)) == 0


def test_missing_sample_id_rejected():
    anonymous = [ManifestRecord("/x.png", "t", "w", "synthetic", "test") for _ in range(4)]
    with pytest.raises(ValueError, match="no sample_id"):
        make_support_query_split(anonymous, query_size=1, seed=1)


def test_query_size_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        make_support_query_split(corpus(3), query_size=0, seed=1)


def test_support_query_split_roundtrips_through_disk(tmp_path: Path):
    split = make_support_query_split(corpus(4, lines_each=8), query_size=3, seed=1)
    assert SupportQuerySplit.load(split.save(tmp_path / "sq.json")).as_dict() == split.as_dict()


# --------------------------------------------------------------------------- group-disjointness


def test_group_disjoint_pools_share_no_passage():
    """CVL writers copy shared passages; support and query must not overlap on text."""
    records = [
        record(f"synthetic/w{w}", i, passage=f"p{i % 4}") for w in range(3) for i in range(12)
    ]
    split = make_support_query_split(records, query_size=3, seed=1, group_of=lambda r: r.passage_id)
    by_id = {r.sample_id: r for r in records}
    for writer in split.writers:
        support_passages = {by_id[s].passage_id for s in split.support_for(writer)}
        query_passages = {by_id[q].passage_id for q in split.query_for(writer)}
        assert not support_passages & query_passages


def test_group_mode_still_disjoint_by_sample():
    records = [
        record(f"synthetic/w{w}", i, passage=f"p{i % 3}") for w in range(2) for i in range(9)
    ]
    make_support_query_split(
        records, query_size=3, seed=1, group_of=lambda r: r.passage_id
    ).assert_disjoint()


def test_single_group_writer_is_skipped_not_leaked(caplog):
    """One passage cannot yield disjoint support and query — skip, never overlap."""
    records = [record("synthetic/w0", i, passage="only") for i in range(6)]
    with caplog.at_level("WARNING", logger="glyphmemory.data.splits"):
        split = make_support_query_split(
            records, query_size=2, seed=1, group_of=lambda r: r.passage_id
        )
    assert not split.writers
