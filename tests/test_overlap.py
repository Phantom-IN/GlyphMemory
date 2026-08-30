"""Text-overlap measurement tests.

Two properties matter more than the arithmetic.

**Constructed overlapping and disjoint sets must produce 100% and 0%.** A measurement that cannot
reach either extreme cannot be trusted in between.

**Support/query overlap must be measured per writer.** Pooling across writers reports ~100% passage
overlap for a split that leaks nothing, because writer A's query passage is writer B's support
passage. That defect was real and is what this file's per-writer tests pin down.
"""

from __future__ import annotations

from glyphmemory.ctc.normalization import IDENTITY, NFC_V1
from glyphmemory.data.manifest import ManifestRecord
from glyphmemory.data.overlap import (
    split_overlaps,
    support_query_overlap,
    text_overlap,
)
from glyphmemory.data.splits import SupportQuerySplit


def record(
    sample_id: str,
    text: str,
    *,
    writer: str = "w1",
    split: str = "train",
    passage: str | None = None,
) -> ManifestRecord:
    return ManifestRecord(
        image=f"/tmp/{sample_id}.png",
        text=text,
        writer_id=writer,
        dataset="test",
        split=split,
        sample_id=sample_id,
        passage_id=passage,
    )


class TestExtremes:
    def test_identical_sets_are_fully_overlapping(self) -> None:
        records = [record("a", "the quick brown fox"), record("b", "jumps over the dog")]
        report = text_overlap(records, records)
        assert report.lines.instance_coverage_of_b == 1.0
        assert report.lines.jaccard == 1.0
        assert report.words.instance_coverage_of_b == 1.0
        assert report.ngrams[3].instance_coverage_of_b == 1.0

    def test_disjoint_sets_have_no_overlap(self) -> None:
        a = [record("a", "alpha beta gamma delta")]
        b = [record("b", "epsilon zeta eta theta")]
        report = text_overlap(a, b)
        assert report.lines.shared == 0
        assert report.words.shared == 0
        assert report.ngrams[3].instance_coverage_of_b == 0.0
        assert report.lines.jaccard == 0.0

    def test_partial_overlap_counts_instances_not_just_types(self) -> None:
        """A line repeated by many writers must weigh once per occurrence."""
        a = [record("a", "shared line")]
        b = [record("b1", "shared line"), record("b2", "shared line"), record("b3", "other line")]
        report = text_overlap(a, b)
        assert report.lines.unique_b == 2
        assert report.lines.shared == 1
        assert report.lines.unique_coverage_of_b == 0.5
        assert report.lines.instances_b == 3
        assert report.lines.instance_coverage_of_b == 2 / 3


class TestGranularity:
    def test_word_overlap_without_line_overlap(self) -> None:
        """Different sentences from a shared vocabulary: 0% line, 100% word."""
        a = [record("a", "the cat sat")]
        b = [record("b", "sat the cat")]
        report = text_overlap(a, b)
        assert report.lines.shared == 0
        assert report.words.instance_coverage_of_b == 1.0

    def test_ngrams_shorter_than_the_order_are_skipped(self) -> None:
        report = text_overlap([record("a", "one two")], [record("b", "one two")])
        assert report.ngrams[3].instances_b == 0
        assert report.ngrams[3].instance_coverage_of_b == 0.0
        assert report.ngrams[5].instances_b == 0

    def test_passages_reported_only_when_both_sides_carry_them(self) -> None:
        with_passage = [record("a", "text", passage="p1")]
        without = [record("b", "text")]
        assert text_overlap(with_passage, with_passage).passages is not None
        assert text_overlap(with_passage, without).passages is None
        assert text_overlap(without, without).passages is None


class TestNormalizationIsRecorded:
    def test_policy_name_travels_with_the_figure(self) -> None:
        report = text_overlap([record("a", "x")], [record("b", "x")], policy=IDENTITY)
        assert report.normalization == "identity"
        assert report.as_dict()["normalization"] == "identity"

    def test_normalization_changes_the_answer(self) -> None:
        """Whitespace collapsing is why the policy must be reported, not assumed."""
        a = [record("a", "hello   world")]
        b = [record("b", "hello world")]
        assert text_overlap(a, b, policy=IDENTITY).lines.shared == 0
        assert text_overlap(a, b, policy=NFC_V1).lines.shared == 1

    def test_extra_normalizer_is_named(self) -> None:
        a = [record("a", "Hello")]
        b = [record("b", "hello")]
        report = text_overlap(a, b, extra_normalizer=str.lower)
        assert report.lines.shared == 1
        assert report.normalization == "nfc_v1+lower"


class TestSplitOverlaps:
    def test_reports_train_to_val_and_train_to_test(self) -> None:
        records = [
            record("t1", "train text", split="train"),
            record("v1", "train text", split="val"),
            record("s1", "unseen text", split="test"),
        ]
        reports = split_overlaps(records)
        assert [(r.label_a, r.label_b) for r in reports] == [
            ("train", "val"),
            ("train", "test"),
        ]
        assert reports[0].lines.instance_coverage_of_b == 1.0
        assert reports[1].lines.instance_coverage_of_b == 0.0

    def test_missing_split_is_skipped_not_reported_as_clean(self) -> None:
        """0% overlap against an empty split looks like a result; it is a missing one."""
        records = [record("t1", "text", split="train")]
        assert split_overlaps(records) == []


class TestPerWriterSupportQuery:
    def build(self) -> tuple[list[ManifestRecord], SupportQuerySplit]:
        """Two writers copying the same two passages, split passage-disjointly.

        Pooled across writers this looks like total passage overlap. Per writer it leaks nothing —
        which is the whole point.
        """
        records = [
            record("w1-s", "alpha beta", writer="w1", passage="p1"),
            record("w1-q", "gamma delta", writer="w1", passage="p2"),
            record("w2-s", "gamma delta", writer="w2", passage="p2"),
            record("w2-q", "alpha beta", writer="w2", passage="p1"),
        ]
        split = SupportQuerySplit(
            support={"w1": ("w1-s",), "w2": ("w2-s",)},
            query={"w1": ("w1-q",), "w2": ("w2-q",)},
        )
        return records, split

    def test_passage_disjoint_split_shows_no_leakage(self) -> None:
        records, split = self.build()
        report = support_query_overlap(records, split)
        assert report.n_writers == 2
        assert report.granularities["exact line"].fraction_shared == 0.0
        assert report.granularities["passage"].fraction_shared == 0.0
        assert report.granularities["passage"].writers_affected == 0

    def test_pooling_across_writers_would_have_hidden_it(self) -> None:
        """The defect this measurement replaced: pooled, the same split reads as 100%."""
        records, _ = self.build()
        support = [r for r in records if r.sample_id in {"w1-s", "w2-s"}]
        query = [r for r in records if r.sample_id in {"w1-q", "w2-q"}]
        pooled = text_overlap(support, query)
        assert pooled.passages is not None
        assert pooled.passages.instance_coverage_of_b == 1.0
        assert pooled.lines.instance_coverage_of_b == 1.0

    def test_random_split_leakage_is_detected(self) -> None:
        records = [
            record("w1-s", "same text", writer="w1", passage="p1"),
            record("w1-q", "same text", writer="w1", passage="p1"),
        ]
        split = SupportQuerySplit(support={"w1": ("w1-s",)}, query={"w1": ("w1-q",)})
        report = support_query_overlap(records, split)
        assert report.granularities["exact line"].fraction_shared == 1.0
        assert report.granularities["exact line"].writers_affected == 1
        assert report.granularities["passage"].fraction_shared == 1.0

    def test_counts_are_summed_over_writers(self) -> None:
        records = [
            record("w1-s", "a b", writer="w1"),
            record("w1-q", "a c", writer="w1"),
            record("w2-s", "x y", writer="w2"),
            record("w2-q", "z w", writer="w2"),
        ]
        split = SupportQuerySplit(
            support={"w1": ("w1-s",), "w2": ("w2-s",)},
            query={"w1": ("w1-q",), "w2": ("w2-q",)},
        )
        report = support_query_overlap(records, split)
        words = report.granularities["word"]
        assert words.instances_query == 4
        assert words.instances_shared == 1
        assert words.writers_affected == 1
        assert report.n_support == 2
        assert report.n_query == 2

    def test_report_is_json_shaped(self) -> None:
        records, split = self.build()
        payload = support_query_overlap(records, split).as_dict()
        assert payload["measurement"] == "per-writer support -> query"
        assert payload["normalization"] == "nfc_v1"
        assert "exact line" in payload["granularities"]
        assert "per writer" in support_query_overlap(records, split).format()
