"""Writer histogram and passage distribution tests."""

from __future__ import annotations

import pytest

from glyphmemory.data.manifest import ManifestRecord
from glyphmemory.data.stats import (
    PassageDistribution,
    WriterHistogram,
    passage_distribution,
    writer_histogram,
)


def record(writer: str, index: int, *, passage: str | None = None) -> ManifestRecord:
    return ManifestRecord(
        image=f"/tmp/{writer}-{index}.png",
        text="text",
        writer_id=writer,
        dataset="test",
        split="test",
        sample_id=f"test/{writer}-{index}",
        passage_id=passage,
    )


def corpus(counts: dict[str, int]) -> list[ManifestRecord]:
    return [record(writer, i) for writer, n in counts.items() for i in range(n)]


class TestWriterHistogram:
    def test_counts_lines_per_writer(self) -> None:
        histogram = writer_histogram(corpus({"a": 3, "b": 1, "c": 7}))
        assert histogram.lines_per_writer == {"a": 3, "b": 1, "c": 7}
        assert histogram.n_writers == 3
        assert histogram.total_lines == 11

    def test_summary_statistics(self) -> None:
        histogram = writer_histogram(corpus({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}))
        assert histogram.minimum == 1
        assert histogram.maximum == 5
        assert histogram.median == 3
        assert histogram.mean == 3.0
        assert histogram.quartiles() == (2.0, 4.0)
        assert histogram.iqr == 2.0

    def test_thresholds(self) -> None:
        histogram = writer_histogram(corpus({"a": 1, "b": 5, "c": 10, "d": 25}))
        assert histogram.writers_with_at_least(1) == 4
        assert histogram.writers_with_at_least(5) == 3
        assert histogram.writers_with_at_least(10) == 2
        assert histogram.writers_with_at_least(20) == 1
        assert histogram.writers_with_at_least(26) == 0

    def test_support_capacity_reserves_the_query_pool_first(self) -> None:
        """A writer with 12 lines cannot do CER@10 while holding back 5 queries.

        This is the distinction that makes a reported ``CER@10`` honest: the naive count says one
        writer qualifies, the correct count says none does.
        """
        histogram = writer_histogram(corpus({"a": 12}))
        assert histogram.writers_with_at_least(10) == 1
        assert histogram.support_capacity(query_size=5)[10] == 0
        assert histogram.support_capacity(query_size=2)[10] == 1

    def test_empty_corpus_does_not_divide_by_zero(self) -> None:
        histogram = writer_histogram([])
        assert histogram.n_writers == 0
        assert histogram.mean == 0.0
        assert histogram.median == 0.0
        assert histogram.quartiles() == (0.0, 0.0)

    def test_single_writer_quartiles(self) -> None:
        """``statistics.quantiles`` raises below two points; the value is both quartiles."""
        histogram = writer_histogram(corpus({"a": 7}))
        assert histogram.quartiles() == (7.0, 7.0)
        assert histogram.iqr == 0.0

    def test_as_dict_is_json_shaped(self) -> None:
        payload = writer_histogram(corpus({"a": 2, "b": 4})).as_dict()
        assert payload["writers"] == 2
        assert payload["lines"] == 6
        assert payload["writers_with_at_least"]["3"] == 1

    def test_format_mentions_every_threshold(self) -> None:
        histogram = WriterHistogram(lines_per_writer={"a": 9}, thresholds=(1, 5))
        text = histogram.format(query_size=3)
        assert "n>=1" in text and "n>=5" in text
        assert "CER@5" in text


class TestPassageDistribution:
    def test_counts_lines_writers_and_passages(self) -> None:
        records = [
            record("a", 0, passage="p1"),
            record("a", 1, passage="p1"),
            record("a", 2, passage="p2"),
            record("b", 0, passage="p2"),
        ]
        distribution = passage_distribution(records)
        assert distribution.lines_per_passage == {"p1": 2, "p2": 2}
        assert distribution.writers_per_passage == {"p1": 1, "p2": 2}
        assert distribution.passages_per_writer == {"a": 2, "b": 1}
        assert distribution.n_passages == 2

    def test_unlabelled_lines_are_counted_not_bucketed(self) -> None:
        """A dataset without passage metadata must report that, not fake one big passage."""
        distribution = passage_distribution([record("a", 0), record("a", 1, passage="p1")])
        assert distribution.unlabelled_lines == 1
        assert distribution.lines_per_passage == {"p1": 1}

    def test_writers_with_at_least_passages_bounds_passage_disjoint_splitting(self) -> None:
        records = [
            record("a", 0, passage="p1"),
            record("a", 1, passage="p2"),
            record("b", 0, passage="p1"),
        ]
        distribution = passage_distribution(records)
        assert distribution.writers_with_at_least_passages(2) == 1
        assert distribution.as_dict()["writers_with_at_least_2_passages"] == 1

    def test_empty(self) -> None:
        distribution = passage_distribution([])
        assert distribution.n_passages == 0
        assert distribution.as_dict()["passages_per_writer"]["median"] == 0
        assert "passages" in distribution.format()

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_format_lists_every_passage(self, n: int) -> None:
        distribution = PassageDistribution(
            lines_per_passage={f"p{i}": i + 1 for i in range(n)},
            writers_per_passage={f"p{i}": 1 for i in range(n)},
            passages_per_writer={"a": n},
        )
        text = distribution.format()
        assert all(f"p{i}" in text for i in range(n))


class TestAgainstSyntheticCorpus:
    def test_histogram_matches_the_generator(self, synthetic_corpus) -> None:
        histogram = writer_histogram(synthetic_corpus.records)
        assert histogram.n_writers == len(synthetic_corpus.writers)
        assert set(histogram.lines_per_writer.values()) == {
            len(synthetic_corpus.records_for(synthetic_corpus.writers[0]))
        }

    def test_passages_are_labelled(self, synthetic_corpus) -> None:
        distribution = passage_distribution(synthetic_corpus.records)
        assert distribution.unlabelled_lines == 0
        assert distribution.n_passages == synthetic_corpus.adapter.n_passages
