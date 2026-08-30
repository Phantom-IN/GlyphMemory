"""Per-writer CER distribution and the heavy-tail statistic."""

from __future__ import annotations

from glyphmemory.evaluation.per_writer import per_writer_distribution


def test_perfect_writer_has_zero_cer():
    records = [("w1", "hello", "hello"), ("w1", "world", "world")]
    dist = per_writer_distribution(records)
    assert dist.n_writers == 1
    assert dist.writers[0].cer == 0.0
    assert dist.writers[0].lines == 2


def test_writer_cer_is_micro_averaged_within_writer():
    # 1 error / 5 chars on line 1, 0 errors / 5 chars on line 2 -> pooled 1/10, not mean(0.2, 0).
    records = [("w1", "hello", "hallo"), ("w1", "world", "world")]
    dist = per_writer_distribution(records)
    assert dist.writers[0].cer == 1 / 10


def test_median_and_worst_decile_over_ten_writers():
    # 9 perfect writers, 1 writer with every character wrong.
    records = []
    for i in range(9):
        records.append((f"w{i}", "cat", "cat"))
    records.append(("w_bad", "cat", "xyz"))
    dist = per_writer_distribution(records)

    assert dist.n_writers == 10
    assert dist.median_cer == 0.0
    assert dist.worst_decile_writers == ("w_bad",)
    assert dist.worst_decile_cer == 1.0
    # ratio against a zero median is undefined, not infinite.
    assert dist.worst_decile_ratio is None
    assert dist.passes_tail_condition is None


def test_worst_decile_ratio_when_median_is_nonzero():
    # 10 writers: one worst decile writer with CER 1.0, the rest with CER 0.2 (median 0.2).
    records = [("w_bad", "cat", "xyz")]
    for i in range(9):
        records.append((f"w{i}", "cats", "cots"))  # 1/4 error rate... adjust below
    dist = per_writer_distribution(records)
    assert dist.n_writers == 10
    assert dist.worst_decile_writers == ("w_bad",)
    assert dist.worst_decile_cer == 1.0
    assert dist.median_cer == 0.25
    assert dist.worst_decile_ratio == 4.0
    assert dist.passes_tail_condition is True


def test_lines_per_writer_is_reported():
    records = [("w1", "a", "a"), ("w1", "b", "b"), ("w2", "c", "c")]
    dist = per_writer_distribution(records)
    by_id = {w.writer_id: w.lines for w in dist.writers}
    assert by_id == {"w1": 2, "w2": 1}


def test_empty_input_has_no_median_or_tail():
    dist = per_writer_distribution([])
    assert dist.n_writers == 0
    assert dist.median_cer is None
    assert dist.worst_decile_ratio is None
