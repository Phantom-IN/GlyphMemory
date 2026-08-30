"""The statistic never computed.

The point of these tests is the distinction the whole of wave 3 rests on: two classifiers with
*identical* aggregate accuracy can have zero headroom (nested) or large headroom (complementary),
and only the joint outcome tells them apart. If `complementarity` got that wrong, every number in
phases 38-42 would be wrong in the same direction and nothing downstream would catch it.
"""

from __future__ import annotations

import pytest
import torch

from glyphmemory.probes import Complementarity, complementarity, readout_scale


def test_nested_and_complementary_are_distinguished_at_equal_accuracy():
    """The core case. Both memories are right on 3 of 5 frames; the base head is right on 3 of 5.
    Aggregate accuracy is identical and cannot tell these apart — the joint outcome can.
    """
    base = [True, True, True, False, False]

    nested = complementarity(base, [True, True, True, False, False])  # right only where base is
    # right on a frame base gets wrong
    complementary = complementarity(base, [True, True, False, True, False])

    assert nested.memory_accuracy == complementary.memory_accuracy == 0.6
    assert nested.base_accuracy == complementary.base_accuracy == 0.6

    assert nested.rescues == 0 and nested.headroom == 0.0
    assert nested.oracle_accuracy == nested.base_accuracy  # no readout can recover anything

    assert complementary.rescues == 1
    assert complementary.headroom == pytest.approx(0.2)
    assert complementary.oracle_accuracy == pytest.approx(0.8)


def test_the_four_counts_partition_every_frame():
    base = [True, False, True, False, True, True]
    memory = [True, True, False, False, True, False]

    result = complementarity(base, memory)

    assert result.n_frames == len(base)
    assert result.both_correct + result.rescues + result.damages + result.both_wrong == len(base)
    assert result.base_correct == sum(base)
    assert result.memory_correct == sum(memory)


def test_rescue_and_damage_rates_use_the_right_denominators():
    # base wrong on 4, memory rescues 3 of them; base right on 6, memory damages 2.
    base = [False] * 4 + [True] * 6
    memory = [*[True, True, True, False], *[True, True, True, True, False, False]]

    result = complementarity(base, memory)

    assert result.rescues == 3
    assert result.damages == 2
    assert result.rescue_rate == pytest.approx(3 / 4)   # of base's errors
    assert result.damage_rate == pytest.approx(2 / 6)   # of base's correct frames
    assert result.rescue_damage_ratio == pytest.approx(1.5)


def test_ratio_below_one_means_indiscriminate_trust_loses_frames():
    """The finding that shaped wave 3: memory can be genuinely complementary and still be a net loss
    if applied everywhere.
    """
    base = [False, True, True, True]
    memory = [True, False, False, True]

    result = complementarity(base, memory)

    assert result.rescues == 1 and result.damages == 2
    assert result.rescue_damage_ratio == pytest.approx(0.5)
    assert result.memory_accuracy < result.base_accuracy
    # headroom exists even though trusting memory everywhere would lose frames
    assert result.oracle_accuracy > result.base_accuracy


def test_no_damages_is_infinite_not_clamped_and_no_signal_is_none():
    assert complementarity([False], [True]).rescue_damage_ratio == float("inf")
    assert complementarity([True], [True]).rescue_damage_ratio is None


def test_empty_input_reports_none_rather_than_dividing_by_zero():
    result = complementarity([], [])

    assert result.n_frames == 0
    for value in (result.base_accuracy, result.oracle_accuracy, result.headroom,
                  result.rescue_rate, result.damage_rate):
        assert value is None


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="same frames in the same order"):
        complementarity([True, False], [True])


def test_as_dict_is_serializable_and_complete():
    payload = Complementarity(both_correct=1, rescues=2, damages=3, both_wrong=4).as_dict()

    assert payload["n_frames"] == 10
    assert payload["rescues"] == 2 and payload["damages"] == 3
    assert set(payload) >= {"base_accuracy", "oracle_accuracy", "headroom", "rescue_damage_ratio"}


# ------------------------------------------------------------------ readout scale


def test_readout_scale_reports_the_margin_to_correction_ratio():
    """A wide-margin base head with bounded scores is structurally inert — the number measured on
    one writer and this makes reproducible.
    """
    logits = torch.tensor([[10.0, 0.0, 0.0], [12.0, 2.0, 0.0]])  # top-2 margins of 10
    scores = torch.full((2, 3), 0.5)

    scale = readout_scale(logits, scores, alpha=0.5)

    assert scale.margin_median == pytest.approx(10.0)
    assert scale.max_correction == pytest.approx(0.25)     # alpha * max|score|
    assert scale.margin_to_correction_ratio == pytest.approx(40.0)


def test_readout_scale_ratio_falls_as_alpha_rises():
    logits = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    scores = torch.full((2, 3), 1.0)

    weak = readout_scale(logits, scores, alpha=0.5)
    strong = readout_scale(logits, scores, alpha=10.0)

    assert weak.margin_to_correction_ratio == pytest.approx(20.0)
    assert strong.margin_to_correction_ratio == pytest.approx(1.0)  # correction ~= the decision


def test_readout_scale_refuses_mismatched_or_degenerate_shapes():
    with pytest.raises(ValueError, match="differ"):
        readout_scale(torch.zeros(2, 3), torch.zeros(2, 4), alpha=0.5)
    with pytest.raises(ValueError, match=r"\[T, V\] with V >= 2"):
        readout_scale(torch.zeros(2, 1), torch.zeros(2, 1), alpha=0.5)
