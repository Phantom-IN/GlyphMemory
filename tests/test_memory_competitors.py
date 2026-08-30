"""The design point this module encodes: outcomes are never read. So the graph must be buildable
from a population where the recognizer is never wrong — which is exactly what
``test_builds_from_slots_where_the_base_is_always_correct`` asserts.
"""

from __future__ import annotations

import json

from glyphmemory.ctc.tokenizer import BLANK_TOKEN
from glyphmemory.evaluation.emitted_spans import EmittedOccurrence
from glyphmemory.memory.competitors import (
    CompetitorGraph,
    accumulate_occurrences,
    build_competitor_graph,
)


def slot(character: str, runner_up: str) -> EmittedOccurrence:
    return EmittedOccurrence(
        index=0,
        character=character,
        start=0,
        end=0,
        peak=0,
        confidence=0.9,
        margin=0.4,
        entropy=0.2,
        candidates=(character, runner_up),
    )


def line(*pairs: tuple[str, str]) -> list[EmittedOccurrence]:
    return [slot(a, b) for a, b in pairs]


class TestBuilding:
    def test_builds_from_slots_where_the_base_is_always_correct(self) -> None:
        """No outcome, no reference text, no error is needed — the whole reason this works."""
        graph = build_competitor_graph(
            [line(("o", "a")) for _ in range(30)], min_observations=20
        )
        assert graph.negatives("o")[0] == "a"

    def test_competitors_are_ordered_by_frequency(self) -> None:
        lines = [line(("e", "a")) for _ in range(30)]
        lines += [line(("e", "o")) for _ in range(25)]
        lines += [line(("e", "c")) for _ in range(21)]
        graph = build_competitor_graph(lines, min_observations=20, negatives_per_character=3)
        assert graph.negatives("e") == ("a", "o", "c")

    def test_pairs_below_the_observation_floor_are_not_promoted_by_it(self) -> None:
        """Below the floor the ordering is sampling noise, so it must not decide a negative."""
        lines = [line(("n", "m")) for _ in range(30)]
        lines += [line(("n", "q")) for _ in range(3)]
        graph = build_competitor_graph(lines, min_observations=20, negatives_per_character=1)
        assert graph.negatives("n") == ("m",)

    def test_blank_runner_ups_are_excluded(self) -> None:
        """A blank has no glyph crop, and such a slot is ineligible at inference anyway."""
        graph = build_competitor_graph(
            [line(("e", BLANK_TOKEN)) for _ in range(50)], min_observations=1
        )
        assert BLANK_TOKEN not in graph.negatives("e")

    def test_self_competition_is_excluded(self) -> None:
        graph = build_competitor_graph([line(("a", "a")) for _ in range(50)], min_observations=1)
        assert "a" not in graph.counts.get("a", {})

    def test_single_candidate_slots_are_skipped(self) -> None:
        occurrence = EmittedOccurrence(
            index=0, character="a", start=0, end=0, peak=0,
            confidence=1.0, margin=1.0, entropy=0.0, candidates=("a",),
        )
        assert build_competitor_graph([[occurrence]]).counts == {}

    def test_empty_input(self) -> None:
        graph = build_competitor_graph([])
        assert graph.counts == {}
        assert graph.pair_count == 0


class TestNegatives:
    def test_padded_up_to_the_requested_count(self) -> None:
        """A short candidate set makes char_loss degenerate, so padding is required."""
        lines = [line(("z", "s")) for _ in range(30)]
        lines += [line(("e", "a")) for _ in range(60)]
        lines += [line(("o", "c")) for _ in range(40)]
        graph = build_competitor_graph(lines, min_observations=20, negatives_per_character=3)
        assert len(graph.negatives("z")) == 3
        assert graph.negatives("z")[0] == "s"

    def test_never_returns_the_character_itself(self) -> None:
        lines = [line(("a", "o")) for _ in range(30)] + [line(("o", "a")) for _ in range(30)]
        graph = build_competitor_graph(lines, min_observations=20, negatives_per_character=8)
        assert "a" not in graph.negatives("a")

    def test_unknown_character_falls_back_globally_rather_than_returning_empty(self) -> None:
        lines = [line(("e", "a")) for _ in range(30)]
        graph = build_competitor_graph(lines, min_observations=20, negatives_per_character=2)
        assert graph.negatives("Q")
        assert "Q" not in graph.negatives("Q")

    def test_negatives_are_deterministic(self) -> None:
        lines = [line(("e", "a")) for _ in range(30)] + [line(("e", "o")) for _ in range(30)]
        first = build_competitor_graph(lines, min_observations=20).negatives("e")
        second = build_competitor_graph(lines, min_observations=20).negatives("e")
        assert first == second

    def test_ties_break_deterministically_by_character(self) -> None:
        lines = [line(("e", "a")) for _ in range(30)] + [line(("e", "o")) for _ in range(30)]
        assert build_competitor_graph(lines, min_observations=20).negatives("e")[:2] == ("a", "o")


class TestAccumulate:
    def test_accumulates_in_place_across_lines(self) -> None:
        counts: dict[str, dict[str, int]] = {}
        accumulate_occurrences(counts, line(("a", "o"), ("b", "l")))
        accumulate_occurrences(counts, line(("a", "o")))
        assert counts["a"]["o"] == 2
        assert counts["b"]["l"] == 1


class TestRoundTrip:
    def test_write_and_read_preserve_negatives(self, tmp_path) -> None:
        lines = [line(("e", "a")) for _ in range(30)] + [line(("o", "a")) for _ in range(25)]
        graph = build_competitor_graph(lines, min_observations=20, negatives_per_character=4)
        path = tmp_path / "competitors.json"
        graph.write(path)
        restored = CompetitorGraph.read(path)
        assert restored.negatives("e") == graph.negatives("e")
        assert restored.min_observations == graph.min_observations

    def test_serialization_is_plain_inspectable_json(self, tmp_path) -> None:
        graph = build_competitor_graph([line(("e", "a")) for _ in range(30)])
        path = tmp_path / "competitors.json"
        graph.write(path)
        assert json.loads(path.read_text())["counts"]["e"]["a"] == 30


class TestPairCount:
    def test_counts_only_pairs_at_or_above_the_floor(self) -> None:
        lines = [line(("e", "a")) for _ in range(30)] + [line(("e", "q")) for _ in range(2)]
        assert build_competitor_graph(lines, min_observations=20).pair_count == 1


def test_outcomes_are_never_consulted() -> None:
    """Guard on the module's whole premise.

    Two graphs from identical slots.
    """
    import inspect

    for function in (accumulate_occurrences, build_competitor_graph):
        parameters = set(inspect.signature(function).parameters)
        assert not parameters & {"reference", "outcomes", "truth", "labels", "help", "damage"}
