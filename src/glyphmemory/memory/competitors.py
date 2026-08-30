"""The base head's top-2 competitor graph — the hard-negative source that actually transfers.

    slot accuracy         train 0.9976   train+augmentation 0.9906   val 0.9268
    train eligible pool   6 HELP / 347 DAMAGE  (HELP-rate 0.0018)

``gm-base-v0`` has memorised its own training writers, so on that population there is almost no
damage structure to mine, and what there is does not transfer:

    per-pair HELP-rate,     train -> val   Spearman rho 0.362
    top-2 competitor graph, train -> val   Spearman rho 0.882, 97.2% of validation mass

**Which characters compete is a stable property of the recognizer. Which competitor is right is
not.** This module mines the former. It deliberately does not weight by outcome, and it deliberately
does not read validation writers, whose confusion pairs are the evaluation cohort's own labels.

Blank competitors are excluded. A blank has no glyph crop, and a slot whose runner-up is blank is
ineligible at inference anyway — memory's candidate is always a real character.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from glyphmemory.ctc.tokenizer import BLANK_TOKEN
from glyphmemory.evaluation.emitted_spans import EmittedOccurrence

#: Minimum observations for a pair to be usable. Below this the ordering is sampling noise.
MIN_PAIR_OBSERVATIONS = 20

#: Hard negatives offered per character.
NEGATIVES_PER_CHARACTER = 8


class CompetitorGraph:
    """``character -> its most frequent top-2 competitors``, mined from emitted slots.

    Built by :func:`build_competitor_graph` over training writers only. Serialization is plain JSON
    so the mined graph is inspectable rather than a pickled blob.
    """

    def __init__(
        self,
        counts: Mapping[str, Mapping[str, int]],
        *,
        min_observations: int = MIN_PAIR_OBSERVATIONS,
        negatives_per_character: int = NEGATIVES_PER_CHARACTER,
    ) -> None:
        self.min_observations = min_observations
        self.negatives_per_character = negatives_per_character
        self.counts = {
            emitted: dict(competitors) for emitted, competitors in counts.items() if competitors
        }

        # Global fallback ordering, for characters with too few competitors of their own.
        totals: Counter[str] = Counter()
        for competitors in self.counts.values():
            totals.update(competitors)
        self.global_order = [character for character, _ in totals.most_common()]

        self._negatives: dict[str, tuple[str, ...]] = {}
        for emitted, competitors in self.counts.items():
            kept = sorted(
                (
                    (competitor, count)
                    for competitor, count in competitors.items()
                    if count >= min_observations
                ),
                key=lambda item: (-item[1], item[0]),
            )
            chosen = [competitor for competitor, _ in kept[:negatives_per_character]]
            for character in self.global_order:
                if len(chosen) >= negatives_per_character:
                    break
                if character != emitted and character not in chosen:
                    chosen.append(character)
            self._negatives[emitted] = tuple(chosen)

    def negatives(self, character: str) -> tuple[str, ...]:
        """Hard negatives for ``character``.

        Falls back to the globally most frequent competitors for a character never seen as an
        emission — rare, but a missing entry must not silently produce an empty candidate set, which
        would make :func:`~glyphmemory.memory.verifier.char_loss` degenerate.
        """
        if character in self._negatives:
            return self._negatives[character]
        return tuple(c for c in self.global_order if c != character)[
            : self.negatives_per_character
        ]

    @property
    def pair_count(self) -> int:
        """Distinct (emitted, competitor) pairs at or above ``min_observations``."""
        return sum(
            1
            for competitors in self.counts.values()
            for count in competitors.values()
            if count >= self.min_observations
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "min_observations": self.min_observations,
            "negatives_per_character": self.negatives_per_character,
            "counts": self.counts,
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))

    @classmethod
    def read(cls, path: str | Path) -> CompetitorGraph:
        payload = json.loads(Path(path).read_text())
        return cls(
            payload["counts"],
            min_observations=payload["min_observations"],
            negatives_per_character=payload["negatives_per_character"],
        )


def accumulate_occurrences(
    counts: dict[str, dict[str, int]],
    occurrences: Iterable[EmittedOccurrence],
) -> dict[str, dict[str, int]]:
    """Add one line's emitted slots to a running competitor tally, in place.

    Every slot contributes, correct or not — that is the whole point. The outcome is not read here
    and no reference text is required, which is why this works on a population where the recognizer
    almost never errs.
    """
    for occurrence in occurrences:
        if len(occurrence.candidates) < 2:
            continue
        runner_up = occurrence.candidates[1]
        if runner_up == BLANK_TOKEN or occurrence.character == BLANK_TOKEN:
            continue
        if runner_up == occurrence.character:
            continue
        competitors = counts.setdefault(occurrence.character, {})
        competitors[runner_up] = competitors.get(runner_up, 0) + 1
    return counts


def build_competitor_graph(
    lines: Iterable[Sequence[EmittedOccurrence]],
    *,
    min_observations: int = MIN_PAIR_OBSERVATIONS,
    negatives_per_character: int = NEGATIVES_PER_CHARACTER,
) -> CompetitorGraph:
    """Mine a :class:`CompetitorGraph` from emitted occurrences, line by line."""
    counts: dict[str, dict[str, int]] = {}
    for occurrences in lines:
        accumulate_occurrences(counts, occurrences)
    return CompetitorGraph(
        counts,
        min_observations=min_observations,
        negatives_per_character=negatives_per_character,
    )
