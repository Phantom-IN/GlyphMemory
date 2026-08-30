"""Deterministic text sampling for synthetic writers.

``coverage`` mode exists because natural text will not reliably supply ``q``, ``x``, ``z`` or ``7``,
and the alignment and prototype work downstream needs every character present for every writer.
Coverage is guaranteed **by construction** — required characters are emitted first — not by sampling
and hoping.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

#: A small embedded vocabulary for readable output. Deliberately tiny: this is a correctness
#: harness, and a large word list would imply linguistic realism the generator does not have.
WORDS: tuple[str, ...] = (
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "and",
    "then",
    "writes",
    "a",
    "short",
    "note",
    "about",
    "letters",
    "words",
    "lines",
    "paper",
    "ink",
    "meeting",
    "scheduled",
    "tomorrow",
    "please",
    "bring",
    "book",
    "hello",
    "committee",
)

CORPUS_MODES = ("coverage", "words")

DEFAULT_MIN_OCCURRENCES = 2
_MIN_CHUNK, _MAX_CHUNK = 2, 7
_MIN_WORDS_PER_LINE, _MAX_WORDS_PER_LINE = 4, 9


def sample_lines(
    characters: Sequence[str],
    *,
    n_lines: int,
    rng: random.Random,
    mode: str = "coverage",
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
) -> list[str]:
    """Produce ``n_lines`` transcripts for one writer.

    Args:
        characters: Encodable characters, typically ``charset.characters``.
        n_lines: How many lines to emit.
        rng: Seeded generator; the same seed always yields the same lines.
        mode: ``coverage`` guarantees every character appears at least ``min_occurrences`` times
            across the writer's lines. ``words`` emits readable text from :data:`WORDS` and makes no
            coverage guarantee.
        min_occurrences: Coverage target per character.

    Raises:
        ValueError: ``n_lines`` is not positive, ``mode`` is unknown, or coverage is requested with
            no usable characters.
    """
    if n_lines < 1:
        raise ValueError(f"n_lines must be at least 1, got {n_lines}")
    if mode not in CORPUS_MODES:
        raise ValueError(f"Unknown corpus mode {mode!r}; expected one of {list(CORPUS_MODES)}")

    if mode == "words":
        return _sample_word_lines(n_lines=n_lines, rng=rng)
    return _sample_coverage_lines(
        characters, n_lines=n_lines, rng=rng, min_occurrences=min_occurrences
    )


def _sample_word_lines(*, n_lines: int, rng: random.Random) -> list[str]:
    lines = []
    for _ in range(n_lines):
        count = rng.randint(_MIN_WORDS_PER_LINE, _MAX_WORDS_PER_LINE)
        lines.append(" ".join(rng.choice(WORDS) for _ in range(count)))
    return lines


def _sample_coverage_lines(
    characters: Sequence[str],
    *,
    n_lines: int,
    rng: random.Random,
    min_occurrences: int,
) -> list[str]:
    """Emit lines containing every character at least ``min_occurrences`` times.

    Space is excluded from the required pool — it is the separator, so it appears everywhere
    regardless and padding it would only lengthen lines.
    """
    required = [c for c in characters if c != " "]
    if not required:
        raise ValueError("Coverage mode needs at least one non-space character.")

    pool: list[str] = []
    for _ in range(max(1, min_occurrences)):
        shuffled = list(required)
        rng.shuffle(shuffled)
        pool.extend(shuffled)

    # Break the pool into pseudo-words, then deal those words across the lines. Dealing round-robin
    # keeps line lengths even; concentrating coverage in line 1 would leave the rest degenerate.
    chunks: list[str] = []
    index = 0
    while index < len(pool):
        size = rng.randint(_MIN_CHUNK, _MAX_CHUNK)
        chunks.append("".join(pool[index : index + size]))
        index += size

    buckets: list[list[str]] = [[] for _ in range(n_lines)]
    for position, chunk in enumerate(chunks):
        buckets[position % n_lines].append(chunk)

    # A line that received nothing (more lines than chunks) still needs content.
    for bucket in buckets:
        if not bucket:
            size = rng.randint(_MIN_CHUNK, _MAX_CHUNK)
            bucket.append("".join(rng.choice(required) for _ in range(size)))

    return [" ".join(bucket) for bucket in buckets]


def coverage_counts(lines: Sequence[str]) -> Counter[str]:
    """Character frequencies across a writer's lines, for verifying coverage."""
    counts: Counter[str] = Counter()
    for line in lines:
        counts.update(line)
    return counts


def missing_coverage(
    lines: Sequence[str], characters: Sequence[str], *, min_occurrences: int
) -> dict[str, int]:
    """Characters that fall short of the coverage target, mapped to their actual count."""
    counts = coverage_counts(lines)
    return {
        character: counts.get(character, 0)
        for character in characters
        if character != " " and counts.get(character, 0) < min_occurrences
    }
