"""S/I/D error taxonomy and the single-glyph-confusion fraction.

**Definition, stated once and used consistently.** A substitution in the character-level alignment
(see :mod:`glyphmemory.evaluation.align`) is a *single-glyph confusion* when it is **isolated**: the
alignment step immediately before it (if any) and immediately after it (if any) are both matches.
Every substitution is, by construction of a character-level alignment, already a
one-character-for-one-character swap — so isolation, not character count, is the condition doing the
work here. An isolated substitution means the model read the entire rest of the line correctly and
only misjudged one glyph, which is exactly the shape a prototype fix addresses. A substitution
sitting next to an insertion, a deletion, or another substitution is part of a longer garbled span —
a segmentation slip or a multi-character misread — which swapping one glyph's representation would
not, on its own, correct.

The confusion matrix counts **every** substitution, isolated or not: its purpose is to show whether
the errors are the kind of consistent per-glyph confusion writer memory could address, not to
pre-filter for that answer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from glyphmemory.ctc.normalization import NFC_V1, NormalizationPolicy, normalize
from glyphmemory.evaluation.align import (
    DELETION,
    INSERTION,
    MATCH,
    SUBSTITUTION,
    AlignOp,
    align_ops,
)


def is_single_glyph_confusion(ops: list[AlignOp], index: int) -> bool:
    """``True`` when ``ops[index]`` is a substitution isolated by matches on both sides."""
    op = ops[index]
    if op.kind != SUBSTITUTION:
        return False
    before_ok = index == 0 or ops[index - 1].kind == MATCH
    after_ok = index == len(ops) - 1 or ops[index + 1].kind == MATCH
    return before_ok and after_ok


@dataclass(frozen=True, slots=True)
class ErrorTaxonomy:
    """The S/I/D split, the single-glyph-confusion fraction, and the confusion matrix.

    ``confusion_counts`` maps ``(reference_char, hypothesis_char)`` to how often that substitution
    occurred, across every line scored.
    """

    substitutions: int
    insertions: int
    deletions: int
    single_glyph_confusions: int
    confusion_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    normalization: str = NFC_V1.name
    lines: int = 0

    @property
    def total_edits(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def single_glyph_fraction(self) -> float | None:
        """``single_glyph_confusions / substitutions``. ``None`` when there are no substitutions."""
        if self.substitutions == 0:
            return None
        return self.single_glyph_confusions / self.substitutions

    def top_confusions(self, n: int = 20) -> list[tuple[tuple[str, str], int]]:
        """The ``n`` most frequent ``(reference_char, hypothesis_char)`` substitution pairs."""
        return self.confusion_counts.most_common(n)

    def as_dict(self, *, top_n: int = 20) -> dict[str, Any]:
        return {
            "normalization": self.normalization,
            "lines": self.lines,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "total_edits": self.total_edits,
            "single_glyph_confusion_definition": (
                "a substitution in the character-level alignment with a match (or line "
                "boundary) immediately before and after it"
            ),
            "single_glyph_confusions": self.single_glyph_confusions,
            "single_glyph_fraction": self.single_glyph_fraction,
            "top_confusions": [
                {"reference": ref, "hypothesis": hyp, "count": count}
                for (ref, hyp), count in self.top_confusions(top_n)
            ],
        }

    def format(self, *, top_n: int = 10) -> str:
        fraction = self.single_glyph_fraction
        fraction_text = "n/a (no substitutions)" if fraction is None else f"{fraction:.4f}"
        lines = [
            f"error taxonomy   ({self.lines:,} lines, normalization {self.normalization!r})",
            f"  substitutions  {self.substitutions:>8,}",
            f"  insertions     {self.insertions:>8,}",
            f"  deletions      {self.deletions:>8,}",
            f"  single-glyph confusions   {self.single_glyph_confusions:>8,}  "
            f"({fraction_text} of substitutions)",
            "  top confusions (reference -> hypothesis : count)",
        ]
        for (ref, hyp), count in self.top_confusions(top_n):
            lines.append(f"    {ref!r:>6} -> {hyp!r:<6} : {count}")
        return "\n".join(lines)


def build_taxonomy(
    pairs: Iterable[tuple[str, str]], *, policy: NormalizationPolicy = NFC_V1
) -> ErrorTaxonomy:
    """Aggregate the error taxonomy over ``(reference, hypothesis)`` pairs.

    Each pair is normalized under ``policy`` before alignment, matching
    :func:`glyphmemory.metrics.text.corpus_cer` so the taxonomy describes the same errors the
    reported CER counts.
    """
    substitutions = insertions = deletions = single_glyph = lines = 0
    confusion_counts: Counter[tuple[str, str]] = Counter()

    for reference, hypothesis in pairs:
        lines += 1
        ref = normalize(reference, policy)
        hyp = normalize(hypothesis, policy)
        ops = align_ops(ref, hyp)
        for index, op in enumerate(ops):
            if op.kind == SUBSTITUTION:
                substitutions += 1
                assert op.ref_char is not None and op.hyp_char is not None
                confusion_counts[(op.ref_char, op.hyp_char)] += 1
                if is_single_glyph_confusion(ops, index):
                    single_glyph += 1
            elif op.kind == INSERTION:
                insertions += 1
            elif op.kind == DELETION:
                deletions += 1

    return ErrorTaxonomy(
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        single_glyph_confusions=single_glyph,
        confusion_counts=confusion_counts,
        normalization=policy.name,
        lines=lines,
    )
