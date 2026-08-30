"""Charset coverage over a corpus.

Answers two questions before training can be trusted:

1. **Which characters does the data contain that the charset does not?** Every one is a sample that
   will be rejected, so the rate must be measured before it is accepted.
2. **Which charset symbols never occur?** Dead classes cost parameters in the CTC head and indicate
   a charset guessed rather than measured.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from glyphmemory.ctc.normalization import DEFAULT_POLICY, NormalizationPolicy, normalize
from glyphmemory.ctc.tokenizer import Charset


@runtime_checkable
class TextSample(Protocol):
    """Anything carrying a transcript. ``ManifestRecord`` satisfies this structurally."""

    text: str
    sample_id: str | None
    image: str


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Character inventory of a corpus measured against a charset."""

    total_samples: int
    total_characters: int
    frequencies: Counter[str] = field(default_factory=Counter)
    missing: Counter[str] = field(default_factory=Counter)
    affected_samples: int = 0
    affected_sample_ids: tuple[str, ...] = ()
    charset_name: str = ""
    charset_fingerprint: str = ""
    policy_name: str = ""

    @property
    def is_covered(self) -> bool:
        """True when every character in the corpus is encodable."""
        return not self.missing

    @property
    def affected_fraction(self) -> float:
        """Share of samples containing at least one unsupported character."""
        return self.affected_samples / self.total_samples if self.total_samples else 0.0

    def unused_symbols(self, charset: Charset) -> tuple[str, ...]:
        """Charset symbols this corpus never exercises."""
        return tuple(c for c in charset.characters if not self.frequencies.get(c))

    def as_dict(self) -> dict[str, Any]:
        return {
            "charset": self.charset_name,
            "charset_fingerprint": self.charset_fingerprint,
            "normalization": self.policy_name,
            "total_samples": self.total_samples,
            "total_characters": self.total_characters,
            "distinct_characters": len(self.frequencies),
            "is_covered": self.is_covered,
            "affected_samples": self.affected_samples,
            "affected_fraction": round(self.affected_fraction, 6),
            "missing": {c: n for c, n in sorted(self.missing.items())},
            "frequencies": {c: n for c, n in self.frequencies.most_common()},
        }

    def format(self, *, charset: Charset | None = None, top: int = 15) -> str:
        lines = [
            f"charset            {self.charset_name} ({self.charset_fingerprint[:12]})",
            f"normalization      {self.policy_name}",
            f"samples            {self.total_samples:>10,}",
            f"characters         {self.total_characters:>10,}",
            f"distinct           {len(self.frequencies):>10,}",
            f"fully covered      {'yes' if self.is_covered else 'NO'}",
        ]
        if self.missing:
            lines.append(
                f"samples affected   {self.affected_samples:>10,} ({self.affected_fraction:.2%})"
            )
            lines.append("unsupported characters:")
            for character, count in self.missing.most_common():
                lines.append(
                    f"  {character!r:<10} U+{ord(character):04X}  {count:>8,} occurrence(s)"
                )
            if self.affected_sample_ids:
                shown = ", ".join(self.affected_sample_ids[:5])
                lines.append(f"  first affected: {shown}")
        if charset is not None:
            unused = self.unused_symbols(charset)
            if unused:
                lines.append(f"unused symbols     {len(unused)}: {''.join(unused)!r}")
        if self.frequencies:
            lines.append(f"most frequent (top {top}):")
            for character, count in self.frequencies.most_common(top):
                lines.append(f"  {character!r:<10} {count:>8,}")
        return "\n".join(lines)


def charset_coverage(
    samples: Iterable[TextSample],
    charset: Charset,
    *,
    policy: NormalizationPolicy = DEFAULT_POLICY,
    max_recorded_ids: int = 50,
) -> CoverageReport:
    """Measure a corpus against a charset.

    Counts everything; records at most ``max_recorded_ids`` affected sample identifiers so the
    report stays bounded on a large corpus while the counts remain exact.
    """
    frequencies: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    affected_ids: list[str] = []
    total_samples = 0
    total_characters = 0
    affected_samples = 0

    for sample in samples:
        total_samples += 1
        text = normalize(sample.text, policy)
        frequencies.update(text)
        total_characters += len(text)

        unsupported = [c for c in text if c not in charset]
        if unsupported:
            affected_samples += 1
            missing.update(unsupported)
            if len(affected_ids) < max_recorded_ids:
                affected_ids.append(getattr(sample, "sample_id", None) or sample.image)

    return CoverageReport(
        total_samples=total_samples,
        total_characters=total_characters,
        frequencies=frequencies,
        missing=missing,
        affected_samples=affected_samples,
        affected_sample_ids=tuple(affected_ids),
        charset_name=charset.name,
        charset_fingerprint=charset.fingerprint(),
        policy_name=policy.name,
    )
