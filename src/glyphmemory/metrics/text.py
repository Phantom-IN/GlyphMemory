"""CER and WER, with aggregation and normalization made explicit.

Every number this project ever reports comes through here, so three conventions are fixed in code
rather than left to a caller:

**Aggregation is micro by default.** ``sum(edits) / sum(reference_length)`` over the corpus, not the
mean of per-line rates. Macro-averaging weights a three-character line the same as a sixty-character
one, so a single error on a short line moves the corpus figure as much as twenty errors on a long
one. Both conventions are defensible and only one is standard; what is indefensible is mixing them,
which makes a project's own results incomparable across time. :func:`macro_cer` exists and is
*named*, so choosing it is a decision rather than an accident, and :attr:`MetricResult.aggregation`
records which one produced the number.

**Normalization travels with the metric.** requires reporting the exact normalization applied before
scoring. It is not possible to obtain a :class:`MetricResult` here without its policy name attached.

**Empty references are defined, not discovered.** An empty reference with a non-empty hypothesis
contributes ``len(hypothesis)`` insertions and zero reference length. Under micro averaging that is
exactly right: the insertions land in the numerator and nothing is divided by zero. The *per-sample*
rate for such a line is ``None`` — not ``1.0``, not ``inf`` — because the rate genuinely does not
exist, and a placeholder would silently distort any macro average taken over it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from glyphmemory.ctc.decode import DEFAULT_DECODER, DecoderConfig
from glyphmemory.ctc.normalization import NFC_V1, NormalizationPolicy, normalize
from glyphmemory.metrics.edit import EditCounts, edit_counts

#: Aggregation conventions. ``micro`` is the default and the one to report.
MICRO = "micro"
MACRO = "macro"


def _prepare(text: str, policy: NormalizationPolicy) -> str:
    return normalize(text, policy)


def character_counts(
    reference: str, hypothesis: str, *, policy: NormalizationPolicy = NFC_V1
) -> EditCounts:
    """Character-level edit counts after normalizing both sides."""
    return edit_counts(_prepare(reference, policy), _prepare(hypothesis, policy))


def word_counts(
    reference: str, hypothesis: str, *, policy: NormalizationPolicy = NFC_V1
) -> EditCounts:
    """Word-level edit counts after normalizing both sides.

    Words are whitespace-separated **after** normalization. Under ``nfc_v1`` that is deterministic
    because the policy has already collapsed whitespace runs to a single ASCII space; under
    ``identity`` it depends on ``str.split``'s handling of runs, which is why the policy is recorded
    alongside every figure rather than assumed.
    """
    return edit_counts(_prepare(reference, policy).split(), _prepare(hypothesis, policy).split())


def cer(reference: str, hypothesis: str, *, policy: NormalizationPolicy = NFC_V1) -> float | None:
    """Character error rate for one pair. ``None`` when the reference is empty."""
    return character_counts(reference, hypothesis, policy=policy).error_rate


def wer(reference: str, hypothesis: str, *, policy: NormalizationPolicy = NFC_V1) -> float | None:
    """Word error rate for one pair. ``None`` when the reference has no words."""
    return word_counts(reference, hypothesis, policy=policy).error_rate


@dataclass(frozen=True, slots=True)
class SampleMetric:
    """One reference/hypothesis pair and its edit breakdown."""

    reference: str
    hypothesis: str
    counts: EditCounts
    sample_id: str | None = None

    @property
    def error_rate(self) -> float | None:
        return self.counts.error_rate

    @property
    def is_exact_match(self) -> bool:
        return self.counts.total == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
            **self.counts.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class MetricResult:
    """An aggregated metric, inseparable from how it was produced."""

    name: str
    counts: EditCounts
    normalization: str
    aggregation: str = MICRO
    decoder: DecoderConfig = DEFAULT_DECODER
    samples: tuple[SampleMetric, ...] = field(default=())
    macro_value: float | None = None

    @property
    def value(self) -> float | None:
        """The reported rate. ``None`` when every reference was empty."""
        if self.aggregation == MACRO:
            return self.macro_value
        return self.counts.error_rate

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def exact_matches(self) -> int:
        return sum(1 for sample in self.samples if sample.is_exact_match)

    def worst(self, n: int = 5) -> tuple[SampleMetric, ...]:
        """The ``n`` samples with the highest error rate. Empty references are excluded."""
        scored = [s for s in self.samples if s.error_rate is not None]
        return tuple(sorted(scored, key=lambda s: s.error_rate or 0.0, reverse=True)[:n])

    def as_dict(self, *, include_samples: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "metric": self.name,
            "value": self.value,
            "aggregation": self.aggregation,
            "normalization": self.normalization,
            "decoder": self.decoder.describe(),
            "samples": self.n_samples,
            "exact_matches": self.exact_matches,
            **self.counts.as_dict(),
        }
        if include_samples:
            payload["per_sample"] = [sample.as_dict() for sample in self.samples]
        return payload

    def format(self) -> str:
        value = "n/a" if self.value is None else f"{self.value:.4f}"
        lines = [
            f"{self.name.upper():<4} {value}   ({self.aggregation}-averaged, "
            f"normalization {self.normalization!r}, decoder {self.decoder.label})",
            f"  samples        {self.n_samples:>8,}   exact {self.exact_matches:>8,}",
            f"  edits          {self.counts.total:>8,}   "
            f"S {self.counts.substitutions:,} / I {self.counts.insertions:,} / "
            f"D {self.counts.deletions:,}",
            f"  reference len  {self.counts.reference_length:>8,}",
        ]
        return "\n".join(lines)


def _score(
    pairs: Iterable[tuple[str, str]],
    *,
    name: str,
    counter: Callable[..., EditCounts],
    policy: NormalizationPolicy,
    decoder: DecoderConfig,
    sample_ids: Sequence[str] | None,
    aggregation: str,
) -> MetricResult:
    samples: list[SampleMetric] = []
    total = EditCounts()

    for index, (reference, hypothesis) in enumerate(pairs):
        counts = counter(reference, hypothesis, policy=policy)
        total = total + counts
        samples.append(
            SampleMetric(
                reference=reference,
                hypothesis=hypothesis,
                counts=counts,
                sample_id=sample_ids[index] if sample_ids is not None else None,
            )
        )

    # Macro deliberately skips samples whose rate is undefined rather than substituting a value for
    # them. That exclusion is part of what makes macro a different measurement, not merely a
    # different average, and it is why the convention is recorded on the result.
    rates = [s.error_rate for s in samples if s.error_rate is not None]
    macro_value = sum(rates) / len(rates) if rates else None

    return MetricResult(
        name=name,
        counts=total,
        normalization=policy.name,
        aggregation=aggregation,
        decoder=decoder,
        samples=tuple(samples),
        macro_value=macro_value,
    )


def corpus_cer(
    pairs: Iterable[tuple[str, str]],
    *,
    policy: NormalizationPolicy = NFC_V1,
    decoder: DecoderConfig = DEFAULT_DECODER,
    sample_ids: Sequence[str] | None = None,
) -> MetricResult:
    """Micro-averaged character error rate over ``(reference, hypothesis)`` pairs."""
    return _score(
        pairs,
        name="cer",
        counter=character_counts,
        policy=policy,
        decoder=decoder,
        sample_ids=sample_ids,
        aggregation=MICRO,
    )


def corpus_wer(
    pairs: Iterable[tuple[str, str]],
    *,
    policy: NormalizationPolicy = NFC_V1,
    decoder: DecoderConfig = DEFAULT_DECODER,
    sample_ids: Sequence[str] | None = None,
) -> MetricResult:
    """Micro-averaged word error rate over ``(reference, hypothesis)`` pairs."""
    return _score(
        pairs,
        name="wer",
        counter=word_counts,
        policy=policy,
        decoder=decoder,
        sample_ids=sample_ids,
        aggregation=MICRO,
    )


def macro_cer(
    pairs: Iterable[tuple[str, str]],
    *,
    policy: NormalizationPolicy = NFC_V1,
    decoder: DecoderConfig = DEFAULT_DECODER,
    sample_ids: Sequence[str] | None = None,
) -> MetricResult:
    """Macro-averaged CER: the mean of per-sample rates.

    **Not the project's reporting convention.** It exists so that choosing it is explicit and
    recorded, and so the difference between the two can be shown rather than argued about. Samples
    with an empty reference are excluded from the mean, because their rate does not exist.
    """
    return _score(
        pairs,
        name="cer",
        counter=character_counts,
        policy=policy,
        decoder=decoder,
        sample_ids=sample_ids,
        aggregation=MACRO,
    )
