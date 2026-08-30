"""Manifest validation and integrity accounting.

Data integrity is part of scientific validity, so the count of what was excluded is itself a result.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from glyphmemory.ctc.normalization import normalizes_to_empty
from glyphmemory.ctc.tokenizer import Tokenizer
from glyphmemory.data.manifest import (
    ERROR_CORRUPTED_RECORD,
    ERROR_INVALID_SPLIT,
    ManifestError,
    ManifestRecord,
    iter_raw_records,
    parse_record,
    unknown_fields,
)
from glyphmemory.runtime.logging import get_logger

logger = get_logger("data.validation")

DEFAULT_LOG_LIMIT_PER_CATEGORY = 20
DEFAULT_MAX_RETAINED_ISSUES = 1000


class IntegrityCategory(StrEnum):
    """Why a sample was rejected."""

    UNREADABLE_IMAGE = "unreadable_image"
    MISSING_TRANSCRIPT = "missing_transcript"
    UNSUPPORTED_CHARACTER = "unsupported_character"
    IMPOSSIBLE_CTC_LENGTH = "impossible_ctc_length"
    CORRUPTED_RECORD = "corrupted_record"
    OVERSIZED_WIDTH = "oversized_width"

    MISSING_IMAGE_FILE = "missing_image_file"
    MISSING_WRITER_ID = "missing_writer_id"
    INVALID_SPLIT = "invalid_split"
    DUPLICATE_SAMPLE_ID = "duplicate_sample_id"

    # Deliberate policy exclusions rather than defects. Each is a separate category precisely
    # so that "we chose not to use this" never gets read as "this data was broken", while
    # still being counted rather than dropped in silence.
    #
    #   EXCLUDED_LANGUAGE  CVL's German passage against an English vocabulary.
    #   STRUCK_OUT_TOKEN   IAM's '#' marker for a word the writer crossed out.
    #                      The ink is a scribble, so the line has no well-defined transcript.
    #   SEGMENTATION_ERROR IAM's 'err' word-segmentation flag. Kept by default;
    #                      this category exists for runs that deliberately exclude them.
    EXCLUDED_LANGUAGE = "excluded_language"
    STRUCK_OUT_TOKEN = "struck_out_token"
    SEGMENTATION_ERROR = "segmentation_error"


_ERROR_KIND_TO_CATEGORY: dict[str, IntegrityCategory] = {
    ERROR_CORRUPTED_RECORD: IntegrityCategory.CORRUPTED_RECORD,
    ERROR_INVALID_SPLIT: IntegrityCategory.INVALID_SPLIT,
}


@dataclass(frozen=True, slots=True)
class IntegrityIssue:
    """One rejected sample. Always carries sample_id, path and reason."""

    category: IntegrityCategory
    sample_id: str | None
    path: str | None
    reason: str
    line_number: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": str(self.category),
            "sample_id": self.sample_id,
            "path": self.path,
            "reason": self.reason,
            "line_number": self.line_number,
        }

    def __str__(self) -> str:
        where = f" (line {self.line_number})" if self.line_number is not None else ""
        return (
            f"[{self.category}] sample_id={self.sample_id} path={self.path}{where}: {self.reason}"
        )


@dataclass
class IntegrityCounters:
    """Exact counts per category, with rate-limited logging and bounded retention."""

    log_limit_per_category: int = DEFAULT_LOG_LIMIT_PER_CATEGORY
    max_retained_issues: int = DEFAULT_MAX_RETAINED_ISSUES
    counts: Counter[str] = field(default_factory=Counter)
    issues: list[IntegrityIssue] = field(default_factory=list)
    _suppression_announced: set[str] = field(default_factory=set, repr=False)

    def record(self, issue: IntegrityIssue) -> None:
        """Count, log (rate-limited) and retain (bounded) one rejection."""
        key = str(issue.category)
        self.counts[key] += 1

        seen = self.counts[key]
        if seen <= self.log_limit_per_category:
            logger.warning("%s", issue)
        elif key not in self._suppression_announced:
            self._suppression_announced.add(key)
            logger.warning(
                "Further %r issues will not be logged individually; counts remain exact.", key
            )

        if len(self.issues) < self.max_retained_issues:
            self.issues.append(issue)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def count_of(self, category: IntegrityCategory) -> int:
        return self.counts[str(category)]

    def as_dict(self) -> dict[str, int]:
        """Every canonical category present, including zeroes, for a stable report shape."""
        return {str(category): self.counts[str(category)] for category in IntegrityCategory}


@dataclass(frozen=True, slots=True)
class ManifestReport:
    """The outcome of validating a manifest."""

    total_records: int
    valid_records: int
    counters: IntegrityCounters
    writers: frozenset[str]
    splits: Counter[str]
    datasets: Counter[str]
    unknown_field_names: frozenset[str]

    @property
    def rejected_records(self) -> int:
        return self.total_records - self.valid_records

    @property
    def is_clean(self) -> bool:
        return self.rejected_records == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "rejected_records": self.rejected_records,
            "writers": len(self.writers),
            "splits": dict(self.splits),
            "datasets": dict(self.datasets),
            "integrity": self.counters.as_dict(),
            "unknown_field_names": sorted(self.unknown_field_names),
        }

    def format(self) -> str:
        lines = [
            f"records          {self.total_records:>8,}",
            f"valid            {self.valid_records:>8,}",
            f"rejected         {self.rejected_records:>8,}",
            f"writers          {len(self.writers):>8,}",
            f"splits           {dict(self.splits)}",
            f"datasets         {dict(self.datasets)}",
        ]
        rejections = {k: v for k, v in self.counters.as_dict().items() if v}
        if rejections:
            lines.append("rejections:")
            width = max(len(name) for name in rejections)
            for name, count in sorted(rejections.items()):
                lines.append(f"  {name:<{width}}  {count:>8,}")
        if self.unknown_field_names:
            lines.append(f"unknown fields   {sorted(self.unknown_field_names)}")
        return "\n".join(lines)


def validate_records(
    records: list[ManifestRecord],
    *,
    check_images: bool = True,
    image_root: Path | None = None,
    counters: IntegrityCounters | None = None,
    tokenizer: Tokenizer | None = None,
) -> ManifestReport:
    """Validate already-parsed records.

    Args:
        records: Parsed records to check.
        check_images: Verify each ``image`` path exists. Disable for structural-only checks on
            manifests whose images are not present locally.
        image_root: Resolve relative image paths against this directory.
        counters: Reuse an existing counter set, e.g. one already holding parse failures.
        tokenizer: When given, transcripts are checked for encodability and emptiness is judged
            **after normalization**. Without it, emptiness falls back to a plain whitespace check
            and ``unsupported_character`` is never raised.
    """
    counters = counters or IntegrityCounters()
    writers: set[str] = set()
    splits: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    seen_sample_ids: dict[str, str] = {}
    valid = 0

    for record in records:
        rejected = False

        # Emptiness is judged after normalization when a tokenizer is available: a transcript of
        # only non-breaking spaces is not empty by str.strip() but carries no transcription.
        if tokenizer is not None:
            empty = normalizes_to_empty(record.text, tokenizer.policy)
            reason = f"transcript is empty after {tokenizer.policy.name} normalization"
        else:
            empty = not record.text.strip()
            reason = "transcript is empty or whitespace-only"

        if empty:
            counters.record(
                IntegrityIssue(
                    IntegrityCategory.MISSING_TRANSCRIPT,
                    record.sample_id,
                    record.image,
                    reason,
                )
            )
            rejected = True
        elif tokenizer is not None:
            unsupported = tokenizer.unsupported_characters(record.text)
            if unsupported:
                listed = ", ".join(f"{c!r} (U+{ord(c):04X})" for c in sorted(unsupported))
                counters.record(
                    IntegrityIssue(
                        IntegrityCategory.UNSUPPORTED_CHARACTER,
                        record.sample_id,
                        record.image,
                        f"transcript contains character(s) absent from charset "
                        f"{tokenizer.charset.name!r}: {listed}",
                    )
                )
                rejected = True

        if not record.writer_id.strip():
            counters.record(
                IntegrityIssue(
                    IntegrityCategory.MISSING_WRITER_ID,
                    record.sample_id,
                    record.image,
                    "writer_id is empty",
                )
            )
            rejected = True

        if record.sample_id is not None:
            previous = seen_sample_ids.get(record.sample_id)
            if previous is not None:
                counters.record(
                    IntegrityIssue(
                        IntegrityCategory.DUPLICATE_SAMPLE_ID,
                        record.sample_id,
                        record.image,
                        f"sample_id already used by {previous!r}",
                    )
                )
                rejected = True
            else:
                seen_sample_ids[record.sample_id] = record.image

        if check_images:
            path = record.image_path
            if image_root is not None and not path.is_absolute():
                path = Path(image_root) / path
            if not path.is_file():
                counters.record(
                    IntegrityIssue(
                        IntegrityCategory.MISSING_IMAGE_FILE,
                        record.sample_id,
                        record.image,
                        "image file does not exist",
                    )
                )
                rejected = True

        if not rejected:
            valid += 1
            writers.add(record.writer_id)
            splits[record.split] += 1
            datasets[record.dataset] += 1

    return ManifestReport(
        total_records=len(records),
        valid_records=valid,
        counters=counters,
        writers=frozenset(writers),
        splits=splits,
        datasets=datasets,
        unknown_field_names=frozenset(),
    )


def validate_manifest(
    path: str | Path,
    *,
    check_images: bool = True,
    image_root: Path | None = None,
    tokenizer: Tokenizer | None = None,
) -> ManifestReport:
    """Validate a manifest file, counting malformed lines rather than raising on them.

    Unlike :func:`glyphmemory.data.manifest.read_manifest`, a corrupted line here is recorded as
    ``CORRUPTED_RECORD`` and validation continues, so one bad line does not hide the state of the
    other 99,999.
    """
    import json

    counters = IntegrityCounters()
    parsed: list[ManifestRecord] = []
    unknown: set[str] = set()
    total = 0

    for line_number, line in iter_raw_records(path):
        total += 1
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            counters.record(
                IntegrityIssue(
                    IntegrityCategory.CORRUPTED_RECORD,
                    None,
                    str(path),
                    f"invalid JSON — {exc}",
                    line_number,
                )
            )
            continue

        if isinstance(payload, dict):
            unknown |= unknown_fields(payload)

        try:
            parsed.append(parse_record(payload, line_number=line_number))
        except ManifestError as exc:
            # Classified by the error's `kind`, never by inspecting its message.
            category = _ERROR_KIND_TO_CATEGORY.get(exc.kind, IntegrityCategory.CORRUPTED_RECORD)
            sample_id = payload.get("sample_id") if isinstance(payload, dict) else None
            image = payload.get("image") if isinstance(payload, dict) else None
            counters.record(IntegrityIssue(category, sample_id, image, str(exc), line_number))

    report = validate_records(
        parsed,
        check_images=check_images,
        image_root=image_root,
        counters=counters,
        tokenizer=tokenizer,
    )

    # Rebuild with the true line total: records that failed to parse never reached validate_records,
    # but they are part of what the file contained.
    return ManifestReport(
        total_records=total,
        valid_records=report.valid_records,
        counters=counters,
        writers=report.writers,
        splits=report.splits,
        datasets=report.datasets,
        unknown_field_names=frozenset(unknown),
    )
