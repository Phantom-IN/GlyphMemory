"""The internal manifest contract.

Every dataset — IAM, CVL, synthetic — converts to this one format.

A manifest is JSONL, one :class:`ManifestRecord` per line::

    {"image":"/data/iam/lines/a01-000u-00.png","text":"A MOVE to stop Mr. Gaitskell",
     "writer_id":"iam_000","dataset":"iam","split":"train"}

Reading is streaming because manifests reach ~10^5 lines.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

# Bumped when the required-field set changes. It is mixed into the fingerprint, so a schema change
# invalidates every recorded manifest identity — the intended loud failure.
MANIFEST_SCHEMA_VERSION = "1"

VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")

REQUIRED_FIELDS: tuple[str, ...] = ("image", "text", "writer_id", "dataset", "split")
OPTIONAL_FIELDS: tuple[str, ...] = (
    "sample_id",
    "source_page",
    "passage_id",
    "language",
    "height",
    "width",
)


# Error kinds carried on ManifestError so callers can classify a failure without parsing its
# message. Message-sniffing is fragile: "missing required field(s): ['split', ...]" contains the
# word "split" but is not a split error.
ERROR_CORRUPTED_RECORD = "corrupted_record"
ERROR_INVALID_SPLIT = "invalid_split"


class ManifestError(ValueError):
    """A manifest record could not be parsed or is structurally invalid.

    Attributes:
        kind: Machine-readable classification, one of the ``ERROR_*`` constants.
    """

    def __init__(self, message: str, *, kind: str = ERROR_CORRUPTED_RECORD) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """One handwritten line.

    Required fields are the contract every adapter must satisfy. Optional fields carry
    dataset-specific provenance without leaking dataset-specific *logic* downstream.

    ``passage_id`` identifies the source text a line was copied from.
    """

    image: str
    text: str
    writer_id: str
    dataset: str
    split: str
    sample_id: str | None = None
    source_page: str | None = None
    passage_id: str | None = None
    language: str | None = None
    height: int | None = None
    width: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain dict with unset optional fields omitted, keeping manifests compact."""
        payload: dict[str, Any] = {name: getattr(self, name) for name in REQUIRED_FIELDS}
        for name in OPTIONAL_FIELDS:
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    @property
    def image_path(self) -> Path:
        return Path(self.image)

    def identity(self) -> str:
        """Stable identifier for logging, falling back to the image path."""
        return self.sample_id or self.image


_FIELD_NAMES = frozenset(f.name for f in fields(ManifestRecord))


def parse_record(payload: Any, *, line_number: int | None = None) -> ManifestRecord:
    """Build a record from a decoded JSON object.

    Unknown keys are ignored so a manifest written by a newer schema stays readable.
    :func:`unknown_fields` exposes them for reporting, so "ignored" never means "invisible".
    """
    where = f"line {line_number}: " if line_number is not None else ""

    if not isinstance(payload, dict):
        raise ManifestError(f"{where}expected a JSON object, got {type(payload).__name__}")

    missing = [name for name in REQUIRED_FIELDS if payload.get(name) in (None, "")]
    if missing:
        raise ManifestError(f"{where}missing or empty required field(s): {sorted(missing)}")

    for name in REQUIRED_FIELDS:
        if not isinstance(payload[name], str):
            raise ManifestError(
                f"{where}field {name!r} must be a string, got {type(payload[name]).__name__}"
            )

    if payload["split"] not in VALID_SPLITS:
        raise ManifestError(
            f"{where}split {payload['split']!r} is not one of {list(VALID_SPLITS)}",
            kind=ERROR_INVALID_SPLIT,
        )

    for name in ("height", "width"):
        value = payload.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ManifestError(f"{where}field {name!r} must be an integer, got {value!r}")
        if value <= 0:
            raise ManifestError(f"{where}field {name!r} must be positive, got {value}")

    known = {name: payload.get(name) for name in _FIELD_NAMES if name in payload}
    return ManifestRecord(**known)


def unknown_fields(payload: dict[str, Any]) -> set[str]:
    """Keys present in a raw record that this schema version does not define."""
    return set(payload) - _FIELD_NAMES


def iter_raw_records(path: str | Path) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, raw_line)`` for every non-blank line.

    Blank lines are skipped without counting — whitespace is not data loss.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, line


def read_manifest(path: str | Path) -> Iterator[ManifestRecord]:
    """Stream records from a JSONL manifest, raising on the first malformed line.

    Strict by design. To *count* malformed records instead of failing, use
    :func:`glyphmemory.data.validation.validate_manifest`, which exists precisely so that tolerating
    bad data is an explicit, counted choice rather than a default.
    """
    for line_number, line in iter_raw_records(path):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"line {line_number}: invalid JSON — {exc}") from exc
        yield parse_record(payload, line_number=line_number)


def write_manifest(path: str | Path, records: Iterable[ManifestRecord]) -> int:
    """Write records as JSONL. Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json())
            handle.write("\n")
            count += 1
    return count


def record_digest(record: ManifestRecord) -> str:
    """SHA-256 of one record's canonical JSON."""
    canonical = json.dumps(
        record.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def records_fingerprint(records: Iterable[ManifestRecord]) -> str:
    """Content fingerprint for a set of records.

    Order-insensitive: per-record digests are sorted before hashing, so regenerating a manifest in a
    different order yields the same identity while any change to any field changes it. Multiplicity
    is preserved, so a duplicated record still shifts the fingerprint. The schema version is mixed
    in.
    """
    digest = hashlib.sha256()
    digest.update(f"glyphmemory-manifest-v{MANIFEST_SCHEMA_VERSION}\n".encode())
    for record_hash in sorted(record_digest(record) for record in records):
        digest.update(record_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def manifest_fingerprint(path: str | Path) -> str:
    """Content fingerprint of a manifest file, recorded with every run."""
    return records_fingerprint(read_manifest(path))
