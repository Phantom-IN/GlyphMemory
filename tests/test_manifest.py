"""Manifest contract, IO, fingerprinting and integrity accounting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glyphmemory.data import (
    IntegrityCategory,
    ManifestError,
    ManifestRecord,
    manifest_fingerprint,
    parse_record,
    read_manifest,
    records_fingerprint,
    validate_manifest,
    validate_records,
    write_manifest,
)
from glyphmemory.data.adapters import make_sample_id, split_sample_id


def record(**overrides) -> ManifestRecord:
    base = {
        "image": "/data/x/line-0.png",
        "text": "the quick brown fox",
        "writer_id": "synthetic/w0",
        "dataset": "synthetic",
        "split": "train",
    }
    return ManifestRecord(**{**base, **overrides})


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "images"
    directory.mkdir()
    for index in range(4):
        (directory / f"line-{index}.png").write_bytes(b"not-a-real-png")
    return directory


def records_with_images(image_dir: Path, count: int = 4) -> list[ManifestRecord]:
    return [
        record(
            image=str(image_dir / f"line-{index}.png"),
            sample_id=make_sample_id("synthetic", f"line-{index}"),
            writer_id=f"synthetic/w{index % 2}",
        )
        for index in range(count)
    ]


# --------------------------------------------------------------------------- record contract


def test_required_fields_present_in_serialised_form():
    payload = record().to_dict()
    assert set(payload) == {"image", "text", "writer_id", "dataset", "split"}


def test_unset_optional_fields_are_omitted():
    assert "passage_id" not in record().to_dict()


def test_optional_fields_roundtrip_without_loss():
    original = record(
        sample_id="synthetic/0",
        source_page="page-7",
        passage_id="p3",
        language="en",
        height=64,
        width=512,
    )
    assert parse_record(json.loads(original.to_json())) == original


@pytest.mark.parametrize("field", ["image", "text", "writer_id", "dataset", "split"])
def test_missing_required_field_rejected(field):
    payload = record().to_dict()
    del payload[field]
    with pytest.raises(ManifestError, match="missing or empty required field"):
        parse_record(payload)


@pytest.mark.parametrize("field", ["image", "text", "writer_id", "dataset", "split"])
def test_empty_required_field_rejected(field):
    payload = record().to_dict()
    payload[field] = ""
    with pytest.raises(ManifestError, match="missing or empty required field"):
        parse_record(payload)


def test_invalid_split_rejected():
    with pytest.raises(ManifestError, match="is not one of"):
        parse_record(record().to_dict() | {"split": "validation"})


def test_non_string_required_field_rejected():
    with pytest.raises(ManifestError, match="must be a string"):
        parse_record(record().to_dict() | {"writer_id": 7})


@pytest.mark.parametrize("value", [64.5, "64", True, 0, -1])
def test_bad_dimension_rejected(value):
    with pytest.raises(ManifestError):
        parse_record(record().to_dict() | {"height": value})


def test_unknown_keys_ignored_for_forward_compatibility():
    """A manifest written by a newer schema must stay readable."""
    parsed = parse_record(record().to_dict() | {"future_field": "whatever"})
    assert parsed.writer_id == "synthetic/w0"


def test_non_object_line_rejected():
    with pytest.raises(ManifestError, match="expected a JSON object"):
        parse_record([1, 2, 3])


# --------------------------------------------------------------------------- file IO


def test_write_then_read_roundtrip(tmp_path: Path):
    original = [record(sample_id=f"synthetic/{i}", passage_id=f"p{i}") for i in range(5)]
    path = tmp_path / "manifest.jsonl"
    assert write_manifest(path, original) == 5
    assert list(read_manifest(path)) == original


def test_blank_lines_skipped_without_error(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(f"{record().to_json()}\n\n   \n{record().to_json()}\n", encoding="utf-8")
    assert len(list(read_manifest(path))) == 2


def test_read_manifest_is_strict_and_names_the_line(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(f"{record().to_json()}\nnot json\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="line 2"):
        list(read_manifest(path))


def test_write_creates_parent_directories(tmp_path: Path):
    path = tmp_path / "nested" / "deeper" / "manifest.jsonl"
    write_manifest(path, [record()])
    assert path.is_file()


def test_unicode_text_survives_roundtrip(tmp_path: Path):
    original = record(text="naïve café — dash")
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [original])
    assert next(read_manifest(path)).text == original.text


# --------------------------------------------------------------------------- fingerprint


def test_fingerprint_stable_across_reads(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [record(sample_id=f"synthetic/{i}") for i in range(3)])
    assert manifest_fingerprint(path) == manifest_fingerprint(path)


def test_fingerprint_changes_when_any_field_changes():
    before = [record(sample_id="synthetic/0")]
    after = [record(sample_id="synthetic/0", text="a different transcript")]
    assert records_fingerprint(before) != records_fingerprint(after)


def test_fingerprint_insensitive_to_record_order():
    """Regenerating a manifest in another order is the same dataset."""
    items = [record(sample_id=f"synthetic/{i}") for i in range(4)]
    assert records_fingerprint(items) == records_fingerprint(list(reversed(items)))


def test_fingerprint_sensitive_to_duplication():
    """Order-insensitivity must not become multiplicity-insensitivity."""
    items = [record(sample_id="synthetic/0")]
    assert records_fingerprint(items) != records_fingerprint(items * 2)


def test_fingerprint_insensitive_to_key_order_and_whitespace(tmp_path: Path):
    canonical = tmp_path / "a.jsonl"
    scrambled = tmp_path / "b.jsonl"
    payload = record().to_dict()
    canonical.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    scrambled.write_text(
        json.dumps(dict(reversed(list(payload.items()))), indent=1).replace("\n", "") + "\n",
        encoding="utf-8",
    )
    assert manifest_fingerprint(canonical) == manifest_fingerprint(scrambled)


def test_empty_manifest_has_a_fingerprint():
    assert records_fingerprint([])


# --------------------------------------------------------------------------- validation


def test_clean_manifest_reports_no_rejections(tmp_path: Path, image_dir: Path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, records_with_images(image_dir))
    report = validate_manifest(path)
    assert report.is_clean
    assert report.valid_records == 4
    assert len(report.writers) == 2


def test_three_malformed_records_are_counted_not_dropped(tmp_path: Path, image_dir: Path):
    """The success criterion: exactly 3 counted, each named, none silently discarded."""
    path = tmp_path / "manifest.jsonl"
    good = records_with_images(image_dir, count=3)
    lines = [r.to_json() for r in good]
    lines.append("{ this is not json")
    lines.append(json.dumps({"image": "x.png", "text": "no writer"}))
    lines.append(json.dumps(record().to_dict() | {"split": "nope"}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = validate_manifest(path)

    assert report.total_records == 6
    assert report.valid_records == 3
    assert report.rejected_records == 3
    assert len(report.counters.issues) == 3
    assert all(issue.reason for issue in report.counters.issues)
    assert report.counters.count_of(IntegrityCategory.INVALID_SPLIT) == 1


def test_every_issue_carries_sample_id_path_and_reason(image_dir: Path):
    report = validate_records([record(sample_id="synthetic/0", text="   ")], check_images=False)
    issue = report.counters.issues[0]
    assert issue.sample_id == "synthetic/0"
    assert issue.path
    assert issue.reason
    assert "synthetic/0" in str(issue)


def test_empty_transcript_counted():
    report = validate_records([record(text="  ")], check_images=False)
    assert report.counters.count_of(IntegrityCategory.MISSING_TRANSCRIPT) == 1
    assert report.valid_records == 0


def test_duplicate_sample_id_counted():
    duplicated = [record(sample_id="synthetic/0"), record(sample_id="synthetic/0")]
    report = validate_records(duplicated, check_images=False)
    assert report.counters.count_of(IntegrityCategory.DUPLICATE_SAMPLE_ID) == 1
    assert report.valid_records == 1


def test_missing_image_file_counted(tmp_path: Path):
    report = validate_records([record(image=str(tmp_path / "absent.png"))], check_images=True)
    assert report.counters.count_of(IntegrityCategory.MISSING_IMAGE_FILE) == 1


def test_image_check_can_be_disabled():
    assert validate_records([record()], check_images=False).is_clean


def test_relative_image_paths_resolve_against_root(image_dir: Path):
    relative = record(image="line-0.png")
    assert validate_records([relative], image_root=image_dir).is_clean


def test_report_shape_stable_with_all_categories_present():
    """Downstream reporting depends on every canonical category existing, including zeroes."""
    counts = validate_records([record()], check_images=False).counters.as_dict()
    assert set(counts) == {str(category) for category in IntegrityCategory}


def test_canonical_category_names_match_documentation():
    """Internal helper."""
    for name in (
        "unreadable_image",
        "missing_transcript",
        "unsupported_character",
        "impossible_ctc_length",
        "corrupted_record",
        "oversized_width",
    ):
        assert name in {str(category) for category in IntegrityCategory}


def test_unknown_fields_are_reported_not_hidden(tmp_path: Path, image_dir: Path):
    path = tmp_path / "manifest.jsonl"
    payload = records_with_images(image_dir, count=1)[0].to_dict() | {"from_the_future": 1}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    report = validate_manifest(path)
    assert report.is_clean
    assert "from_the_future" in report.unknown_field_names


def test_logging_is_rate_limited_but_counts_stay_exact(caplog):
    """Internal helper."""
    many = [record(text="") for _ in range(50)]
    with caplog.at_level("WARNING", logger="glyphmemory.data.validation"):
        report = validate_records(many, check_images=False)
    assert report.counters.count_of(IntegrityCategory.MISSING_TRANSCRIPT) == 50
    assert len(caplog.records) < 50
    assert any("counts remain exact" in r.message for r in caplog.records)


def test_report_formats_and_serialises(tmp_path: Path, image_dir: Path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, records_with_images(image_dir))
    report = validate_manifest(path)
    assert "records" in report.format()
    assert json.loads(json.dumps(report.as_dict()))["valid_records"] == 4


# --------------------------------------------------------------------------- sample ids


def test_sample_ids_are_dataset_prefixed():
    assert make_sample_id("cvl", "0001-1-1") == "cvl/0001-1-1"


def test_sample_id_roundtrip():
    assert split_sample_id(make_sample_id("iam", "a01-000u-00")) == ("iam", "a01-000u-00")


def test_ids_from_different_datasets_cannot_collide():
    """Cross-dataset evaluation (Protocol B) puts two corpora in one split."""
    assert make_sample_id("iam", "0001") != make_sample_id("cvl", "0001")


def test_unprefixed_sample_id_rejected():
    with pytest.raises(ValueError, match="not dataset-prefixed"):
        split_sample_id("0001")


@pytest.mark.parametrize("args", [("", "x"), ("iam", ""), ("bad/name", "x")])
def test_invalid_sample_id_components_rejected(args):
    with pytest.raises(ValueError):
        make_sample_id(*args)
