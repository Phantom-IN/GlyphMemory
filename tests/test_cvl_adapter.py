"""CVL adapter tests.

Every test runs against a generated fake release (``tests/conftest.py``). No CVL data is committed:
the corpus is CC BY-NC licensed, 5 GB, and CI must never need it.

What these tests defend is the set of properties that were *established by inspecting the real
release* and would silently corrupt the corpus if they regressed:

- transcripts come from word filenames, joined in word-index order;
- the page index is the passage identity;
- the German passage is excluded **and counted**;
- a line with no word images is rejected **and counted**, never emitted with empty text.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from glyphmemory.data.adapters.cvl import (
    CVL_PASSAGES,
    TRANSCRIPT_LIMITATION,
    CVLAdapter,
    passage_id,
)
from glyphmemory.data.manifest import read_manifest
from glyphmemory.data.validation import IntegrityCategory, validate_records


def prepare(source: Path, out: Path, **kwargs: object) -> tuple[CVLAdapter, list]:
    adapter = CVLAdapter(**kwargs)  # type: ignore[arg-type]
    manifest = adapter.prepare(source, out)
    return adapter, list(read_manifest(manifest))


class TestLayout:
    def test_accepts_release_directory(self, fake_cvl: Path) -> None:
        assert CVLAdapter.resolve_root(fake_cvl) == fake_cvl

    def test_accepts_parent_of_release_directory(
        self, make_cvl: Callable[..., Path], tmp_path: Path
    ) -> None:
        """Pointing at ``datasets/CVL`` must work as well as at the release itself."""
        base = make_cvl(
            tmp_path / "CVL",
            layout={"trainset": {"0001": {1: {0: ["hello"]}}}},
            nest_release=True,
        )
        assert CVLAdapter.resolve_root(tmp_path / "CVL") == base

    def test_rejects_unrelated_directory(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="does not look like a CVL release"):
            CVLAdapter.resolve_root(tmp_path / "empty")

    def test_rejects_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            CVLAdapter.resolve_root(tmp_path / "nope")


class TestTranscripts:
    def test_text_joins_words_in_index_order(self, fake_cvl: Path, tmp_path: Path) -> None:
        _, records = prepare(fake_cvl, tmp_path / "out")
        by_id = {record.sample_id: record for record in records}
        assert by_id["cvl/0001-1-0"].text == "Imagine a vast sheet"
        assert by_id["cvl/0052-2-0"].text == "Show'd like a rebel's whore"

    def test_word_index_order_survives_filesystem_order(
        self, make_cvl: Callable[..., Path], tmp_path: Path
    ) -> None:
        """Index 10 must follow index 9, not sort between 1 and 2.

        Directory listings are lexicographic; word indices are numeric. Sorting the wrong way
        silently reorders every line of ten or more words, which is most of them.
        """
        source = make_cvl(
            tmp_path / "cvl",
            layout={
                "trainset": {
                    "0001": {1: {0: [f"w{index}" for index in range(12)]}},
                }
            },
        )
        _, records = prepare(source, tmp_path / "out")
        assert records[0].text == " ".join(f"w{index}" for index in range(12))

    def test_gapped_word_indices_drop_a_token_without_reordering(
        self, make_cvl: Callable[..., Path], tmp_path: Path
    ) -> None:
        """9.4% of real CVL lines have gaps; the surviving tokens must stay in order."""
        source = make_cvl(tmp_path / "cvl", layout={"trainset": {"0001": {1: {0: ["x"]}}}})
        words = source / "trainset" / "words" / "0001"
        (words / "0001-1-0-0-x.tif").rename(words / "0001-1-0-0-alpha.tif")
        for index, word in ((2, "beta"), (5, "gamma")):
            (words / f"0001-1-0-{index}-{word}.tif").write_bytes(
                (words / "0001-1-0-0-alpha.tif").read_bytes()
            )
        _, records = prepare(source, tmp_path / "out")
        assert records[0].text == "alpha beta gamma"

    def test_line_without_word_images_is_rejected_and_counted(
        self, make_cvl: Callable[..., Path], tmp_path: Path
    ) -> None:
        source = make_cvl(
            tmp_path / "cvl",
            layout={"trainset": {"0001": {1: {0: ["kept"], 1: []}}}},
        )
        adapter, records = prepare(source, tmp_path / "out")
        assert [r.sample_id for r in records] == ["cvl/0001-1-0"]
        assert adapter.counters.count_of(IntegrityCategory.MISSING_TRANSCRIPT) == 1
        assert all(record.text for record in records)

    def test_filename_text_is_nfc_composed(
        self, make_cvl: Callable[..., Path], tmp_path: Path
    ) -> None:
        """macOS hands back decomposed filenames; Linux may not.

        Without NFC the same release would produce different manifests — and different manifest
        fingerprints — on different machines.
        """
        source = make_cvl(
            tmp_path / "cvl",
            layout={"trainset": {"0001": {1: {0: ["Mailüfterl"]}}}},
        )
        _, records = prepare(source, tmp_path / "out")
        assert records[0].text == "Mailüfterl"
        assert "̈" not in records[0].text


class TestPassagesAndLanguage:
    def test_passage_id_comes_from_the_page_index(self, fake_cvl: Path, tmp_path: Path) -> None:
        _, records = prepare(fake_cvl, tmp_path / "out")
        by_id = {record.sample_id: record for record in records}
        assert by_id["cvl/0001-1-0"].passage_id == "p1"
        assert by_id["cvl/0001-2-0"].passage_id == "p2"
        assert by_id["cvl/0052-1-1"].source_page == "0052-1"

    def test_german_is_excluded_and_counted(self, fake_cvl: Path, tmp_path: Path) -> None:
        adapter, records = prepare(fake_cvl, tmp_path / "out")
        assert not [r for r in records if r.passage_id == "p6"]
        assert adapter.counters.count_of(IntegrityCategory.EXCLUDED_LANGUAGE) == 2
        assert {record.language for record in records} == {"en"}

    def test_german_can_be_kept_explicitly(self, fake_cvl: Path, tmp_path: Path) -> None:
        adapter, records = prepare(fake_cvl, tmp_path / "out", include_german=True)
        german = [r for r in records if r.passage_id == "p6"]
        assert len(german) == 2
        assert {record.language for record in german} == {"de"}
        assert adapter.counters.count_of(IntegrityCategory.EXCLUDED_LANGUAGE) == 0

    def test_page_five_is_not_a_passage(self) -> None:
        """The real release numbers pages 1-4 and 6-8; a page 5 would mean a layout change."""
        assert 5 not in CVL_PASSAGES
        assert sorted(CVL_PASSAGES) == [1, 2, 3, 4, 6, 7, 8]
        assert [p.language for p in CVL_PASSAGES.values()].count("de") == 1

    def test_unknown_page_index_is_rejected_not_guessed(
        self, make_cvl: Callable[..., Path], tmp_path: Path
    ) -> None:
        source = make_cvl(
            tmp_path / "cvl",
            layout={"trainset": {"0001": {1: {0: ["ok"]}, 9: {0: ["surprise"]}}}},
        )
        adapter, records = prepare(source, tmp_path / "out")
        assert [r.sample_id for r in records] == ["cvl/0001-1-0"]
        assert adapter.counters.count_of(IntegrityCategory.CORRUPTED_RECORD) == 1


class TestManifestContract:
    def test_manifest_validates(self, fake_cvl: Path, tmp_path: Path) -> None:
        _, records = prepare(fake_cvl, tmp_path / "out")
        report = validate_records(records)
        assert report.is_clean
        assert report.rejected_records == 0

    def test_ids_are_dataset_prefixed(self, fake_cvl: Path, tmp_path: Path) -> None:
        _, records = prepare(fake_cvl, tmp_path / "out")
        assert all(record.sample_id and record.sample_id.startswith("cvl/") for record in records)
        assert all(record.writer_id.startswith("cvl/") for record in records)

    def test_writers_span_both_source_sets(self, fake_cvl: Path, tmp_path: Path) -> None:
        _, records = prepare(fake_cvl, tmp_path / "out")
        assert {record.writer_id for record in records} == {"cvl/0001", "cvl/0052"}

    def test_default_split_is_test(self, fake_cvl: Path, tmp_path: Path) -> None:
        """CVL evaluates; it does not train."""
        _, records = prepare(fake_cvl, tmp_path / "out")
        assert {record.split for record in records} == {"test"}

    def test_image_size_is_recorded(self, fake_cvl: Path, tmp_path: Path) -> None:
        _, records = prepare(fake_cvl, tmp_path / "out")
        assert all(record.width == 240 and record.height == 48 for record in records)

    def test_image_size_can_be_skipped(self, fake_cvl: Path, tmp_path: Path) -> None:
        _, records = prepare(fake_cvl, tmp_path / "out", read_image_size=False)
        assert all(record.width is None and record.height is None for record in records)

    def test_images_are_referenced_in_place(self, fake_cvl: Path, tmp_path: Path) -> None:
        """A 5 GB corpus is never copied into an output directory."""
        out = tmp_path / "out"
        _, records = prepare(fake_cvl, out)
        assert all(Path(record.image).is_file() for record in records)
        assert all(Path(record.image).is_relative_to(fake_cvl) for record in records)
        assert not list(out.rglob("*.tif"))

    def test_prepare_is_deterministic(self, fake_cvl: Path, tmp_path: Path) -> None:
        _, first = prepare(fake_cvl, tmp_path / "a")
        _, second = prepare(fake_cvl, tmp_path / "b")
        assert [r.to_dict() for r in first] == [r.to_dict() for r in second]


class TestProvenance:
    def test_summary_records_exclusions(self, fake_cvl: Path, tmp_path: Path) -> None:
        import json

        out = tmp_path / "out"
        prepare(fake_cvl, out)
        summary = json.loads((out / "cvl_summary.json").read_text())
        assert summary["records_written"] == 6
        assert summary["excluded_language"] == {"de": 2}
        assert summary["writers_per_source_set"] == {"testset": 1, "trainset": 1}
        assert summary["lines_per_passage"] == {"p1": 4, "p2": 2}
        assert summary["lines_per_source_set"] == {"testset": 3, "trainset": 3}

    def test_describe_carries_the_ground_truth_limitation(self) -> None:
        """A limitation recorded nowhere machine-readable is a limitation that gets lost."""
        described = CVLAdapter().describe()
        assert described["known_limitations"] == [TRANSCRIPT_LIMITATION]
        assert "non-commercial" in described["license"]
        assert "never trains a recognizer" in described["role"]

    def test_passage_id_helper(self) -> None:
        assert passage_id(6) == "p6"
