"""IAM adapter tests.

Every test runs against a generated fake release (``tests/conftest.py``). No IAM data is committed:
the corpus is licensed for non-commercial research, is 1.5 GB, and CI must never need it.

- the transcript is de-tokenised, so punctuation is not preceded by a space;
- the XML is double-escaped and needs a second unescape;
- the two sources are cross-checked, and a disagreement is a counted rejection;
- writer identity is joined from ``forms.txt`` and never inferred from the line id;
- ``err`` lines are kept by default and counted either way;
- ``#`` marks a struck-out word and is excluded **and counted**.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from glyphmemory.data.adapters.iam import IAMAdapter, passage_id_for
from glyphmemory.data.manifest import read_manifest
from glyphmemory.data.validation import IntegrityCategory, validate_records


def prepare(source: Path, out: Path, **kwargs: object) -> tuple[IAMAdapter, list]:
    adapter = IAMAdapter(**kwargs)  # type: ignore[arg-type]
    manifest = adapter.prepare(source, out)
    return adapter, list(read_manifest(manifest))


def by_id(records: list) -> dict:
    return {record.sample_id: record for record in records}


class TestLayout:
    def test_accepts_release_directory(self, fake_iam: Path) -> None:
        assert IAMAdapter.resolve_root(fake_iam) == fake_iam

    def test_accepts_parent_of_release_directory(
        self, make_iam: Callable[..., Path], tmp_path: Path
    ) -> None:
        base = make_iam(
            tmp_path / "IAMroot",
            forms={"a01-000u": {"writer": "000", "lines": {0: ("ok", "hello")}}},
            nest_release=True,
        )
        assert IAMAdapter.resolve_root(tmp_path / "IAMroot") == base

    def test_rejects_unrelated_directory(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="does not look like an IAM release"):
            IAMAdapter.resolve_root(tmp_path / "empty")

    def test_rejects_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            IAMAdapter.resolve_root(tmp_path / "nope")


class TestTranscripts:
    def test_punctuation_is_not_preceded_by_a_space(self, fake_iam: Path, tmp_path: Path) -> None:
        """The whole reason the transcript comes from the XML rather than ``lines.txt``.

        ``lines.txt`` stores ``support|,|a``; joining its tokens with spaces yields ``'support ,
        a'``. Measuring the component boxes on the real release showed the gap before punctuation is
        0.33x the line's own inter-word gap, so that space is not in the ink and would train the
        recogniser to emit one.
        """
        _, records = prepare(fake_iam, tmp_path / "out")
        text = by_id(records)["iam/a01-003u-00"].text
        assert text == "Though they may gather some Left-wing support, a"
        assert " ," not in text
        assert " ." not in text

    def test_double_escaped_entities_are_fully_resolved(
        self, fake_iam: Path, tmp_path: Path
    ) -> None:
        """IAM stores ``&amp;quot;``; one unescape leaves the literal string ``&quot;``."""
        _, records = prepare(fake_iam, tmp_path / "out")
        text = by_id(records)["iam/a01-000x-00"].text
        assert text == 'He said "no" to the & motion.'
        assert "&quot;" not in text
        assert "&amp;" not in text

    def test_transcripts_encode_under_the_default_charset(
        self, fake_iam: Path, tmp_path: Path
    ) -> None:
        """A transcript that cannot round-trip is a transcript training will reject."""
        from glyphmemory.ctc.tokenizer import Charset, Tokenizer

        tokenizer = Tokenizer(Charset.english_v1())
        _, records = prepare(fake_iam, tmp_path / "out", include_struck_out=True)
        for record in records:
            assert tokenizer.decode(tokenizer.encode(record.text)) == record.text


class TestCrossSourceChecks:
    """The two sources are read independently and required to agree."""

    def test_character_disagreement_is_rejected_and_counted(
        self, make_iam: Callable[..., Path], tmp_path: Path
    ) -> None:
        """Neither source is silently preferred when they conflict."""
        base = make_iam(
            tmp_path / "iam",
            forms={"a01-000u": {"writer": "000", "lines": {0: ("ok", "hello world")}}},
        )
        xml = base / "xml" / "a01-000u.xml"
        xml.write_text(
            xml.read_text(encoding="iso-8859-1").replace("hello world", "goodbye world"),
            encoding="iso-8859-1",
        )
        adapter, records = prepare(base, tmp_path / "out")
        assert records == []
        assert adapter.counters.as_dict()[IntegrityCategory.CORRUPTED_RECORD] == 1

    def test_whitespace_only_difference_is_accepted(self, fake_iam: Path, tmp_path: Path) -> None:
        """The sources differ in whitespace on 8,891 real lines; that is expected, not an error."""
        adapter, records = prepare(fake_iam, tmp_path / "out")
        assert records
        assert adapter.counters.as_dict()[IntegrityCategory.CORRUPTED_RECORD] == 0

    def test_writer_disagreement_is_rejected_and_counted(
        self, make_iam: Callable[..., Path], tmp_path: Path
    ) -> None:
        base = make_iam(
            tmp_path / "iam",
            forms={"a01-000u": {"writer": "000", "lines": {0: ("ok", "hello world")}}},
        )
        xml = base / "xml" / "a01-000u.xml"
        xml.write_text(
            xml.read_text(encoding="iso-8859-1").replace('writer-id="000"', 'writer-id="999"'),
            encoding="iso-8859-1",
        )
        adapter, records = prepare(base, tmp_path / "out")
        assert records == []
        assert adapter.counters.as_dict()[IntegrityCategory.MISSING_WRITER_ID] == 1


class TestWriterJoin:
    def test_writer_comes_from_forms_not_from_the_line_id(
        self, fake_iam: Path, tmp_path: Path
    ) -> None:
        """``a01-000u`` and ``a01-000x`` share an id prefix but are different writers.

        This is the real release's behaviour — ``a01-000u`` and ``a01-003u`` are both writer ``000``
        while ``a01-000x`` is writer ``001`` — and it is why the join is explicit. A prefix rule
        would mis-assign writers with nothing raising.
        """
        _, records = prepare(fake_iam, tmp_path / "out")
        writers = by_id(records)
        assert writers["iam/a01-000u-00"].writer_id == "iam/000"
        assert writers["iam/a01-000x-00"].writer_id == "iam/001"
        assert writers["iam/a01-003u-00"].writer_id == "iam/000"

    def test_line_with_no_form_entry_is_rejected_not_guessed(
        self, make_iam: Callable[..., Path], tmp_path: Path
    ) -> None:
        base = make_iam(
            tmp_path / "iam",
            forms={"a01-000u": {"writer": "000", "lines": {0: ("ok", "hello world")}}},
        )
        forms = base / "ascii" / "forms.txt"
        forms.write_text("#--- forms.txt ---\n", encoding="utf-8")
        adapter, records = prepare(base, tmp_path / "out")
        assert records == []
        assert adapter.counters.as_dict()[IntegrityCategory.MISSING_WRITER_ID] == 1

    def test_source_page_is_the_form_id(self, fake_iam: Path, tmp_path: Path) -> None:
        _, records = prepare(fake_iam, tmp_path / "out")
        assert by_id(records)["iam/a01-000u-00"].source_page == "a01-000u"


class TestSegmentationPolicy:
    def test_err_lines_are_kept_by_default(self, fake_iam: Path, tmp_path: Path) -> None:
        """``err`` marks *word* segmentation, which line recognition does not use.

        The file's own header states a segmentation error should not affect the line's transcription
        or extraction. Dropping them would remove the harder samples and inflate every number that
        follows.
        """
        _, records = prepare(fake_iam, tmp_path / "out")
        assert "iam/a01-000u-01" in by_id(records)

    def test_err_lines_can_be_dropped_and_are_then_counted(
        self, fake_iam: Path, tmp_path: Path
    ) -> None:
        adapter, records = prepare(fake_iam, tmp_path / "out", keep_segmentation_errors=False)
        assert "iam/a01-000u-01" not in by_id(records)
        assert adapter.counters.as_dict()[IntegrityCategory.SEGMENTATION_ERROR] == 1

    def test_both_counts_are_reported_either_way(self, fake_iam: Path, tmp_path: Path) -> None:
        """Whatever the policy, the summary states how many of each there were."""
        import json

        out = tmp_path / "out"
        prepare(fake_iam, out, keep_segmentation_errors=False)
        summary = json.loads((out / "iam_summary.json").read_text())
        assert summary["segmentation_all"] == {"err": 1, "ok": 4}
        assert summary["segmentation_kept"] == {"ok": 3}


class TestStruckOutPolicy:
    def test_struck_out_lines_are_excluded_and_counted(
        self, fake_iam: Path, tmp_path: Path
    ) -> None:
        """``#`` is IAM's marker for a word the writer crossed out.

        Verified by inspecting the real line images: the ink is a scribble, so keeping the line
        teaches the recogniser to emit a glyph for it.
        """
        adapter, records = prepare(fake_iam, tmp_path / "out")
        assert "iam/a01-003u-01" not in by_id(records)
        assert adapter.counters.as_dict()[IntegrityCategory.STRUCK_OUT_TOKEN] == 1

    def test_struck_out_lines_can_be_kept(self, fake_iam: Path, tmp_path: Path) -> None:
        adapter, records = prepare(fake_iam, tmp_path / "out", include_struck_out=True)
        assert "iam/a01-003u-01" in by_id(records)
        assert adapter.counters.as_dict()[IntegrityCategory.STRUCK_OUT_TOKEN] == 0

    def test_hash_inside_a_word_is_not_treated_as_a_marker(
        self, make_iam: Callable[..., Path], tmp_path: Path
    ) -> None:
        """The marker is a standalone token; ``#4`` is ordinary text."""
        base = make_iam(
            tmp_path / "iam",
            forms={"a01-000u": {"writer": "000", "lines": {0: ("ok", "ranked #4 today")}}},
        )
        adapter, records = prepare(base, tmp_path / "out")
        assert len(records) == 1
        assert adapter.counters.as_dict()[IntegrityCategory.STRUCK_OUT_TOKEN] == 0


class TestPassages:
    def test_forms_sharing_a_prompt_share_a_passage_id(
        self, fake_iam: Path, tmp_path: Path
    ) -> None:
        """Two writers copying one LOB prompt must be recognisable as the same source text.

        On the real release 1,539 forms carry 1,280 distinct prompts and one prompt is copied by 17
        forms, so without this a support and a query line could be the same sentence.
        """
        _, records = prepare(fake_iam, tmp_path / "out")
        records_by_id = by_id(records)
        shared = records_by_id["iam/a01-000u-00"].passage_id
        assert shared == records_by_id["iam/a01-000x-00"].passage_id
        assert shared != records_by_id["iam/a01-003u-00"].passage_id

    def test_passage_id_ignores_prompt_line_wrapping(self) -> None:
        assert passage_id_for("a b  c") == passage_id_for(" a\nb c ")

    def test_empty_prompt_has_no_passage_id(self) -> None:
        assert passage_id_for("   ") == ""


class TestManifestIntegrity:
    def test_manifest_passes_phase_01_validation(self, fake_iam: Path, tmp_path: Path) -> None:
        _, records = prepare(fake_iam, tmp_path / "out")
        report = validate_records(records)
        assert report.is_clean, report.counters.as_dict()

    def test_missing_image_is_rejected_and_counted(
        self, make_iam: Callable[..., Path], tmp_path: Path
    ) -> None:
        base = make_iam(
            tmp_path / "iam",
            forms={
                "a01-000u": {
                    "writer": "000",
                    "lines": {0: ("ok", "kept line"), 1: ("ok", "lost line")},
                }
            },
            skip_images=("a01-000u-01",),
        )
        adapter, records = prepare(base, tmp_path / "out")
        assert [r.sample_id for r in records] == ["iam/a01-000u-00"]
        assert adapter.counters.as_dict()[IntegrityCategory.MISSING_IMAGE_FILE] == 1

    def test_image_dimensions_are_recorded(self, fake_iam: Path, tmp_path: Path) -> None:
        """The bucket sampler needs real dimensions, not the manifest's word for it."""
        _, records = prepare(fake_iam, tmp_path / "out")
        assert all(r.width == 400 and r.height == 60 for r in records)

    def test_sample_and_writer_ids_are_dataset_prefixed(
        self, fake_iam: Path, tmp_path: Path
    ) -> None:
        _, records = prepare(fake_iam, tmp_path / "out")
        assert all(r.sample_id.startswith("iam/") for r in records)
        assert all(r.writer_id.startswith("iam/") for r in records)

    def test_describe_records_the_decisions_that_shaped_the_corpus(
        self, fake_iam: Path, tmp_path: Path
    ) -> None:
        adapter, _ = prepare(fake_iam, tmp_path / "out")
        described = adapter.describe()
        assert described["dataset"] == "iam"
        assert "double-escaped" in described["transcript_source"]
        assert "forms.txt" in described["writer_id_source"]
        assert described["include_struck_out"] is False
