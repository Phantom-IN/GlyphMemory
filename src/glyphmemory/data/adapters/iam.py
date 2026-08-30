"""IAM Handwriting Database adapter.

Three findings from measuring the real release drive the design; each is defended in the function
that acts on it:

* the ASCII transcription is word-tokenised, so expanding its ``|`` to a space inserts a space
  before punctuation that is not in the ink (:func:`_read_ascii_lines`, :data:`TOKENISATION_NOTE`)
* the XML carries the de-tokenised text but is **double-escaped** (:func:`_read_xml_forms`)
* ``#`` is a corpus marker for a struck-out word, not a glyph (:data:`STRUCK_OUT_MARKER`)

Writer identity is never derived from the line ID. It is joined from ``forms.txt`` and cross-checked
against the XML, because a wrong join corrupts every writer-disjoint split built on top of it and
nothing downstream would raise.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from glyphmemory.data.adapters.base import make_sample_id, make_writer_id
from glyphmemory.data.manifest import ManifestRecord, write_manifest
from glyphmemory.data.validation import IntegrityCategory, IntegrityCounters, IntegrityIssue
from glyphmemory.runtime.logging import get_logger

logger = get_logger("data.adapters.iam")

DATASET_NAME = "iam"
MANIFEST_FILENAME = "manifest.jsonl"
SUMMARY_FILENAME = "iam_summary.json"

#: Prefix for the passage identifier derived from a form's machine-printed prompt.
PASSAGE_PREFIX = "lob"

#: ``a01-000u-00`` -> form ``a01-000u``, group ``a01``.
LINE_ID_PATTERN = re.compile(r"^(?P<form>(?P<group>[a-z]\d+)-[\w-]+?)-(?P<line>\d+)$")

#: Word token IAM uses for a word the writer struck out. Verified by inspecting the line images: the
#: ink is a scribble, not a ``#`` glyph. Keeping it teaches the recogniser to emit ``#`` for
#: crossed-out ink, so these lines are excluded by default and counted.
STRUCK_OUT_MARKER = "#"

#: ``lines.txt`` marks *word* segmentation quality. Its own header states that a segmentation error
#: "should not affect the transcription and extraction of the whole line negatively", and
#: GlyphMemory never uses word boxes — so ``err`` lines are kept. They are the harder samples;
#: dropping them would inflate every number that follows.
SEGMENTATION_NOTE = (
    "lines.txt 'ok'/'err' marks word-segmentation quality, which line-level recognition does "
    "not use; per the file's own header it does not indicate a bad line image or transcript. "
    "Both counts are reported and err lines are kept by default."
)

#: Why the transcript comes from the XML rather than the ASCII index.
TOKENISATION_NOTE = (
    "Transcripts come from the XML line/@text attribute, not from joining lines.txt word "
    "tokens with spaces. The two agree character-for-character on all 13,353 lines but "
    "differ in whitespace on 8,891 of them, because the ASCII form is word-tokenised: "
    "joining it yields 'Exchange .' where the XML has 'Exchange.'. Measuring the component "
    "bounding boxes shows the gap before punctuation is 0.33x the line's own median "
    "inter-word gap (p10 negative, i.e. overlapping) against 1.00x between ordinary words, "
    "so the tokenised spacing is not in the ink. Every line is still cross-checked against "
    "lines.txt ignoring whitespace, and a disagreement is a counted rejection."
)


#: Why IAM lines carry a passage identifier at all.
PASSAGE_NOTE = (
    "IAM forms are copies of LOB corpus prompts, which the XML preserves as "
    "machine-print-line elements. 1,539 forms carry 1,280 distinct prompts, and one prompt is "
    "copied by as many as 17 forms - so two writers can transcribe the same source text. "
    "passage_id is the hash of that prompt, which makes the sharing measurable by "
    "'data stats' and lets support/query pools be drawn passage-disjoint as they are for CVL. "
    "Without it, same-text support and query lines would be indistinguishable from genuine "
    "few-shot adaptation."
)


def passage_id_for(prompt: str) -> str:
    """Stable passage identifier for a machine-printed prompt.

    Whitespace is collapsed before hashing so that a prompt is identified by its text rather than by
    how the form happened to wrap it across lines.
    """
    normalised = " ".join(prompt.split())
    if not normalised:
        return ""
    digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:8]
    return f"{PASSAGE_PREFIX}-{digest}"


@dataclass(frozen=True, slots=True)
class IAMLine:
    """One line as the ASCII index describes it."""

    line_id: str
    segmentation: str
    text: str


@dataclass(frozen=True, slots=True)
class IAMForm:
    """One form as ``forms.txt`` describes it."""

    form_id: str
    writer_id: str


def _strip_whitespace(text: str) -> str:
    """Text with all whitespace removed, for whitespace-insensitive comparison."""
    return "".join(text.split())


@dataclass(slots=True)
class IAMAdapter:
    """Converts an IAM release directory into a GlyphMemory manifest.

    Attributes:
        split: Value written to every record's ``split`` field. A writer-disjoint split is applied
            afterwards by :mod:`glyphmemory.data.splits`.
        include_struck_out: Keep lines whose transcript contains :data:`STRUCK_OUT_MARKER`. Off by
            default; exclusions are counted, never silent.
        keep_segmentation_errors: Keep lines flagged ``err``. On by default — see
            :data:`SEGMENTATION_NOTE`.
        read_image_size: Record each line image's pixel size in the manifest. Reads PNG headers only
            (no decode), so width-aware bucketing works without opening every image again.
    """

    split: str = "train"
    include_struck_out: bool = False
    keep_segmentation_errors: bool = True
    read_image_size: bool = True
    counters: IntegrityCounters = field(default_factory=IntegrityCounters)

    name: str = DATASET_NAME

    # ------------------------------------------------------------------ layout

    @staticmethod
    def resolve_root(source_dir: str | Path) -> Path:
        """Find the release root, accepting either it or a directory containing it."""
        root = Path(source_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"IAM source directory does not exist: {root}")
        for candidate in (root, root / "IAM", root / "iam"):
            if (candidate / "ascii" / "lines.txt").is_file() and (candidate / "xml").is_dir():
                return candidate
        raise FileNotFoundError(
            f"{root} does not look like an IAM release: expected 'ascii/lines.txt' and an "
            "'xml/' directory, either directly or under an 'IAM/' subdirectory."
        )

    # ------------------------------------------------------------------ reading

    def _read_ascii_lines(self, root: Path) -> dict[str, IAMLine]:
        """Parse ``ascii/lines.txt``.

        Format, from the file's own header::

            a01-000u-00 ok 154 19 408 746 1663 91 A|MOVE|to|stop|Mr.|Gaitskell|from
            <id>       <seg> <gray> <ncomp> <x> <y> <w> <h> <transcription>

        The transcription is kept **tokenised** here. It is used only to verify the XML text, never
        as the transcript itself: a word token may itself contain a space (36 lines, e.g. ``M Ps``),
        so the token stream cannot be recovered by splitting on whitespace and the ``|`` boundaries
        carry no information the XML does not already have.
        """
        path = root / "ascii" / "lines.txt"
        lines: dict[str, IAMLine] = {}
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if raw.startswith("#") or not raw.strip():
                continue
            fields = raw.split(" ", 8)
            if len(fields) < 9:
                self.counters.record(
                    IntegrityIssue(
                        IntegrityCategory.CORRUPTED_RECORD,
                        None,
                        f"{path}:{number}",
                        f"expected 9 space-separated fields, found {len(fields)}",
                    )
                )
                continue
            line_id, segmentation, transcription = fields[0], fields[1], fields[8]
            lines[line_id] = IAMLine(line_id, segmentation, transcription.replace("|", " "))
        return lines

    def _read_forms(self, root: Path) -> dict[str, IAMForm]:
        """Parse ``ascii/forms.txt`` into form -> writer.

        This is the only source of writer identity. It is **not** derivable from the line ID:
        ``a01-000u`` and ``a01-003u`` are both writer ``000`` while ``a01-000x`` is writer ``001``,
        so any rule based on the ID prefix would silently mis-assign writers.
        """
        path = root / "ascii" / "forms.txt"
        forms: dict[str, IAMForm] = {}
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if raw.startswith("#") or not raw.strip():
                continue
            fields = raw.split()
            if len(fields) < 2:
                self.counters.record(
                    IntegrityIssue(
                        IntegrityCategory.CORRUPTED_RECORD,
                        None,
                        f"{path}:{number}",
                        f"expected at least 2 fields (form id, writer id), found {len(fields)}",
                    )
                )
                continue
            forms[fields[0]] = IAMForm(fields[0], fields[1])
        return forms

    def _read_xml_forms(self, root: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        """Parse ``xml/*.xml`` into ``(line_id -> text, form_id -> writer, form_id -> passage)``.

        **The XML is double-escaped.** The files literally contain ``&amp;quot;``, which an XML
        parser correctly resolves to the *string* ``&quot;`` rather than to ``"``. One further
        :func:`html.unescape` is therefore applied, and it is not a guess: before it, the XML text
        disagrees with ``lines.txt`` on 963 lines; after it, on **0 of 13,353**. The cross-check in
        :meth:`prepare` re-verifies that on every run.
        """
        texts: dict[str, str] = {}
        writers: dict[str, str] = {}
        passages: dict[str, str] = {}
        for path in sorted((root / "xml").glob("*.xml")):
            try:
                form = ET.parse(path).getroot()
            except ET.ParseError as exc:
                self.counters.record(
                    IntegrityIssue(
                        IntegrityCategory.CORRUPTED_RECORD, None, str(path), f"invalid XML: {exc}"
                    )
                )
                continue
            form_id, writer_id = form.get("id"), form.get("writer-id")
            if form_id and writer_id:
                writers[form_id] = writer_id
            if form_id:
                prompt = " ".join(
                    html.unescape(element.get("text", ""))
                    for element in form.iter("machine-print-line")
                )
                passages[form_id] = passage_id_for(prompt)
            for line in form.iter("line"):
                line_id, text = line.get("id"), line.get("text")
                if line_id is None or text is None:
                    self.counters.record(
                        IntegrityIssue(
                            IntegrityCategory.CORRUPTED_RECORD,
                            line_id,
                            str(path),
                            "line element is missing an 'id' or 'text' attribute",
                        )
                    )
                    continue
                texts[line_id] = html.unescape(text)
        return texts, writers, passages

    def _image_path(self, root: Path, line_id: str, form_id: str, group: str) -> Path:
        """``lines/a01/a01-000u/a01-000u-00.png``."""
        return root / "lines" / group / form_id / f"{line_id}.png"

    def _image_size(self, path: Path, sample_id: str) -> tuple[int, int] | None:
        """``(width, height)`` from the header, or ``None`` if unreadable (counted)."""
        try:
            with Image.open(path) as handle:
                return handle.size
        except (OSError, UnidentifiedImageError) as exc:
            self.counters.record(
                IntegrityIssue(
                    IntegrityCategory.UNREADABLE_IMAGE,
                    sample_id,
                    str(path),
                    f"{type(exc).__name__}: {exc}",
                )
            )
            return None

    # ------------------------------------------------------------------ prepare

    def prepare(self, source_dir: str | Path, output_dir: str | Path) -> Path:
        """Convert the release at ``source_dir`` into ``output_dir/manifest.jsonl``.

        Images are referenced in place — IAM is never copied or committed.
        """
        root = self.resolve_root(source_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ascii_lines = self._read_ascii_lines(root)
        forms = self._read_forms(root)
        xml_texts, xml_writers, xml_passages = self._read_xml_forms(root)

        records: list[ManifestRecord] = []
        segmentation_counts: Counter[str] = Counter()
        kept_segmentation: Counter[str] = Counter()
        lines_per_writer: Counter[str] = Counter()
        lines_per_passage: Counter[str] = Counter()
        struck_out = text_disagreements = writer_disagreements = 0

        for line_id, ascii_line in sorted(ascii_lines.items()):
            sample_id = make_sample_id(DATASET_NAME, line_id)
            segmentation_counts[ascii_line.segmentation] += 1

            match = LINE_ID_PATTERN.match(line_id)
            if match is None:
                self.counters.record(
                    IntegrityIssue(
                        IntegrityCategory.CORRUPTED_RECORD,
                        sample_id,
                        str(root / "ascii" / "lines.txt"),
                        f"line id {line_id!r} does not match '<group>-<form>-<line>'",
                    )
                )
                continue
            form_id, group = match["form"], match["group"]
            image_path = self._image_path(root, line_id, form_id, group)

            # ---- transcript: XML text, verified against the ASCII index
            text = xml_texts.get(line_id)
            if text is None:
                self.counters.record(
                    IntegrityIssue(
                        IntegrityCategory.MISSING_TRANSCRIPT,
                        sample_id,
                        str(image_path),
                        "line is in ascii/lines.txt but has no matching XML line element, so "
                        "its de-tokenised transcript is unavailable",
                    )
                )
                continue
            if _strip_whitespace(text) != _strip_whitespace(ascii_line.text):
                text_disagreements += 1
                self.counters.record(
                    IntegrityIssue(
                        IntegrityCategory.CORRUPTED_RECORD,
                        sample_id,
                        str(image_path),
                        f"XML text and ascii/lines.txt disagree on characters (not merely "
                        f"whitespace): xml={text!r} ascii={ascii_line.text!r}",
                    )
                )
                continue
            if not text.strip():
                self.counters.record(
                    IntegrityIssue(
                        IntegrityCategory.MISSING_TRANSCRIPT,
                        sample_id,
                        str(image_path),
                        "transcript is empty",
                    )
                )
                continue

            # ---- writer: forms.txt, verified against the XML
            form = forms.get(form_id)
            if form is None:
                self.counters.record(
                    IntegrityIssue(
                        IntegrityCategory.MISSING_WRITER_ID,
                        sample_id,
                        str(image_path),
                        f"form {form_id!r} has no entry in ascii/forms.txt, so this line's "
                        "writer is unknown; it is not guessed from the line id",
                    )
                )
                continue
            xml_writer = xml_writers.get(form_id)
            if xml_writer is not None and xml_writer != form.writer_id:
                writer_disagreements += 1
                self.counters.record(
                    IntegrityIssue(
                        IntegrityCategory.MISSING_WRITER_ID,
                        sample_id,
                        str(image_path),
                        f"forms.txt says writer {form.writer_id!r} but the XML says "
                        f"{xml_writer!r}; refusing to guess which is right because a wrong "
                        "writer id silently corrupts every writer-disjoint split",
                    )
                )
                continue

            # ---- policy exclusions
            if STRUCK_OUT_MARKER in text.split() and not self.include_struck_out:
                struck_out += 1
                self.counters.record(
                    IntegrityIssue(
                        IntegrityCategory.STRUCK_OUT_TOKEN,
                        sample_id,
                        str(image_path),
                        f"transcript contains the {STRUCK_OUT_MARKER!r} marker, which IAM uses "
                        "for a word the writer struck out; the ink is a scribble, so keeping "
                        "the line would train the recogniser to emit a glyph for it. Pass "
                        "include_struck_out=True to keep it.",
                    )
                )
                continue
            if ascii_line.segmentation == "err" and not self.keep_segmentation_errors:
                self.counters.record(
                    IntegrityIssue(
                        IntegrityCategory.SEGMENTATION_ERROR,
                        sample_id,
                        str(image_path),
                        "line is flagged 'err' in lines.txt and keep_segmentation_errors is "
                        f"off. {SEGMENTATION_NOTE}",
                    )
                )
                continue

            if not image_path.is_file():
                self.counters.record(
                    IntegrityIssue(
                        IntegrityCategory.MISSING_IMAGE_FILE,
                        sample_id,
                        str(image_path),
                        "line image file does not exist",
                    )
                )
                continue

            size = self._image_size(image_path, sample_id) if self.read_image_size else None
            if self.read_image_size and size is None:
                continue

            writer_id = make_writer_id(DATASET_NAME, form.writer_id)
            passage = xml_passages.get(form_id) or None
            if passage is None:
                self.counters.record(
                    IntegrityIssue(
                        IntegrityCategory.CORRUPTED_RECORD,
                        sample_id,
                        str(image_path),
                        f"form {form_id!r} has no machine-printed prompt, so its passage "
                        "cannot be identified and passage-disjoint sampling would silently "
                        "treat it as unique",
                    )
                )
                continue
            records.append(
                ManifestRecord(
                    image=str(image_path),
                    text=text,
                    writer_id=writer_id,
                    dataset=DATASET_NAME,
                    split=self.split,
                    sample_id=sample_id,
                    source_page=form_id,
                    passage_id=passage,
                    language="en",
                    width=size[0] if size else None,
                    height=size[1] if size else None,
                )
            )
            kept_segmentation[ascii_line.segmentation] += 1
            lines_per_writer[writer_id] += 1
            lines_per_passage[passage] += 1

        manifest_path = output_dir / MANIFEST_FILENAME
        write_manifest(manifest_path, records)

        summary = {
            "adapter": self.describe(),
            "source_root": str(root),
            "lines_in_ascii_index": len(ascii_lines),
            "forms_in_ascii_index": len(forms),
            "records_written": len(records),
            "writers": len(lines_per_writer),
            "passages": len(lines_per_passage),
            "max_forms_sharing_a_passage": max(
                Counter(
                    passage
                    for passage, _form in {
                        (record.passage_id, record.source_page)
                        for record in records
                        if record.passage_id and record.source_page
                    }
                ).values(),
                default=0,
            ),
            "segmentation_all": dict(sorted(segmentation_counts.items())),
            "segmentation_kept": dict(sorted(kept_segmentation.items())),
            "struck_out_excluded": struck_out,
            "cross_source_disagreements": {
                "text": text_disagreements,
                "writer_id": writer_disagreements,
            },
            "integrity": self.counters.as_dict(),
        }
        (output_dir / SUMMARY_FILENAME).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        logger.info(
            "IAM: %d line(s) from %d writer(s); segmentation kept %s; %d struck-out excluded; "
            "%d rejected.",
            len(records),
            len(lines_per_writer),
            dict(sorted(kept_segmentation.items())),
            struck_out,
            self.counters.total,
        )
        if text_disagreements or writer_disagreements:
            logger.warning(
                "IAM cross-source check failed on %d transcript(s) and %d writer id(s). "
                "Both were 0 on release 3.0; a non-zero count means the release on disk "
                "differs from the one this adapter was measured against.",
                text_disagreements,
                writer_disagreements,
            )
        return manifest_path

    def describe(self) -> dict[str, Any]:
        """Provenance for the run record."""
        return {
            "dataset": DATASET_NAME,
            "split": self.split,
            "include_struck_out": self.include_struck_out,
            "keep_segmentation_errors": self.keep_segmentation_errors,
            "read_image_size": self.read_image_size,
            "transcript_source": "xml line/@text, unescaped twice (the XML is double-escaped)",
            "passage_source": "sha256 of the form's machine-printed LOB prompt, first 8 hex",
            "passage_note": PASSAGE_NOTE,
            "transcript_verification": (
                "every line cross-checked against ascii/lines.txt ignoring whitespace; a "
                "character-level disagreement is a counted rejection, not a silent preference"
            ),
            "writer_id_source": "ascii/forms.txt, joined on form id",
            "writer_id_verification": "cross-checked against xml form/@writer-id",
            "text_normalization_applied": "none beyond XML entity resolution",
            "tokenisation_note": TOKENISATION_NOTE,
            "segmentation_note": SEGMENTATION_NOTE,
            "role": "trains the generic recogniser; the standard benchmark for GM-Base",
            "license": (
                "IAM Handwriting Database is free for non-commercial research use. Cite "
                "Marti and Bunke, IJDAR 2002. Not redistributed by this repository."
            ),
        }
