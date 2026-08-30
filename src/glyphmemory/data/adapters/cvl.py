"""CVL Database adapter.

Converts the CVL Database 1.1 (TU Wien Computer Vision Lab) into the internal manifest.

**CVL is an evaluation corpus, never a training corpus**. Every writer copies the same small passage
set, so a recognizer trained on it memorizes sentences instead of learning handwriting. Records are
therefore emitted with ``split="test"`` by default, and any model trained on them is a throwaway
diagnostic that must be named ``diag_``.

On-disk layout, verified against the real release rather than assumed::

    cvl-database-1-1/
      trainset/          27 writers x 7 pages     (CVL's own writer-identification split)
        lines/0001/0001-1-0.tif                   writer-page-line
        words/0001/0001-1-0-0-Imagine.tif         writer-page-line-word-TEXT
        pages/0001-1.tif
        xml/0001-1_attributes.xml
      testset/          283 writers x 5 pages

The two sets are writer-disjoint (train IDs 0001-0050, test IDs 0052-1139), but that is *CVL's*
split for writer identification, not ours. GlyphMemory derives its own writer splits from
:mod:`glyphmemory.data.splits`; both sets are ingested as one corpus.

Three properties of this corpus drive the whole adapter:

**1. The page index is the passage identity.** Page indices run ``1,2,3,4,6,7,8`` — there is no page
5 — and each index denotes the same source text for every writer. Measured across all 310 writers,
within-page vocabulary agreement has median Jaccard 1.00 while cross-page agreement is at most 0.10.

**2. Transcriptions live in the word image filenames, not the XML.** The ``*_attributes.xml`` files
carry geometry only — no ``TextEquiv``, no ``Unicode``, no text of any kind — and their region IDs
collide between the printed sample block and the handwritten block. The word filenames are the only
transcription CVL ships.

**3. Those filenames are punctuation-free.** See :data:`TRANSCRIPT_LIMITATION`. This is the single
most important thing to know about CVL as HTR ground truth and it is repeated in the adapter's
warning log, its :meth:`CVLAdapter.describe`, and the summary written beside every manifest, because
a limitation recorded in only one place is a limitation that gets forgotten.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from glyphmemory.data.adapters.base import make_sample_id, make_writer_id
from glyphmemory.data.manifest import ManifestRecord, write_manifest
from glyphmemory.data.validation import IntegrityCategory, IntegrityCounters, IntegrityIssue
from glyphmemory.runtime.logging import get_logger

logger = get_logger("data.adapters.cvl")

DATASET_NAME = "cvl"
MANIFEST_FILENAME = "manifest.jsonl"
SUMMARY_FILENAME = "cvl_summary.json"

#: Directory name of the release as distributed on Zenodo (record 1492267).
RELEASE_DIRNAME = "cvl-database-1-1"

#: CVL's own writer-identification sets. Both are ingested; our splits are made separately.
SOURCE_SETS: tuple[str, ...] = ("trainset", "testset")

#: ``{writer}-{page}-{line}.tif``
LINE_PATTERN = re.compile(r"^(?P<writer>\d+)-(?P<page>\d+)-(?P<line>\d+)\.tif$", re.IGNORECASE)

#: ``{writer}-{page}-{line}-{word}-{TEXT}.tif``. ``text`` is greedy-free: it may contain hyphens, so
#: anchoring on the extension is what keeps the split unambiguous.
WORD_PATTERN = re.compile(
    r"^(?P<writer>\d+)-(?P<page>\d+)-(?P<line>\d+)-(?P<word>\d+)-(?P<text>.*)\.tif$",
    re.IGNORECASE,
)

TRANSCRIPT_LIMITATION = (
    "CVL ships no line-level transcription. Line text is reconstructed by joining the word "
    "image filenames in word-index order, and that reconstruction is a LOWER BOUND on the "
    "ink for two measured reasons. (1) The filenames carry no sentence punctuation: across "
    "all 13,473 line images the reconstructed inventory contains no '.', ',', ';', ':', "
    "'?', '!' or quotation marks, only the intra-word apostrophe and hyphen - while the ink "
    "plainly does contain them. (2) A small tail of lines carries ink no word image covers: "
    "summed word width over line width has median 0.82 but a p1 of 0.60, and the low tail "
    "contains lines whose trailing words are absent from the transcript entirely. "
    "CVL absolute CER is therefore inflated by systematic ground-truth omission and is NOT "
    "comparable with IAM absolute CER. Adaptation *deltas* measured on a fixed query pool "
    "are far less affected, because the omission is constant across conditions - but the "
    "effect is not zero and must be stated wherever a CVL number appears."
)


@dataclass(frozen=True, slots=True)
class CVLPassage:
    """One of the seven fixed source texts every CVL writer copies."""

    page: int
    title: str
    language: str


#: Page index -> passage. Titles are from the release's own ``readme.txt``; languages were confirmed
#: by reading the reconstructed text. Page 5 does not exist in the release.
CVL_PASSAGES: dict[int, CVLPassage] = {
    1: CVLPassage(1, "Edwin A. Abbott - Flatland", "en"),
    2: CVLPassage(2, "William Shakespeare - Macbeth", "en"),
    3: CVLPassage(3, "Wikipedia - Mailufterl", "en"),
    4: CVLPassage(4, "Charles Darwin - Origin of Species", "en"),
    6: CVLPassage(6, "Johann Wolfgang von Goethe - Faust", "de"),
    7: CVLPassage(7, "Oscar Wilde - The Picture of Dorian Gray", "en"),
    8: CVLPassage(8, "Edgar Allan Poe - The Fall of the House of Usher", "en"),
}


def passage_id(page: int) -> str:
    """Manifest ``passage_id`` for a page index."""
    return f"p{page}"


@dataclass(slots=True)
class CVLAdapter:
    """Converts a CVL release directory into a GlyphMemory manifest.

    Attributes:
        split: Value written to every record's ``split`` field. Defaults to ``test`` because CVL's
            job is evaluation; a writer-disjoint split is applied afterwards by
            :mod:`glyphmemory.data.splits` if a diagnostic split is needed.
        include_german: Keep the German passage (page 6). Exclusions are **counted**, never silent.
        source_sets: Which of CVL's own sets to ingest.
        read_image_size: Record each line image's pixel size in the manifest. Reads TIFF headers
            only (no decode), so the cost is small and the widths let width-aware bucketing work
            without opening every image again.
    """

    split: str = "test"
    include_german: bool = False
    source_sets: tuple[str, ...] = SOURCE_SETS
    read_image_size: bool = True
    counters: IntegrityCounters = field(default_factory=IntegrityCounters)

    name: str = DATASET_NAME

    # ------------------------------------------------------------------ layout

    @staticmethod
    def resolve_root(source_dir: str | Path) -> Path:
        """Find the release root, accepting either it or a directory containing it.

        Both of these work, because which one a user points at is a coin flip::

            datasets/CVL                      (contains cvl-database-1-1/)
            datasets/CVL/cvl-database-1-1
        """
        root = Path(source_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"CVL source directory does not exist: {root}")
        if any((root / name).is_dir() for name in SOURCE_SETS):
            return root
        nested = root / RELEASE_DIRNAME
        if any((nested / name).is_dir() for name in SOURCE_SETS):
            return nested
        raise FileNotFoundError(
            f"{root} does not look like a CVL release: expected a subdirectory named one of "
            f"{list(SOURCE_SETS)}, either directly or under {RELEASE_DIRNAME!r}."
        )

    # ------------------------------------------------------------------ reading

    def _read_transcripts(self, set_dir: Path) -> dict[tuple[str, int, int], str]:
        """Reconstruct line text from word image filenames.

        Word indices are frequently non-contiguous — 9.4% of lines have at least one gap, 1,719
        slots in total. A gap is a token CVL segmented but did not export. Inspecting examples
        against their line images shows punctuation in those positions, but that is a spot check,
        not a proof that no gap ever drops a word.

        Sorting by index and joining preserves reading order: verified by reading line images across
        writers, pages and both languages, so a gap drops a token rather than reordering the line.
        """
        words_dir = set_dir / "words"
        if not words_dir.is_dir():
            return {}

        collected: dict[tuple[str, int, int], dict[int, str]] = defaultdict(dict)
        for writer_dir in sorted(words_dir.iterdir()):
            if not writer_dir.is_dir():
                continue
            for entry in writer_dir.iterdir():
                match = WORD_PATTERN.match(entry.name)
                if match is None:
                    self.counters.record(
                        IntegrityIssue(
                            IntegrityCategory.CORRUPTED_RECORD,
                            None,
                            str(entry),
                            "word filename does not match "
                            "'{writer}-{page}-{line}-{word}-{text}.tif'",
                        )
                    )
                    continue
                key = (match["writer"], int(match["page"]), int(match["line"]))
                collected[key][int(match["word"])] = match["text"]

        # NFC composition only. The macOS filesystem hands back decomposed forms (u + U+0308) where
        # Linux may hand back precomposed ones; without this the same release would produce
        # different manifests on different platforms. It is a lossless decoding of the filename, not
        # a normalization of the transcript.
        return {
            key: unicodedata.normalize("NFC", " ".join(text for _, text in sorted(words.items())))
            for key, words in collected.items()
        }

    def _iter_line_images(self, set_dir: Path) -> list[tuple[tuple[str, int, int], Path]]:
        """Every line image in a set, in deterministic order."""
        lines_dir = set_dir / "lines"
        if not lines_dir.is_dir():
            return []

        found: list[tuple[tuple[str, int, int], Path]] = []
        for writer_dir in sorted(lines_dir.iterdir()):
            if not writer_dir.is_dir():
                continue
            for entry in sorted(writer_dir.iterdir()):
                match = LINE_PATTERN.match(entry.name)
                if match is None:
                    self.counters.record(
                        IntegrityIssue(
                            IntegrityCategory.CORRUPTED_RECORD,
                            None,
                            str(entry),
                            "line filename does not match '{writer}-{page}-{line}.tif'",
                        )
                    )
                    continue
                key = (match["writer"], int(match["page"]), int(match["line"]))
                found.append((key, entry))
        return found

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

        Images are referenced in place — CVL is 5 GB and is never copied or committed.
        """
        root = self.resolve_root(source_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        records: list[ManifestRecord] = []
        per_set: Counter[str] = Counter()
        per_passage: Counter[str] = Counter()
        writers_per_set: dict[str, set[str]] = defaultdict(set)
        excluded_language: Counter[str] = Counter()
        unknown_pages: Counter[int] = Counter()

        for set_name in self.source_sets:
            set_dir = root / set_name
            if not set_dir.is_dir():
                logger.warning("CVL set %r not present under %s; skipping.", set_name, root)
                continue

            transcripts = self._read_transcripts(set_dir)

            for (writer, page, line), image_path in self._iter_line_images(set_dir):
                local_id = f"{writer}-{page}-{line}"
                sample_id = make_sample_id(DATASET_NAME, local_id)

                passage = CVL_PASSAGES.get(page)
                if passage is None:
                    unknown_pages[page] += 1
                    self.counters.record(
                        IntegrityIssue(
                            IntegrityCategory.CORRUPTED_RECORD,
                            sample_id,
                            str(image_path),
                            f"page index {page} is not one of the known CVL passages "
                            f"{sorted(CVL_PASSAGES)}",
                        )
                    )
                    continue

                if passage.language == "de" and not self.include_german:
                    excluded_language[passage.language] += 1
                    self.counters.record(
                        IntegrityIssue(
                            IntegrityCategory.EXCLUDED_LANGUAGE,
                            sample_id,
                            str(image_path),
                            f"passage {passage_id(page)} ({passage.title}) is "
                            f"{passage.language}; the English V1 vocabulary excludes it "
                            "Pass include_german=True to keep it.",
                        )
                    )
                    continue

                text = transcripts.get((writer, page, line), "")
                if not text.strip():
                    self.counters.record(
                        IntegrityIssue(
                            IntegrityCategory.MISSING_TRANSCRIPT,
                            sample_id,
                            str(image_path),
                            "no word images exist for this line, so CVL provides no "
                            "transcription for it",
                        )
                    )
                    continue

                size = self._image_size(image_path, sample_id) if self.read_image_size else None
                if self.read_image_size and size is None:
                    continue

                records.append(
                    ManifestRecord(
                        image=str(image_path),
                        text=text,
                        writer_id=make_writer_id(DATASET_NAME, writer),
                        dataset=DATASET_NAME,
                        split=self.split,
                        sample_id=sample_id,
                        source_page=f"{writer}-{page}",
                        passage_id=passage_id(page),
                        language=passage.language,
                        width=size[0] if size else None,
                        height=size[1] if size else None,
                    )
                )
                per_set[set_name] += 1
                per_passage[passage_id(page)] += 1
                writers_per_set[set_name].add(writer)

        manifest_path = output_dir / MANIFEST_FILENAME
        write_manifest(manifest_path, records)

        summary = {
            "adapter": self.describe(),
            "source_root": str(root),
            "lines_per_source_set": dict(per_set),
            "writers_per_source_set": {k: len(v) for k, v in sorted(writers_per_set.items())},
            "lines_per_passage": dict(sorted(per_passage.items())),
            "excluded_language": dict(excluded_language),
            "unknown_page_indices": dict(unknown_pages),
            "integrity": self.counters.as_dict(),
            "records_written": len(records),
        }
        (output_dir / SUMMARY_FILENAME).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        logger.info(
            "CVL: %d line(s) from %d writer(s) across %d passage(s); %d excluded, %d rejected.",
            len(records),
            len({r.writer_id for r in records}),
            len(per_passage),
            sum(excluded_language.values()),
            self.counters.total - sum(excluded_language.values()),
        )
        logger.warning("CVL ground-truth limitation: %s", TRANSCRIPT_LIMITATION)
        return manifest_path

    def describe(self) -> dict[str, Any]:
        """Provenance for the run record."""
        return {
            "dataset": DATASET_NAME,
            "release": RELEASE_DIRNAME,
            "source_sets": list(self.source_sets),
            "split": self.split,
            "include_german": self.include_german,
            "read_image_size": self.read_image_size,
            "transcript_source": "word image filenames, joined in word-index order",
            "text_normalization_applied": "unicode NFC composition only",
            "passages": {
                passage_id(page): {"title": p.title, "language": p.language}
                for page, p in sorted(CVL_PASSAGES.items())
            },
            "role": "held-out external personalization benchmark; never trains a recognizer",
            "known_limitations": [TRANSCRIPT_LIMITATION],
            "license": (
                "CC BY-NC 4.0 - non-commercial research only. Cite Kleber, Fiel, Diem and "
                "Sablatnig, ICDAR 2013. Not redistributed by this repository."
            ),
        }
