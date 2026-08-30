"""Synthetic dataset adapter.

**Every record is labelled ``dataset="synthetic"``.** Synthetic output is correctness evidence only:
it never enters a reported training set, never appears in a headline table, and never supports a
claim about handwriting.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from glyphmemory.ctc.tokenizer import Charset
from glyphmemory.data.adapters.base import make_sample_id, make_writer_id
from glyphmemory.data.manifest import ManifestRecord, write_manifest
from glyphmemory.data.preprocessing import DEFAULT_MAX_WIDTH
from glyphmemory.data.synthetic.corpus import DEFAULT_MIN_OCCURRENCES, sample_lines
from glyphmemory.data.synthetic.fonts import FontSource, resolve_fonts
from glyphmemory.data.synthetic.render import DEFAULT_HEIGHT, WriterStyle, render_line
from glyphmemory.runtime.logging import get_logger

logger = get_logger("data.adapters.synthetic")

DATASET_NAME = "synthetic"
MANIFEST_FILENAME = "manifest.jsonl"
STYLES_FILENAME = "writer_styles.json"
IMAGE_DIRNAME = "images"


@dataclass(slots=True)
class SyntheticAdapter:
    """Generates a synthetic corpus. Fully determined by its parameters.

    Attributes:
        n_writers: How many synthetic writers to create.
        n_lines: Lines per writer.
        seed: Master seed. Same seed and parameters always produce identical output.
        corpus_mode: ``coverage`` (default) or ``words`` — see
            :func:`~glyphmemory.data.synthetic.corpus.sample_lines`.
        n_passages: Lines are labelled with a rotating ``passage_id`` so passage-disjoint
            support/query splitting is exercisable before CVL arrives.
        split: Value written to every record's ``split`` field. Writer-disjoint splits are applied
            afterwards by :mod:`glyphmemory.data.splits`.
    """

    n_writers: int = 3
    n_lines: int = 4
    seed: int = 1337
    corpus_mode: str = "coverage"
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES
    n_passages: int = 2
    height: int = DEFAULT_HEIGHT
    max_width: int = DEFAULT_MAX_WIDTH
    split: str = "train"
    charset: Charset = field(default_factory=Charset.english_v1)
    fonts: Sequence[FontSource] | None = None

    name: str = DATASET_NAME

    def __post_init__(self) -> None:
        if self.n_writers < 1:
            raise ValueError(f"n_writers must be at least 1, got {self.n_writers}")
        if self.n_lines < 1:
            raise ValueError(f"n_lines must be at least 1, got {self.n_lines}")
        if self.n_passages < 1:
            raise ValueError(f"n_passages must be at least 1, got {self.n_passages}")

    # ------------------------------------------------------------------ generation

    def writer_styles(self) -> list[WriterStyle]:
        """Style vector per writer. Deterministic given ``seed`` and ``n_writers``."""
        fonts = resolve_fonts(self.fonts)
        return [
            WriterStyle.sample(
                random.Random(f"{self.seed}:style:{index}"),
                writer_index=index,
                n_writers=self.n_writers,
                n_fonts=len(fonts),
            )
            for index in range(self.n_writers)
        ]

    def prepare(self, source_dir: Path | None = None, output_dir: Path | None = None) -> Path:
        """Generate images and a manifest under ``output_dir``.

        Args:
            source_dir: Unused. Present to satisfy the
                :class:`~glyphmemory.data.adapters.base.DatasetAdapter` protocol — a synthetic
                corpus has no source to read.
            output_dir: Destination directory. Required.

        Returns:
            Path to the written ``manifest.jsonl``.
        """
        if output_dir is None:
            raise ValueError("output_dir is required.")

        output_dir = Path(output_dir)
        image_dir = output_dir / IMAGE_DIRNAME
        image_dir.mkdir(parents=True, exist_ok=True)

        fonts = resolve_fonts(self.fonts)
        styles = self.writer_styles()
        records: list[ManifestRecord] = []

        for writer_index, style in enumerate(styles):
            writer_local = f"w{writer_index:03d}"
            writer_id = make_writer_id(DATASET_NAME, writer_local)
            font = fonts[style.font_index % len(fonts)]

            text_rng = random.Random(f"{self.seed}:text:{writer_index}")
            lines = sample_lines(
                self.charset.characters,
                n_lines=self.n_lines,
                rng=text_rng,
                mode=self.corpus_mode,
                min_occurrences=self.min_occurrences,
            )

            for line_index, text in enumerate(lines):
                # Jitter is seeded per (writer, line) so a single line is reproducible independently
                # of how many lines precede it.
                jitter_rng = random.Random(f"{self.seed}:jitter:{writer_index}:{line_index}")
                line_style = style.jitter(jitter_rng)

                image = render_line(text, font, line_style, height=self.height)
                local_id = f"{writer_local}-l{line_index:03d}"
                image_path = image_dir / f"{local_id}.png"
                image.save(image_path, format="PNG", optimize=False, compress_level=6)

                records.append(
                    ManifestRecord(
                        image=str(image_path),
                        text=text,
                        writer_id=writer_id,
                        dataset=DATASET_NAME,
                        split=self.split,
                        sample_id=make_sample_id(DATASET_NAME, local_id),
                        passage_id=f"p{line_index % self.n_passages}",
                        language="en",
                        height=image.height,
                        width=image.width,
                    )
                )

        manifest_path = output_dir / MANIFEST_FILENAME
        write_manifest(manifest_path, records)
        self._write_styles(output_dir, fonts, styles)

        logger.info(
            "Generated %d synthetic line(s) for %d writer(s) into %s",
            len(records),
            self.n_writers,
            output_dir,
        )
        self._warn_if_oversized(records)
        return manifest_path

    def _warn_if_oversized(self, records: Sequence[ManifestRecord]) -> None:
        """Warn when generated lines exceed the preprocessing width guard.

        ``coverage`` mode distributes a fixed character budget (charset size x ``min_occurrences``)
        across ``n_lines``, so **fewer lines means wider lines**. At ``n_lines=2`` every line lands
        around 2600 px against a 1600 px guard, and the whole corpus would be flagged as oversized
        downstream. Measured, not assumed.
        """
        oversized = [r for r in records if (r.width or 0) > self.max_width]
        if not oversized:
            return
        widest = max(r.width or 0 for r in oversized)
        logger.warning(
            "%d/%d generated line(s) exceed max_width=%d (widest %d px). Coverage mode packs "
            "~%d characters across %d line(s) per writer, so raise --lines or lower "
            "min_occurrences rather than widening the guard.",
            len(oversized),
            len(records),
            self.max_width,
            widest,
            len(self.charset.characters) * self.min_occurrences,
            self.n_lines,
        )

    def _write_styles(
        self, output_dir: Path, fonts: Sequence[FontSource], styles: Sequence[WriterStyle]
    ) -> None:
        """Record the generating parameters beside the data, so a corpus is explicable."""
        payload = {
            "generator": self.describe(),
            "fonts": [font.describe() for font in fonts],
            "writers": {
                make_writer_id(DATASET_NAME, f"w{index:03d}"): style.describe()
                for index, style in enumerate(styles)
            },
        }
        (output_dir / STYLES_FILENAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def describe(self) -> dict[str, Any]:
        """Provenance for the run record."""
        return {
            "dataset": DATASET_NAME,
            "synthetic": True,
            "n_writers": self.n_writers,
            "n_lines": self.n_lines,
            "seed": self.seed,
            "corpus_mode": self.corpus_mode,
            "min_occurrences": self.min_occurrences,
            "n_passages": self.n_passages,
            "height": self.height,
            "max_width": self.max_width,
            "split": self.split,
            "charset": self.charset.name,
            "charset_fingerprint": self.charset.fingerprint(),
            "note": "Correctness harness only. Never a performance benchmark.",
        }
