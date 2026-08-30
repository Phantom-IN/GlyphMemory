"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from glyphmemory.data.adapters.synthetic import SyntheticAdapter
from glyphmemory.data.manifest import ManifestRecord, read_manifest

SYNTHETIC_WRITERS = 3
SYNTHETIC_LINES = 4
SYNTHETIC_SEED = 20260817


@dataclass(frozen=True)
class SyntheticCorpus:
    """A generated corpus plus the adapter that produced it."""

    root: Path
    manifest_path: Path
    records: tuple[ManifestRecord, ...]
    adapter: SyntheticAdapter

    @property
    def writers(self) -> tuple[str, ...]:
        return tuple(sorted({record.writer_id for record in self.records}))

    def records_for(self, writer_id: str) -> tuple[ManifestRecord, ...]:
        return tuple(r for r in self.records if r.writer_id == writer_id)


@pytest.fixture(scope="session")
def synthetic_corpus(tmp_path_factory: pytest.TempPathFactory) -> SyntheticCorpus:
    """A small synthetic corpus shared across the session."""
    root = tmp_path_factory.mktemp("synthetic")
    adapter = SyntheticAdapter(
        n_writers=SYNTHETIC_WRITERS, n_lines=SYNTHETIC_LINES, seed=SYNTHETIC_SEED
    )
    manifest_path = adapter.prepare(output_dir=root)
    return SyntheticCorpus(
        root=root,
        manifest_path=manifest_path,
        records=tuple(read_manifest(manifest_path)),
        adapter=adapter,
    )


# --------------------------------------------------------------------------- fake CVL


def write_fake_cvl(
    root: Path,
    *,
    layout: dict[str, dict[str, dict[int, dict[int, list[str]]]]],
    nest_release: bool = False,
    line_size: tuple[int, int] = (240, 48),
) -> Path:
    """Build a miniature CVL-shaped tree from ``{set: {writer: {page: {line: [words]}}}}``.

    The shape — not the content — is what these tests exercise.
    """
    from PIL import Image

    base = root / "cvl-database-1-1" if nest_release else root
    for set_name, writers in layout.items():
        for writer, pages in writers.items():
            lines_dir = base / set_name / "lines" / writer
            words_dir = base / set_name / "words" / writer
            lines_dir.mkdir(parents=True, exist_ok=True)
            words_dir.mkdir(parents=True, exist_ok=True)
            for page, lines in pages.items():
                for line, words in lines.items():
                    Image.new("L", line_size, color=255).save(
                        lines_dir / f"{writer}-{page}-{line}.tif"
                    )
                    for index, word in enumerate(words):
                        Image.new("L", (40, 40), color=255).save(
                            words_dir / f"{writer}-{page}-{line}-{index}-{word}.tif"
                        )
    return base


@pytest.fixture
def make_cvl() -> Callable[..., Path]:
    """The fake-CVL builder itself, for tests that need a custom layout.

    Exposed as a fixture rather than imported: ``tests/`` is not a package, so ``from tests.conftest
    import ...`` fails at collection.
    """
    return write_fake_cvl


@pytest.fixture
def fake_cvl(tmp_path: Path) -> Path:
    """A fake CVL release: two writers, English pages 1 and 2 plus German page 6."""
    return write_fake_cvl(
        tmp_path / "cvl",
        layout={
            "trainset": {
                "0001": {
                    1: {0: ["Imagine", "a", "vast", "sheet"], 1: ["of", "paper"]},
                    2: {0: ["And", "fortune", "on", "his"]},
                    6: {0: ["Werd", "ich", "zum", "Augenblicke"]},
                }
            },
            "testset": {
                "0052": {
                    1: {0: ["Imagine", "a", "vast", "sheet"], 1: ["move", "freely", "about"]},
                    2: {0: ["Show'd", "like", "a", "rebel's", "whore"]},
                    6: {0: ["Verweile", "doch", "du", "bist"]},
                }
            },
        },
    )


# --------------------------------------------------------------------------- fake IAM


def write_fake_iam(
    root: Path,
    *,
    forms: dict[str, dict],
    nest_release: bool = False,
    line_size: tuple[int, int] = (400, 60),
    skip_images: tuple[str, ...] = (),
) -> Path:
    """Build a miniature IAM-shaped tree.

    ``forms`` maps a form id to ``{"writer": str, "prompt": str, "lines": {index: (seg, text)}}``
    where ``text`` is the **de-tokenised** transcript. The builder derives both source files from it
    the way the real release does — ``ascii/lines.txt`` gets the word-tokenised form with ``|``
    separators and no space before punctuation, the XML gets the de-tokenised form
    **double-escaped** — so a test that breaks either reader fails here rather than on data no one
    can commit.
    """
    from PIL import Image

    base = root / "IAM" if nest_release else root
    (base / "ascii").mkdir(parents=True, exist_ok=True)
    (base / "xml").mkdir(parents=True, exist_ok=True)

    ascii_lines = ["#--- lines.txt ---", "#"]
    ascii_forms = ["#--- forms.txt ---", "#"]

    for form_id, spec in forms.items():
        writer = spec["writer"]
        prompt = spec.get("prompt", f"prompt for {form_id}")
        lines = spec["lines"]
        ascii_forms.append(f"{form_id} {writer} 2 prt {len(lines)} {len(lines)} 50 50")

        xml = [
            '<?xml version="1.0" encoding="ISO-8859-1"?>',
            f'<form id="{form_id}" writer-id="{writer}">',
            "  <machine-printed-part>",
            f'    <machine-print-line text="{_xml_attr(prompt)}" />',
            "  </machine-printed-part>",
            "  <handwritten-part>",
        ]
        group = form_id.split("-")[0]
        for index, (segmentation, text) in sorted(lines.items()):
            line_id = f"{form_id}-{index:02d}"
            ascii_lines.append(f"{line_id} {segmentation} 154 19 408 746 1663 91 {_tokenise(text)}")
            xml.append(
                f'    <line id="{line_id}" segmentation="{segmentation}" '
                f'text="{_xml_attr(text)}" />'
            )
            if line_id in skip_images:
                continue
            image_dir = base / "lines" / group / form_id
            image_dir.mkdir(parents=True, exist_ok=True)
            Image.new("L", line_size, color=255).save(image_dir / f"{line_id}.png")
        xml += ["  </handwritten-part>", "</form>", ""]
        (base / "xml" / f"{form_id}.xml").write_text("\n".join(xml), encoding="iso-8859-1")

    (base / "ascii" / "lines.txt").write_text("\n".join(ascii_lines) + "\n", encoding="utf-8")
    (base / "ascii" / "forms.txt").write_text("\n".join(ascii_forms) + "\n", encoding="utf-8")
    return base


def _tokenise(text: str) -> str:
    """De-tokenised text -> IAM's ``|``-separated word tokens.

    Inverse of what the real ``lines.txt`` stores: punctuation becomes its own token, so
    ``'projects.'`` round-trips to ``'projects|.'`` and a reader that simply replaces ``|`` with a
    space produces ``'projects .'`` — the defect this fixture exists to expose.
    """
    import re

    tokens: list[str] = []
    for word in text.split(" "):
        match = re.fullmatch(r"(.*?)([.,;:!?]*)", word)
        assert match is not None
        stem, trailing = match.groups()
        if stem:
            tokens.append(stem)
        tokens.extend(trailing)
    return "|".join(tokens)


def _xml_attr(text: str) -> str:
    """Escape for an XML attribute the way the real IAM release does — inconsistently.

    The release mixes escaping depths: it stores 8,436 single-escaped ``&apos;`` alongside 2,789
    **double**-escaped ``&amp;quot;``, so a parser that unescapes once leaves the literal string
    ``&quot;`` in 984 lines. This helper reproduces both depths in one file — ``&`` and ``<``
    singly, ``"`` and ``'`` doubly — because a reader that handles only one of them passes a fixture
    that uses only the other.
    """
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
    return escaped.replace("&quot;", "&amp;quot;").replace("&apos;", "&amp;apos;")


@pytest.fixture
def make_iam() -> Callable[..., Path]:
    """The fake-IAM builder, for tests that need a custom layout."""
    return write_fake_iam


@pytest.fixture
def fake_iam(tmp_path: Path) -> Path:
    """A fake IAM release: three forms, two writers, one shared prompt, one struck-out line.

    ``a01-000u`` and ``a01-000x`` deliberately share a prompt while belonging to *different*
    writers, and writer ``000`` deliberately owns two forms whose ids differ in more than a suffix —
    the two facts that make the writer join and the passage grouping non-trivial.
    """
    shared = "A MOVE to stop Mr. Gaitskell from nominating any more Labour life Peers."
    return write_fake_iam(
        tmp_path / "iam",
        forms={
            "a01-000u": {
                "writer": "000",
                "prompt": shared,
                "lines": {
                    0: ("ok", "A MOVE to stop Mr. Gaitskell from"),
                    1: ("err", "nominating any more Labour life Peers."),
                },
            },
            "a01-000x": {
                "writer": "001",
                "prompt": shared,
                "lines": {0: ("ok", 'He said "no" to the & motion.')},
            },
            "a01-003u": {
                "writer": "000",
                "prompt": "Though they may gather some Left-wing support, a large majority.",
                "lines": {
                    0: ("ok", "Though they may gather some Left-wing support, a"),
                    1: ("ok", "been # particularly intensified in the past year"),
                },
            },
        },
    )
