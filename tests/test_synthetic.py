"""Synthetic font-as-writer generator.

Assertions are **structural**, never golden-pixel: rendering differs between macOS and the Linux CI
runner, so the tests verify determinism *within* a platform, separation *between* writers, and the
manifest contract — not exact bytes.
"""

from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

import pytest
from PIL import Image

from glyphmemory.ctc import Charset
from glyphmemory.data import validate_manifest
from glyphmemory.data.adapters.synthetic import SyntheticAdapter
from glyphmemory.data.manifest import read_manifest
from glyphmemory.data.synthetic import (
    BUNDLED_FONT_NAME,
    CORPUS_MODES,
    WriterStyle,
    bundled_font,
    coverage_counts,
    discover_fonts,
    image_difference,
    missing_coverage,
    render_line,
    resolve_fonts,
    sample_lines,
)

TEXT = "handwriting sample 42"


# --------------------------------------------------------------------------- fonts


def test_bundled_font_is_always_available():
    """The whole cross-platform guarantee rests on this."""
    fonts = discover_fonts()
    assert fonts
    assert fonts[0].name == BUNDLED_FONT_NAME
    assert fonts[0].path is None


def test_bundled_font_loads_and_renders():
    assert render_line("abc", bundled_font(), WriterStyle()).height == 64


def test_discovery_excludes_system_fonts_by_default():
    """System paths differ between macOS and the CI runner; never depend on them."""
    assert all(font.origin != "system" for font in discover_fonts())


def test_discovery_is_deterministic():
    assert [f.name for f in discover_fonts()] == [f.name for f in discover_fonts()]


def test_resolve_fonts_defaults_to_bundled():
    assert resolve_fonts(None)[0].name == BUNDLED_FONT_NAME
    assert resolve_fonts([bundled_font()])[0].name == BUNDLED_FONT_NAME


def test_extra_font_dir_is_searched(tmp_path: Path):
    (tmp_path / "nothing.txt").write_text("not a font", encoding="utf-8")
    assert len(discover_fonts(extra_dir=tmp_path)) == 1  # bundled only; .txt ignored


# --------------------------------------------------------------------------- rendering


def test_render_is_deterministic():
    style = WriterStyle(font_size=36, slant_degrees=5, letter_spacing=1)
    first = render_line(TEXT, bundled_font(), style)
    second = render_line(TEXT, bundled_font(), style)
    assert first.tobytes() == second.tobytes()


def test_render_output_contract():
    image = render_line(TEXT, bundled_font(), WriterStyle())
    assert image.height == 64
    assert image.mode == "L"
    assert image.width > 64  # a line is wider than it is tall


def test_width_varies_with_text_length():
    font, style = bundled_font(), WriterStyle()
    short = render_line("ab", font, style)
    long = render_line("ab" * 20, font, style)
    assert long.width > short.width * 5


def test_aspect_ratio_is_preserved_not_stretched():
    """Internal helper."""
    font, style = bundled_font(), WriterStyle()
    widths = [render_line("a" * n, font, style).width for n in (5, 10, 20)]
    assert widths[0] < widths[1] < widths[2]


@pytest.mark.parametrize("height", [32, 64, 96])
def test_height_is_exact(height):
    assert render_line(TEXT, bundled_font(), WriterStyle(), height=height).height == height


def test_empty_text_rejected():
    with pytest.raises(ValueError, match="empty line"):
        render_line("", bundled_font(), WriterStyle())


def test_whitespace_only_render_does_not_crash():
    """The corpus never emits this, but a caller might."""
    assert render_line("   ", bundled_font(), WriterStyle()).height == 64


def test_style_parameters_change_the_image():
    font, base = bundled_font(), WriterStyle(font_size=36)
    for changed in (
        WriterStyle(font_size=36, slant_degrees=12),
        WriterStyle(font_size=36, letter_spacing=3),
        WriterStyle(font_size=36, stroke_width=1),
        WriterStyle(font_size=36, ink_level=90),
    ):
        assert image_difference(render_line(TEXT, font, base), render_line(TEXT, font, changed)) > 0


# --------------------------------------------------------------------------- writer identity


def _styles(n: int, seed: str = "s") -> list[WriterStyle]:
    return [
        WriterStyle.sample(random.Random(f"{seed}:{i}"), writer_index=i, n_writers=n, n_fonts=1)
        for i in range(n)
    ]


def test_two_writers_render_measurably_differently():
    styles = _styles(3)
    images = [render_line(TEXT, bundled_font(), s) for s in styles]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert image_difference(images[i], images[j]) > 0.02


def test_one_writer_repeated_is_similar_but_not_identical():
    """Jitter must be live: a writer is a distribution, not a constant."""
    style = _styles(3)[0]
    first = render_line(TEXT, bundled_font(), style.jitter(random.Random("a")))
    second = render_line(TEXT, bundled_font(), style.jitter(random.Random("b")))
    assert first.tobytes() != second.tobytes()
    assert image_difference(first, second) < 0.10


def test_within_writer_variation_is_smaller_than_between_writer():
    """Writer identity must dominate line-to-line noise, or the oracle proves nothing."""
    styles = _styles(3)
    within = image_difference(
        render_line(TEXT, bundled_font(), styles[0].jitter(random.Random("a"))),
        render_line(TEXT, bundled_font(), styles[0].jitter(random.Random("b"))),
    )
    between = image_difference(
        render_line(TEXT, bundled_font(), styles[0]),
        render_line(TEXT, bundled_font(), styles[1]),
    )
    assert within < between


def test_slant_is_stratified_so_writers_cannot_collide():
    slants = sorted(s.slant_degrees for s in _styles(6))
    assert all(b - a > 1.0 for a, b in itertools.pairwise(slants))


def test_different_seeds_give_different_styles():
    assert _styles(3, "one")[0] != _styles(3, "two")[0]


def test_same_seed_gives_identical_styles():
    assert _styles(3, "same") == _styles(3, "same")


# --------------------------------------------------------------------------- corpus


def test_coverage_mode_guarantees_every_character():
    characters = Charset.english_v1().characters
    lines = sample_lines(characters, n_lines=4, rng=random.Random(1), min_occurrences=2)
    assert missing_coverage(lines, characters, min_occurrences=2) == {}


@pytest.mark.parametrize("min_occurrences", [1, 3])
def test_coverage_respects_the_requested_target(min_occurrences):
    characters = Charset.english_v1().characters
    lines = sample_lines(
        characters, n_lines=5, rng=random.Random(2), min_occurrences=min_occurrences
    )
    assert missing_coverage(lines, characters, min_occurrences=min_occurrences) == {}


def test_coverage_includes_rare_characters():
    """Natural text will not reliably supply these; prototypes need them."""
    characters = Charset.english_v1().characters
    counts = coverage_counts(sample_lines(characters, n_lines=4, rng=random.Random(3)))
    for rare in "qxz7#&":
        assert counts[rare] >= 2


def test_every_line_is_non_empty():
    characters = Charset.english_v1().characters
    lines = sample_lines(characters, n_lines=12, rng=random.Random(4))
    assert len(lines) == 12
    assert all(line.strip() for line in lines)


def test_corpus_is_deterministic():
    characters = Charset.english_v1().characters
    assert sample_lines(characters, n_lines=3, rng=random.Random(5)) == sample_lines(
        characters, n_lines=3, rng=random.Random(5)
    )


def test_words_mode_is_readable():
    lines = sample_lines(("a",), n_lines=3, rng=random.Random(6), mode="words")
    assert all(" " in line for line in lines)
    assert "the" in " ".join(lines) or "fox" in " ".join(lines)


@pytest.mark.parametrize("mode", CORPUS_MODES)
def test_all_modes_produce_encodable_text(mode):
    from glyphmemory.ctc import Tokenizer

    tokenizer = Tokenizer.english_v1()
    lines = sample_lines(tokenizer.charset.characters, n_lines=3, rng=random.Random(7), mode=mode)
    for line in lines:
        assert tokenizer.encode(line)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="Unknown corpus mode"):
        sample_lines(("a",), n_lines=1, rng=random.Random(8), mode="prose")


def test_zero_lines_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        sample_lines(("a",), n_lines=0, rng=random.Random(9))


# --------------------------------------------------------------------------- adapter


def test_adapter_output_validates_cleanly(synthetic_corpus):
    """Success criterion: zero integrity-counter hits."""
    report = validate_manifest(synthetic_corpus.manifest_path)
    assert report.is_clean
    assert report.rejected_records == 0
    assert report.counters.total == 0


def test_adapter_record_count(synthetic_corpus):
    assert len(synthetic_corpus.records) == 12
    assert len(synthetic_corpus.writers) == 3


def test_every_record_is_labelled_synthetic(synthetic_corpus):
    """Synthetic output must never be mistakable for real data."""
    for record in synthetic_corpus.records:
        assert record.dataset == "synthetic"
        assert record.writer_id.startswith("synthetic/")
        assert record.sample_id and record.sample_id.startswith("synthetic/")


def test_images_match_the_declared_contract(synthetic_corpus):
    widths = set()
    for record in synthetic_corpus.records:
        image = Image.open(record.image)
        assert image.height == 64
        assert image.mode == "L"
        assert image.height == record.height
        assert image.width == record.width
        widths.add(image.width)
    assert len(widths) > 1  # variable width, not a fixed canvas


def test_passage_ids_are_assigned(synthetic_corpus):
    """Passage-disjoint support/query must be exercisable before CVL arrives."""
    passages = {record.passage_id for record in synthetic_corpus.records}
    assert len(passages) == 2
    assert None not in passages


def test_same_seed_reproduces_byte_identical_output(tmp_path: Path):
    first = SyntheticAdapter(n_writers=2, n_lines=2, seed=99).prepare(output_dir=tmp_path / "a")
    second = SyntheticAdapter(n_writers=2, n_lines=2, seed=99).prepare(output_dir=tmp_path / "b")

    left = [r.to_dict() | {"image": Path(r.image).name} for r in read_manifest(first)]
    right = [r.to_dict() | {"image": Path(r.image).name} for r in read_manifest(second)]
    assert left == right

    for record in read_manifest(first):
        name = Path(record.image).name
        assert (tmp_path / "a" / "images" / name).read_bytes() == (
            tmp_path / "b" / "images" / name
        ).read_bytes()


def test_different_seeds_produce_different_output(tmp_path: Path):
    a = SyntheticAdapter(n_writers=2, n_lines=2, seed=1).prepare(output_dir=tmp_path / "a")
    b = SyntheticAdapter(n_writers=2, n_lines=2, seed=2).prepare(output_dir=tmp_path / "b")
    assert [r.text for r in read_manifest(a)] != [r.text for r in read_manifest(b)]


def test_writer_styles_are_recorded_beside_the_data(synthetic_corpus):
    """A generated corpus must be explicable after the fact."""
    payload = json.loads((synthetic_corpus.root / "writer_styles.json").read_text())
    assert payload["generator"]["synthetic"] is True
    assert "never a performance benchmark" in payload["generator"]["note"].lower()
    assert len(payload["writers"]) == 3
    assert payload["fonts"][0]["name"] == BUNDLED_FONT_NAME


def test_adapter_describes_itself(synthetic_corpus):
    described = synthetic_corpus.adapter.describe()
    assert described["dataset"] == "synthetic"
    assert described["synthetic"] is True
    assert described["charset_fingerprint"]


def test_coverage_holds_per_writer_in_generated_corpus(synthetic_corpus):
    characters = Charset.english_v1().characters
    for writer in synthetic_corpus.writers:
        lines = [r.text for r in synthetic_corpus.records_for(writer)]
        assert missing_coverage(lines, characters, min_occurrences=2) == {}


def test_generated_writers_render_differently(synthetic_corpus):
    first, second = synthetic_corpus.writers[0], synthetic_corpus.writers[1]
    a = Image.open(synthetic_corpus.records_for(first)[0].image)
    b = Image.open(synthetic_corpus.records_for(second)[0].image)
    assert image_difference(a, b) > 0.02


@pytest.mark.parametrize("kwargs", [{"n_writers": 0}, {"n_lines": 0}, {"n_passages": 0}])
def test_invalid_adapter_parameters_rejected(kwargs):
    with pytest.raises(ValueError, match="at least 1"):
        SyntheticAdapter(**kwargs)


def test_output_dir_is_required():
    with pytest.raises(ValueError, match="output_dir is required"):
        SyntheticAdapter().prepare()


def test_fixture_generation_is_fast(tmp_path: Path):
    """CI runs this on every test session; it must stay cheap."""
    start = time.perf_counter()
    SyntheticAdapter(n_writers=3, n_lines=4, seed=1).prepare(output_dir=tmp_path)
    assert time.perf_counter() - start < 5.0


def test_no_images_are_committed_to_the_repository():
    """Fixtures are generated at test time; the repo carries no image data."""
    repo = Path(__file__).resolve().parents[1]
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        assert not list((repo / "tests").rglob(pattern))
        assert not list((repo / "src").rglob(pattern))
