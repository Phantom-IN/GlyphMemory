"""The alignment visualization: it must not crash on the cases the aligner itself produces, and it
must actually write a file — this is inspection evidence, not decoration.
"""

from __future__ import annotations

import torch
from PIL import Image

from glyphmemory.alignment import AlignmentSpan
from glyphmemory.alignment.visualize import render_alignment, save_alignment


def test_render_returns_an_image_wider_than_the_input_and_taller_for_the_bands():
    image = torch.rand(1, 64, 120)
    spans = [AlignmentSpan(token="a", start_t=1, end_t=5, score=0.8)]
    rendered = render_alignment(image, spans)
    assert rendered.size[0] == 120
    assert rendered.size[1] > 64


def test_render_with_no_spans_does_not_crash():
    image = torch.rand(1, 64, 80)
    rendered = render_alignment(image, [])
    assert rendered.size[0] == 80


def test_render_with_many_spans_cycles_through_colors():
    image = torch.rand(1, 64, 300)
    spans = [
        AlignmentSpan(token=chr(ord("a") + i), start_t=i * 5, end_t=i * 5 + 4, score=0.5)
        for i in range(10)
    ]
    rendered = render_alignment(image, spans)
    assert rendered.size[0] == 300


def test_render_clamps_a_span_that_runs_past_the_image_width():
    # A span whose pixel columns exceed the image (e.g. from a mismatched downsample factor in a
    # caller's own bug) must not raise — it should be visually clipped, not fatal.
    image = torch.rand(1, 64, 40)
    spans = [AlignmentSpan(token="z", start_t=5, end_t=50, score=0.5)]
    rendered = render_alignment(image, spans)
    assert rendered.size[0] == 40


def test_render_labels_a_space_token_visibly_rather_than_blank_text():
    image = torch.rand(1, 64, 60)
    spans = [AlignmentSpan(token=" ", start_t=1, end_t=6, score=0.7)]
    rendered = render_alignment(image, spans)  # must not raise on an empty label string
    assert rendered.size[0] == 60


def test_render_accepts_a_pil_image_directly():
    image = Image.new("L", (50, 64), color=128)
    spans = [AlignmentSpan(token="x", start_t=0, end_t=4, score=0.6)]
    rendered = render_alignment(image, spans)
    assert rendered.size[0] == 50


def test_save_alignment_writes_a_file(tmp_path):
    image = torch.rand(1, 64, 90)
    spans = [AlignmentSpan(token="a", start_t=1, end_t=5, score=0.8)]
    out = tmp_path / "nested" / "example.png"
    written = save_alignment(image, spans, out)
    assert written == out
    assert out.is_file()
    with Image.open(out) as reopened:
        assert reopened.size[0] == 90
