"""The central assertion is the one the whole design turns on: a midpoint-tiled cell recovers the
per-character width that a CTC span does not.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch

from glyphmemory.alignment.spans import AlignmentSpan
from glyphmemory.data.preprocessing import HORIZONTAL_DOWNSAMPLE
from glyphmemory.evaluation.emitted_spans import EmittedOccurrence
from glyphmemory.memory.glyph_regions import (
    CROP_WIDTH,
    EDGE_HALF_FRAMES,
    aligned_centers,
    crops_from_occurrences,
    crops_from_spans,
    emitted_centers,
    extract_crops,
    midpoint_cells,
    region_statistics,
)


def occurrence(index: int, character: str, start: int, end: int, peak: int) -> EmittedOccurrence:
    return EmittedOccurrence(
        index=index,
        character=character,
        start=start,
        end=end,
        peak=peak,
        confidence=0.9,
        margin=0.5,
        entropy=0.1,
        candidates=(character, "x"),
    )


class TestMidpointCells:
    def test_a_cell_is_far_wider_than_the_span_it_replaces(self) -> None:
        """The measured motivation, asserted rather than trusted.

        Emitted spans are one frame (4 px) for 92% of eligible slots. Centers spaced 5 frames apart
        — the measured median gap — must produce ~5-frame cells, not 1-frame ones.
        """
        centers = [2.0, 7.0, 12.0, 17.0]
        cells = midpoint_cells(centers)
        interior = [high - low for low, high in cells[1:-1]]
        assert interior == [5.0, 5.0]
        assert all(width * HORIZONTAL_DOWNSAMPLE == 20.0 for width in interior)

    def test_cells_tile_without_gap_or_overlap(self) -> None:
        """An ambiguous mask channel would make the second input channel meaningless."""
        cells = midpoint_cells([1.0, 4.0, 9.0, 11.0])
        for (_, high), (low, _) in pairwise(cells):
            assert high == low

    def test_edge_glyphs_extend_by_the_stated_half_width(self) -> None:
        cells = midpoint_cells([3.0, 9.0])
        assert cells[0][0] == pytest.approx(3.0 - EDGE_HALF_FRAMES)
        assert cells[-1][1] == pytest.approx(9.0 + EDGE_HALF_FRAMES)

    def test_single_center_is_symmetric_about_itself(self) -> None:
        (low, high), = midpoint_cells([10.0])
        assert low == pytest.approx(10.0 - EDGE_HALF_FRAMES)
        assert high == pytest.approx(10.0 + EDGE_HALF_FRAMES)

    def test_empty_input(self) -> None:
        assert midpoint_cells([]) == []


class TestCenters:
    def test_emitted_centers_use_the_peak_not_the_span_midpoint(self) -> None:
        """The peak is where the model committed; on an asymmetric run they differ."""
        occ = occurrence(0, "a", start=4, end=8, peak=5)
        assert emitted_centers([occ]) == [5.0]

    def test_aligned_centers_respect_the_exclusive_end(self) -> None:
        """``end_t`` is exclusive, so frames 4..6 have center 5.0, not 5.5."""
        span = AlignmentSpan(token="a", start_t=4, end_t=7, score=0.9)
        assert aligned_centers([span]) == [5.0]


class TestExtractCrops:
    def test_shape_and_channels(self) -> None:
        line = torch.rand(1, 64, 200)
        crops = extract_crops(line, [5.0, 12.0, 20.0], true_width=200)
        assert crops.shape == (3, 2, 64, CROP_WIDTH)

    def test_the_window_is_centered_on_the_glyph(self) -> None:
        """A single bright column at the center's pixel must land in the middle of the window."""
        line = torch.zeros(1, 64, 200)
        center_frame = 10
        center_px = center_frame * HORIZONTAL_DOWNSAMPLE + HORIZONTAL_DOWNSAMPLE // 2
        line[0, :, center_px] = 1.0
        crops = extract_crops(line, [float(center_frame)], true_width=200)
        column = int(crops[0, 0, 0].argmax())
        assert abs(column - CROP_WIDTH // 2) <= 1

    def test_mask_marks_the_cell_and_nothing_else(self) -> None:
        crops = extract_crops(torch.zeros(1, 64, 400), [20.0, 25.0, 30.0], true_width=400)
        mask = crops[1, 1, 0]
        # Middle glyph: cell is frames 22.5..27.5 = 20 px wide, centered in the window.
        assert mask.sum() == pytest.approx(20.0)
        # Cell spans window columns 8..27, so both margins are background.
        assert mask[:8].sum() == 0.0
        assert mask[28:].sum() == 0.0

    def test_mask_is_constant_down_every_row(self) -> None:
        """The cell is a column range; a row-varying mask would mean a bug in the fill."""
        crops = extract_crops(torch.zeros(1, 64, 400), [20.0, 25.0, 30.0], true_width=400)
        mask = crops[1, 1]
        assert torch.equal(mask, mask[0].expand_as(mask))

    def test_padding_uses_the_line_background_not_zero(self) -> None:
        """Under inverted polarity the background is not 0.0, and a zero pad would be ink."""
        line = torch.full((1, 64, 100), 0.7)
        line[0, :, 40:60] = 0.95
        crops = extract_crops(line, [0.0], true_width=100)
        assert crops[0, 0, :, 0].min() == pytest.approx(0.7)

    def test_columns_past_true_width_are_treated_as_padding(self) -> None:
        """The collator pads to a width multiple; that padding is not handwriting."""
        line = torch.full((1, 64, 200), 0.5)
        line[0, :, 120:] = 99.0  # obvious sentinel in the padded region
        crops = extract_crops(line, [32.0], true_width=120)
        assert crops[0, 0].max() < 99.0

    def test_line_start_clips_without_shifting_the_glyph_off_centre(self) -> None:
        line = torch.zeros(1, 64, 200)
        line[0, :, 2] = 1.0
        crops = extract_crops(line, [0.0], true_width=200)
        assert crops.shape == (1, 2, 64, CROP_WIDTH)
        assert float(crops[0, 0].max()) == pytest.approx(1.0)

    def test_accepts_two_dimensional_lines(self) -> None:
        assert extract_crops(torch.zeros(64, 100), [5.0]).shape == (1, 2, 64, CROP_WIDTH)

    def test_rejects_multichannel_input(self) -> None:
        with pytest.raises(ValueError, match="single-channel"):
            extract_crops(torch.zeros(3, 64, 100), [5.0])

    def test_no_centers_returns_an_empty_batch_not_an_error(self) -> None:
        assert extract_crops(torch.zeros(1, 64, 100), []).shape == (0, 2, 64, CROP_WIDTH)


class TestConvenienceWrappers:
    def test_occurrence_and_span_wrappers_agree_when_centers_agree(self) -> None:
        """Support and query paths differ only in where centers come from, by design."""
        line = torch.rand(1, 64, 200)
        occ = [occurrence(0, "a", 4, 6, 5), occurrence(1, "b", 9, 11, 10)]
        spans = [
            AlignmentSpan(token="a", start_t=4, end_t=7, score=0.9),
            AlignmentSpan(token="b", start_t=9, end_t=12, score=0.9),
        ]
        assert emitted_centers(occ) == aligned_centers(spans)
        assert torch.equal(
            crops_from_occurrences(line, occ, true_width=200),
            crops_from_spans(line, spans, true_width=200),
        )


class TestRegionStatistics:
    def test_reports_widths_in_pixels_against_the_measured_reference(self) -> None:
        stats = region_statistics([2.0, 7.0, 12.0, 17.0, 22.0])
        assert stats["count"] == 5.0
        assert stats["p50_px"] == pytest.approx(20.0)

    def test_empty(self) -> None:
        assert region_statistics([])["count"] == 0.0
