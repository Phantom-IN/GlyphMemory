"""Image preprocessing and the input contract.

The highest-value assertions here are about **widths**, not pixels: `temporal_length` being the sole
computation of `T`, and the true unpadded width surviving separately from the padded one. Both are
silent-failure paths — a model trained with padding counted as valid input decodes truncated text
and never raises.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from PIL import Image

from glyphmemory.data import (
    DEFAULT_HEIGHT,
    DEFAULT_MAX_WIDTH,
    DEFAULT_WIDTH_MULTIPLE,
    HORIZONTAL_DOWNSAMPLE,
    PixelNormalization,
    UnreadableImageError,
    check_max_width,
    load_line_image,
    pad_width_to_multiple,
    preprocess_image,
    preprocess_path,
    resize_to_height,
    temporal_length,
    to_uint8_tensor,
)


def grey(width: int, height: int, value: int = 255) -> Image.Image:
    return Image.new("L", (width, height), value)


def inked(width: int, height: int) -> Image.Image:
    """A pale page with a dark bar, so ink and background are distinguishable."""
    image = Image.new("L", (width, height), 240)
    for x in range(width // 4, width // 2):
        for y in range(height // 4, height // 2):
            image.putpixel((x, y), 20)
    return image


# --------------------------------------------------------------------------- temporal length


@pytest.mark.parametrize(
    ("width", "expected"),
    [(0, 0), (1, 1), (4, 1), (16, 4), (63, 16), (64, 16), (512, 128), (1024, 256), (1600, 400)],
)
def test_temporal_length_matches_hand_computed_values(width, expected):
    assert temporal_length(width) == expected


def test_temporal_length_uses_ceiling_for_non_multiples():
    """Matches stride-2 conv arithmetic, which is ceil(in/2) per stage."""
    for width in (17, 18, 19, 20):
        assert temporal_length(width) == math.ceil(width / HORIZONTAL_DOWNSAMPLE)


def test_temporal_length_exact_for_padded_widths():
    """Widths always reach the model as multiples of 16, so ceil and floor coincide."""
    for multiple in range(1, 40):
        width = multiple * DEFAULT_WIDTH_MULTIPLE
        assert temporal_length(width) == width // HORIZONTAL_DOWNSAMPLE


def test_temporal_length_honours_custom_downsample():
    assert temporal_length(512, downsample=8) == 64


@pytest.mark.parametrize(("width", "downsample"), [(-1, 4), (16, 0)])
def test_temporal_length_rejects_invalid_input(width, downsample):
    with pytest.raises(ValueError):
        temporal_length(width, downsample=downsample)


def test_downsample_constant_matches_documented_architecture():
    """Internal helper."""
    assert HORIZONTAL_DOWNSAMPLE == 4


# --------------------------------------------------------------------------- loading


def test_load_produces_grayscale(tmp_path: Path):
    path = tmp_path / "line.png"
    Image.new("RGB", (40, 20), (10, 200, 30)).save(path)
    assert load_line_image(path).mode == "L"


def test_missing_file_raises_typed_error(tmp_path: Path):
    with pytest.raises(UnreadableImageError, match="does not exist"):
        load_line_image(tmp_path / "absent.png")


def test_non_image_raises_typed_error(tmp_path: Path):
    path = tmp_path / "not_an_image.png"
    path.write_text("definitely not a png", encoding="utf-8")
    with pytest.raises(UnreadableImageError, match="not a recognisable image"):
        load_line_image(path)


def test_unreadable_error_carries_path_and_reason(tmp_path: Path):
    try:
        load_line_image(tmp_path / "absent.png")
    except UnreadableImageError as exc:
        assert exc.path.endswith("absent.png")
        assert exc.reason


# --------------------------------------------------------------------------- resize


def test_resize_hits_exact_height():
    assert resize_to_height(to_uint8_tensor(grey(300, 97))).shape[-2] == DEFAULT_HEIGHT


def test_resize_preserves_aspect_ratio_within_a_pixel():
    original_w, original_h = 300, 97
    resized = resize_to_height(to_uint8_tensor(grey(original_w, original_h)))
    expected = round(original_w * DEFAULT_HEIGHT / original_h)
    assert abs(int(resized.shape[-1]) - expected) <= 1


def test_resize_never_stretches_to_a_fixed_width():
    """Internal helper."""
    widths = [
        int(resize_to_height(to_uint8_tensor(grey(w, 100))).shape[-1]) for w in (200, 400, 800)
    ]
    assert widths[1] == pytest.approx(widths[0] * 2, rel=0.02)
    assert widths[2] == pytest.approx(widths[0] * 4, rel=0.02)


def test_resize_is_a_noop_at_target_height():
    tensor = to_uint8_tensor(grey(120, DEFAULT_HEIGHT))
    assert torch.equal(resize_to_height(tensor), tensor)


def test_resize_rejects_bad_height():
    with pytest.raises(ValueError, match="at least 1"):
        resize_to_height(to_uint8_tensor(grey(10, 10)), 0)


# --------------------------------------------------------------------------- padding


@pytest.mark.parametrize("width", [1, 15, 16, 17, 100, 1000])
def test_padding_reaches_a_multiple_and_reports_true_width(width):
    padded, true_width = pad_width_to_multiple(to_uint8_tensor(grey(width, 64)))
    assert true_width == width
    assert int(padded.shape[-1]) % DEFAULT_WIDTH_MULTIPLE == 0
    assert int(padded.shape[-1]) >= width


def test_padding_is_a_noop_when_already_aligned():
    tensor = to_uint8_tensor(grey(64, 64))
    padded, true_width = pad_width_to_multiple(tensor)
    assert torch.equal(padded, tensor)
    assert true_width == 64


def test_padding_is_applied_on_the_right_only():
    """Right-padding keeps content prefix-aligned with input_lengths."""
    tensor = to_uint8_tensor(inked(20, 64))
    padded, true_width = pad_width_to_multiple(tensor)
    assert torch.equal(padded[..., :true_width], tensor)
    assert (padded[..., true_width:] == 255).all()


def test_padding_rejects_invalid_multiple():
    with pytest.raises(ValueError, match="at least 1"):
        pad_width_to_multiple(to_uint8_tensor(grey(10, 10)), 0)


# --------------------------------------------------------------------------- normalization


def test_normalization_inverts_so_background_is_zero():
    """Background 0 matches the zero that conv and batch padding already use."""
    normalization = PixelNormalization()
    white = normalization.apply(to_uint8_tensor(grey(4, 4, 255)))
    black = normalization.apply(to_uint8_tensor(grey(4, 4, 0)))
    assert white.max().item() == pytest.approx(0.0)
    assert black.min().item() == pytest.approx(1.0)


def test_normalization_can_skip_inversion():
    normalization = PixelNormalization(invert=False)
    assert normalization.apply(to_uint8_tensor(grey(4, 4, 255))).max().item() == pytest.approx(1.0)


def test_background_value_matches_actual_padding():
    for normalization in (
        PixelNormalization(),
        PixelNormalization(invert=False, mean=0.5, std=0.5),
    ):
        padded = normalization.apply(to_uint8_tensor(grey(4, 4, 255)))
        assert padded.flatten()[0].item() == pytest.approx(normalization.background_value)


def test_normalization_is_not_per_image():
    """Two images differing only in ink density must stay different after normalization.

    Per-image standardisation would rescale both to the same statistics, erasing exactly the signal
    the writer-memory hypothesis depends on.
    """
    normalization = PixelNormalization()
    faint = normalization.apply(to_uint8_tensor(grey(8, 8, 200)))
    heavy = normalization.apply(to_uint8_tensor(grey(8, 8, 60)))
    assert not torch.allclose(faint.mean(), heavy.mean())


def test_normalization_applies_mean_and_std():
    normalization = PixelNormalization(invert=False, mean=0.5, std=0.5)
    assert normalization.apply(to_uint8_tensor(grey(2, 2, 255))).max().item() == pytest.approx(1.0)
    assert normalization.apply(to_uint8_tensor(grey(2, 2, 0))).min().item() == pytest.approx(-1.0)


def test_zero_std_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        PixelNormalization(std=0.0)


# --------------------------------------------------------------------------- max width guard


def test_width_within_guard_passes():
    assert check_max_width(1000) is True


def test_oversized_width_is_flagged_and_logged_not_resized(caplog):
    with caplog.at_level("WARNING", logger="glyphmemory.data.preprocessing"):
        result = check_max_width(2000, sample_id="synthetic/0", path="/tmp/x.png")
    assert result is False
    message = caplog.records[0].getMessage()
    assert "synthetic/0" in message
    assert "/tmp/x.png" in message
    assert "not resized" in message


def test_oversized_image_keeps_its_width():
    """The guard flags; it must never crush the aspect ratio."""
    result = preprocess_image(grey(3000, 64), max_width=DEFAULT_MAX_WIDTH)
    assert result.oversized is True
    assert result.true_width == 3000


def test_max_width_default_matches_documentation():
    assert DEFAULT_MAX_WIDTH == 1600


# --------------------------------------------------------------------------- full pipeline


def test_pipeline_output_contract():
    result = preprocess_image(inked(300, 97))
    assert result.tensor.shape[0] == 1
    assert result.tensor.shape[1] == DEFAULT_HEIGHT
    assert int(result.tensor.shape[-1]) % DEFAULT_WIDTH_MULTIPLE == 0
    assert result.tensor.dtype == torch.float32


def test_true_width_is_distinct_from_padded_width():
    result = preprocess_image(grey(101, 64))
    assert result.true_width < result.padded_width
    assert result.padded_width % DEFAULT_WIDTH_MULTIPLE == 0


def test_input_length_derives_from_true_width_not_padded():
    """The bug this prevents: padding counted as valid CTC input."""
    result = preprocess_image(grey(101, 64))
    assert result.input_length == temporal_length(result.true_width)
    assert result.input_length <= result.padded_length


def test_padded_region_holds_the_background_value():
    result = preprocess_image(inked(101, 64))
    padding = result.tensor[..., result.true_width :]
    assert padding.numel() > 0
    assert torch.allclose(padding, torch.zeros_like(padding))


def test_pipeline_is_deterministic():
    image = inked(220, 80)
    assert torch.equal(preprocess_image(image).tensor, preprocess_image(image).tensor)


def test_pipeline_records_original_size():
    assert preprocess_image(grey(321, 97)).original_size == (321, 97)


def test_pipeline_accepts_a_tensor():
    assert preprocess_image(to_uint8_tensor(grey(120, 64))).height == DEFAULT_HEIGHT


def test_pipeline_rejects_wrong_tensor_shape():
    with pytest.raises(ValueError, match=r"\[1, H, W\]"):
        preprocess_image(torch.zeros(3, 64, 100, dtype=torch.uint8))


def test_augmentation_hook_is_applied():
    calls: list[int] = []

    def spy(tensor):
        calls.append(1)
        return tensor

    preprocess_image(grey(100, 64), augmentation=spy)
    assert calls == [1]


def test_no_augmentation_by_default():
    """Evaluation path must be untouched."""
    a = preprocess_image(inked(200, 64))
    b = preprocess_image(inked(200, 64))
    assert torch.equal(a.tensor, b.tensor)


def test_preprocess_path_round_trip(tmp_path: Path):
    path = tmp_path / "line.png"
    inked(240, 80).save(path)
    result = preprocess_path(path)
    assert result.height == DEFAULT_HEIGHT
    assert result.true_width > 0


# --------------------------------------------------------------------------- against fixtures


def test_every_synthetic_fixture_satisfies_the_contract(synthetic_corpus):
    """Success criterion: [1, 64, W] with W % 16 == 0 for every fixture image."""
    for record in synthetic_corpus.records:
        result = preprocess_path(record.image, sample_id=record.sample_id)
        assert result.tensor.shape[0] == 1
        assert result.tensor.shape[1] == DEFAULT_HEIGHT
        assert int(result.tensor.shape[-1]) % DEFAULT_WIDTH_MULTIPLE == 0
        assert result.input_length == temporal_length(result.true_width)


def test_fixture_widths_are_within_the_guard(synthetic_corpus):
    """The session fixture must not itself be oversized, or inherits flagged data."""
    for record in synthetic_corpus.records:
        assert not preprocess_path(record.image).oversized


def test_fixture_widths_vary(synthetic_corpus):
    widths = {preprocess_path(r.image).true_width for r in synthetic_corpus.records}
    assert len(widths) > 1
