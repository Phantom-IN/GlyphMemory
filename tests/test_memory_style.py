"""Global writer style.

Two properties carry this phase and are asserted behaviourally rather than trusted:

1. **Invariant 4** — enrollment is forward passes only. No optimizer, no `backward()`, no gradient.
   Proven by making `Tensor.backward` raise for the duration, not by reading the source.
2. **The frozen model is never mutated.** `gm-base-v0` must be byte-identical after enrollment
   (ADR-0008), so every buffer touched is restored — including on exception.
"""

from __future__ import annotations

import pytest
import torch

from glyphmemory.ctc import DEFAULT_CHARSET_PATH, load_tokenizer
from glyphmemory.memory.style import (
    base_style,
    batchnorm_modules,
    blend_style,
    compile_global_style,
    describe_style,
    style_bytes,
    style_dimension,
    writer_style,
)
from glyphmemory.model import GMBase


def _model():
    torch.manual_seed(0)
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    model = GMBase(vocab_size=tokenizer.vocab_size)
    model.eval()
    return model


def _lines(n=3, width=192):
    torch.manual_seed(1)
    return [torch.rand(1, 64, width + 32 * i) for i in range(n)]


def _buffers(model):
    return {name: buffer.detach().clone() for name, buffer in model.named_buffers()}


def _identical(before, after):
    return all(torch.equal(value, after[name]) for name, value in before.items())


# ------------------------------------------------------------------ invariant 4


def test_enrollment_uses_no_gradient_no_backward_no_optimizer(monkeypatch):
    """Enforced by making the two ways to get one fail loudly for the duration of the call."""
    model = _model()

    def forbidden(*args, **kwargs):
        raise AssertionError("enrollment must not call backward()")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)

    style = compile_global_style(model, _lines())

    assert not style.requires_grad
    assert style.grad_fn is None
    assert all(p.grad is None for p in model.parameters())


def test_style_has_the_layout_the_model_implies():
    model = _model()

    style = compile_global_style(model, _lines())

    modules = batchnorm_modules(model)
    assert style.ndim == 1
    assert style.numel() == style_dimension(model) == sum(2 * m.num_features for m in modules)
    assert torch.isfinite(style).all()


def test_variances_are_positive_and_means_are_not_all_zero():
    """A reset BatchNorm has mean 0 / var 1; a real estimate must have moved off that."""
    model = _model()
    modules = batchnorm_modules(model)

    style = compile_global_style(model, _lines())

    offset, saw_nonzero_mean, saw_shifted_var = 0, False, False
    for module in modules:
        channels = module.num_features
        mean = style[offset : offset + channels]
        var = style[offset + channels : offset + 2 * channels]
        offset += 2 * channels
        assert (var > 0).all(), "a non-positive variance would divide by ~zero at inference"
        saw_nonzero_mean |= bool((mean.abs() > 1e-6).any())
        saw_shifted_var |= bool((var - 1.0).abs().max() > 1e-6)
    assert saw_nonzero_mean and saw_shifted_var


# ------------------------------------------------------------------ model integrity


def test_compiling_a_style_leaves_the_model_byte_identical():
    model = _model()
    before = _buffers(model)

    compile_global_style(model, _lines())

    assert _identical(before, _buffers(model))


def test_compiling_restores_buffers_even_when_the_forward_pass_raises():
    model = _model()
    before = _buffers(model)

    # second line has the wrong channel count, so the encoder raises partway through
    with pytest.raises(ValueError, match="input channel"):
        compile_global_style(model, [torch.rand(1, 64, 128), torch.rand(3, 64, 128)])

    assert _identical(before, _buffers(model))


def test_compiling_restores_the_models_training_mode():
    model = _model()
    model.train()

    compile_global_style(model, _lines())

    assert model.training
    assert all(m.training for m in batchnorm_modules(model))


def test_applying_a_style_changes_outputs_and_restores_on_exit():
    model = _model()
    style = compile_global_style(model, _lines())
    image = torch.rand(1, 1, 64, 224)
    before = _buffers(model)

    with torch.no_grad():
        plain = model(image).logits
        with writer_style(model, style):
            personalized = model(image).logits
        after = model(image).logits

    assert not torch.allclose(plain, personalized), "the style must actually do something"
    assert torch.equal(plain, after), "leaving the context must restore the base model exactly"
    assert _identical(before, _buffers(model))


def test_applying_restores_buffers_even_when_the_body_raises():
    model = _model()
    style = compile_global_style(model, _lines())
    before = _buffers(model)

    with pytest.raises(ValueError, match="boom"), writer_style(model, style):
        raise ValueError("boom")

    assert _identical(before, _buffers(model))


# ------------------------------------------------------------------ contract


def test_a_none_style_is_a_no_op_so_callers_need_no_branch():
    """Graceful degradation, the same shape fusion already has for a profile with no prototypes."""
    model = _model()
    image = torch.rand(1, 1, 64, 224)

    with torch.no_grad():
        plain = model(image).logits
        with writer_style(model, None):
            unchanged = model(image).logits

    assert torch.equal(plain, unchanged)


def test_a_style_from_a_different_architecture_is_refused():
    model = _model()

    with pytest.raises(ValueError, match="different architecture"), writer_style(
        model, torch.zeros(7)
    ):
        pass


def test_empty_support_is_refused_rather_than_returning_reset_statistics():
    model = _model()

    with pytest.raises(ValueError, match="at least one support line"):
        compile_global_style(model, [])


def test_more_support_lines_change_the_estimate():
    """Cumulative averaging must actually consume every line, not just the first."""
    model = _model()

    one = compile_global_style(model, _lines(1))
    three = compile_global_style(model, _lines(3))

    assert not torch.allclose(one, three)


def test_storage_cost_fits_objective_4s_budget():
    """Objective 4: tens-to-hundreds of KB, not a checkpoint."""
    model = _model()

    described = describe_style(model)

    assert described["bytes"] == style_bytes(model) == style_dimension(model) * 4
    assert 10_000 < described["bytes"] < 500_000
    assert described["n_batchnorm_layers"] > 0


# ------------------------------------------------------------------ blending


def test_base_style_matches_the_models_own_statistics():
    model = _model()

    packed = base_style(model)

    assert packed.numel() == style_dimension(model)
    with torch.no_grad():
        image = torch.rand(1, 1, 64, 224)
        plain = model(image).logits
        with writer_style(model, packed):
            reapplied = model(image).logits
    assert torch.equal(plain, reapplied), "re-applying the model's own statistics is a no-op"


def test_blend_interpolates_between_the_two_endpoints():
    model = _model()
    base = base_style(model)
    writer = compile_global_style(model, _lines())

    assert torch.allclose(blend_style(base, writer, 0.0), base)
    assert torch.allclose(blend_style(base, writer, 1.0), writer)
    half = blend_style(base, writer, 0.5)
    assert torch.allclose(half, 0.5 * base + 0.5 * writer)


def test_blend_refuses_a_weight_outside_the_unit_interval_or_a_shape_mismatch():
    model = _model()
    base = base_style(model)

    with pytest.raises(ValueError, match=r"weight must be in \[0, 1\]"):
        blend_style(base, base, 1.5)
    with pytest.raises(ValueError, match="differ"):
        blend_style(base, torch.zeros(3), 0.5)


# ------------------------------------------------------------------ profile integration


def test_compile_profile_populates_global_style_only_when_asked(synthetic_corpus):
    """The field specifies has been `None` for the project's whole life."""
    from glyphmemory.data.preprocessing import preprocess_path
    from glyphmemory.memory.compiler import compile_profile

    model, tokenizer = _model(), load_tokenizer(DEFAULT_CHARSET_PATH)
    records = list(synthetic_corpus.records)[:3]
    lines = [(preprocess_path(r.image).tensor, r.text) for r in records]

    without = compile_profile(model, tokenizer.charset, lines, model_fingerprint="d" * 16)
    with_style = compile_profile(
        model, tokenizer.charset, lines, model_fingerprint="d" * 16, with_global_style=True
    )

    assert without.global_style is None
    assert with_style.global_style is not None
    assert with_style.global_style.numel() == style_dimension(model)
    assert with_style.glyphs.keys() == without.glyphs.keys()


def test_a_profile_carrying_a_style_round_trips_through_disk(synthetic_corpus, tmp_path):
    from glyphmemory.data.preprocessing import preprocess_path
    from glyphmemory.memory.compiler import compile_profile
    from glyphmemory.memory.profile import WriterProfile

    model, tokenizer = _model(), load_tokenizer(DEFAULT_CHARSET_PATH)
    lines = [(preprocess_path(r.image).tensor, r.text) for r in list(synthetic_corpus.records)[:3]]
    profile = compile_profile(
        model, tokenizer.charset, lines, model_fingerprint="d" * 16, with_global_style=True
    )

    path = profile.save(tmp_path / "writer.pt")
    reloaded = WriterProfile.load(path)

    assert reloaded.global_style is not None
    assert torch.equal(reloaded.global_style, profile.global_style)
