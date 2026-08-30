"""Global writer style: the second half of the profile.

    WriterProfile
    |-- global_style   <- this module: one compact vector of writer-level distribution statistics
    `-- glyph_memory      per-character visual prototypes (memory/compiler.py, M7)

Only the second half has ever been built.

**What is stored, and why this shape.** BatchNorm running statistics, re-estimated from the writer's
own support lines. So this is not a new idea bolted on; it is the gradient-free form of the one
adaptation this project already measured working.

**Invariant 4 holds by construction.** Estimation is forward passes only -- no optimizer, no
``backward()``, no gradient anywhere. PyTorch's own BatchNorm accumulator does the arithmetic
(``momentum=None``, its cumulative-moving-average path), so the statistic is the library's, not a
hand-rolled reimplementation. Asserted behaviourally in this module's tests, not merely intended.

**The frozen model is never mutated.** Both:func:`compile_global_style` and:func:`writer_style` save
every buffer they touch and restore it on the way out, including on exception -- `gm-base-v0` must
be byte-identical after enrollment (ADR-0008).

**No schema-version bump, deliberately.** `memory/profile.py` added ``global_style`` to the schema
up front with the stated intent that "adding it later is not a schema-version bump", and the layout
is recoverable from the model rather than stored, so no new field is needed either.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm

from glyphmemory.model.htr import GMBase

#: Bytes per stored scalar. Two vectors (mean, variance) per BatchNorm channel, float32.
BYTES_PER_SCALAR = 4


def batchnorm_modules(model: nn.Module) -> list[_BatchNorm]:
    """Every BatchNorm module, in ``named_modules`` order.

    Order is the contract between :func:`compile_global_style` and :func:`writer_style`: the vector
    is packed and unpacked against this sequence, so both must walk it identically.
    ``named_modules`` is deterministic for a fixed architecture, and the architecture is pinned by
    the profile's ``model_fingerprint`` -- which is why the layout does not need storing.
    """
    return [module for _, module in model.named_modules() if isinstance(module, _BatchNorm)]


def style_dimension(model: nn.Module) -> int:
    """Length of the vector :func:`compile_global_style` produces for ``model``."""
    return sum(2 * module.num_features for module in batchnorm_modules(model))


def style_bytes(model: nn.Module) -> int:
    """Storage cost of one writer's global style, against Objective 4."""
    return style_dimension(model) * BYTES_PER_SCALAR


def describe_style(model: nn.Module) -> dict[str, Any]:
    """Layout summary for a run record -- what was measured, not just how big it is."""
    modules = batchnorm_modules(model)
    return {
        "n_batchnorm_layers": len(modules),
        "total_channels": sum(m.num_features for m in modules),
        "dimension": style_dimension(model),
        "bytes": style_bytes(model),
    }


def base_style(model: nn.Module) -> Tensor:
    """The model's *own* BatchNorm statistics, packed in the same layout as a writer's.

    Estimated over the whole training corpus, so it is the low-variance reference a few-line writer
    estimate can be blended toward (:func:`blend_style`).
    """
    return torch.cat(
        [
            torch.cat([module.running_mean.detach(), module.running_var.detach()])
            for module in batchnorm_modules(model)
        ]
    ).clone()


def blend_style(base: Tensor, writer: Tensor, weight: float) -> Tensor:
    """``(1 - weight) * base + weight * writer`` -- a bias/variance dial on the style estimate.

    Intermediate weights keep most of the stable estimate while still moving toward the writer, the
    same shape `memory/fusion.py`'s ``alpha`` gives the prototype path.

    Raises:
        ValueError: ``weight`` outside ``[0, 1]``, or the two styles differ in length.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    if base.shape != writer.shape:
        raise ValueError(f"base {tuple(base.shape)} and writer {tuple(writer.shape)} differ.")
    return (1.0 - weight) * base + weight * writer


@contextmanager
def _statistic_estimation(model: GMBase) -> Iterator[list[_BatchNorm]]:
    """Put BatchNorm into cumulative-average estimation mode and restore everything afterwards.

    ``model.eval()`` first, then only the BatchNorm modules back to train mode: dropout must stay
    off (its noise would perturb the activations feeding later BatchNorm layers) while BatchNorm
    must use batch statistics and accumulate them. ``momentum=None`` selects PyTorch's cumulative
    moving average, so the result is the average over all support lines rather than an
    exponentially-weighted trace dominated by the last one.
    """
    modules = batchnorm_modules(model)
    was_training = model.training
    saved = [
        (
            module.training,
            module.momentum,
            module.running_mean.detach().clone() if module.running_mean is not None else None,
            module.running_var.detach().clone() if module.running_var is not None else None,
            module.num_batches_tracked.detach().clone()
            if module.num_batches_tracked is not None
            else None,
        )
        for module in modules
    ]
    model.eval()
    try:
        for module in modules:
            module.reset_running_stats()
            module.momentum = None
            module.train()
        yield modules
    finally:
        for module, (training, momentum, mean, var, tracked) in zip(modules, saved, strict=True):
            module.momentum = momentum
            module.train(training)
            if mean is not None:
                module.running_mean.copy_(mean)
            if var is not None:
                module.running_var.copy_(var)
            if tracked is not None:
                module.num_batches_tracked.copy_(tracked)
        model.train(was_training)


def compile_global_style(
    model: GMBase,
    support_images: Sequence[Tensor],
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Estimate one writer's BatchNorm statistics from their support lines. Forward passes only.

    Args:
        support_images: Preprocessed ``[1, H, W]`` line tensors. Run one at a time because IAM lines
            differ in width and padding a batch would fold background pixels into the very
            statistics being estimated.

    Returns:
        A detached 1-D tensor, ``[sum(2 * C)]``, packed as ``mean, var`` per BatchNorm module in
        :func:`batchnorm_modules` order.

    Raises:
        ValueError: ``support_images`` is empty -- statistics estimated from nothing would silently
            be the reset (0, 1) values, which is not a writer's style but a broken profile.
    """
    if not support_images:
        raise ValueError(
            "compile_global_style needs at least one support line; estimating BatchNorm "
            "statistics from an empty support set would return reset values, not a writer style."
        )
    resolved = device if isinstance(device, torch.device) else torch.device(device)
    model.to(resolved)

    with _statistic_estimation(model) as modules, torch.no_grad():
        for image in support_images:
            model(image.unsqueeze(0).to(resolved))
        style = torch.cat(
            [
                torch.cat([module.running_mean.detach(), module.running_var.detach()])
                for module in modules
            ]
        ).clone()
    return style


@contextmanager
def writer_style(model: GMBase, style: Tensor | None) -> Iterator[GMBase]:
    """Run ``model`` with one writer's BatchNorm statistics in place, then restore the originals.

    ``style=None`` is a no-op, so a caller can wrap an inference path unconditionally and let the
    profile decide whether personalization happens -- the same graceful-degradation shape
    `memory/fusion.py` already has for a profile with no prototypes.

    Raises:
        ValueError: ``style`` does not match ``model``'s BatchNorm layout. A length mismatch means
            the style came from a different architecture, and silently applying a prefix of it would
            corrupt normalization with no error.
    """
    if style is None:
        yield model
        return

    modules = batchnorm_modules(model)
    expected = style_dimension(model)
    if style.numel() != expected:
        raise ValueError(
            f"global_style has {style.numel()} values but this model's BatchNorm layout needs "
            f"{expected} ({len(modules)} layers); the style was compiled against a different "
            "architecture."
        )

    saved = [
        (module.running_mean.detach().clone(), module.running_var.detach().clone())
        for module in modules
    ]
    flat = style.detach().to(next(model.parameters()).device)
    offset = 0
    try:
        for module in modules:
            channels = module.num_features
            module.running_mean.copy_(flat[offset : offset + channels])
            offset += channels
            module.running_var.copy_(flat[offset : offset + channels])
            offset += channels
        yield model
    finally:
        for module, (mean, var) in zip(modules, saved, strict=True):
            module.running_mean.copy_(mean)
            module.running_var.copy_(var)
