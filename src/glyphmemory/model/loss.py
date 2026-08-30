"""CTC loss.

```text
logits [B,T,C] -> log_softmax(dim=-1) -> transpose(0,1) -> [T,B,C]
nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)(
    log_probs, targets, input_lengths, target_lengths)
```

None raises. All of them produce a model that trains to a plausible-looking loss and decodes
nonsense.

**On ``zero_infinity=True``.** It replaces the infinite loss of an unalignable sample with zero,
which keeps a training run alive. It also makes the sample *disappear* — no exception, no log line,
just a batch that contributed less than it appears to.

:func:`ctc_loss` therefore returns diagnostics alongside the loss rather than only a scalar. The
alternative is a training loop that reaches into the loss internals to find out what happened, which
is how the counting stops happening.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from glyphmemory.ctc.tokenizer import BLANK_INDEX
from glyphmemory.runtime.logging import get_logger

logger = get_logger("model.loss")

#: Backends with no native CTC kernel. Measured on 2026-08-18 with torch 2.13.0: MPS raises
#: ``NotImplementedError: The operator 'aten::_ctc_loss' is not currently implemented for the MPS
#: device``. The loss is therefore computed on the CPU for these backends — see :func:`ctc_loss`.
BACKENDS_WITHOUT_CTC: frozenset[str] = frozenset({"mps"})

_fallback_announced: set[str] = set()


@dataclass(frozen=True, slots=True)
class CTCDiagnostics:
    """What the loss saw.

    ``infeasible`` counts samples whose ``input_length`` cannot accommodate their target — the ones
    ``zero_infinity`` would silently zero. It is computed from the lengths directly, **not**
    inferred from the loss value, so it stays exact even when the flag has already swallowed the
    evidence.
    """

    batch_size: int
    time_steps: int
    infeasible: int
    input_length_min: int
    input_length_max: int
    target_length_min: int
    target_length_max: int
    loss_is_finite: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "time_steps": self.time_steps,
            "infeasible": self.infeasible,
            "input_length_min": self.input_length_min,
            "input_length_max": self.input_length_max,
            "target_length_min": self.target_length_min,
            "target_length_max": self.target_length_max,
            "loss_is_finite": self.loss_is_finite,
        }


def required_alignment_lengths(targets: Tensor, target_lengths: Tensor) -> Tensor:
    """Minimum frames CTC needs per sample: ``len(target) + adjacent_repeats``.

    A blank must separate identical consecutive labels, so ``ll`` needs three frames, not two. This
    mirrors :func:`glyphmemory.data.dataset.required_ctc_length`, which operates on a single
    sample's list of IDs before batching; here the same rule is applied to the flattened batch
    tensor so the check can run without unpacking it.
    """
    required = torch.zeros_like(target_lengths)
    offset = 0
    for index, length in enumerate(target_lengths.tolist()):
        segment = targets[offset : offset + length]
        repeats = int((segment[1:] == segment[:-1]).sum()) if length > 1 else 0
        required[index] = length + repeats
        offset += length
    return required


def count_infeasible(
    targets: Tensor, target_lengths: Tensor, input_lengths: Tensor
) -> tuple[int, list[int]]:
    """``(count, indices)`` of samples CTC cannot align. Should always be ``(0, [])``."""
    required = required_alignment_lengths(targets, target_lengths).to(input_lengths.device)
    mask = input_lengths < required
    return int(mask.sum()), torch.nonzero(mask).flatten().tolist()


def ctc_loss(
    logits: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    *,
    blank: int = BLANK_INDEX,
    reduction: str = "mean",
    zero_infinity: bool = True,
    strict: bool = True,
) -> tuple[Tensor, CTCDiagnostics]:
    """CTC loss over per-frame logits.

    Args:
        logits: ``[B, T, C]``, **unnormalized**. ``log_softmax`` is applied here, so a model that
            already normalized would be normalized twice.
        targets: Flattened ``[sum(target_lengths)]`` label IDs, the layout:class:`torch.nn.CTCLoss`
            expects and the one collator produces.
        input_lengths: ``[B]`` valid frames per sample, from true unpadded widths.
        target_lengths: ``[B]`` label counts per sample.
        strict: Raise when a sample is unalignable. Set ``False`` only to exercise the
            ``zero_infinity`` path deliberately.

    Returns:
        ``(loss, diagnostics)``.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B, T, C], got {tuple(logits.shape)}")

    batch_size, time_steps, vocab_size = logits.shape
    if not 0 <= blank < vocab_size:
        raise ValueError(f"blank index {blank} is outside the vocabulary of size {vocab_size}")
    if input_lengths.shape != (batch_size,) or target_lengths.shape != (batch_size,):
        raise ValueError(
            f"Expected input_lengths and target_lengths of shape [{batch_size}], got "
            f"{tuple(input_lengths.shape)} and {tuple(target_lengths.shape)}"
        )
    if int(target_lengths.sum()) != targets.numel():
        raise ValueError(
            f"target_lengths sum to {int(target_lengths.sum())} but targets holds "
            f"{targets.numel()} label(s); the flattened layout is inconsistent."
        )
    if batch_size and int(input_lengths.max()) > time_steps:
        raise ValueError(
            f"input_lengths max {int(input_lengths.max())} exceeds T={time_steps}. This is "
            "the padded-width bug: lengths must come from true widths and describe these logits."
        )

    infeasible, offenders = count_infeasible(targets, target_lengths, input_lengths)
    if infeasible:
        message = (
            f"{infeasible}/{batch_size} sample(s) cannot be aligned by CTC "
            f"(batch indices {offenders[:10]}). collator should have rejected "
            "them; zero_infinity would hide this rather than fix it."
        )
        if strict:
            raise ValueError(message)
        logger.warning("%s", message)

    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [T, B, C]

    # MPS has no CTC kernel (torch 2.13.0), so the loss runs on the CPU there. The transfer is
    # differentiable, so autograd carries the gradient back to the original device and the result is
    # numerically identical to a CPU run — verified in tests/test_loss.py. This is done in code
    # rather than by asking the user to set PYTORCH_ENABLE_MPS_FALLBACK=1, because a silent global
    # env var is a worse contract than an explicit, logged fallback.
    device_type = log_probs.device.type
    if device_type in BACKENDS_WITHOUT_CTC:
        if device_type not in _fallback_announced:
            _fallback_announced.add(device_type)
            logger.info(
                "%s has no native CTC kernel; computing the CTC loss on the CPU. Gradients "
                "flow back to %s and the value is unchanged, but each step pays a "
                "[T, B, C] transfer.",
                device_type,
                device_type,
            )
        log_probs = log_probs.cpu()
        targets = targets.cpu()

    loss = nn.functional.ctc_loss(
        log_probs,
        targets,
        # CTC wants its length tensors on the CPU; passing device tensors works but forces a
        # synchronization on every call.
        input_lengths.detach().to("cpu", dtype=torch.long),
        target_lengths.detach().to("cpu", dtype=torch.long),
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )
    if device_type in BACKENDS_WITHOUT_CTC:
        loss = loss.to(logits.device)

    diagnostics = CTCDiagnostics(
        batch_size=batch_size,
        time_steps=time_steps,
        infeasible=infeasible,
        input_length_min=int(input_lengths.min()) if batch_size else 0,
        input_length_max=int(input_lengths.max()) if batch_size else 0,
        target_length_min=int(target_lengths.min()) if batch_size else 0,
        target_length_max=int(target_lengths.max()) if batch_size else 0,
        loss_is_finite=bool(torch.isfinite(loss).all()),
    )
    return loss, diagnostics


def ctc_loss_for(
    output: Any,
    targets: Tensor,
    target_lengths: Tensor,
    **kwargs: Any,
) -> tuple[Tensor, CTCDiagnostics]:
    """:func:`ctc_loss` against an :class:`~glyphmemory.model.htr.HTROutput`.

    Takes ``input_lengths`` from the output itself, so a caller cannot pair one pass's logits with
    another's lengths — the reason lengths travel inside ``HTROutput`` at all.
    """
    return ctc_loss(output.logits, targets, output.input_lengths, target_lengths, **kwargs)
