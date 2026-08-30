"""`gm-base-v0` is frozen against ADR-0008 — a checkpoint whose recorded CER cannot be reproduced is
not frozen in any useful sense, it is a filename.

Neither the checkpoint nor the IAM manifest is committed to git (`artifacts/*` and `*.pt` are
gitignored by design, and IAM is licensed data this repo never redistributes), so these tests skip
rather than fail when either is absent — that is the expected state on a fresh clone or in CI, not a
bug. They run for real on a machine that has done the freeze locally.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from glyphmemory.config.loader import load_config
from glyphmemory.ctc import load_tokenizer
from glyphmemory.evaluation import evaluate_checkpoint
from glyphmemory.model import GMBase
from glyphmemory.training.checkpoint import load_checkpoint

FROZEN_CHECKPOINT = Path("artifacts/gm-base-v0.pt")
FROZEN_CONFIG = Path("artifacts/gm-base-v0.config.yaml")
IAM_MANIFEST = Path("artifacts/iam/splits/manifest.jsonl")
CHARSET = Path("artifacts/charset_en_v1.json")

#: ADR-0008's recorded numbers — this test's job is to prove they still hold, not to redefine them,
#: so they are pinned as literals rather than read back out of the ADR.
RECORDED_MODEL_FINGERPRINT = "fea77c9aaafd52d4"
RECORDED_PARAMETER_COUNT = 1_544_560
RECORDED_TEST_CER = 0.07237160852713179

_FREEZE_MISSING = not (
    FROZEN_CHECKPOINT.exists() and FROZEN_CONFIG.exists() and IAM_MANIFEST.exists()
)
_SKIP_REASON = (
    "gm-base-v0 freeze artifacts or the IAM manifest are not present locally — neither is "
    "committed (see ADR-0008); this test only runs where the freeze has actually been done"
)

pytestmark = pytest.mark.skipif(_FREEZE_MISSING, reason=_SKIP_REASON)


def test_frozen_checkpoint_fingerprint_is_unchanged():
    """The freeze's whole point: if this ever changes, the checkpoint moved and every + delta
    measured against the old one stops being interpretable (ADR-0008).
    """
    from glyphmemory.benchmark.context import checkpoint_fingerprint

    assert checkpoint_fingerprint(FROZEN_CHECKPOINT) == RECORDED_MODEL_FINGERPRINT


def test_frozen_checkpoint_loads_and_matches_the_recorded_parameter_count():
    tokenizer = load_tokenizer(CHARSET)
    loaded = load_checkpoint(FROZEN_CHECKPOINT, charset_fingerprint=tokenizer.charset.fingerprint())
    model = GMBase.from_config(load_config(FROZEN_CONFIG).model, tokenizer.vocab_size)
    model.load_state_dict(loaded.model_state)
    assert sum(p.numel() for p in model.parameters()) == RECORDED_PARAMETER_COUNT


@pytest.mark.slow
def test_frozen_checkpoint_reproduces_its_recorded_test_cer():
    """The full-split re-evaluation ADR-0008 exists to make checkable, not just claim."""
    config = load_config(FROZEN_CONFIG)
    tokenizer = load_tokenizer(CHARSET)

    report = evaluate_checkpoint(
        FROZEN_CHECKPOINT,
        IAM_MANIFEST,
        config=config,
        tokenizer=tokenizer,
        split="test",
        device=torch.device("cpu"),
    )

    assert report.cer.value == pytest.approx(RECORDED_TEST_CER, abs=1e-6)
