"""Miniature tiny-overfit regression test.

**It asserts in train mode.** That is not a shortcut — measured that on a corpus this small,
BatchNorm running statistics cannot reproduce batch statistics, because with only a couple of
batches BatchNorm leaks batch identity and the model learns to use it. Asserting in eval mode here
would encode that artifact as a requirement. What the test is for is the recognition chain — data,
labels, lengths, CTC layout, decoder — and that chain is exercised identically in either mode.
"""

from __future__ import annotations

import pytest
import torch

from glyphmemory.config.schema import AugmentationConfig, Config, DataConfig, ModelConfig
from glyphmemory.ctc import DEFAULT_CHARSET_PATH, decode_output, load_tokenizer
from glyphmemory.data import build_dataloader, build_dataset
from glyphmemory.metrics import corpus_cer
from glyphmemory.model import GMBase, ctc_loss_for

#: Loose on purpose. The gate's real threshold is 2%; this guards against a broken pipeline, not
#: against a small regression in convergence speed.
THRESHOLD = 0.10
STEPS = 400


@pytest.fixture(scope="module")
def gate_config() -> Config:
    """The gate's conditions: no augmentation, no dropout."""
    return Config(
        data=DataConfig(augmentation=AugmentationConfig(enabled=False)),
        model=ModelConfig(gru_dropout=0.0, head_dropout=0.0),
    )


@pytest.mark.slow
def test_model_memorizes_a_handful_of_synthetic_lines(synthetic_corpus, gate_config) -> None:
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    torch.manual_seed(1337)

    dataset = build_dataset(
        synthetic_corpus.manifest_path, tokenizer, gate_config, training=False
    ).take(4)
    loader = build_dataloader(
        dataset, gate_config, training=False, batch_size=4, bucket=False, num_workers=0
    )

    model = GMBase.from_config(gate_config.model, tokenizer.vocab_size).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = next(iter(loader))
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = ctc_loss_for(
            model(batch.images, batch.input_lengths), batch.targets, batch.target_lengths
        )
        assert diagnostics.infeasible == 0
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        output = model(batch.images, batch.input_lengths)
    result = corpus_cer(list(zip(batch.texts, decode_output(output, tokenizer), strict=True)))

    assert result.value is not None
    assert result.value <= THRESHOLD, (
        f"CER {result.value:.4f} after {STEPS} steps on {batch.batch_size} lines. The model "
        "cannot memorize a handful of samples, so the failure is in the recognition chain — "
        "Input_lengths -> target_lengths -> CTC layout -> "
        "blank index -> decoder collapse."
    )


@pytest.mark.slow
def test_a_single_line_is_memorized_exactly(synthetic_corpus, gate_config) -> None:
    """One sample first: if this fails it is not a capacity problem."""
    tokenizer = load_tokenizer(DEFAULT_CHARSET_PATH)
    torch.manual_seed(0)

    dataset = build_dataset(
        synthetic_corpus.manifest_path, tokenizer, gate_config, training=False
    ).take(1)
    loader = build_dataloader(
        dataset, gate_config, training=False, batch_size=1, bucket=False, num_workers=0
    )
    batch = next(iter(loader))

    model = GMBase.from_config(gate_config.model, tokenizer.vocab_size).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = ctc_loss_for(
            model(batch.images, batch.input_lengths), batch.targets, batch.target_lengths
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        decoded = decode_output(model(batch.images, batch.input_lengths), tokenizer)
    assert decoded[0] == batch.texts[0], f"got {decoded[0]!r}, want {batch.texts[0]!r}"


def test_gate_conditions_are_what_the_committed_config_says() -> None:
    """The gate's conditions live in a committed file, not in a memory of how it was run."""
    import json
    from pathlib import Path

    from glyphmemory.config.loader import load_config

    config = load_config(Path("configs/tiny_overfit.yaml"))
    assert config.data.augmentation.enabled is False
    assert config.model.gru_dropout == 0.0
    assert config.model.head_dropout == 0.0
    assert config.model.name.startswith("diag_")

    samples = json.loads(Path("configs/tiny_overfit_samples.json").read_text())
    assert len(samples["sample_ids"]) == 64
    assert len(set(samples["sample_ids"])) == 64
    assert samples["criterion"]["threshold"] == 0.02
    assert samples["criterion"]["max_epochs"] == 500
