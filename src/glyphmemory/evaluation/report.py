"""Runs a checkpoint over a manifest split and assembles the evaluation report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from glyphmemory.config.schema import Config
from glyphmemory.ctc.decode import DEFAULT_DECODER, decode_output
from glyphmemory.ctc.tokenizer import Tokenizer
from glyphmemory.data import build_dataloader, build_dataset
from glyphmemory.evaluation.per_writer import PerWriterDistribution, per_writer_distribution
from glyphmemory.evaluation.taxonomy import ErrorTaxonomy, build_taxonomy
from glyphmemory.metrics.text import MetricResult, SampleMetric, corpus_cer, corpus_wer
from glyphmemory.model.htr import GMBase
from glyphmemory.training.checkpoint import load_checkpoint

CER_GATE_THRESHOLD = 0.10
TAIL_GATE_THRESHOLD = 2.0


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """The four exit-gate conditions, evaluated where a number can decide them.

    Condition 2 — "error analysis shows no systematic pipeline bug" — is a judgment call about
    whether errors look like handwriting mistakes or like a length/decode/alignment bug. This
    computes supporting evidence (the worst lines are attached to the report) but does **not**
    decide it: ``pipeline_bug_reviewed`` stays ``None``, and so does the overall verdict, until a
    human records one. Auto-passing a qualitative check would defeat the reason it exists.
    """

    cer_threshold: float
    cer_value: float | None
    cer_pass: bool | None

    tail_threshold: float
    worst_decile_ratio: float | None
    tail_pass: bool | None

    taxonomy_recorded: bool

    pipeline_bug_reviewed: bool | None = None
    pipeline_bug_notes: str | None = None

    @property
    def passed(self) -> bool | None:
        """Overall verdict. ``None`` until the pipeline-bug review is recorded."""
        if self.pipeline_bug_reviewed is None or self.cer_pass is None or self.tail_pass is None:
            return None
        return bool(
            self.cer_pass
            and self.tail_pass
            and self.taxonomy_recorded
            and self.pipeline_bug_reviewed
        )

    def reviewed(self, *, no_pipeline_bug: bool, notes: str) -> GateVerdict:
        """Record the human review for condition 2, returning the completed verdict."""
        return GateVerdict(
            cer_threshold=self.cer_threshold,
            cer_value=self.cer_value,
            cer_pass=self.cer_pass,
            tail_threshold=self.tail_threshold,
            worst_decile_ratio=self.worst_decile_ratio,
            tail_pass=self.tail_pass,
            taxonomy_recorded=self.taxonomy_recorded,
            pipeline_bug_reviewed=no_pipeline_bug,
            pipeline_bug_notes=notes,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition_1_cer": {
                "threshold": self.cer_threshold,
                "value": self.cer_value,
                "passed": self.cer_pass,
            },
            "condition_2_no_pipeline_bug": {
                "reviewed": self.pipeline_bug_reviewed,
                "notes": self.pipeline_bug_notes,
            },
            "condition_3_heavy_tail": {
                "threshold": self.tail_threshold,
                "worst_decile_ratio": self.worst_decile_ratio,
                "passed": self.tail_pass,
            },
            "condition_4_taxonomy_recorded": self.taxonomy_recorded,
            "overall_passed": self.passed,
        }

    def format(self) -> str:
        def mark(value: bool | None) -> str:
            return "n/a" if value is None else ("PASS" if value else "FAIL")

        cer_v = "n/a" if self.cer_value is None else f"{self.cer_value:.4f}"
        ratio_v = "n/a" if self.worst_decile_ratio is None else f"{self.worst_decile_ratio:.2f}x"
        cond1 = (
            f"  1. CER <= {self.cer_threshold:.2f}                value {cer_v}      "
            f"{mark(self.cer_pass)}"
        )
        cond2 = (
            f"  2. no systematic pipeline bug     reviewed {self.pipeline_bug_reviewed!r}   "
            f"{mark(self.pipeline_bug_reviewed)}"
        )
        cond3 = (
            f"  3. worst-decile >= {self.tail_threshold:.1f}x median   ratio {ratio_v}   "
            f"{mark(self.tail_pass)}"
        )
        cond4 = f"  4. taxonomy recorded              {self.taxonomy_recorded}"
        overall = f"  overall                           {mark(self.passed)}"
        return "\n".join(["gate", cond1, cond2, cond3, cond4, overall])


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Everything requires for one checkpoint on one split."""

    checkpoint: str
    manifest: str
    split: str
    device: str
    cer: MetricResult
    wer: MetricResult
    per_writer: PerWriterDistribution
    taxonomy: ErrorTaxonomy
    gate: GateVerdict
    worst_examples: tuple[SampleMetric, ...] = ()

    def as_dict(self, *, include_samples: bool = False) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "manifest": self.manifest,
            "split": self.split,
            "device": self.device,
            "cer": self.cer.as_dict(include_samples=include_samples),
            "wer": self.wer.as_dict(include_samples=include_samples),
            "per_writer": self.per_writer.as_dict(),
            "taxonomy": self.taxonomy.as_dict(),
            "gate": self.gate.as_dict(),
            "worst_examples": [s.as_dict() for s in self.worst_examples],
        }

    def format(self) -> str:
        lines = [
            f"checkpoint   {self.checkpoint}",
            f"manifest     {self.manifest}   split {self.split}",
            f"device       {self.device}",
            "",
            self.cer.format(),
            "",
            self.wer.format(),
            "",
            self.per_writer.format(),
            "",
            self.taxonomy.format(),
            "",
            self.gate.format(),
            "",
            "worst lines (for the pipeline-bug review):",
        ]
        for sample in self.worst_examples:
            rate = "n/a" if sample.error_rate is None else f"{sample.error_rate:.4f}"
            lines.append(f"  [{rate}] {sample.sample_id}")
            lines.append(f"    ref  {sample.reference!r}")
            lines.append(f"    hyp  {sample.hypothesis!r}")
        return "\n".join(lines)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    config: Config,
    tokenizer: Tokenizer,
    split: str = "test",
    device: torch.device | str = "cpu",
    batch_size: int | None = None,
    num_workers: int | None = None,
    worst_n: int = 10,
) -> EvaluationReport:
    """Load ``checkpoint_path``, run it over ``split`` of ``manifest_path``, and score it.

    Raises the same :class:`~glyphmemory.training.checkpoint.CheckpointCompatibilityError` as
    training if the checkpoint's charset does not match ``tokenizer`` — a mismatched checkpoint is
    not silently evaluated against the wrong vocabulary.
    """
    resolved_device = device if isinstance(device, torch.device) else torch.device(device)

    loaded = load_checkpoint(checkpoint_path, charset_fingerprint=tokenizer.charset.fingerprint())
    model = GMBase.from_config(config.model, tokenizer.vocab_size)
    model.load_state_dict(loaded.model_state)
    model.to(resolved_device)
    model.eval()

    dataset = build_dataset(manifest_path, tokenizer, config, split=split, training=False)
    if len(dataset) == 0:
        raise ValueError(f"no records for split={split!r} in {manifest_path}")
    loader = build_dataloader(
        dataset,
        config,
        training=False,
        batch_size=batch_size,
        bucket=False,
        num_workers=num_workers,
    )

    pairs: list[tuple[str, str]] = []
    sample_ids: list[str] = []
    writer_records: list[tuple[str, str, str]] = []

    with torch.no_grad():
        for batch in loader:
            if batch.is_empty:
                continue
            batch = batch.to(resolved_device)
            output = model(batch.images, batch.input_lengths)
            predictions = decode_output(output, tokenizer)
            for writer_id, sample_id, reference, hypothesis in zip(
                batch.writer_ids, batch.sample_ids, batch.texts, predictions, strict=True
            ):
                pairs.append((reference, hypothesis))
                sample_ids.append(sample_id)
                writer_records.append((writer_id, reference, hypothesis))

    cer = corpus_cer(pairs, policy=tokenizer.policy, decoder=DEFAULT_DECODER, sample_ids=sample_ids)
    wer = corpus_wer(pairs, policy=tokenizer.policy, decoder=DEFAULT_DECODER, sample_ids=sample_ids)
    per_writer = per_writer_distribution(writer_records, policy=tokenizer.policy)
    taxonomy = build_taxonomy(pairs, policy=tokenizer.policy)

    gate = GateVerdict(
        cer_threshold=CER_GATE_THRESHOLD,
        cer_value=cer.value,
        cer_pass=None if cer.value is None else cer.value <= CER_GATE_THRESHOLD,
        tail_threshold=TAIL_GATE_THRESHOLD,
        worst_decile_ratio=per_writer.worst_decile_ratio,
        tail_pass=per_writer.passes_tail_condition,
        taxonomy_recorded=True,
    )

    return EvaluationReport(
        checkpoint=str(checkpoint_path),
        manifest=str(manifest_path),
        split=split,
        device=str(resolved_device),
        cer=cer,
        wer=wer,
        per_writer=per_writer,
        taxonomy=taxonomy,
        gate=gate,
        worst_examples=cer.worst(worst_n),
    )
