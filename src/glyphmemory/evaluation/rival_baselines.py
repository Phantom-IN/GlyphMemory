"""Rival baselines: required comparison table.

Every fine-tune baseline restores `gm-base-v0`'s exact original weights after evaluating each
writer; forgetting that would silently leak one writer's adaptation into the next writer's
"baseline" — the single highest-risk bug in this module, guarded structurally by a context manager
rather than left to caller discipline.
"""

from __future__ import annotations

import copy
import json
import statistics
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from glyphmemory.config.schema import MemoryConfig
from glyphmemory.ctc.decode import ctc_collapse, greedy_decode
from glyphmemory.ctc.normalization import NFC_V1, normalize
from glyphmemory.ctc.tokenizer import BLANK_INDEX, Charset, Tokenizer
from glyphmemory.data.manifest import ManifestRecord
from glyphmemory.data.preprocessing import preprocess_path
from glyphmemory.data.splits import SupportQuerySplit
from glyphmemory.evaluation.few_shot import (
    DEFAULT_SHOTS,
    DEFAULT_SUPPORT_SEEDS,
    FORM_MODES,
    partition_by_form,
    sample_support_subset,
)
from glyphmemory.memory.compiler import compile_profile
from glyphmemory.memory.fusion import personalize
from glyphmemory.metrics.text import corpus_cer
from glyphmemory.model.htr import GMBase, HTROutput
from glyphmemory.model.loss import ctc_loss_for
from glyphmemory.probes.geometry import l2_normalize

#: The three gradient-based rows, plus `class_bias`.
#:
#: `class_bias` is M11-T0 (`docs/results/m11-tiny-adaptation-protocol.md` section 3): the output
#: projection's bias vector alone -- one learned scalar per output class, 80 parameters at the
#: frozen charset. It is appended rather than inserted so the three M8 rows keep their identity.
PARAMETER_GROUPS: tuple[str, ...] = ("head", "batchnorm", "full", "class_bias")

#: Every method this module scores.
ADAPTIVE_METHODS: tuple[str, ...] = ("glyphmemory", "head_ft", "batchnorm_ft", "full_ft", "replay")

_PARAMETER_GROUP_FOR_METHOD: dict[str, str] = {
    "head_ft": "head",
    "batchnorm_ft": "batchnorm",
    "full_ft": "full",
}

#: Tuned on validation-split writers (never test), per work plan step 7 -- see Log for the sanity
#: grid that picked these. One shared step count works for every group; the learning rate does not
#: -- `full` touches ~50x more parameters than `head` and destabilizes catastrophically (CER 0.97)
#: at a rate `head`/`batchnorm` tolerate fine, so the rate is per parameter-group, not a single
#: shared number forced onto all three.
DEFAULT_FT_STEPS = 20
DEFAULT_FT_LR: dict[str, float] = {"head": 1e-3, "batchnorm": 1e-3, "full": 1e-4}

#: Decoupled weight decay. **This default is torch's own, and is kept deliberately** so the
#: rival-baseline numbers (`m8-rival-baseline-table-001`, the one sanctioned IAM test-split spend)
#: stay reproducible -- they were produced by `torch.optim.AdamW(params, lr=lr)` with this value
#: implicit.
DEFAULT_FT_WEIGHT_DECAY = 0.01


def _resolve_ft_lr(ft_lr: float | dict[str, float], group: str) -> float:
    """A single float applies to every group (an explicit, deliberate override); a mapping is read
    per parameter-group, falling back to `DEFAULT_FT_LR`'s rate for a group it omits.
    """
    if isinstance(ft_lr, dict):
        return ft_lr.get(group, DEFAULT_FT_LR[group])
    return ft_lr


# ------------------------------------------------------------------ parameter groups


def _select_parameters(model: GMBase, group: str) -> list[nn.Parameter]:
    if group == "class_bias":
        # The output bias ONLY -- not `head.projection.weight` (30,720), not the head LayerNorm
        # affine (768). This is the exact functional analogue of V0 fusion, which also adds a
        # per-class constant to the logits; only the source of the constant differs.
        return [model.head.projection.bias]
    if group == "head":
        return list(model.head.parameters())
    if group == "batchnorm":
        params: list[nn.Parameter] = []
        for module in model.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                params.append(module.weight)
                params.append(module.bias)
        return params
    if group == "full":
        return list(model.parameters())
    raise ValueError(f"unknown parameter group {group!r}, expected one of {PARAMETER_GROUPS}")


def parameter_storage_bytes(model: GMBase, group: str) -> int:
    """Internal helper."""
    return sum(p.numel() for p in _select_parameters(model, group)) * 4


def trainable_parameter_count(model: GMBase, group: str) -> int:
    """Exact count of what a group would train."""
    return sum(p.numel() for p in _select_parameters(model, group))


# ------------------------------------------------------------------ fine-tuning


def _encode_lines(
    lines: Sequence[tuple[Tensor, str]], tokenizer: Tokenizer
) -> list[tuple[Tensor, list[int]]]:
    return [(image, tokenizer.encode(normalize(text, NFC_V1))) for image, text in lines]


@contextmanager
def fine_tuned_model(
    model: GMBase,
    original_state: dict[str, Any],
    parameter_group: str,
    lines: Sequence[tuple[Tensor, str]],
    tokenizer: Tokenizer,
    *,
    steps: int = DEFAULT_FT_STEPS,
    lr: float = DEFAULT_FT_LR,
    weight_decay: float = DEFAULT_FT_WEIGHT_DECAY,
    device: torch.device | str = "cpu",
):
    """Fine-tune `model` in place on `lines` for `steps` AdamW steps over `parameter_group`'s
    parameters, yield it, then **always** restore `original_state` exactly — even if the caller's
    block raises. This is the one guarantee the whole rival-baseline table depends on: forgetting it
    would leak one writer's adaptation into the next writer's "baseline" state, silently, with no
    error.

    `model` stays in `.eval()` mode throughout the fine-tune loop, deliberately never `.train()` --
    `.eval()` does not disable gradients or `requires_grad`, only Dropout and BatchNorm's
    forward-pass numerics. Keeping BatchNorm on its frozen running statistics (rather than
    re-estimating them from a 1-10 line batch) avoids a confound where "head-only fine-tune" would
    silently also be re-normalizing the encoder's activations.
    """
    resolved_device = device if isinstance(device, torch.device) else torch.device(device)
    model.load_state_dict(original_state)
    model.to(resolved_device)
    model.eval()

    original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    target_params = _select_parameters(model, parameter_group)
    target_ids = {id(p) for p in target_params}
    for p in model.parameters():
        p.requires_grad = id(p) in target_ids

    optimizer = torch.optim.AdamW(target_params, lr=lr, weight_decay=weight_decay)
    encoded = _encode_lines(lines, tokenizer)

    try:
        for _ in range(steps):
            optimizer.zero_grad()
            losses = []
            for image, target_ids_ in encoded:
                batch = image.unsqueeze(0).to(resolved_device)
                output = model(batch)
                targets = torch.tensor(target_ids_, dtype=torch.long, device=resolved_device)
                target_lengths = torch.tensor([len(target_ids_)], dtype=torch.long)
                loss, _ = ctc_loss_for(output, targets, target_lengths)
                losses.append(loss)
            torch.stack(losses).mean().backward()
            optimizer.step()
        yield model
    finally:
        model.load_state_dict(original_state)
        for name, p in model.named_parameters():
            p.requires_grad = original_requires_grad[name]
        model.eval()


# ------------------------------------------------------------------ support replay / in-context


@dataclass(frozen=True, slots=True)
class ReplayPool:
    """Individual labeled support frames — not averaged into a prototype, the methodological
    contrast with GlyphMemory's `memory/prototypes.py`.
    """

    features: Tensor  # [N, D]
    labels: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return self.features.shape[0] == 0

    def estimated_bytes(self) -> int:
        return self.features.element_size() * self.features.nelement()


def build_replay_pool(
    model: GMBase,
    charset: Charset,
    lines: Sequence[tuple[Tensor, str]],
    *,
    device: torch.device | str = "cpu",
) -> ReplayPool:
    """One `(feature, character)` row per aligned support frame — reuses the compiler's alignment
    machinery, deliberately skips its averaging step.
    """
    if not lines:
        raise ValueError("build_replay_pool requires at least one support line, got zero.")

    from glyphmemory.alignment import forced_align

    resolved_device = device if isinstance(device, torch.device) else torch.device(device)
    was_training = model.training
    model.eval()
    feature_rows: list[Tensor] = []
    labels: list[str] = []
    feature_dim: int | None = None

    try:
        with torch.no_grad():
            for image, transcript in lines:
                reference = normalize(transcript, NFC_V1)
                batch = image.unsqueeze(0).to(resolved_device)
                output = model(batch)
                length = int(output.input_lengths[0])
                features = output.sequence_features[0, :length]
                if feature_dim is None:
                    feature_dim = int(features.shape[-1])
                logits = output.logits[0, :length]
                log_probs = torch.log_softmax(logits, dim=-1)
                alignment = forced_align(log_probs, reference, charset)
                for span in alignment.spans:
                    for t in range(span.start_t, span.end_t):
                        feature_rows.append(features[t].detach().cpu())
                        labels.append(span.token)
    finally:
        model.train(was_training)

    assert feature_dim is not None  # `lines` is non-empty, so the loop ran at least once.
    if not feature_rows:
        return ReplayPool(features=torch.zeros((0, feature_dim)), labels=())
    return ReplayPool(features=torch.stack(feature_rows), labels=tuple(labels))


def replay_decode(output: HTROutput, pool: ReplayPool, tokenizer: Tokenizer) -> str:
    """Personalize one query line by hard 1-NN retrieval, not fusion.

    Segmentation (which frames emit a character at all) comes from the base model's own argmax — the
    same signal `memory/fusion.py`'s `blank_emission_gate` reads, used here as a hard switch rather
    than a soft multiplier. Only the **character identity** at a non-blank frame is replaced, by the
    nearest support frame's label. An empty pool leaves every frame exactly as the base model
    predicted it.
    """
    logits = output.logits[0]
    length = int(output.input_lengths[0])
    base_argmax = logits[:length].argmax(dim=-1)

    if pool.is_empty:
        frame_classes = base_argmax.tolist()
    else:
        features = output.sequence_features[0, :length]
        # `pool.features` is always CPU (`build_replay_pool` detaches every row there); move it to
        # match the query features' device rather than the reverse -- there is exactly one pool per
        # call here, never a batch of them (the same reasoning `memory/retrieval.py`'s
        # `memory_scores` already applies to WriterProfile prototypes).
        pool_features = pool.features.to(device=features.device, dtype=features.dtype)
        similarities = l2_normalize(features) @ l2_normalize(pool_features).T
        nearest = similarities.argmax(dim=-1)
        frame_classes = []
        for t in range(length):
            if int(base_argmax[t]) == BLANK_INDEX:
                frame_classes.append(BLANK_INDEX)
            else:
                label = pool.labels[int(nearest[t])]
                frame_classes.append(tokenizer.charset.index_of(label))

    return tokenizer.decode(ctc_collapse(frame_classes, blank=BLANK_INDEX))


# ------------------------------------------------------------------ per-writer orchestration


@dataclass(frozen=True, slots=True)
class RivalShotResult:
    """One method's outcome for one `(form_mode, n, seed)` draw."""

    method: str
    writer_id: str
    form_mode: str
    n: int
    seed: int
    sample_ids: tuple[str, ...]
    cer: float | None
    n_query_lines: int
    storage_bytes: int
    adaptation_steps: int
    wall_clock_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "writer_id": self.writer_id,
            "form_mode": self.form_mode,
            "n": self.n,
            "seed": self.seed,
            "sample_ids": list(self.sample_ids),
            "cer": self.cer,
            "n_query_lines": self.n_query_lines,
            "storage_bytes": self.storage_bytes,
            "adaptation_steps": self.adaptation_steps,
            "wall_clock_ms": self.wall_clock_ms,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RivalShotResult:
        return cls(
            method=payload["method"],
            writer_id=payload["writer_id"],
            form_mode=payload["form_mode"],
            n=payload["n"],
            seed=payload["seed"],
            sample_ids=tuple(payload["sample_ids"]),
            cer=payload["cer"],
            n_query_lines=payload["n_query_lines"],
            storage_bytes=payload["storage_bytes"],
            adaptation_steps=payload["adaptation_steps"],
            wall_clock_ms=payload["wall_clock_ms"],
        )


@dataclass(frozen=True, slots=True)
class RivalWriterCurve:
    """Every method's shots for one writer, plus the shared `CER@0` baseline."""

    writer_id: str
    cer_at_0: float | None
    n_query_lines: int
    shots: tuple[RivalShotResult, ...] = ()
    unavailable: tuple[str, ...] = ()

    def cer_values_at(self, method: str, form_mode: str, n: int) -> list[float]:
        return [
            s.cer
            for s in self.shots
            if s.method == method and s.form_mode == form_mode and s.n == n and s.cer is not None
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "writer_id": self.writer_id,
            "cer_at_0": self.cer_at_0,
            "n_query_lines": self.n_query_lines,
            "shots": [s.as_dict() for s in self.shots],
            "unavailable": list(self.unavailable),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RivalWriterCurve:
        return cls(
            writer_id=payload["writer_id"],
            cer_at_0=payload["cer_at_0"],
            n_query_lines=payload["n_query_lines"],
            shots=tuple(RivalShotResult.from_dict(s) for s in payload["shots"]),
            unavailable=tuple(payload["unavailable"]),
        )


def _run_query_forward(
    model: GMBase, query_records: Sequence[ManifestRecord], device: torch.device
) -> dict[str, HTROutput]:
    outputs: dict[str, HTROutput] = {}
    with torch.no_grad():
        for record in query_records:
            key = record.sample_id or record.image
            tensor = preprocess_path(record.image).tensor.unsqueeze(0).to(device)
            outputs[key] = model(tensor)
    return outputs


def _decode_pairs(
    query_records: Sequence[ManifestRecord], outputs: dict[str, HTROutput], tokenizer: Tokenizer
) -> list[tuple[str, str]]:
    pairs = []
    for record in query_records:
        key = record.sample_id or record.image
        output = outputs[key]
        length = int(output.input_lengths[0])
        pairs.append((record.text, greedy_decode(output.logits[0], tokenizer, length)))
    return pairs


def run_writer_rivals(
    model: GMBase,
    charset: Charset,
    tokenizer: Tokenizer,
    writer_id: str,
    support_pool: Sequence[str],
    query_pool: Sequence[str],
    records_by_id: dict[str, ManifestRecord],
    *,
    model_fingerprint: str,
    methods: Sequence[str] = ADAPTIVE_METHODS,
    shots: Sequence[int] = DEFAULT_SHOTS,
    seeds: Sequence[int] = DEFAULT_SUPPORT_SEEDS,
    memory_config: MemoryConfig | None = None,
    ft_steps: int = DEFAULT_FT_STEPS,
    ft_lr: float | dict[str, float] = DEFAULT_FT_LR,
    device: torch.device | str = "cpu",
) -> RivalWriterCurve:
    """Run every requested method for one writer, against identical support draws."""
    resolved_config = memory_config or MemoryConfig(enabled=True)
    resolved_device = device if isinstance(device, torch.device) else torch.device(device)

    # A freshly constructed nn.Module defaults to .train() mode. Every method in this sweep relies
    # on BatchNorm running its frozen running statistics (never re-estimating them from a 1-10 line
    # batch, and never drifting between writers) -- fixed here, once, rather than trusted to
    # whatever mode the caller happened to leave the model in.
    model.eval()

    query_records = [records_by_id[qid] for qid in query_pool]
    outputs = _run_query_forward(model, query_records, resolved_device)
    base_pairs = _decode_pairs(query_records, outputs, tokenizer)
    cer_at_0 = corpus_cer(base_pairs, policy=tokenizer.policy).value

    partition = partition_by_form(support_pool, query_pool, records_by_id)
    original_state = copy.deepcopy(model.state_dict())

    shot_results: list[RivalShotResult] = []
    unavailable: list[str] = []

    for form_mode in FORM_MODES:
        pool = partition.pool_for(form_mode)
        for n in shots:
            if len(pool) < n:
                unavailable.append(f"{form_mode}:n={n}")
                continue
            for seed in seeds:
                sample_ids = sample_support_subset(pool, n, seed, key=f"{writer_id}:{form_mode}")
                lines = [
                    (preprocess_path(records_by_id[sid].image).tensor, records_by_id[sid].text)
                    for sid in sample_ids
                ]

                for method in methods:
                    start = time.perf_counter()
                    if method == "glyphmemory":
                        profile = compile_profile(
                            model, charset, lines, model_fingerprint=model_fingerprint,
                            config=resolved_config, device=resolved_device,
                        )
                        pairs = [
                            (
                                record.text,
                                greedy_decode(
                                    personalize(
                                        outputs[record.sample_id or record.image],
                                        profile, charset, resolved_config,
                                    )[0],
                                    tokenizer,
                                    int(outputs[record.sample_id or record.image].input_lengths[0]),
                                ),
                            )
                            for record in query_records
                        ]
                        storage_bytes = profile.estimated_bytes()
                        steps = 0
                    elif method == "replay":
                        pool_obj = build_replay_pool(model, charset, lines, device=resolved_device)
                        pairs = [
                            (
                                record.text,
                                replay_decode(
                                    outputs[record.sample_id or record.image], pool_obj, tokenizer
                                ),
                            )
                            for record in query_records
                        ]
                        storage_bytes = pool_obj.estimated_bytes()
                        steps = 0
                    elif method in _PARAMETER_GROUP_FOR_METHOD:
                        group = _PARAMETER_GROUP_FOR_METHOD[method]
                        with fine_tuned_model(
                            model, original_state, group, lines, tokenizer,
                            steps=ft_steps, lr=_resolve_ft_lr(ft_lr, group),
                            device=resolved_device,
                        ) as ft_model:
                            ft_outputs = _run_query_forward(
                                ft_model, query_records, resolved_device
                            )
                            pairs = _decode_pairs(query_records, ft_outputs, tokenizer)
                        storage_bytes = parameter_storage_bytes(model, group)
                        steps = ft_steps
                    else:
                        raise ValueError(
                            f"unknown method {method!r}, expected one of {ADAPTIVE_METHODS}"
                        )

                    wall_ms = (time.perf_counter() - start) * 1000.0
                    shot_results.append(
                        RivalShotResult(
                            method=method, writer_id=writer_id, form_mode=form_mode, n=n,
                            seed=seed, sample_ids=sample_ids,
                            cer=corpus_cer(pairs, policy=tokenizer.policy).value,
                            n_query_lines=len(query_records), storage_bytes=storage_bytes,
                            adaptation_steps=steps, wall_clock_ms=wall_ms,
                        )
                    )

    return RivalWriterCurve(
        writer_id=writer_id, cer_at_0=cer_at_0, n_query_lines=len(query_records),
        shots=tuple(shot_results), unavailable=tuple(unavailable),
    )


# ------------------------------------------------------------------ the comparison table


@dataclass(frozen=True, slots=True)
class MethodSummary:
    """One row of the comparison table, for one method at one `(form_mode, n)`: adaptation
    cost, storage per writer, and the `CER@n` gain it actually achieved through the identical
    query pools and support draws every other method used.
    """

    method: str
    form_mode: str
    n: int
    n_writers: int
    mean_cer: float | None
    mean_gain: float | None
    median_gain: float | None
    mean_storage_bytes: float | None
    mean_adaptation_steps: float | None
    mean_wall_clock_ms: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "form_mode": self.form_mode,
            "n": self.n,
            "n_writers": self.n_writers,
            "mean_cer": self.mean_cer,
            "mean_gain": self.mean_gain,
            "median_gain": self.median_gain,
            "mean_storage_bytes": self.mean_storage_bytes,
            "mean_adaptation_steps": self.mean_adaptation_steps,
            "mean_wall_clock_ms": self.mean_wall_clock_ms,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MethodSummary:
        return cls(**payload)


def summarize_methods(
    curves: Sequence[RivalWriterCurve],
    *,
    methods: Sequence[str] = ADAPTIVE_METHODS,
    shots: Sequence[int] = DEFAULT_SHOTS,
    form_modes: Sequence[str] = FORM_MODES,
) -> tuple[MethodSummary, ...]:
    """One `MethodSummary` per `(method, form_mode, n)`, pooling every writer's curve."""
    results: list[MethodSummary] = []
    for method in methods:
        for form_mode in form_modes:
            for n in shots:
                cers: list[float] = []
                gains: list[float] = []
                storages: list[int] = []
                step_counts: list[int] = []
                wall_clocks: list[float] = []

                for curve in curves:
                    matching = [
                        s for s in curve.shots if s.method == method and s.form_mode == form_mode
                        and s.n == n and s.cer is not None
                    ]
                    if not matching or curve.cer_at_0 is None:
                        continue
                    mean_cer_for_writer = statistics.mean(s.cer for s in matching)
                    cers.append(mean_cer_for_writer)
                    gains.append(curve.cer_at_0 - mean_cer_for_writer)
                    storages.extend(s.storage_bytes for s in matching)
                    step_counts.extend(s.adaptation_steps for s in matching)
                    wall_clocks.extend(s.wall_clock_ms for s in matching)

                results.append(
                    MethodSummary(
                        method=method,
                        form_mode=form_mode,
                        n=n,
                        n_writers=len(cers),
                        mean_cer=statistics.mean(cers) if cers else None,
                        mean_gain=statistics.mean(gains) if gains else None,
                        median_gain=statistics.median(gains) if gains else None,
                        mean_storage_bytes=statistics.mean(storages) if storages else None,
                        mean_adaptation_steps=(
                            statistics.mean(step_counts) if step_counts else None
                        ),
                        mean_wall_clock_ms=statistics.mean(wall_clocks) if wall_clocks else None,
                    )
                )
    return tuple(results)


@dataclass(frozen=True, slots=True)
class RivalBaselineReport:
    """A complete rival-baseline run: every writer's curve across every method, plus the per-method
    summary table.
    """

    checkpoint: str
    manifest: str
    split: str
    device: str
    methods: tuple[str, ...]
    shots: tuple[int, ...]
    seeds: tuple[int, ...]
    ft_steps: int
    ft_lr: float | dict[str, float]
    curves: tuple[RivalWriterCurve, ...]
    summaries: tuple[MethodSummary, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "manifest": self.manifest,
            "split": self.split,
            "device": self.device,
            "methods": list(self.methods),
            "shots": list(self.shots),
            "seeds": list(self.seeds),
            "ft_steps": self.ft_steps,
            "ft_lr": self.ft_lr,
            "curves": [c.as_dict() for c in self.curves],
            "summaries": [s.as_dict() for s in self.summaries],
        }

    def format(self) -> str:
        lines = [
            f"checkpoint   {self.checkpoint}",
            f"manifest     {self.manifest}   split {self.split}",
            f"device       {self.device}   ft_steps {self.ft_steps}   ft_lr {self.ft_lr}",
            f"writers      {len(self.curves)}   methods {list(self.methods)}",
            "",
        ]
        for s in self.summaries:
            cer = "n/a" if s.mean_cer is None else f"{s.mean_cer:.4f}"
            gain = "n/a" if s.mean_gain is None else f"{s.mean_gain:+.4f}"
            storage = "n/a" if s.mean_storage_bytes is None else f"{s.mean_storage_bytes:,.0f}B"
            steps = "n/a" if s.mean_adaptation_steps is None else f"{s.mean_adaptation_steps:.0f}"
            lines.append(
                f"[{s.method:>12} {s.form_mode:>10} n={s.n:>2}] writers {s.n_writers:>3}   "
                f"CER {cer}   gain {gain}   storage {storage}   steps {steps}"
            )
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> RivalBaselineReport:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            checkpoint=payload["checkpoint"],
            manifest=payload["manifest"],
            split=payload["split"],
            device=payload["device"],
            methods=tuple(payload["methods"]),
            shots=tuple(payload["shots"]),
            seeds=tuple(payload["seeds"]),
            ft_steps=payload["ft_steps"],
            ft_lr=payload["ft_lr"],
            curves=tuple(RivalWriterCurve.from_dict(c) for c in payload["curves"]),
            summaries=tuple(MethodSummary.from_dict(s) for s in payload["summaries"]),
        )


def build_rival_baseline_report(
    model: GMBase,
    charset: Charset,
    tokenizer: Tokenizer,
    support_query: SupportQuerySplit,
    records_by_id: dict[str, ManifestRecord],
    *,
    checkpoint_label: str,
    manifest_label: str,
    split_name: str,
    model_fingerprint: str,
    writers: Sequence[str] | None = None,
    methods: Sequence[str] = ADAPTIVE_METHODS,
    shots: Sequence[int] = DEFAULT_SHOTS,
    seeds: Sequence[int] = DEFAULT_SUPPORT_SEEDS,
    memory_config: MemoryConfig | None = None,
    ft_steps: int = DEFAULT_FT_STEPS,
    ft_lr: float | dict[str, float] = DEFAULT_FT_LR,
    device: torch.device | str = "cpu",
) -> RivalBaselineReport:
    """Run every method over every requested writer and assemble the comparison report."""
    resolved_device = device if isinstance(device, torch.device) else torch.device(device)
    target_writers = writers or sorted(
        w
        for w in support_query.writers
        if support_query.query_for(w) and support_query.support_for(w)
    )

    curves = tuple(
        run_writer_rivals(
            model,
            charset,
            tokenizer,
            writer_id,
            support_query.support_for(writer_id),
            support_query.query_for(writer_id),
            records_by_id,
            model_fingerprint=model_fingerprint,
            methods=methods,
            shots=shots,
            seeds=seeds,
            memory_config=memory_config,
            ft_steps=ft_steps,
            ft_lr=ft_lr,
            device=resolved_device,
        )
        for writer_id in target_writers
    )

    return RivalBaselineReport(
        checkpoint=checkpoint_label,
        manifest=manifest_label,
        split=split_name,
        device=str(resolved_device),
        methods=tuple(methods),
        shots=tuple(shots),
        seeds=tuple(seeds),
        ft_steps=ft_steps,
        ft_lr=ft_lr,
        curves=curves,
        summaries=summarize_methods(curves, methods=methods, shots=shots, form_modes=FORM_MODES),
    )
