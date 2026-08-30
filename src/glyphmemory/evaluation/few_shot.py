"""The few-shot evaluation harness, turned into code.

    0. query pool fixed first, per writer, never re-derived           (SupportQuerySplit)
    1. CER@0 over that fixed query pool, no profile
    2. for n in 1, 3, 5, 10: draw an n-line support subset, several deterministic seeds
    3. compile a profile from exactly that subset, personalize the SAME query pool -> CER@n
    4. repeat for same-form and cross-form support separately

**The query pool never changes.** `SupportQuerySplit` already reserves it once, before support is
ever drawn; this module never recomputes "the query set" per `n`, because that would change the
denominator with `n` and mix genuine adaptation with query-set drift (named failure mode).

**The base forward pass runs once per query line, not once per shot.** `personalize` is a post-hoc
transform of a frozen model's output (`memory/fusion.py`), so every shot for a writer reuses the
same cached `HTROutput` for that writer's query lines — only the compiled profile differs between
shots. Recomputing the base forward pass per shot would be correct but wasteful, and at up to
`len(FORM_MODES) * len(shots) * len(seeds)` shots per writer, the difference is not marginal.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from glyphmemory.config.schema import MemoryConfig
from glyphmemory.ctc.decode import greedy_decode
from glyphmemory.ctc.tokenizer import Charset, Tokenizer
from glyphmemory.data.manifest import ManifestRecord
from glyphmemory.data.preprocessing import preprocess_path
from glyphmemory.data.splits import SupportQuerySplit
from glyphmemory.memory.compiler import compile_profile
from glyphmemory.memory.fusion import personalize
from glyphmemory.memory.projection import GlyphProjection
from glyphmemory.metrics.text import corpus_cer
from glyphmemory.model.htr import GMBase, HTROutput

#: "repeat support sampling with several deterministic seeds" -- a number fixed once here so
#: rival-baseline comparison uses the identical seeds, not a re-derived set that could silently
#: drift between phases.
DEFAULT_SUPPORT_SEEDS: tuple[int, ...] = (1337, 1338, 1339, 1340, 1341)

DEFAULT_SHOTS: tuple[int, ...] = (1, 3, 5, 10)

FORM_MODES: tuple[str, ...] = ("same_form", "cross_form")


def sample_support_subset(
    pool: Sequence[str], n: int, seed: int, *, key: str = ""
) -> tuple[str, ...]:
    """A deterministic `n`-line subset of `pool`.

    Args:
        pool: Candidate sample IDs to draw from.
        n: How many to draw.
        seed: Determinism.
        key: Extra text mixed into the RNG seed -- writer ID, form mode, whatever distinguishes this
            draw from another one at the same numeric `seed` -- so two draws are never silently
            coupled just because they share a seed number. The same reasoning `data/splits.py`'s
            `f"{seed}:{writer}"` keying already applies one level up.

    Raises:
        ValueError: `n < 1`, or `pool` has fewer than `n` candidates. Never silently returns fewer
            lines than requested -- a caller must decide what "not enough support" means for its own
            reporting, not have it happen invisibly here.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if len(pool) < n:
        raise ValueError(f"pool has {len(pool)} candidate(s), cannot draw n={n}")
    ordered = sorted(pool)
    rng = random.Random(f"{seed}:{key}")
    return tuple(sorted(rng.sample(ordered, n)))


@dataclass(frozen=True, slots=True)
class FormPartition:
    """A writer's support pool, split by whether a line shares a scanned form/page with any query
    line.
    """

    same_form: tuple[str, ...]
    cross_form: tuple[str, ...]
    unknown_form: tuple[str, ...]

    def pool_for(self, form_mode: str) -> tuple[str, ...]:
        if form_mode == "same_form":
            return self.same_form
        if form_mode == "cross_form":
            return self.cross_form
        raise ValueError(f"unknown form_mode {form_mode!r}, expected one of {FORM_MODES}")


def partition_by_form(
    support_ids: Sequence[str],
    query_ids: Sequence[str],
    records_by_id: dict[str, ManifestRecord],
) -> FormPartition:
    """Split `support_ids` into same-form / cross-form relative to `query_ids`."""
    query_forms = {
        records_by_id[qid].source_page
        for qid in query_ids
        if records_by_id[qid].source_page is not None
    }
    same_form: list[str] = []
    cross_form: list[str] = []
    unknown_form: list[str] = []
    for sid in support_ids:
        form = records_by_id[sid].source_page
        if form is None:
            unknown_form.append(sid)
        elif form in query_forms:
            same_form.append(sid)
        else:
            cross_form.append(sid)
    return FormPartition(
        same_form=tuple(sorted(same_form)),
        cross_form=tuple(sorted(cross_form)),
        unknown_form=tuple(sorted(unknown_form)),
    )


@dataclass(frozen=True, slots=True)
class ProfileStats:
    """Internal helper."""

    support_lines: int
    characters_observed: int
    unique_characters_observed: int
    estimated_bytes: int
    compile_ms: float
    feature_dim: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "support_lines": self.support_lines,
            "characters_observed": self.characters_observed,
            "unique_characters_observed": self.unique_characters_observed,
            "estimated_bytes": self.estimated_bytes,
            "compile_ms": self.compile_ms,
            "feature_dim": self.feature_dim,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProfileStats:
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ShotResult:
    """One `(form_mode, n, seed)` draw's outcome for one writer."""

    writer_id: str
    form_mode: str
    n: int
    seed: int
    sample_ids: tuple[str, ...]
    cer: float | None
    n_query_lines: int
    profile: ProfileStats

    def as_dict(self) -> dict[str, Any]:
        return {
            "writer_id": self.writer_id,
            "form_mode": self.form_mode,
            "n": self.n,
            "seed": self.seed,
            "sample_ids": list(self.sample_ids),
            "cer": self.cer,
            "n_query_lines": self.n_query_lines,
            "profile": self.profile.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ShotResult:
        return cls(
            writer_id=payload["writer_id"],
            form_mode=payload["form_mode"],
            n=payload["n"],
            seed=payload["seed"],
            sample_ids=tuple(payload["sample_ids"]),
            cer=payload["cer"],
            n_query_lines=payload["n_query_lines"],
            profile=ProfileStats.from_dict(payload["profile"]),
        )


@dataclass(frozen=True, slots=True)
class WriterCurve:
    """Everything measured for one writer: `CER@0`, every `(form_mode, n, seed)` shot, and which
    combinations could not be drawn at all.
    """

    writer_id: str
    cer_at_0: float | None
    n_query_lines: int
    shots: tuple[ShotResult, ...] = ()
    unavailable: tuple[str, ...] = ()

    def cer_values_at(self, form_mode: str, n: int) -> list[float]:
        return [
            s.cer
            for s in self.shots
            if s.form_mode == form_mode and s.n == n and s.cer is not None
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
    def from_dict(cls, payload: dict[str, Any]) -> WriterCurve:
        return cls(
            writer_id=payload["writer_id"],
            cer_at_0=payload["cer_at_0"],
            n_query_lines=payload["n_query_lines"],
            shots=tuple(ShotResult.from_dict(s) for s in payload["shots"]),
            unavailable=tuple(payload["unavailable"]),
        )


def _cer_over_pairs(pairs: list[tuple[str, str]], tokenizer: Tokenizer) -> float | None:
    return corpus_cer(pairs, policy=tokenizer.policy).value


def run_writer_curve(
    model: GMBase,
    charset: Charset,
    tokenizer: Tokenizer,
    writer_id: str,
    support_pool: Sequence[str],
    query_pool: Sequence[str],
    records_by_id: dict[str, ManifestRecord],
    *,
    model_fingerprint: str,
    shots: Sequence[int] = DEFAULT_SHOTS,
    seeds: Sequence[int] = DEFAULT_SUPPORT_SEEDS,
    memory_config: MemoryConfig | None = None,
    device: torch.device | str = "cpu",
    projection: GlyphProjection | None = None,
    projection_fingerprint: str | None = None,
) -> WriterCurve:
    """Run the full curve for one writer: `CER@0`, then every `(form_mode, n, seed)` shot.

    Args:
        support_pool: Candidate support sample IDs for this writer (their whole support pool --
            `SupportQuerySplit.support_for(writer_id)`).
        query_pool: This writer's fixed query sample IDs (`SupportQuerySplit.query_for(writer_id)`)
            -- identical across every `n` and every `form_mode`, by construction: computed once, in
            this function, at the top.
        records_by_id: Every support and query sample ID for this writer, resolved to its
            `ManifestRecord`.
        model_fingerprint: Threaded into every compiled `WriterProfile` this call builds
            (`compile_profile`'s own required field) -- these profiles are never saved or reloaded
            by this harness, but the field stays meaningful rather than a placeholder.
        projection: A trained `GlyphProjection`, required exactly when
            ``memory_config.feature_layer`` is a projected layer
            (`memory.compiler.PROJECTED_FEATURE_LAYERS`) -- passed through to both `compile_profile`
            and `personalize` unchanged, so every shot's profile and its retrieval live in the same
            feature space.
        projection_fingerprint: Threaded into every compiled profile alongside ``projection``.
    """
    resolved_config = memory_config or MemoryConfig(enabled=True)
    resolved_device = device if isinstance(device, torch.device) else torch.device(device)

    query_records = [records_by_id[qid] for qid in query_pool]

    outputs: dict[str, HTROutput] = {}
    with torch.no_grad():
        for record in query_records:
            key = record.sample_id or record.image
            tensor = preprocess_path(record.image).tensor.unsqueeze(0).to(resolved_device)
            outputs[key] = model(tensor)

    base_pairs = [
        (
            record.text,
            greedy_decode(
                outputs[record.sample_id or record.image].logits[0],
                tokenizer,
                int(outputs[record.sample_id or record.image].input_lengths[0]),
            ),
        )
        for record in query_records
    ]
    cer_at_0 = _cer_over_pairs(base_pairs, tokenizer)

    partition = partition_by_form(support_pool, query_pool, records_by_id)

    shot_results: list[ShotResult] = []
    unavailable: list[str] = []

    for form_mode in FORM_MODES:
        pool = partition.pool_for(form_mode)
        for n in shots:
            if len(pool) < n:
                unavailable.append(f"{form_mode}:n={n}")
                continue
            for seed in seeds:
                sample_ids = sample_support_subset(
                    pool, n, seed, key=f"{writer_id}:{form_mode}"
                )
                lines = [
                    (preprocess_path(records_by_id[sid].image).tensor, records_by_id[sid].text)
                    for sid in sample_ids
                ]

                start = time.perf_counter()
                profile = compile_profile(
                    model,
                    charset,
                    lines,
                    model_fingerprint=model_fingerprint,
                    config=resolved_config,
                    device=resolved_device,
                    projection=projection,
                    projection_fingerprint=projection_fingerprint,
                )
                compile_ms = (time.perf_counter() - start) * 1000.0

                pairs: list[tuple[str, str]] = []
                with torch.no_grad():
                    for record in query_records:
                        key = record.sample_id or record.image
                        output = outputs[key]
                        corrected = personalize(
                            output, profile, charset, resolved_config, projection=projection
                        )
                        length = int(output.input_lengths[0])
                        pairs.append((record.text, greedy_decode(corrected[0], tokenizer, length)))

                shot_results.append(
                    ShotResult(
                        writer_id=writer_id,
                        form_mode=form_mode,
                        n=n,
                        seed=seed,
                        sample_ids=sample_ids,
                        cer=_cer_over_pairs(pairs, tokenizer),
                        n_query_lines=len(query_records),
                        profile=ProfileStats(
                            support_lines=len(sample_ids),
                            characters_observed=sum(profile.counts.values()),
                            unique_characters_observed=len(profile.glyphs),
                            estimated_bytes=profile.estimated_bytes(),
                            compile_ms=compile_ms,
                            feature_dim=profile.feature_dim,
                        ),
                    )
                )

    return WriterCurve(
        writer_id=writer_id,
        cer_at_0=cer_at_0,
        n_query_lines=len(query_records),
        shots=tuple(shot_results),
        unavailable=tuple(unavailable),
    )


@dataclass(frozen=True, slots=True)
class ShotStatistics:
    """Aggregate adaptation-gain statistics at one `(form_mode, n)`, over every writer with a usable
    draw there.
    """

    form_mode: str
    n: int
    n_writers_available: int
    n_writers_unavailable: int
    mean_gain: float | None
    median_gain: float | None
    mean_relative_gain: float | None
    pct_improved: float | None
    pct_regressed: float | None
    mean_seed_std: float | None
    worst_regressions: tuple[tuple[str, float], ...]
    bucket_mean_gain: dict[str, float | None]

    def as_dict(self) -> dict[str, Any]:
        return {
            "form_mode": self.form_mode,
            "n": self.n,
            "n_writers_available": self.n_writers_available,
            "n_writers_unavailable": self.n_writers_unavailable,
            "mean_gain": self.mean_gain,
            "median_gain": self.median_gain,
            "mean_relative_gain": self.mean_relative_gain,
            "pct_improved": self.pct_improved,
            "pct_regressed": self.pct_regressed,
            "mean_seed_std": self.mean_seed_std,
            "worst_regressions": [[w, g] for w, g in self.worst_regressions],
            "bucket_mean_gain": dict(self.bucket_mean_gain),
        }


def _bucket_writers(writers: Sequence[tuple[str, float]]) -> dict[str, list[str]]:
    """Split `(writer_id, cer_at_0)` pairs into easy/medium/hard tertiles by `cer_at_0` (ascending
    -- lower baseline CER is "easier"). Degrades gracefully below 3 writers.
    """
    ordered = sorted(writers, key=lambda item: item[1])
    n = len(ordered)
    if n == 0:
        return {"easy": [], "medium": [], "hard": []}
    third = max(1, round(n / 3))
    easy = ordered[:third]
    hard = ordered[-third:] if n > third else []
    medium = ordered[third : n - third] if n > 2 * third else []
    return {
        "easy": [w for w, _ in easy],
        "medium": [w for w, _ in medium],
        "hard": [w for w, _ in hard],
    }


def aggregate_shot_statistics(
    curves: Sequence[WriterCurve],
    *,
    shots: Sequence[int] = DEFAULT_SHOTS,
    form_modes: Sequence[str] = FORM_MODES,
    worst_n: int = 5,
) -> tuple[ShotStatistics, ...]:
    """One `ShotStatistics` per `(form_mode, n)`, from every writer's curve."""
    baseline_by_writer = {c.writer_id: c.cer_at_0 for c in curves if c.cer_at_0 is not None}
    buckets = _bucket_writers(list(baseline_by_writer.items()))

    results: list[ShotStatistics] = []
    for form_mode in form_modes:
        for n in shots:
            gains: dict[str, float] = {}
            relative_gains: dict[str, float] = {}
            seed_stds: list[float] = []
            n_unavailable = 0

            for curve in curves:
                if f"{form_mode}:n={n}" in curve.unavailable:
                    n_unavailable += 1
                    continue
                values = curve.cer_values_at(form_mode, n)
                if not values or curve.cer_at_0 is None:
                    continue
                mean_cer_n = statistics.mean(values)
                gains[curve.writer_id] = curve.cer_at_0 - mean_cer_n
                if curve.cer_at_0 > 0:
                    relative_gains[curve.writer_id] = (curve.cer_at_0 - mean_cer_n) / curve.cer_at_0
                if len(values) > 1:
                    seed_stds.append(statistics.stdev(values))

            gain_values = list(gains.values())
            ranked_worst = sorted(gains.items(), key=lambda item: item[1])[:worst_n]

            bucket_mean_gain: dict[str, float | None] = {}
            for name in ("easy", "medium", "hard"):
                bucket_gains = [gains[w] for w in buckets.get(name, []) if w in gains]
                bucket_mean_gain[name] = statistics.mean(bucket_gains) if bucket_gains else None

            results.append(
                ShotStatistics(
                    form_mode=form_mode,
                    n=n,
                    n_writers_available=len(gain_values),
                    n_writers_unavailable=n_unavailable,
                    mean_gain=statistics.mean(gain_values) if gain_values else None,
                    median_gain=statistics.median(gain_values) if gain_values else None,
                    mean_relative_gain=(
                        statistics.mean(relative_gains.values()) if relative_gains else None
                    ),
                    pct_improved=(
                        sum(1 for g in gain_values if g > 0) / len(gain_values)
                        if gain_values
                        else None
                    ),
                    pct_regressed=(
                        sum(1 for g in gain_values if g < 0) / len(gain_values)
                        if gain_values
                        else None
                    ),
                    mean_seed_std=statistics.mean(seed_stds) if seed_stds else None,
                    worst_regressions=tuple(ranked_worst),
                    bucket_mean_gain=bucket_mean_gain,
                )
            )
    return tuple(results)


@dataclass(frozen=True, slots=True)
class FewShotReport:
    """A complete few-shot run: every writer's curve, plus the aggregate statistics."""

    checkpoint: str
    manifest: str
    split: str
    device: str
    shots: tuple[int, ...]
    seeds: tuple[int, ...]
    curves: tuple[WriterCurve, ...]
    statistics: tuple[ShotStatistics, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "manifest": self.manifest,
            "split": self.split,
            "device": self.device,
            "shots": list(self.shots),
            "seeds": list(self.seeds),
            "curves": [c.as_dict() for c in self.curves],
            "statistics": [s.as_dict() for s in self.statistics],
        }

    def format(self) -> str:
        lines = [
            f"checkpoint   {self.checkpoint}",
            f"manifest     {self.manifest}   split {self.split}",
            f"device       {self.device}",
            f"writers      {len(self.curves)}   shots {list(self.shots)}   "
            f"seeds {len(self.seeds)}",
            "",
        ]
        for stat in self.statistics:
            mean_g = "n/a" if stat.mean_gain is None else f"{stat.mean_gain:+.4f}"
            median_g = "n/a" if stat.median_gain is None else f"{stat.median_gain:+.4f}"
            improved = "n/a" if stat.pct_improved is None else f"{stat.pct_improved:.1%}"
            regressed = "n/a" if stat.pct_regressed is None else f"{stat.pct_regressed:.1%}"
            lines.append(
                f"[{stat.form_mode:>10} n={stat.n:>2}] writers {stat.n_writers_available:>3} "
                f"(unavailable {stat.n_writers_unavailable:>2})   "
                f"mean gain {mean_g}   median gain {median_g}   "
                f"improved {improved}   regressed {regressed}"
            )
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> FewShotReport:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            checkpoint=payload["checkpoint"],
            manifest=payload["manifest"],
            split=payload["split"],
            device=payload["device"],
            shots=tuple(payload["shots"]),
            seeds=tuple(payload["seeds"]),
            curves=tuple(WriterCurve.from_dict(c) for c in payload["curves"]),
            statistics=tuple(
                ShotStatistics(
                    form_mode=s["form_mode"],
                    n=s["n"],
                    n_writers_available=s["n_writers_available"],
                    n_writers_unavailable=s["n_writers_unavailable"],
                    mean_gain=s["mean_gain"],
                    median_gain=s["median_gain"],
                    mean_relative_gain=s["mean_relative_gain"],
                    pct_improved=s["pct_improved"],
                    pct_regressed=s["pct_regressed"],
                    mean_seed_std=s["mean_seed_std"],
                    worst_regressions=tuple((w, g) for w, g in s["worst_regressions"]),
                    bucket_mean_gain=s["bucket_mean_gain"],
                )
                for s in payload["statistics"]
            ),
        )


def build_few_shot_report(
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
    shots: Sequence[int] = DEFAULT_SHOTS,
    seeds: Sequence[int] = DEFAULT_SUPPORT_SEEDS,
    memory_config: MemoryConfig | None = None,
    device: torch.device | str = "cpu",
    projection: GlyphProjection | None = None,
    projection_fingerprint: str | None = None,
) -> FewShotReport:
    """Run the full harness over every requested writer and assemble the report.

    Args:
        support_query: The fixed query/support partition (`SupportQuerySplit.load(...)` or
            `make_support_query_split(...)`) -- reserved once, outside this function, so re-running
            this call never redraws the query pool.
        records_by_id: Every support and query sample this run will touch, resolved to its
            `ManifestRecord`.
        writers: Restrict to these writers. Defaults to every writer in `support_query` with a
            non-empty query pool and at least one support line.
        projection: Passed through unchanged to every writer's `run_writer_curve` call -- see its
            own docstring.
        projection_fingerprint: Passed through unchanged alongside ``projection``.
    """
    resolved_device = device if isinstance(device, torch.device) else torch.device(device)
    target_writers = writers or sorted(
        w
        for w in support_query.writers
        if support_query.query_for(w) and support_query.support_for(w)
    )

    curves = tuple(
        run_writer_curve(
            model,
            charset,
            tokenizer,
            writer_id,
            support_query.support_for(writer_id),
            support_query.query_for(writer_id),
            records_by_id,
            model_fingerprint=model_fingerprint,
            shots=shots,
            seeds=seeds,
            memory_config=memory_config,
            device=resolved_device,
            projection=projection,
            projection_fingerprint=projection_fingerprint,
        )
        for writer_id in target_writers
    )

    return FewShotReport(
        checkpoint=checkpoint_label,
        manifest=manifest_label,
        split=split_name,
        device=str(resolved_device),
        shots=tuple(shots),
        seeds=tuple(seeds),
        curves=curves,
        statistics=aggregate_shot_statistics(curves, shots=shots, form_modes=FORM_MODES),
    )
