"""GlyphMemory command line interface.

Later milestones added ``data``, ``train``, ``evaluate``, ``benchmark``, ``enroll`` and
``transcribe``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
import typer
from torchvision.transforms.v2 import functional as TF

from glyphmemory import __version__
from glyphmemory.benchmark import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_WIDTHS,
    run_benchmark,
)
from glyphmemory.config.loader import dump_config, load_config
from glyphmemory.config.schema import Config, ConfigError, to_dict
from glyphmemory.ctc import (
    DEFAULT_CHARSET_PATH,
    DEFAULT_POLICY,
    Charset,
    charset_coverage,
    get_policy,
    load_tokenizer,
)
from glyphmemory.ctc.decode import greedy_decode
from glyphmemory.data import (
    ManifestError,
    SupportQuerySplit,
    UnreadableImageError,
    apply_writer_split,
    augment_deterministically,
    build_augmentation,
    build_dataloader,
    build_dataset,
    load_line_image,
    make_support_query_split,
    make_writer_disjoint_split,
    passage_distribution,
    preprocess_path,
    read_manifest,
    split_overlaps,
    split_statistics,
    support_query_overlap,
    to_uint8_tensor,
    write_manifest,
    writer_histogram,
)
from glyphmemory.data.adapters import (
    TRANSCRIPT_LIMITATION,
    CVLAdapter,
    IAMAdapter,
    SyntheticAdapter,
)
from glyphmemory.data.validation import IntegrityCounters
from glyphmemory.evaluation import (
    ADAPTIVE_METHODS,
    DEFAULT_FT_LR,
    DEFAULT_FT_STEPS,
    DEFAULT_SHOTS,
    DEFAULT_SUPPORT_SEEDS,
    build_few_shot_report,
    build_rival_baseline_report,
    evaluate_checkpoint,
)
from glyphmemory.memory import (
    ProfileCompatibilityError,
    WriterProfile,
    compile_profile,
    personalize,
)
from glyphmemory.model import GMBase, stage_output_shapes
from glyphmemory.runtime.device import available_devices, resolve_device
from glyphmemory.runtime.environment import Environment
from glyphmemory.runtime.experiment import ExperimentDir
from glyphmemory.runtime.fingerprint import checkpoint_fingerprint
from glyphmemory.runtime.logging import setup_logging
from glyphmemory.runtime.seed import seed_everything
from glyphmemory.training import (
    SELECTION_METRIC,
    CheckpointCompatibilityError,
    Trainer,
    build_run_record,
    build_scheduler,
    load_checkpoint,
    manifest_fingerprints,
)

app = typer.Typer(
    name="glyphmemory",
    help="Lightweight HTR with persistent few-shot writer adaptation.",
    no_args_is_help=True,
    add_completion=False,
)

config_app = typer.Typer(name="config", help="Inspect configuration files.", no_args_is_help=True)
app.add_typer(config_app)

data_app = typer.Typer(name="data", help="Inspect datasets and manifests.", no_args_is_help=True)
app.add_typer(data_app)

model_app = typer.Typer(name="model", help="Inspect the model.", no_args_is_help=True)
app.add_typer(model_app)


@app.command()
def version() -> None:
    """Print the GlyphMemory version."""
    typer.echo(__version__)


@app.command()
def info(
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps | cuda"),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show the resolved device and captured environment."""
    setup_logging()
    try:
        resolved = resolve_device(device, log=not as_json)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--device") from exc

    env = Environment.capture(cwd=Path.cwd())
    payload = {"device": resolved.as_dict(), "environment": env.as_dict()}

    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    git = env.git
    git_line = "not a git repository or no commits yet"
    if git.commit:
        git_line = f"{git.commit[:12]} ({git.branch}){' [dirty]' if git.dirty else ''}"

    typer.echo(f"glyphmemory      {env.glyphmemory_version}")
    typer.echo(f"python           {env.python_version}")
    typer.echo(f"torch            {env.torch_version}")
    typer.echo(f"platform         {env.platform}")
    typer.echo(f"machine          {env.machine}")
    typer.echo(f"devices          {', '.join(available_devices())}")
    typer.echo(f"resolved device  {resolved.kind}  ({resolved.reason})")
    typer.echo(f"git              {git_line}")


@config_app.command("show")
def config_show(
    path: Path = typer.Argument(..., help="Path to a YAML config file."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Load, validate and print a configuration file."""
    try:
        config = load_config(path)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc), param_hint="PATH") from exc

    data = to_dict(config)
    if as_json:
        typer.echo(json.dumps(data, indent=2, sort_keys=True))
        return

    for section, values in data.items():
        if not isinstance(values, dict):
            typer.echo(f"{section}: {values}")
            continue
        typer.echo(f"\n[{section}]")
        width = max(len(key) for key in values)
        for key, value in values.items():
            typer.echo(f"  {key:<{width}}  {value}")


@data_app.command("charset-report")
def data_charset_report(
    manifest: Path = typer.Argument(..., help="Path to a manifest.jsonl file."),
    charset: Path = typer.Option(
        DEFAULT_CHARSET_PATH, "--charset", help="Charset artifact to measure against."
    ),
    policy: str = typer.Option(DEFAULT_POLICY.name, "--policy", help="Normalization policy name."),
    top: int = typer.Option(15, "--top", help="How many frequent characters to list."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Measure a manifest's character inventory against a charset.

    Reports characters present in the data but absent from the charset (each one a sample that would
    be rejected), and charset symbols the data never uses.
    """
    setup_logging()
    try:
        loaded = Charset.load(charset)
        normalization = get_policy(policy)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        records = list(read_manifest(manifest))
    except (ManifestError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="MANIFEST") from exc

    report = charset_coverage(records, loaded, policy=normalization)

    if as_json:
        typer.echo(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        return
    typer.echo(report.format(charset=loaded, top=top))


@data_app.command("make-synthetic")
def data_make_synthetic(
    out: Path = typer.Option(..., "--out", help="Directory to write images and manifest into."),
    writers: int = typer.Option(3, "--writers", help="Number of synthetic writers."),
    lines: int = typer.Option(4, "--lines", help="Lines per writer."),
    seed: int = typer.Option(1337, "--seed", help="Master seed; same seed, same output."),
    mode: str = typer.Option("coverage", "--mode", help="coverage | words"),
    passages: int = typer.Option(2, "--passages", help="Distinct passage IDs to rotate through."),
    split: str = typer.Option("train", "--split", help="train | val | test"),
) -> None:
    """Generate a synthetic font-as-writer corpus.

    A correctness harness, never a performance benchmark: every record is labelled
    ``dataset="synthetic"`` and must not appear in a reported result.
    """
    setup_logging()
    try:
        adapter = SyntheticAdapter(
            n_writers=writers,
            n_lines=lines,
            seed=seed,
            corpus_mode=mode,
            n_passages=passages,
            split=split,
        )
        manifest_path = adapter.prepare(output_dir=out)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"wrote {manifest_path}")
    typer.echo(f"  writers   {writers}")
    typer.echo(f"  lines     {writers * lines}")
    typer.echo(f"  images    {out}/images")
    typer.echo(f"  styles    {out}/writer_styles.json")
    typer.echo("  note      synthetic data is correctness evidence only, never a benchmark")


@model_app.command("summary")
def model_summary(
    config_path: Path | None = typer.Option(None, "--config", help="Config file to build from."),
    charset: Path = typer.Option(DEFAULT_CHARSET_PATH, "--charset", help="Charset artifact."),
    width: int = typer.Option(512, "--width", help="Probe width for the shape table."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Print GM-Base's exact parameter count, per-module breakdown and shape table."""
    setup_logging()
    try:
        config = load_config(config_path) if config_path else Config()
        tokenizer = load_tokenizer(charset)
    except (ConfigError, ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    model = GMBase.from_config(config.model, tokenizer.vocab_size)
    report = model.parameter_report()

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "architecture": model.describe(),
                    "parameters": {
                        "total": report.total,
                        "fp32_megabytes": round(report.fp32_megabytes, 3),
                        "within_preferred": report.within_preferred,
                        "within_hard_ceiling": report.within_hard_ceiling,
                        "by_module": report.by_module,
                    },
                    "time_steps_at_width": {str(width): model.output_length(width)},
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    typer.echo(f"model       {config.model.name}   vocab {tokenizer.vocab_size}")
    typer.echo(f"charset     {tokenizer.charset.name} ({tokenizer.charset.fingerprint()[:12]})")
    typer.echo("")
    typer.echo(report.format())
    typer.echo("\nshapes:")
    for name, shape in stage_output_shapes(config.model.input_height, width):
        typer.echo(f"  {name:<15} [B, {shape[0]}, {shape[1]}, {shape[2]}]")
    typer.echo(
        f"  {'sequence':<15} [B, {model.output_length(width)}, {model.sequence.output_size}]"
    )
    typer.echo(f"  {'logits':<15} [B, {model.output_length(width)}, {tokenizer.vocab_size}]")


@app.command()
def train(
    manifest: Path = typer.Argument(..., help="Manifest to train on."),
    config_path: Path | None = typer.Option(None, "--config", help="Config file."),
    out: Path = typer.Option(Path("runs"), "--out", help="Base directory for run folders."),
    name: str | None = typer.Option(None, "--name", help="Experiment name for the run ID."),
    charset: Path = typer.Option(DEFAULT_CHARSET_PATH, "--charset", help="Charset artifact."),
    epochs: int | None = typer.Option(None, "--epochs", help="Override config.training.epochs."),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Override batch size."),
    limit: int | None = typer.Option(
        None, "--limit", help="Use only the first N records. For tiny-overfit probes."
    ),
    val_manifest: Path | None = typer.Option(
        None, "--val-manifest", help="Separate validation manifest. Defaults to the val split."
    ),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps | cuda"),
    seed: int | None = typer.Option(None, "--seed", help="Override config.runtime.seed."),
    no_augment: bool = typer.Option(False, "--no-augment", help="Disable training augmentation."),
    workers: int | None = typer.Option(None, "--workers", help="DataLoader worker processes."),
    samples: Path | None = typer.Option(
        None, "--samples", help="JSON file of frozen sample_ids to train on, in order."
    ),
    overfit: bool = typer.Option(
        False, "--overfit", help="Tiny-overfit correctness gate: fixes and asserts its conditions."
    ),
) -> None:
    """Train GM-Base, validating by CER and recording everything needed to reproduce the run.

    Writes ``run.json``, ``metrics.jsonl`` and checkpoints into a timestamped run directory. The
    best checkpoint is selected by **validation CER**, never by loss.
    """
    setup_logging()
    try:
        config = load_config(config_path) if config_path else Config()
        tokenizer = load_tokenizer(charset)
        resolved = resolve_device(device)
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    resolved_seed = seed if seed is not None else config.runtime.seed
    seed_everything(resolved_seed)

    frozen_ids: list[str] | None = None
    criterion: dict | None = None
    if samples is not None:
        try:
            payload = json.loads(samples.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--samples") from exc
        frozen_ids = payload["sample_ids"] if isinstance(payload, dict) else list(payload)
        criterion = payload.get("criterion") if isinstance(payload, dict) else None

    if overfit:
        # The gate's conditions are enforced here rather than trusted to a config the caller
        # remembered to pass.
        problems = []
        if config.data.augmentation.enabled and not no_augment:
            problems.append("data.augmentation.enabled must be false (it prevents memorization)")
        if config.model.gru_dropout or config.model.head_dropout:
            problems.append("model.gru_dropout and model.head_dropout must be 0.0")
        if problems:
            raise typer.BadParameter(
                "--overfit requires the gate conditions; use configs/tiny_overfit.yaml. "
                + "; ".join(problems),
                param_hint="--overfit",
            )
        no_augment = True

    try:
        train_dataset = build_dataset(
            manifest, tokenizer, config, split="train", training=not no_augment
        )
        if len(train_dataset) == 0:
            train_dataset = build_dataset(manifest, tokenizer, config, training=not no_augment)
        if frozen_ids is not None:
            train_dataset = train_dataset.select(frozen_ids)
        if limit is not None:
            train_dataset = train_dataset.take(limit)

        validation_source = val_manifest or manifest
        val_dataset = build_dataset(
            validation_source, tokenizer, config, split="val", training=False
        )
        if len(val_dataset) == 0:
            # No val split in the manifest: validate on the training lines. Correct for the
            # tiny-overfit gate, which deliberately trains and validates on the same lines, and
            # loudly wrong for anything else — so it says so.
            val_dataset = build_dataset(validation_source, tokenizer, config, training=False)
            if frozen_ids is not None:
                val_dataset = val_dataset.select(frozen_ids)
            if limit is not None:
                val_dataset = val_dataset.take(limit)
            typer.echo(
                "note        no 'val' split found; validating on the training lines. "
                "This is correct only for a memorization probe."
            )
    except (ManifestError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="MANIFEST") from exc

    if len(train_dataset) == 0:
        raise typer.BadParameter("no training records matched", param_hint="MANIFEST")

    train_loader = build_dataloader(
        train_dataset,
        config,
        training=True,
        batch_size=batch_size,
        bucket=not overfit,
        num_workers=workers,
    )
    val_loader = build_dataloader(
        val_dataset,
        config,
        training=False,
        batch_size=batch_size,
        bucket=False,
        num_workers=workers,
    )

    model = GMBase.from_config(config.model, tokenizer.vocab_size)
    report = model.parameter_report()
    if not report.within_hard_ceiling:
        raise typer.BadParameter(
            f"model has {report.total:,} parameters, over the ceiling", param_hint="--config"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    total_epochs = epochs if epochs is not None else config.training.epochs
    total_steps = max(len(train_loader) * total_epochs, 1)
    scheduler, schedule = build_scheduler(optimizer, config.training, total_steps=total_steps)

    experiment = ExperimentDir.create(out, name or config.model.name)
    record = build_run_record(
        run_id=experiment.run_id,
        config=config,
        tokenizer=tokenizer,
        device=resolved,
        model=model,
        seed=resolved_seed,
        manifests=manifest_fingerprints(train=manifest, val=val_manifest),
        extra={
            "schedule": schedule.describe(),
            "optimizer": "adamw",
            "total_steps": total_steps,
            "epochs": total_epochs,
            "train_records": len(train_dataset),
            "val_records": len(val_dataset),
            "augmentation": not no_augment,
            "overfit_gate": overfit,
            "frozen_sample_ids": frozen_ids,
            "criterion": criterion,
            "bucketing": not overfit,
        },
    )
    # Written BEFORE the first step, so a run that crashes in epoch one is still attributable.
    experiment.write_json(experiment.root / "run.json", record)
    dump_config(config, experiment.config_path)

    typer.echo(f"run         {experiment.run_id}")
    typer.echo(f"device      {resolved.kind}  ({resolved.reason})")
    typer.echo(f"records     train {len(train_dataset):,}   val {len(val_dataset):,}")
    typer.echo(f"model       {report.total:,} parameters   vocab {tokenizer.vocab_size}")
    typer.echo(f"schedule    {total_steps:,} steps, {schedule.warmup_steps:,} warmup")
    typer.echo(f"selection   {SELECTION_METRIC} (never loss)")
    if overfit:
        threshold = (criterion or {}).get("threshold")
        max_epochs = (criterion or {}).get("max_epochs")
        typer.echo(
            "gate        tiny-overfit correctness gate — augmentation off, dropout 0, "
            "bucketing off, train==val"
        )
        if threshold is not None:
            typer.echo(f"criterion   train CER <= {threshold} within {max_epochs} epochs")
    typer.echo(f"artifacts   {experiment.root}\n")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=resolved.torch_device,
        experiment=experiment,
        checkpoint_meta={
            "config": to_dict(config),
            "parameter_count": report.total,
            "git_commit": record["git_commit"],
            "seed": resolved_seed,
            "run_id": experiment.run_id,
            "manifest_fingerprints": record["manifest_fingerprints"],
        },
    )
    trainer.trace_path = experiment.predictions_path if overfit else None
    history = trainer.fit(total_epochs)

    experiment.write_json(
        experiment.metrics_path,
        {"history": history, "best": {SELECTION_METRIC: trainer.best_value}},
    )
    best = "n/a" if trainer.best_value is None else f"{trainer.best_value:.4f}"
    typer.echo(f"\nbest {SELECTION_METRIC}  {best}")
    typer.echo(f"checkpoints    {experiment.checkpoints_dir}")

    if overfit and criterion is not None:
        threshold = criterion["threshold"]
        passed = trainer.best_value is not None and trainer.best_value <= threshold
        verdict = "PASS" if passed else "FAIL"
        typer.echo(f"\ngate        {verdict}  (best {best} vs threshold {threshold})")
        if not passed:
            typer.echo(
                "            Do NOT launch full training. Work the diagnosis order in "
                "input_lengths -> target_lengths -> CTC layout "
                "-> blank index -> decoder collapse."
            )
        typer.echo(f"trace       {experiment.predictions_path}")


@app.command()
def evaluate(
    checkpoint: Path = typer.Argument(..., help="Checkpoint file to evaluate."),
    manifest: Path = typer.Argument(..., help="Manifest containing the split to evaluate."),
    config_path: Path | None = typer.Option(
        None, "--config", help="Config file. Must match the one the checkpoint was trained with."
    ),
    charset: Path = typer.Option(DEFAULT_CHARSET_PATH, "--charset", help="Charset artifact."),
    split: str = typer.Option("test", "--split", help="train | val | test"),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps | cuda"),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Override batch size."),
    workers: int | None = typer.Option(None, "--workers", help="DataLoader worker processes."),
    worst_n: int = typer.Option(10, "--worst-n", help="Number of worst lines to print."),
    out: Path | None = typer.Option(None, "--out", help="Also write the full JSON report here."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of the table."),
) -> None:
    """Evaluate a checkpoint on a manifest split.

    Reports CER/WER, per-writer CER, and the error taxonomy.
    """
    setup_logging()
    try:
        config = load_config(config_path) if config_path else Config()
        tokenizer = load_tokenizer(charset)
        resolved = resolve_device(device)
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        report = evaluate_checkpoint(
            checkpoint,
            manifest,
            config=config,
            tokenizer=tokenizer,
            split=split,
            device=resolved.torch_device,
            batch_size=batch_size,
            num_workers=workers,
            worst_n=worst_n,
        )
    except (CheckpointCompatibilityError, ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")

    if as_json:
        typer.echo(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        return

    typer.echo(report.format())
    if out is not None:
        typer.echo(f"\nreport      {out}")


def _parse_int_list(value: str, *, param_hint: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise typer.BadParameter(f"expected comma-separated integers, got {value!r}") from exc


@app.command("few-shot")
def few_shot(
    checkpoint: Path = typer.Argument(..., help="Checkpoint to run the harness against."),
    manifest: Path = typer.Argument(..., help="Manifest containing the split to evaluate."),
    support_query: Path = typer.Option(
        ..., "--support-query", help="A stored SupportQuerySplit (from `glyphmemory data split`)."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", help="Config file. Must match the one the checkpoint was trained with."
    ),
    charset: Path = typer.Option(DEFAULT_CHARSET_PATH, "--charset", help="Charset artifact."),
    split: str = typer.Option("val", "--split", help="train | val | test."),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps | cuda"),
    shots: str = typer.Option(
        ",".join(str(n) for n in DEFAULT_SHOTS), "--shots", help="Comma-separated support sizes."
    ),
    seeds: str = typer.Option(
        ",".join(str(s) for s in DEFAULT_SUPPORT_SEEDS),
        "--seeds",
        help="Comma-separated deterministic support-sampling seeds.",
    ),
    writers: str | None = typer.Option(
        None, "--writers", help="Comma-separated writer IDs. Defaults to every usable writer."
    ),
    out: Path | None = typer.Option(None, "--out", help="Also write the full JSON report here."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of the table."),
) -> None:
    """`CER@n` curves and adaptation-gain statistics."""
    setup_logging()
    try:
        config = load_config(config_path) if config_path else Config()
        tokenizer = load_tokenizer(charset)
        resolved = resolve_device(device)
        shot_tuple = _parse_int_list(shots, param_hint="--shots")
        seed_tuple = _parse_int_list(seeds, param_hint="--seeds")
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        model = _load_model_for_inference(checkpoint, config, tokenizer, resolved.torch_device)
    except (CheckpointCompatibilityError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="CHECKPOINT") from exc

    try:
        records = [r for r in read_manifest(manifest) if r.split == split]
    except (ManifestError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="MANIFEST") from exc
    if not records:
        raise typer.BadParameter(
            f"no records for split={split!r} in {manifest}", param_hint="MANIFEST"
        )

    try:
        loaded_split = SupportQuerySplit.load(support_query)
    except OSError as exc:
        raise typer.BadParameter(str(exc), param_hint="--support-query") from exc

    records_by_id = {r.sample_id: r for r in records if r.sample_id}
    writer_list = [w.strip() for w in writers.split(",")] if writers else None

    report = build_few_shot_report(
        model,
        tokenizer.charset,
        tokenizer,
        loaded_split,
        records_by_id,
        checkpoint_label=str(checkpoint),
        manifest_label=str(manifest),
        split_name=split,
        model_fingerprint=checkpoint_fingerprint(checkpoint),
        writers=writer_list,
        shots=shot_tuple,
        seeds=seed_tuple,
        memory_config=replace(config.memory, enabled=True),
        device=resolved.torch_device,
    )

    if out is not None:
        report.save(out)

    if as_json:
        typer.echo(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        return

    typer.echo(report.format())
    if out is not None:
        typer.echo(f"\nreport      {out}")


@app.command("rival-baselines")
def rival_baselines(
    checkpoint: Path = typer.Argument(..., help="Checkpoint to run the comparison against."),
    manifest: Path = typer.Argument(..., help="Manifest containing the split to evaluate."),
    support_query: Path = typer.Option(
        ..., "--support-query", help="A stored SupportQuerySplit (from `glyphmemory data split`)."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", help="Config file. Must match the one the checkpoint was trained with."
    ),
    charset: Path = typer.Option(DEFAULT_CHARSET_PATH, "--charset", help="Charset artifact."),
    split: str = typer.Option("val", "--split", help="train | val | test."),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps | cuda"),
    methods: str = typer.Option(
        ",".join(ADAPTIVE_METHODS),
        "--methods",
        help="Comma-separated methods: glyphmemory, head_ft, batchnorm_ft, full_ft, replay.",
    ),
    shots: str = typer.Option(
        ",".join(str(n) for n in DEFAULT_SHOTS), "--shots", help="Comma-separated support sizes."
    ),
    seeds: str = typer.Option(
        ",".join(str(s) for s in DEFAULT_SUPPORT_SEEDS),
        "--seeds",
        help="Comma-separated deterministic support-sampling seeds.",
    ),
    ft_steps: int = typer.Option(
        DEFAULT_FT_STEPS, "--ft-steps", help="Gradient steps for each fine-tune baseline."
    ),
    ft_lr: float | None = typer.Option(
        None,
        "--ft-lr",
        help=(
            "AdamW learning rate, applied to every fine-tune baseline alike. Defaults to "
            "per-parameter-group tuned rates (head/batchnorm 1e-3, full 1e-4) -- full "
            "fine-tune destabilizes badly at the rate head/batchnorm tolerate fine."
        ),
    ),
    writers: str | None = typer.Option(
        None, "--writers", help="Comma-separated writer IDs. Defaults to every usable writer."
    ),
    out: Path | None = typer.Option(None, "--out", help="Also write the full JSON report here."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of the table."),
) -> None:
    """Head-only / BatchNorm-only / full fine-tune and support-replay vs. GlyphMemory: required
    gain-vs-cost comparison.

    Every method is scored against the identical query pool and support draws `glyphmemory few-shot`
    uses. Fine-tune baselines restore the checkpoint's exact original weights after every writer —
    `gm-base-v0` itself is never modified.
    """
    setup_logging()
    try:
        config = load_config(config_path) if config_path else Config()
        tokenizer = load_tokenizer(charset)
        resolved = resolve_device(device)
        method_tuple = tuple(m.strip() for m in methods.split(","))
        shot_tuple = _parse_int_list(shots, param_hint="--shots")
        seed_tuple = _parse_int_list(seeds, param_hint="--seeds")
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    unknown_methods = set(method_tuple) - set(ADAPTIVE_METHODS)
    if unknown_methods:
        raise typer.BadParameter(
            f"unknown method(s) {sorted(unknown_methods)}; expected a subset of "
            f"{list(ADAPTIVE_METHODS)}",
            param_hint="--methods",
        )

    try:
        model = _load_model_for_inference(checkpoint, config, tokenizer, resolved.torch_device)
    except (CheckpointCompatibilityError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="CHECKPOINT") from exc

    try:
        records = [r for r in read_manifest(manifest) if r.split == split]
    except (ManifestError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="MANIFEST") from exc
    if not records:
        raise typer.BadParameter(
            f"no records for split={split!r} in {manifest}", param_hint="MANIFEST"
        )

    try:
        loaded_split = SupportQuerySplit.load(support_query)
    except OSError as exc:
        raise typer.BadParameter(str(exc), param_hint="--support-query") from exc

    records_by_id = {r.sample_id: r for r in records if r.sample_id}
    writer_list = [w.strip() for w in writers.split(",")] if writers else None

    report = build_rival_baseline_report(
        model,
        tokenizer.charset,
        tokenizer,
        loaded_split,
        records_by_id,
        checkpoint_label=str(checkpoint),
        manifest_label=str(manifest),
        split_name=split,
        model_fingerprint=checkpoint_fingerprint(checkpoint),
        writers=writer_list,
        methods=method_tuple,
        shots=shot_tuple,
        seeds=seed_tuple,
        memory_config=replace(config.memory, enabled=True),
        ft_steps=ft_steps,
        ft_lr=ft_lr if ft_lr is not None else DEFAULT_FT_LR,
        device=resolved.torch_device,
    )

    if out is not None:
        report.save(out)

    if as_json:
        typer.echo(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        return

    typer.echo(report.format())
    if out is not None:
        typer.echo(f"\nreport      {out}")


@app.command()
def benchmark(
    checkpoint: Path = typer.Argument(..., help="Checkpoint file to benchmark."),
    config_path: Path | None = typer.Option(
        None, "--config", help="Config file. Must match the one the checkpoint was trained with."
    ),
    charset: Path = typer.Option(DEFAULT_CHARSET_PATH, "--charset", help="Charset artifact."),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps | cuda"),
    threads: int | None = typer.Option(
        None, "--threads", help="torch.set_num_threads(...). Always recorded; never left implicit."
    ),
    widths: str = typer.Option(
        ",".join(str(w) for w in DEFAULT_WIDTHS), "--widths", help="Comma-separated input widths."
    ),
    batch_sizes: str = typer.Option(
        ",".join(str(b) for b in DEFAULT_BATCH_SIZES),
        "--batch-sizes",
        help="Comma-separated batch sizes.",
    ),
    warmup: int = typer.Option(5, "--warmup", help="Warmup iterations, discarded."),
    iterations: int = typer.Option(20, "--iterations", help="Measured iterations."),
    out: Path | None = typer.Option(None, "--out", help="Also write the full JSON report here."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of the table."),
) -> None:
    """Latency, throughput and the MPS CTC round-trip cost — never without their context."""
    setup_logging()
    try:
        config = load_config(config_path) if config_path else Config()
        tokenizer = load_tokenizer(charset)
        resolved = resolve_device(device)
        width_tuple = _parse_int_list(widths, param_hint="--widths")
        batch_tuple = _parse_int_list(batch_sizes, param_hint="--batch-sizes")
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        report = run_benchmark(
            checkpoint,
            config=config,
            tokenizer=tokenizer,
            device=resolved.torch_device,
            widths=width_tuple,
            batch_sizes=batch_tuple,
            warmup_iterations=warmup,
            measurement_iterations=iterations,
            threads=threads,
        )
    except (CheckpointCompatibilityError, ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")

    if as_json:
        typer.echo(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        return

    typer.echo(report.format())
    if out is not None:
        typer.echo(f"\nreport      {out}")


@data_app.command("prepare")
def data_prepare(
    dataset: str = typer.Option(..., "--dataset", help="Corpus to convert: cvl or iam."),
    source: Path = typer.Option(..., "--source", help="Release directory on disk."),
    out: Path = typer.Option(..., "--out", help="Directory to write the manifest into."),
    split: str = typer.Option(
        "",
        "--split",
        help="Value for every record's split field. Default: test for cvl, train for iam.",
    ),
    include_german: bool = typer.Option(
        False, "--include-german", help="cvl only. Keep CVL's German passage (page 6)."
    ),
    include_struck_out: bool = typer.Option(
        False,
        "--include-struck-out",
        help="iam only. Keep lines whose transcript contains IAM's '#' struck-out marker.",
    ),
    drop_segmentation_errors: bool = typer.Option(
        False,
        "--drop-segmentation-errors",
        help="iam only. Exclude lines flagged 'err' in lines.txt. Off by default: the flag "
        "marks word segmentation, which line recognition does not use.",
    ),
    no_image_size: bool = typer.Option(
        False, "--no-image-size", help="Skip reading image headers for width/height."
    ),
) -> None:
    """Convert a source corpus into a GlyphMemory manifest."""
    setup_logging()
    if dataset not in {"cvl", "iam"}:
        raise typer.BadParameter(
            f"unknown dataset {dataset!r}; adapters with a source layout are 'cvl' and 'iam' "
            "(synthetic data is generated by 'data make-synthetic')",
            param_hint="--dataset",
        )

    adapter: CVLAdapter | IAMAdapter
    if dataset == "cvl":
        adapter = CVLAdapter(
            split=split or "test",
            include_german=include_german,
            read_image_size=not no_image_size,
        )
    else:
        adapter = IAMAdapter(
            split=split or "train",
            include_struck_out=include_struck_out,
            keep_segmentation_errors=not drop_segmentation_errors,
            read_image_size=not no_image_size,
        )

    try:
        manifest_path = adapter.prepare(source, out)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--source") from exc

    records = list(read_manifest(manifest_path))
    rejections = {name: count for name, count in adapter.counters.as_dict().items() if count}

    typer.echo(f"wrote {manifest_path}")
    typer.echo(f"  records    {len(records):,}")
    typer.echo(f"  writers    {len({r.writer_id for r in records}):,}")
    if dataset == "cvl":
        typer.echo(f"  passages   {len({r.passage_id for r in records if r.passage_id}):,}")
    else:
        typer.echo(f"  forms      {len({r.source_page for r in records if r.source_page}):,}")
    typer.echo(f"  split      {adapter.split}")
    typer.echo(f"  summary    {out}/{dataset}_summary.json")
    typer.echo(f"  rejections {rejections or 'none'}")
    if dataset == "cvl":
        typer.echo(f"\nlimitation  {TRANSCRIPT_LIMITATION}")


@data_app.command("stats")
def data_stats(
    manifest: Path = typer.Argument(..., help="Path to a manifest.jsonl file."),
    query_size: int = typer.Option(
        5, "--query-size", help="Query lines reserved per writer, for support-capacity."
    ),
    policy: str = typer.Option(
        DEFAULT_POLICY.name, "--policy", help="Normalization used for overlap measurement."
    ),
    support_query: Path | None = typer.Option(
        None, "--support-query", help="A stored SupportQuerySplit to measure overlap for."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of tables."),
) -> None:
    """Writer/line histogram, passage distribution and text overlap.

    Overlap is reported under a named normalization, without which the figures cannot be quoted.
    """
    setup_logging()
    try:
        normalization = get_policy(policy)
        records = list(read_manifest(manifest))
    except (ManifestError, ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="MANIFEST") from exc

    if not records:
        raise typer.BadParameter("manifest contains no records", param_hint="MANIFEST")

    histogram = writer_histogram(records)
    passages = passage_distribution(records)
    overlaps = split_overlaps(records, policy=normalization)

    per_writer = None
    if support_query is not None:
        stored = SupportQuerySplit.load(support_query)
        per_writer = support_query_overlap(records, stored, policy=normalization)

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "manifest": str(manifest),
                    "writers": histogram.as_dict(),
                    "support_capacity": {
                        str(n): count
                        for n, count in histogram.support_capacity(query_size=query_size).items()
                    },
                    "passages": passages.as_dict(),
                    "overlap": [report.as_dict() for report in overlaps],
                    "support_query_overlap": per_writer.as_dict() if per_writer else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    typer.echo(f"manifest    {manifest}\n")
    typer.echo("[writers]")
    typer.echo(histogram.format(query_size=query_size))
    typer.echo("\n[passages]")
    typer.echo(passages.format())
    typer.echo("\n[text overlap]")
    if not overlaps and per_writer is None:
        typer.echo("  no split pair to compare (populate splits, or pass --support-query)")
    for report in overlaps:
        typer.echo(report.format())
    if per_writer is not None:
        typer.echo(per_writer.format())


@data_app.command("split")
def data_split(
    manifest: Path = typer.Argument(..., help="Path to a manifest.jsonl file."),
    out: Path = typer.Option(..., "--out", help="Directory for split artifacts."),
    ratios: str = typer.Option("0.8,0.1,0.1", "--ratios", help="train,val,test writer ratios."),
    query_size: int = typer.Option(5, "--query-size", help="Query lines reserved per writer."),
    seed: int = typer.Option(1337, "--seed", help="Determinism."),
    passage_disjoint: bool = typer.Option(
        True,
        "--passage-disjoint/--random-support",
        help="Draw support and query from disjoint passages.",
    ),
    eval_split: str = typer.Option(
        "test", "--eval-split", help="Split the support/query pools are built from."
    ),
) -> None:
    """Generate and store a writer-disjoint split plus support/query pools."""
    setup_logging()
    try:
        parts = tuple(float(value) for value in ratios.split(","))
    except ValueError as exc:
        raise typer.BadParameter(f"expected three comma-separated numbers: {ratios!r}") from exc
    if len(parts) != 3:
        raise typer.BadParameter(f"expected three ratios, got {len(parts)}", param_hint="--ratios")

    try:
        records = list(read_manifest(manifest))
    except (ManifestError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="MANIFEST") from exc

    try:
        writer_split = make_writer_disjoint_split(records, ratios=parts, seed=seed)  # type: ignore[arg-type]
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--ratios") from exc

    assigned = apply_writer_split(records, writer_split)
    out.mkdir(parents=True, exist_ok=True)

    writer_split_path = writer_split.save(out / "writer_split.json")
    manifest_path = out / "manifest.jsonl"
    write_manifest(manifest_path, assigned)

    evaluation = [record for record in assigned if record.split == eval_split]
    support_query = make_support_query_split(
        evaluation,
        query_size=query_size,
        seed=seed,
        group_of=(lambda record: record.passage_id) if passage_disjoint else None,
    )
    support_query_path = support_query.save(out / "support_query.json")

    typer.echo(f"wrote {manifest_path}")
    typer.echo(f"      {writer_split_path}")
    typer.echo(f"      {support_query_path}")
    typer.echo(f"\nwriter split (seed {seed}, ratios {parts})")
    for name, values in split_statistics(assigned, writer_split).items():
        typer.echo(f"  {name:<6} {values['writers']:>5,} writer(s)  {values['lines']:>7,} line(s)")
    typer.echo(
        f"\nsupport/query on {eval_split!r}  "
        f"({'passage-disjoint' if passage_disjoint else 'random'}, query_size {query_size})"
    )
    typer.echo(f"  writers        {len(support_query.writers):,}")
    for shot in (1, 3, 5, 10):
        supporting = len(support_query.writers_supporting(shot))
        typer.echo(f"  CER@{shot:<3}       {supporting:,} writer(s)")


@data_app.command("preview-augmentations")
def data_preview_augmentations(
    manifest: Path = typer.Argument(..., help="Path to a manifest.jsonl file."),
    out: Path = typer.Option(..., "--out", help="Directory to write preview images into."),
    n: int = typer.Option(8, "--n", help="How many samples to preview."),
    config_path: Path | None = typer.Option(
        None, "--config", help="Config file supplying the augmentation section."
    ),
    seed: int = typer.Option(1337, "--seed", help="Seed, so a preview is reproducible."),
) -> None:
    """Write before/after augmentation previews for visual inspection.

    Conservative-looking numbers in a config are not evidence that the transform preserved what the
    line says.
    """
    setup_logging()
    try:
        config = load_config(config_path) if config_path else Config()
    except ConfigError as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from exc

    try:
        records = list(read_manifest(manifest))[:n]
    except (ManifestError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="MANIFEST") from exc

    if not records:
        raise typer.BadParameter("manifest contains no records", param_hint="MANIFEST")

    pipeline = build_augmentation(config.data.augmentation, training=True)
    out.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(records):
        try:
            original = to_uint8_tensor(load_line_image(record.image))
        except UnreadableImageError as exc:
            typer.echo(f"skipped {record.sample_id}: {exc}")
            continue
        augmented = augment_deterministically(pipeline, original, seed=seed + index)
        stem = (record.sample_id or f"sample{index}").replace("/", "_")
        TF.to_pil_image(original).save(out / f"{stem}_before.png")
        TF.to_pil_image(augmented).save(out / f"{stem}_after.png")

    typer.echo(f"wrote {len(records) * 2} preview image(s) to {out}")
    typer.echo(f"  families  {', '.join(pipeline.families) or 'none (identity)'}")
    typer.echo(f"  seed      {seed}")
    typer.echo("  inspect these before training — conservative config values are not evidence")


@data_app.command("inspect")
def data_inspect(
    manifest: Path = typer.Argument(..., help="Path to a manifest.jsonl file."),
    batch_size: int = typer.Option(4, "--batch-size", help="Samples per batch."),
    batches: int = typer.Option(1, "--batches", help="How many batches to show."),
    config_path: Path | None = typer.Option(None, "--config", help="Config file to use."),
    charset: Path = typer.Option(DEFAULT_CHARSET_PATH, "--charset", help="Charset artifact."),
    split: str | None = typer.Option(None, "--split", help="Keep only this split."),
    no_bucket: bool = typer.Option(False, "--no-bucket", help="Disable width bucketing."),
    training: bool = typer.Option(False, "--training", help="Apply augmentation."),
    save_grid: Path | None = typer.Option(None, "--save-grid", help="Write batch images here."),
) -> None:
    """Inspect batches: shapes, CTC lengths, decoded text, writer IDs, padding efficiency.

    **This is the exit gate**. It exists so a developer can confirm by eye that images, transcripts,
    lengths and writer IDs line up before any training run consumes them.
    """
    setup_logging()
    try:
        config = load_config(config_path) if config_path else Config()
        tokenizer = load_tokenizer(charset)
    except (ConfigError, ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        dataset = build_dataset(manifest, tokenizer, config, split=split, training=training)
    except (ManifestError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="MANIFEST") from exc

    if len(dataset) == 0:
        raise typer.BadParameter("no records matched", param_hint="MANIFEST")

    counters = IntegrityCounters()
    loader = build_dataloader(
        dataset,
        config,
        training=training,
        batch_size=batch_size,
        bucket=not no_bucket,
        counters=counters,
        num_workers=0,
    )

    typer.echo(f"manifest    {manifest}")
    typer.echo(
        f"records     {len(dataset)}   tokenizer {tokenizer.charset.name} "
        f"(vocab {tokenizer.vocab_size})"
    )
    typer.echo(
        f"bucketing   {'off' if no_bucket else 'on'}   augmentation {'on' if training else 'off'}"
    )

    for index, batch in enumerate(loader):
        if index >= batches:
            break
        typer.echo(f"\n=== batch {index} ===")
        typer.echo(
            f"images {tuple(batch.images.shape)}  targets {tuple(batch.targets.shape)}  "
            f"padding efficiency {batch.padding_efficiency:.1%}"
        )
        for position in range(batch.batch_size):
            decoded = tokenizer.decode(batch.targets_for(position).tolist())
            matches = "ok" if decoded == batch.texts[position] else "MISMATCH"
            typer.echo(
                f"  [{position}] {batch.sample_ids[position]}  writer={batch.writer_ids[position]}"
            )
            typer.echo(
                f"      width {batch.true_widths[position]:>5} -> input_length "
                f"{int(batch.input_lengths[position]):>4}   target_length "
                f"{int(batch.target_lengths[position]):>3}   roundtrip {matches}"
            )
            typer.echo(f"      text  {decoded[:70]!r}")
        if batch.rejected:
            typer.echo(f"  rejected {len(batch.rejected)}:")
            for rejection in batch.rejected[:5]:
                typer.echo(
                    f"      [{rejection.category}] {rejection.sample_id}: {rejection.reason[:90]}"
                )

        if save_grid is not None:
            save_grid.mkdir(parents=True, exist_ok=True)
            for position in range(batch.batch_size):
                image = batch.images[position]
                if config.data.invert_pixels:
                    image = 1.0 - image
                stem = batch.sample_ids[position].replace("/", "_")
                TF.to_pil_image(image.clamp(0, 1)).save(save_grid / f"b{index}_{stem}.png")

    rejections = {name: count for name, count in counters.as_dict().items() if count}
    typer.echo(f"\nrejections  {rejections or 'none'}")
    if save_grid is not None:
        typer.echo(f"images      {save_grid}")


def _load_model_for_inference(
    checkpoint: Path, config: Config, tokenizer, device: torch.device
) -> GMBase:
    loaded = load_checkpoint(checkpoint, charset_fingerprint=tokenizer.charset.fingerprint())
    model = GMBase.from_config(config.model, tokenizer.vocab_size)
    model.load_state_dict(loaded.model_state)
    model.to(device)
    model.eval()
    return model


@app.command()
def enroll(
    checkpoint: Path = typer.Argument(..., help="Checkpoint to enroll against."),
    manifest: Path = typer.Argument(..., help="Manifest containing the enrollment lines."),
    writer_id: str = typer.Option(..., "--writer-id", help="Writer to compile a profile for."),
    out: Path = typer.Option(..., "--out", help="Where to write the compiled WriterProfile."),
    config_path: Path | None = typer.Option(
        None, "--config", help="Config file. Must match the one the checkpoint was trained with."
    ),
    charset: Path = typer.Option(DEFAULT_CHARSET_PATH, "--charset", help="Charset artifact."),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps | cuda"),
    n: int | None = typer.Option(
        None, "--n", help="Use only the writer's first N lines (support-size control)."
    ),
    feature_layer: str | None = typer.Option(
        None, "--feature-layer", help="Override the config's memory.feature_layer."
    ),
    pooling: str | None = typer.Option(
        None, "--pooling", help="Override the config's memory.pooling."
    ),
) -> None:
    """Compile a WriterProfile from labeled enrollment lines.

    Gradient-free: forward passes, forced alignment and prototype averaging only.
    """
    setup_logging()
    try:
        config = load_config(config_path) if config_path else Config()
        tokenizer = load_tokenizer(charset)
        resolved = resolve_device(device)
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        model = _load_model_for_inference(checkpoint, config, tokenizer, resolved.torch_device)
    except (CheckpointCompatibilityError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="CHECKPOINT") from exc

    try:
        records = [r for r in read_manifest(manifest) if r.writer_id == writer_id]
    except (ManifestError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="MANIFEST") from exc
    if n is not None:
        records = records[:n]
    if not records:
        raise typer.BadParameter(
            f"no lines found for writer {writer_id!r} in {manifest}", param_hint="--writer-id"
        )

    memory_config = config.memory
    if feature_layer is not None:
        memory_config = replace(memory_config, feature_layer=feature_layer)
    if pooling is not None:
        memory_config = replace(memory_config, pooling=pooling)

    try:
        lines = [(preprocess_path(r.image).tensor, r.text) for r in records]
        profile = compile_profile(
            model,
            tokenizer.charset,
            lines,
            model_fingerprint=checkpoint_fingerprint(checkpoint),
            config=memory_config,
            device=resolved.torch_device,
        )
    except (UnreadableImageError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    profile.save(out)

    typer.echo(f"wrote {out}")
    typer.echo(f"  writer          {writer_id}")
    typer.echo(f"  support lines   {len(lines)}")
    typer.echo(f"  characters      {len(profile.glyphs)} / {tokenizer.charset.size - 1}")
    typer.echo(f"  feature layer   {profile.feature_layer} ({profile.feature_dim}D)")
    typer.echo(f"  estimated size  ~{profile.estimated_bytes():,} bytes")


@app.command()
def transcribe(
    checkpoint: Path = typer.Argument(..., help="Checkpoint to transcribe with."),
    manifest: Path = typer.Argument(..., help="Manifest containing the lines to transcribe."),
    config_path: Path | None = typer.Option(
        None, "--config", help="Config file. Must match the one the checkpoint was trained with."
    ),
    charset: Path = typer.Option(DEFAULT_CHARSET_PATH, "--charset", help="Charset artifact."),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps | cuda"),
    profile: Path | None = typer.Option(
        None, "--profile", help="A compiled WriterProfile; personalize when given."
    ),
    writer_id: str | None = typer.Option(None, "--writer-id", help="Only this writer's lines."),
    n: int | None = typer.Option(None, "--n", help="Limit to the first N matching lines."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of a table."),
) -> None:
    """Transcribe lines; generic vs. personalized side by side when ``--profile`` is given."""
    setup_logging()
    try:
        config = load_config(config_path) if config_path else Config()
        tokenizer = load_tokenizer(charset)
        resolved = resolve_device(device)
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        model = _load_model_for_inference(checkpoint, config, tokenizer, resolved.torch_device)
    except (CheckpointCompatibilityError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="CHECKPOINT") from exc

    loaded_profile: WriterProfile | None = None
    if profile is not None:
        try:
            loaded_profile = WriterProfile.load(
                profile, expected_model_fingerprint=checkpoint_fingerprint(checkpoint)
            )
        except (ProfileCompatibilityError, OSError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--profile") from exc

    memory_config = (
        replace(config.memory, enabled=True) if loaded_profile is not None else config.memory
    )

    try:
        records = list(read_manifest(manifest))
    except (ManifestError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="MANIFEST") from exc
    if writer_id is not None:
        records = [r for r in records if r.writer_id == writer_id]
    if n is not None:
        records = records[:n]
    if not records:
        raise typer.BadParameter("no matching lines found", param_hint="MANIFEST")

    results: list[dict[str, str]] = []
    with torch.no_grad():
        for record in records:
            try:
                tensor = preprocess_path(record.image).tensor
            except UnreadableImageError as exc:
                raise typer.BadParameter(str(exc), param_hint="MANIFEST") from exc
            batch = tensor.unsqueeze(0).to(resolved.torch_device)
            output = model(batch)
            length = int(output.input_lengths[0])
            entry = {
                "sample_id": record.sample_id or record.image,
                "writer_id": record.writer_id,
                "reference": record.text,
                "generic": greedy_decode(output.logits[0], tokenizer, length),
            }
            if loaded_profile is not None:
                corrected = personalize(output, loaded_profile, tokenizer.charset, memory_config)
                entry["personalized"] = greedy_decode(corrected[0], tokenizer, length)
            results.append(entry)

    if as_json:
        typer.echo(json.dumps(results, indent=2))
        return

    for entry in results:
        typer.echo(f"[{entry['sample_id']}] writer={entry['writer_id']}")
        typer.echo(f"  ref            {entry['reference']!r}")
        typer.echo(f"  generic        {entry['generic']!r}")
        if "personalized" in entry:
            typer.echo(f"  personalized   {entry['personalized']!r}")


if __name__ == "__main__":  # pragma: no cover
    app()
