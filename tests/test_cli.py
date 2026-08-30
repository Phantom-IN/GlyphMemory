"""CLI tests. Every command the README documents must be exercised here."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from glyphmemory import __version__
from glyphmemory.cli import app

REPO_ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def flatten(text: str) -> str:
    """Reduce Rich-rendered CLI output to plain, whitespace-normalised text.

    Typer renders help and errors through Rich, which does three things that break a literal
    substring assertion, all of them presentation rather than content:

    * hard-wraps a message to the terminal width;
    * draws a box around it, so ``│`` lands mid-sentence;
    * emits ANSI colour codes **when the environment enables them** - which CI does and a developer
      machine usually does not, so a test can pass locally and fail in CI on styling alone. That is
      exactly how ``test_train_is_advertised_now_that_it_works`` broke.

    Stripping here rather than setting ``NO_COLOR`` is deliberate: ``FORCE_COLOR`` overrides
    ``NO_COLOR``, so an environment variable cannot be relied on to win.
    """
    return " ".join(_ANSI.sub("", text).replace("│", " ").split())


def test_help_lists_implemented_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("version", "info", "config"):
        assert command in result.stdout


def test_version_matches_package():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_info_reports_environment():
    result = runner.invoke(app, ["info", "--device", "cpu"])
    assert result.exit_code == 0
    assert "resolved device  cpu" in result.stdout
    assert "torch" in result.stdout


def test_info_json_is_machine_readable():
    result = runner.invoke(app, ["info", "--device", "cpu", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["device"]["kind"] == "cpu"
    assert "torch_version" in payload["environment"]


def test_info_rejects_unknown_device():
    result = runner.invoke(app, ["info", "--device", "tpu"])
    assert result.exit_code != 0


def test_config_show_renders_sections():
    result = runner.invoke(app, ["config", "show", str(REPO_ROOT / "configs" / "default.yaml")])
    assert result.exit_code == 0
    for section in ("[runtime]", "[model]", "[memory]"):
        assert section in result.stdout


def test_config_show_json():
    result = runner.invoke(
        app, ["config", "show", str(REPO_ROOT / "configs" / "default.yaml"), "--json"]
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["model"]["max_parameters"] == 3_000_000


def test_config_show_reports_invalid_config(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("training:\n  learning_rat: 0.1\n", encoding="utf-8")
    result = runner.invoke(app, ["config", "show", str(bad)])
    assert result.exit_code != 0


def _write_manifest(tmp_path: Path, texts: list[str]) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "image": f"/tmp/l{i}.png",
                    "text": text,
                    "writer_id": "synthetic/w0",
                    "dataset": "synthetic",
                    "split": "train",
                    "sample_id": f"synthetic/{i}",
                }
            )
            for i, text in enumerate(texts)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_charset_report_on_covered_manifest(tmp_path):
    manifest = _write_manifest(tmp_path, ["The quick brown fox!", "jumps over (the) dog?"])
    result = runner.invoke(app, ["data", "charset-report", str(manifest)])
    assert result.exit_code == 0
    assert "charset_en_v1" in result.stdout
    assert "fully covered      yes" in result.stdout


def test_charset_report_names_unsupported_characters(tmp_path):
    manifest = _write_manifest(tmp_path, ["plain text", "costs 50€"])
    result = runner.invoke(app, ["data", "charset-report", str(manifest)])
    assert result.exit_code == 0
    assert "fully covered      NO" in result.stdout
    assert "U+20AC" in result.stdout


def test_charset_report_json(tmp_path):
    manifest = _write_manifest(tmp_path, ["costs 50€"])
    result = runner.invoke(app, ["data", "charset-report", str(manifest), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["is_covered"] is False
    assert payload["normalization"] == "nfc_v1"


def test_charset_report_rejects_bad_manifest(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    assert runner.invoke(app, ["data", "charset-report", str(bad)]).exit_code != 0


def test_charset_report_rejects_unknown_policy(tmp_path):
    manifest = _write_manifest(tmp_path, ["text"])
    result = runner.invoke(
        app, ["data", "charset-report", str(manifest), "--policy", "aggressive_v9"]
    )
    assert result.exit_code != 0


# ------------------------------------------------------------------ data prepare/stats/split


def test_data_prepare_cvl_writes_manifest_and_summary(fake_cvl, tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["data", "prepare", "--dataset", "cvl", "--source", str(fake_cvl), "--out", str(out)]
    )
    assert result.exit_code == 0, result.stdout
    assert (out / "manifest.jsonl").is_file()
    assert (out / "cvl_summary.json").is_file()
    assert "excluded_language" in result.stdout
    assert "limitation" in result.stdout


def test_data_prepare_rejects_unknown_dataset(fake_cvl, tmp_path):
    """A name with no adapter fails on the *name*, before any source is touched.

    Asserting the message keeps it honest if a third adapter lands.
    """
    result = runner.invoke(
        app,
        ["data", "prepare", "--dataset", "nope", "--source", str(fake_cvl), "--out", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "unknown dataset" in flatten(result.stderr)


def test_data_prepare_iam_writes_manifest_and_summary(fake_iam, tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["data", "prepare", "--dataset", "iam", "--source", str(fake_iam), "--out", str(out)]
    )
    assert result.exit_code == 0, result.stdout
    assert (out / "manifest.jsonl").is_file()
    assert (out / "iam_summary.json").is_file()
    assert "struck_out_token" in result.stdout
    assert "forms" in result.stdout


def test_data_prepare_iam_rejects_a_cvl_source(fake_cvl, tmp_path):
    """Pointing the IAM adapter at the wrong corpus fails loudly, not with an empty manifest."""
    result = runner.invoke(
        app,
        ["data", "prepare", "--dataset", "iam", "--source", str(fake_cvl), "--out", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "does not look like an IAM release" in flatten(result.stderr)


def test_data_prepare_rejects_non_cvl_source(tmp_path):
    (tmp_path / "src").mkdir()
    result = runner.invoke(
        app,
        [
            "data",
            "prepare",
            "--dataset",
            "cvl",
            "--source",
            str(tmp_path / "src"),
            "--out",
            str(tmp_path / "o"),
        ],
    )
    assert result.exit_code != 0


def test_data_stats_reports_histogram_and_passages(fake_cvl, tmp_path):
    out = tmp_path / "out"
    runner.invoke(
        app, ["data", "prepare", "--dataset", "cvl", "--source", str(fake_cvl), "--out", str(out)]
    )
    result = runner.invoke(app, ["data", "stats", str(out / "manifest.jsonl")])
    assert result.exit_code == 0, result.stdout
    assert "[writers]" in result.stdout
    assert "[passages]" in result.stdout
    assert "p1" in result.stdout


def test_data_stats_json(fake_cvl, tmp_path):
    out = tmp_path / "out"
    runner.invoke(
        app, ["data", "prepare", "--dataset", "cvl", "--source", str(fake_cvl), "--out", str(out)]
    )
    result = runner.invoke(app, ["data", "stats", str(out / "manifest.jsonl"), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["writers"]["writers"] == 2
    assert payload["passages"]["passages"] == 2


def test_data_split_writes_all_three_artifacts(fake_cvl, tmp_path):
    """A recomputed pool is a pool whose denominator can drift; all of it goes to disk."""
    out = tmp_path / "out"
    runner.invoke(
        app, ["data", "prepare", "--dataset", "cvl", "--source", str(fake_cvl), "--out", str(out)]
    )
    split_dir = tmp_path / "split"
    result = runner.invoke(
        app,
        [
            "data",
            "split",
            str(out / "manifest.jsonl"),
            "--out",
            str(split_dir),
            "--ratios",
            "0.5,0.0,0.5",
            "--query-size",
            "1",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (split_dir / "manifest.jsonl").is_file()
    assert (split_dir / "writer_split.json").is_file()
    assert (split_dir / "support_query.json").is_file()


def test_data_split_rejects_malformed_ratios(fake_cvl, tmp_path):
    out = tmp_path / "out"
    runner.invoke(
        app, ["data", "prepare", "--dataset", "cvl", "--source", str(fake_cvl), "--out", str(out)]
    )
    result = runner.invoke(
        app,
        [
            "data",
            "split",
            str(out / "manifest.jsonl"),
            "--out",
            str(tmp_path / "s"),
            "--ratios",
            "0.5,0.5",
        ],
    )
    assert result.exit_code != 0


# ------------------------------------------------------------------ model summary


def test_model_summary_prints_the_exact_parameter_count():
    """Internal helper."""
    result = runner.invoke(app, ["model", "summary"])
    assert result.exit_code == 0, result.stdout
    assert "1,544,560" in result.stdout
    assert "ceiling   <= 3,000,000   ok" in result.stdout
    for module in ("encoder", "sequence", "head"):
        assert module in result.stdout


def test_model_summary_shows_the_shape_table():
    result = runner.invoke(app, ["model", "summary", "--width", "512"])
    assert result.exit_code == 0
    assert "[B, 192, 1, 128]" in result.stdout
    assert "[B, 128, 80]" in result.stdout


def test_model_summary_json():
    result = runner.invoke(app, ["model", "summary", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["parameters"]["total"] == 1_544_560
    assert payload["parameters"]["within_hard_ceiling"] is True
    assert payload["architecture"]["head"]["applies_softmax"] is False


def test_model_summary_rejects_an_inconsistent_config(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("data:\n  image_height: 48\n", encoding="utf-8")
    assert runner.invoke(app, ["model", "summary", "--config", str(bad)]).exit_code != 0


# ------------------------------------------------------------------ train


def _synthetic(tmp_path):
    out = tmp_path / "corpus"
    result = runner.invoke(
        app,
        [
            "data",
            "make-synthetic",
            "--out",
            str(out),
            "--writers",
            "2",
            "--lines",
            "3",
            "--seed",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stdout
    return out / "manifest.jsonl"


@pytest.mark.slow
def test_train_writes_a_complete_run_directory(tmp_path):
    manifest = _synthetic(tmp_path)
    result = runner.invoke(
        app,
        [
            "train",
            str(manifest),
            "--out",
            str(tmp_path / "runs"),
            "--name",
            "gm_base_cli",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.stdout

    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    run = runs[0]
    for artifact in ("run.json", "config.yaml", "metrics.jsonl", "metrics.json"):
        assert (run / artifact).is_file(), artifact
    assert (run / "checkpoints" / "last.pt").is_file()

    record = json.loads((run / "run.json").read_text())
    from glyphmemory.training import missing_fields

    assert missing_fields(record) == []
    assert record["device"] == "cpu"
    assert record["parameter_count"] == 1_544_560


@pytest.mark.slow
def test_train_reports_selection_metric_and_never_loss(tmp_path):
    manifest = _synthetic(tmp_path)
    result = runner.invoke(
        app,
        [
            "train",
            str(manifest),
            "--out",
            str(tmp_path / "runs"),
            "--name",
            "gm_base_cli",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0
    assert "val_cer (never loss)" in result.stdout


@pytest.mark.slow
def test_train_limit_restricts_the_corpus(tmp_path):
    manifest = _synthetic(tmp_path)
    result = runner.invoke(
        app,
        [
            "train",
            str(manifest),
            "--out",
            str(tmp_path / "runs"),
            "--name",
            "gm_base_cli",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--limit",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "train 2" in result.stdout


def test_train_rejects_a_missing_manifest(tmp_path):
    result = runner.invoke(
        app, ["train", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path / "runs")]
    )
    assert result.exit_code != 0


# ------------------------------------------------------------------ evaluate


def _trained_checkpoint(tmp_path) -> tuple[Path, Path]:
    """A one-epoch checkpoint on the synthetic corpus, for evaluate's plumbing tests."""
    manifest = _synthetic(tmp_path)
    result = runner.invoke(
        app,
        [
            "train",
            str(manifest),
            "--out",
            str(tmp_path / "runs"),
            "--name",
            "gm_base_cli",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.stdout
    run = next((tmp_path / "runs").iterdir())
    return run / "checkpoints" / "last.pt", manifest


@pytest.mark.slow
def test_evaluate_prints_gate_and_taxonomy(tmp_path):
    checkpoint, manifest = _trained_checkpoint(tmp_path)
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(checkpoint),
            str(manifest),
            "--split",
            "train",
            "--device",
            "cpu",
            "--workers",
            "0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "gate" in result.stdout
    assert "single-glyph confusions" in result.stdout
    assert "per-writer CER" in result.stdout


@pytest.mark.slow
def test_evaluate_json_matches_the_report(tmp_path):
    checkpoint, manifest = _trained_checkpoint(tmp_path)
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(checkpoint),
            str(manifest),
            "--split",
            "train",
            "--device",
            "cpu",
            "--workers",
            "0",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert "single_glyph_confusion_definition" in payload["taxonomy"]
    assert payload["gate"]["condition_2_no_pipeline_bug"]["reviewed"] is None


@pytest.mark.slow
def test_evaluate_writes_the_json_report_to_out(tmp_path):
    checkpoint, manifest = _trained_checkpoint(tmp_path)
    out = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(checkpoint),
            str(manifest),
            "--split",
            "train",
            "--device",
            "cpu",
            "--workers",
            "0",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out.is_file()
    payload = json.loads(out.read_text())
    assert "per_writer" in payload


def test_evaluate_rejects_a_missing_checkpoint(tmp_path):
    manifest = _synthetic(tmp_path)
    result = runner.invoke(
        app, ["evaluate", str(tmp_path / "nope.pt"), str(manifest), "--device", "cpu"]
    )
    assert result.exit_code != 0


# ------------------------------------------------------------------ benchmark


@pytest.mark.slow
def test_benchmark_prints_the_context_and_grid(tmp_path):
    checkpoint, _ = _trained_checkpoint(tmp_path)
    result = runner.invoke(
        app,
        [
            "benchmark",
            str(checkpoint),
            "--device",
            "cpu",
            "--threads",
            "1",
            "--widths",
            "64,128",
            "--batch-sizes",
            "1,2",
            "--warmup",
            "1",
            "--iterations",
            "2",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "context" in result.stdout
    assert "MPS CTC round-trip cost" in result.stdout
    assert "AMP" in result.stdout


@pytest.mark.slow
def test_benchmark_json_has_a_self_contained_grid(tmp_path):
    checkpoint, _ = _trained_checkpoint(tmp_path)
    result = runner.invoke(
        app,
        [
            "benchmark",
            str(checkpoint),
            "--device",
            "cpu",
            "--threads",
            "1",
            "--widths",
            "64",
            "--batch-sizes",
            "1",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    point = payload["grid"][0]
    for field in ("cpu_model", "os", "num_threads", "input_width", "batch_size", "p50_ms"):
        assert field in point


@pytest.mark.slow
def test_benchmark_writes_the_json_report_to_out(tmp_path):
    checkpoint, _ = _trained_checkpoint(tmp_path)
    out = tmp_path / "bench.json"
    result = runner.invoke(
        app,
        [
            "benchmark",
            str(checkpoint),
            "--device",
            "cpu",
            "--threads",
            "1",
            "--widths",
            "64",
            "--batch-sizes",
            "1",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out.is_file()
    payload = json.loads(out.read_text())
    assert "roundtrip_on_device" in payload


def test_benchmark_rejects_malformed_widths(tmp_path):
    checkpoint, _ = _trained_checkpoint(tmp_path)
    result = runner.invoke(
        app, ["benchmark", str(checkpoint), "--device", "cpu", "--widths", "not-a-number"]
    )
    assert result.exit_code != 0


def test_benchmark_rejects_a_missing_checkpoint(tmp_path):
    result = runner.invoke(
        app, ["benchmark", str(tmp_path / "nope.pt"), "--device", "cpu"]
    )
    assert result.exit_code != 0


def test_benchmark_is_advertised_now_that_it_works():
    assert " benchmark " in flatten(runner.invoke(app, ["--help"]).stdout)


def test_no_unimplemented_commands_are_advertised():
    """Only working commands are advertised.

    Anything still here is a command the CLI must not mention until it does something. Empty for now
    — every planned top-level command exists.
    """
    result = runner.invoke(app, ["--help"])
    for future in ():
        assert f" {future} " not in result.stdout


def test_train_is_advertised_now_that_it_works():
    """The command list is Rich-rendered, so compare against normalised text, not raw stdout."""
    assert " train " in flatten(runner.invoke(app, ["--help"]).stdout)


def test_evaluate_is_advertised_now_that_it_works():
    assert " evaluate " in flatten(runner.invoke(app, ["--help"]).stdout)


# ------------------------------------------------------------------ enroll / transcribe


def _first_writer(manifest: Path) -> str:
    from glyphmemory.data import read_manifest

    return next(iter(read_manifest(manifest))).writer_id


@pytest.mark.slow
def test_enroll_writes_a_profile_with_the_expected_metadata(tmp_path):
    checkpoint, manifest = _trained_checkpoint(tmp_path)
    writer = _first_writer(manifest)
    out = tmp_path / "writer.profile.pt"

    result = runner.invoke(
        app,
        ["enroll", str(checkpoint), str(manifest), "--writer-id", writer, "--out", str(out)],
    )

    assert result.exit_code == 0, result.stdout
    assert out.is_file()

    from glyphmemory.memory import WriterProfile
    from glyphmemory.runtime.fingerprint import checkpoint_fingerprint

    profile = WriterProfile.load(out, expected_model_fingerprint=checkpoint_fingerprint(checkpoint))
    assert profile.glyphs
    assert profile.feature_layer == "sequence"
    assert f"characters      {len(profile.glyphs)}" in result.stdout


@pytest.mark.slow
def test_enroll_respects_n_and_pooling_overrides(tmp_path):
    checkpoint, manifest = _trained_checkpoint(tmp_path)
    writer = _first_writer(manifest)
    out = tmp_path / "writer.profile.pt"

    result = runner.invoke(
        app,
        [
            "enroll",
            str(checkpoint),
            str(manifest),
            "--writer-id",
            writer,
            "--out",
            str(out),
            "--n",
            "1",
            "--pooling",
            "uniform",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "support lines   1" in result.stdout


def test_enroll_rejects_an_unknown_writer(tmp_path):
    checkpoint, manifest = _trained_checkpoint(tmp_path)
    result = runner.invoke(
        app,
        [
            "enroll",
            str(checkpoint),
            str(manifest),
            "--writer-id",
            "not-a-real-writer",
            "--out",
            str(tmp_path / "writer.profile.pt"),
        ],
    )
    assert result.exit_code != 0


@pytest.mark.slow
def test_transcribe_without_profile_prints_generic_only(tmp_path):
    checkpoint, manifest = _trained_checkpoint(tmp_path)

    result = runner.invoke(app, ["transcribe", str(checkpoint), str(manifest), "--n", "1"])

    assert result.exit_code == 0, result.stdout
    assert "generic" in result.stdout
    assert "personalized" not in result.stdout


@pytest.mark.slow
def test_transcribe_with_profile_prints_generic_and_personalized(tmp_path):
    checkpoint, manifest = _trained_checkpoint(tmp_path)
    writer = _first_writer(manifest)
    profile_path = tmp_path / "writer.profile.pt"
    enroll_result = runner.invoke(
        app,
        [
            "enroll",
            str(checkpoint),
            str(manifest),
            "--writer-id",
            writer,
            "--out",
            str(profile_path),
        ],
    )
    assert enroll_result.exit_code == 0, enroll_result.stdout

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(checkpoint),
            str(manifest),
            "--writer-id",
            writer,
            "--profile",
            str(profile_path),
            "--n",
            "1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "generic" in result.stdout
    assert "personalized" in result.stdout


@pytest.mark.slow
def test_transcribe_json_includes_both_transcripts(tmp_path):
    checkpoint, manifest = _trained_checkpoint(tmp_path)
    writer = _first_writer(manifest)
    profile_path = tmp_path / "writer.profile.pt"
    runner.invoke(
        app,
        [
            "enroll",
            str(checkpoint),
            str(manifest),
            "--writer-id",
            writer,
            "--out",
            str(profile_path),
        ],
    )

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(checkpoint),
            str(manifest),
            "--writer-id",
            writer,
            "--profile",
            str(profile_path),
            "--n",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload[0]["generic"]
    assert payload[0]["personalized"]


@pytest.mark.slow
def test_transcribe_rejects_a_profile_from_a_different_checkpoint(tmp_path):
    checkpoint_a, manifest = _trained_checkpoint(tmp_path)

    # A second checkpoint on the same manifest but a different seed, so its weights (and therefore
    # its content fingerprint) differ from checkpoint_a's -- proving the rejection is a real
    # fingerprint mismatch, not an artifact of two identical runs.
    second_run = tmp_path / "runs2"
    result = runner.invoke(
        app,
        [
            "train",
            str(manifest),
            "--out",
            str(second_run),
            "--name",
            "gm_base_cli_2",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
            "--seed",
            "99",
        ],
    )
    assert result.exit_code == 0, result.stdout
    checkpoint_b = next(second_run.iterdir()) / "checkpoints" / "last.pt"

    writer = _first_writer(manifest)
    profile_path = tmp_path / "writer.profile.pt"
    runner.invoke(
        app,
        [
            "enroll",
            str(checkpoint_a),
            str(manifest),
            "--writer-id",
            writer,
            "--out",
            str(profile_path),
        ],
    )

    result = runner.invoke(
        app,
        ["transcribe", str(checkpoint_b), str(manifest), "--profile", str(profile_path)],
    )

    assert result.exit_code != 0


def test_transcribe_rejects_a_missing_checkpoint(tmp_path):
    result = runner.invoke(
        app, ["transcribe", str(tmp_path / "nope.pt"), str(tmp_path / "m.jsonl")]
    )
    assert result.exit_code != 0


def test_enroll_is_advertised_now_that_it_works():
    assert " enroll " in flatten(runner.invoke(app, ["--help"]).stdout)


def test_transcribe_is_advertised_now_that_it_works():
    assert " transcribe " in flatten(runner.invoke(app, ["--help"]).stdout)


# ------------------------------------------------------------------ few-shot


@pytest.mark.slow
def test_few_shot_runs_end_to_end_and_reports_cer_at_0(tmp_path):
    """Plain synthetic data carries no source_page (form field), so every same-form/cross-form shot
    is correctly reported unavailable -- this checks the CLI wiring runs cleanly end to end even
    when no shot is possible, not the form-partition logic itself (covered directly in
    tests/test_evaluation_few_shot.py).
    """
    checkpoint, manifest = _trained_checkpoint(tmp_path)
    split_out = tmp_path / "split"
    split_result = runner.invoke(
        app,
        [
            "data",
            "split",
            str(manifest),
            "--out",
            str(split_out),
            "--ratios",
            "0.5,0.5,0",
            "--query-size",
            "1",
            "--eval-split",
            "val",
        ],
    )
    assert split_result.exit_code == 0, split_result.stdout

    result = runner.invoke(
        app,
        [
            "few-shot",
            str(checkpoint),
            str(split_out / "manifest.jsonl"),
            "--support-query",
            str(split_out / "support_query.json"),
            "--split",
            "val",
            "--shots",
            "1",
            "--seeds",
            "1337",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "writers" in result.stdout


@pytest.mark.slow
def test_few_shot_exercises_real_shots_when_forms_are_present(tmp_path):
    """A bigger corpus with source_page assigned, so same-form/cross-form draws are actually
    possible -- the literal protocol path, run through the real CLI.
    """
    make_result = runner.invoke(
        app,
        [
            "data",
            "make-synthetic",
            "--out",
            str(tmp_path / "corpus"),
            "--writers",
            "4",
            "--lines",
            "10",
            "--seed",
            "7",
        ],
    )
    assert make_result.exit_code == 0, make_result.stdout
    raw_manifest = tmp_path / "corpus" / "manifest.jsonl"

    from glyphmemory.data.manifest import read_manifest, write_manifest

    records = list(read_manifest(raw_manifest))
    with_forms = [
        dataclasses.replace(r, source_page=f"{r.writer_id}-page{i % 2}")
        for i, r in enumerate(records)
    ]
    formed_manifest = tmp_path / "corpus" / "manifest_with_forms.jsonl"
    write_manifest(formed_manifest, with_forms)

    train_result = runner.invoke(
        app,
        [
            "train",
            str(formed_manifest),
            "--out",
            str(tmp_path / "runs"),
            "--name",
            "gm_base_few_shot",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    assert train_result.exit_code == 0, train_result.stdout
    checkpoint = next((tmp_path / "runs").iterdir()) / "checkpoints" / "last.pt"

    split_out = tmp_path / "split"
    split_result = runner.invoke(
        app,
        [
            "data",
            "split",
            str(formed_manifest),
            "--out",
            str(split_out),
            "--ratios",
            "0.5,0.5,0",
            "--query-size",
            "2",
            "--eval-split",
            "val",
        ],
    )
    assert split_result.exit_code == 0, split_result.stdout

    result = runner.invoke(
        app,
        [
            "few-shot",
            str(checkpoint),
            str(split_out / "manifest.jsonl"),
            "--support-query",
            str(split_out / "support_query.json"),
            "--split",
            "val",
            "--shots",
            "1,3",
            "--seeds",
            "1337",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["curves"]
    # At least one writer should have a real, scored shot -- not everything unavailable.
    assert any(curve["shots"] for curve in payload["curves"])


def test_few_shot_writes_the_json_report_to_out(tmp_path):
    checkpoint, manifest = _trained_checkpoint(tmp_path)
    split_out = tmp_path / "split"
    runner.invoke(
        app,
        [
            "data",
            "split",
            str(manifest),
            "--out",
            str(split_out),
            "--ratios",
            "0.5,0.5,0",
            "--query-size",
            "1",
            "--eval-split",
            "val",
        ],
    )
    out = tmp_path / "few_shot.json"

    result = runner.invoke(
        app,
        [
            "few-shot",
            str(checkpoint),
            str(split_out / "manifest.jsonl"),
            "--support-query",
            str(split_out / "support_query.json"),
            "--split",
            "val",
            "--shots",
            "1",
            "--seeds",
            "1337",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert out.is_file()
    payload = json.loads(out.read_text())
    assert "curves" in payload


def test_few_shot_rejects_a_missing_support_query_file(tmp_path):
    checkpoint, manifest = _trained_checkpoint(tmp_path)
    result = runner.invoke(
        app,
        [
            "few-shot",
            str(checkpoint),
            str(manifest),
            "--support-query",
            str(tmp_path / "nope.json"),
        ],
    )
    assert result.exit_code != 0


def test_few_shot_rejects_a_missing_checkpoint(tmp_path):
    result = runner.invoke(
        app,
        [
            "few-shot",
            str(tmp_path / "nope.pt"),
            str(tmp_path / "m.jsonl"),
            "--support-query",
            str(tmp_path / "sq.json"),
        ],
    )
    assert result.exit_code != 0


def test_few_shot_is_advertised_now_that_it_works():
    assert " few-shot " in flatten(runner.invoke(app, ["--help"]).stdout)


# ------------------------------------------------------------------ rival-baselines


@pytest.mark.slow
def test_rival_baselines_runs_end_to_end_and_reports_cer_at_0(tmp_path):
    checkpoint, manifest = _trained_checkpoint(tmp_path)
    split_out = tmp_path / "split"
    split_result = runner.invoke(
        app,
        [
            "data",
            "split",
            str(manifest),
            "--out",
            str(split_out),
            "--ratios",
            "0.5,0.5,0",
            "--query-size",
            "1",
            "--eval-split",
            "val",
        ],
    )
    assert split_result.exit_code == 0, split_result.stdout

    result = runner.invoke(
        app,
        [
            "rival-baselines",
            str(checkpoint),
            str(split_out / "manifest.jsonl"),
            "--support-query",
            str(split_out / "support_query.json"),
            "--split",
            "val",
            "--methods",
            "glyphmemory,replay",
            "--shots",
            "1",
            "--seeds",
            "1337",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "writers" in result.stdout


@pytest.mark.slow
def test_rival_baselines_exercises_every_method_when_forms_are_present(tmp_path):
    make_result = runner.invoke(
        app,
        [
            "data",
            "make-synthetic",
            "--out",
            str(tmp_path / "corpus"),
            "--writers",
            "4",
            "--lines",
            "10",
            "--seed",
            "11",
        ],
    )
    assert make_result.exit_code == 0, make_result.stdout
    raw_manifest = tmp_path / "corpus" / "manifest.jsonl"

    from glyphmemory.data.manifest import read_manifest, write_manifest

    records = list(read_manifest(raw_manifest))
    with_forms = [
        dataclasses.replace(r, source_page=f"{r.writer_id}-page{i % 2}")
        for i, r in enumerate(records)
    ]
    formed_manifest = tmp_path / "corpus" / "manifest_with_forms.jsonl"
    write_manifest(formed_manifest, with_forms)

    train_result = runner.invoke(
        app,
        [
            "train",
            str(formed_manifest),
            "--out",
            str(tmp_path / "runs"),
            "--name",
            "gm_base_rivals",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    assert train_result.exit_code == 0, train_result.stdout
    checkpoint = next((tmp_path / "runs").iterdir()) / "checkpoints" / "last.pt"

    split_out = tmp_path / "split"
    split_result = runner.invoke(
        app,
        [
            "data",
            "split",
            str(formed_manifest),
            "--out",
            str(split_out),
            "--ratios",
            "0.5,0.5,0",
            "--query-size",
            "2",
            "--eval-split",
            "val",
        ],
    )
    assert split_result.exit_code == 0, split_result.stdout

    result = runner.invoke(
        app,
        [
            "rival-baselines",
            str(checkpoint),
            str(split_out / "manifest.jsonl"),
            "--support-query",
            str(split_out / "support_query.json"),
            "--split",
            "val",
            "--methods",
            "glyphmemory,head_ft,replay",
            "--shots",
            "1",
            "--seeds",
            "1337",
            "--ft-steps",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    methods_seen = {s["method"] for c in payload["curves"] for s in c["shots"]}
    assert methods_seen == {"glyphmemory", "head_ft", "replay"}


def test_rival_baselines_writes_the_json_report_to_out(tmp_path):
    checkpoint, manifest = _trained_checkpoint(tmp_path)
    split_out = tmp_path / "split"
    runner.invoke(
        app,
        [
            "data",
            "split",
            str(manifest),
            "--out",
            str(split_out),
            "--ratios",
            "0.5,0.5,0",
            "--query-size",
            "1",
            "--eval-split",
            "val",
        ],
    )
    out = tmp_path / "rivals.json"

    result = runner.invoke(
        app,
        [
            "rival-baselines",
            str(checkpoint),
            str(split_out / "manifest.jsonl"),
            "--support-query",
            str(split_out / "support_query.json"),
            "--split",
            "val",
            "--methods",
            "glyphmemory",
            "--shots",
            "1",
            "--seeds",
            "1337",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert out.is_file()
    payload = json.loads(out.read_text())
    assert "curves" in payload


def test_rival_baselines_rejects_an_unknown_method(tmp_path):
    checkpoint, manifest = _trained_checkpoint(tmp_path)
    result = runner.invoke(
        app,
        [
            "rival-baselines",
            str(checkpoint),
            str(manifest),
            "--support-query",
            str(tmp_path / "sq.json"),
            "--methods",
            "not-a-real-method",
        ],
    )
    assert result.exit_code != 0


def test_rival_baselines_rejects_a_missing_support_query_file(tmp_path):
    checkpoint, manifest = _trained_checkpoint(tmp_path)
    result = runner.invoke(
        app,
        [
            "rival-baselines",
            str(checkpoint),
            str(manifest),
            "--support-query",
            str(tmp_path / "nope.json"),
        ],
    )
    assert result.exit_code != 0


def test_rival_baselines_rejects_a_missing_checkpoint(tmp_path):
    result = runner.invoke(
        app,
        [
            "rival-baselines",
            str(tmp_path / "nope.pt"),
            str(tmp_path / "m.jsonl"),
            "--support-query",
            str(tmp_path / "sq.json"),
        ],
    )
    assert result.exit_code != 0


def test_rival_baselines_is_advertised_now_that_it_works():
    assert " rival-baselines " in flatten(runner.invoke(app, ["--help"]).stdout)
