"""Configuration loading tests.

The behaviour that matters scientifically is strictness: a mistyped key must fail loudly rather than
silently leave a default in place.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from glyphmemory.config import Config, ConfigError, dump_config, load_config, to_dict
from glyphmemory.config.loader import check_consistency

REPO_ROOT = Path(__file__).resolve().parents[1]


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_shipped_default_config_loads():
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert config.name == "default"
    assert config.runtime.device == "auto"
    assert config.model.max_parameters == 3_000_000


def test_defaults_apply_to_empty_config(tmp_path):
    config = load_config(write(tmp_path, ""))
    assert config == Config()


def test_partial_config_keeps_other_defaults(tmp_path):
    config = load_config(write(tmp_path, "runtime:\n  seed: 7\n"))
    assert config.runtime.seed == 7
    assert config.runtime.device == "auto"
    assert config.training.learning_rate == pytest.approx(3e-4)


def test_unknown_top_level_key_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, "modle:\n  name: gm_base\n"))


def test_unknown_nested_key_rejected(tmp_path):
    """A typo in a hyperparameter name must not silently keep the default."""
    with pytest.raises(ConfigError, match=r"training.*unknown key"):
        load_config(write(tmp_path, "training:\n  learning_rat: 0.1\n"))


def test_int_is_widened_to_float(tmp_path):
    config = load_config(write(tmp_path, "training:\n  learning_rate: 1\n"))
    assert isinstance(config.training.learning_rate, float)


def test_scientific_notation_string_parsed(tmp_path):
    config = load_config(write(tmp_path, "training:\n  learning_rate: '3e-4'\n"))
    assert config.training.learning_rate == pytest.approx(3e-4)


def test_float_where_int_expected_is_rejected(tmp_path):
    """Silently truncating an integer hyperparameter is the failure mode this prevents."""
    with pytest.raises(ConfigError, match="expected an integer"):
        load_config(write(tmp_path, "training:\n  epochs: 10.5\n"))


def test_section_must_be_a_mapping(tmp_path):
    with pytest.raises(ConfigError, match="expected a mapping"):
        load_config(write(tmp_path, "training: 5\n"))


def test_invalid_yaml_reports_path(tmp_path):
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(write(tmp_path, "training:\n  - [unbalanced\n"))


def test_missing_file_reports_clearly(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_roundtrip_through_dump(tmp_path):
    original = load_config(REPO_ROOT / "configs" / "default.yaml")
    path = dump_config(original, tmp_path / "snapshot.yaml")
    assert load_config(path) == original


def test_to_dict_is_plain_data():
    data = to_dict(Config())
    assert isinstance(data, dict)
    assert isinstance(data["runtime"], dict)
    assert data["runtime"]["device"] == "auto"


def test_all_six_config_sections_present():
    """Internal helper."""
    data = to_dict(Config())
    assert {"data", "model", "training", "evaluation", "memory", "runtime"} <= set(data)


def test_memory_defaults_match_fusion_v0_policy():
    """Internal helper."""
    config = Config()
    assert config.memory.protect_blank is True
    assert config.memory.gate_by_emission is True
    assert config.memory.pooling == "posterior_weighted"


# ------------------------------------------------------------------ cross-section checks
# Per-field types are validated during construction, but relationships
# *between* sections are invisible to any single section.


def _write(tmp_path, body: str):
    path = tmp_path / "c.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_mismatched_heights_are_rejected(tmp_path):
    """A run-time shape error, caught at config load instead.

    The visual encoder's height reducer is sized for one input height, so disagreeing values do not
    train something slightly different — they crash on the first batch.
    """
    path = _write(tmp_path, "data:\n  image_height: 48\nmodel:\n  input_height: 64\n")
    with pytest.raises(ConfigError, match=r"must equal model\.input_height"):
        load_config(path)


def test_matching_heights_are_accepted(tmp_path):
    path = _write(tmp_path, "data:\n  image_height: 48\nmodel:\n  input_height: 48\n")
    assert load_config(path).model.input_height == 48


def test_max_width_must_be_a_multiple_of_width_multiple(tmp_path):
    path = _write(tmp_path, "data:\n  max_width: 1601\n")
    with pytest.raises(ConfigError, match="multiple of"):
        load_config(path)


def test_non_positive_dimensions_are_rejected(tmp_path):
    path = _write(tmp_path, "model:\n  gru_hidden: 0\n")
    with pytest.raises(ConfigError, match="must both be positive"):
        load_config(path)


def test_the_shipped_default_config_is_consistent():
    """configs/default.yaml must satisfy every cross-section rule."""
    check_consistency(load_config(REPO_ROOT / "configs" / "default.yaml"))


def test_dataclass_defaults_are_consistent():
    check_consistency(Config())
