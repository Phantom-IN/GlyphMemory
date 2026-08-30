"""YAML configuration loading with strict validation.

Strictness is the point. An unrecognised key is an error, not a warning: a typo such as
``learning_rat`` would otherwise leave the default silently in place and produce an experiment whose
recorded config does not describe what actually ran.

Note on annotations: the schema module uses ``from __future__ import annotations``, so
``dataclasses.fields()`` reports type *strings*. Types are resolved with ``typing.get_type_hints``
before use — without that, nested sections silently arrive as raw dictionaries.
"""

from __future__ import annotations

import types
from dataclasses import fields, is_dataclass
from functools import cache
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from glyphmemory.config.schema import Config, ConfigError, to_dict

T = TypeVar("T")

NoneType = type(None)


@cache
def _resolved_hints(cls: type) -> dict[str, Any]:
    """Field name -> concrete type, with string annotations resolved."""
    return get_type_hints(cls)


def _base_types(annotation: Any) -> set[type]:
    """Flatten an annotation into the set of concrete types it permits."""
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        found: set[type] = set()
        for arg in get_args(annotation):
            found |= _base_types(arg)
        return found
    if isinstance(annotation, type):
        return {annotation}
    return set()


def _build(cls: type[T], payload: Any, path: str) -> T:
    """Construct a config dataclass from a mapping, rejecting unknown/mistyped keys."""
    if not isinstance(payload, dict):
        raise ConfigError(f"{path or 'config'}: expected a mapping, got {type(payload).__name__}.")

    hints = _resolved_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        valid = ", ".join(sorted(known))
        raise ConfigError(f"{path or 'config'}: unknown key(s) {unknown}. Valid keys are: {valid}.")

    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in payload:
            continue
        annotation = hints[name]
        child = f"{path}.{name}" if path else name
        if is_dataclass(annotation) and isinstance(annotation, type):
            kwargs[name] = _build(annotation, payload[name], child)
        else:
            kwargs[name] = _coerce(payload[name], annotation, child)
    return cls(**kwargs)


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Apply the narrow numeric coercion YAML makes necessary.

    YAML yields ``3`` for a float field written without a decimal point, so int -> float is
    permitted. float -> int is not: silently truncating a research hyperparameter is exactly the
    class of error this loader exists to prevent.
    """
    allowed = _base_types(annotation)

    if value is None:
        if allowed and NoneType not in allowed:
            expected = " | ".join(sorted(t.__name__ for t in allowed))
            raise ConfigError(f"{path}: null is not allowed; expected {expected}.")
        return None

    wants_float = float in allowed
    wants_int = int in allowed
    is_bool = isinstance(value, bool)

    if wants_float and isinstance(value, int) and not is_bool:
        return float(value)

    if wants_float and isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ConfigError(f"{path}: cannot interpret {value!r} as a float.") from exc

    if wants_int and not wants_float and isinstance(value, float):
        raise ConfigError(
            f"{path}: expected an integer, got {value!r}. "
            "Write it without a decimal point rather than relying on truncation."
        )

    if allowed and not is_bool and bool in allowed and not isinstance(value, bool):
        raise ConfigError(f"{path}: expected true or false, got {value!r}.")

    return value


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML configuration file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML — {exc}") from exc

    if raw is None:
        raw = {}
    config = _build(Config, raw, "")
    check_consistency(config)
    return config


def check_consistency(config: Config) -> None:
    """Validate constraints that span sections.

    Per-field types are checked during construction; these are the relationships *between* fields,
    which no single section can see.
    """
    if config.data.image_height != config.model.input_height:
        raise ConfigError(
            f"data.image_height ({config.data.image_height}) must equal model.input_height "
            f"({config.model.input_height}). The visual encoder's height reducer is sized for "
            "one input height, so a mismatch is a run-time shape error rather than a "
            "configuration that trains something slightly different."
        )
    if config.model.visual_dim < 1 or config.model.gru_hidden < 1:
        raise ConfigError(
            f"model.visual_dim ({config.model.visual_dim}) and model.gru_hidden "
            f"({config.model.gru_hidden}) must both be positive."
        )
    if config.data.max_width % config.data.width_multiple:
        raise ConfigError(
            f"data.max_width ({config.data.max_width}) must be a multiple of "
            f"data.width_multiple ({config.data.width_multiple}); otherwise the guard rejects "
            "widths the padder would legitimately produce."
        )


def dump_config(config: Config, path: str | Path) -> Path:
    """Write a config back to YAML — used to snapshot the exact config of every run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(to_dict(config), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return path
