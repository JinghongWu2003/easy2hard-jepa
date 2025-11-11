"""Configuration utilities for loading YAML files and applying overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def _parse_override(value: str) -> Any:
    """Attempt to parse a primitive override value."""
    true_values = {"true", "yes", "on"}
    false_values = {"false", "no", "off"}

    lower = value.lower()
    if lower in true_values:
        return True
    if lower in false_values:
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    if value.lower() == "none":
        return None
    return value


def _apply_override(config: Dict[str, Any], key_path: Iterable[str], value: Any) -> None:
    current = config
    keys = list(key_path)
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if overrides:
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Invalid override '{override}', expected key=value")
            key, raw_value = override.split("=", 1)
            value = _parse_override(raw_value)
            _apply_override(config, key.split("."), value)
    return config
