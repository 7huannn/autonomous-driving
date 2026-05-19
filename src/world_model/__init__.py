"""World-model utilities for Stage 13 EPLS integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


VALID_STAGE13_MODES = {"offline", "live_planning", "iterative"}


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def stage13_mode(config: dict[str, Any]) -> str:
    stage13 = config.get("stage13", {})
    if not isinstance(stage13, dict):
        raise ValueError("world model config key 'stage13' must be a mapping")
    mode = str(stage13.get("mode", "offline"))
    if mode not in VALID_STAGE13_MODES:
        raise ValueError(f"Unsupported stage13.mode '{mode}'. Expected one of {sorted(VALID_STAGE13_MODES)}")
    return mode


def ensure_stage13_scope(config: dict[str, Any], allow_stage13_control: bool) -> str:
    mode = stage13_mode(config)
    if mode in {"live_planning", "iterative"} and not allow_stage13_control:
        raise PermissionError(
            "stage13.mode requests simulator control. Re-run with --allow-stage13-control to acknowledge "
            "the Stage 13 scope exception."
        )
    return mode
