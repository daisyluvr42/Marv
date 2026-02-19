from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


VALID_EXECUTION_MODES = {"auto", "local", "sandbox"}


def get_execution_config_path() -> Path:
    value = os.getenv("EDGE_EXECUTION_CONFIG_PATH")
    if value:
        return Path(value).expanduser().resolve()
    data_dir = Path(os.getenv("EDGE_DATA_DIR", "./data")).expanduser().resolve()
    return data_dir / "execution-config.json"


def normalize_execution_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    defaults = {
        "mode": os.getenv("EDGE_EXECUTION_MODE", "auto").strip().lower() or "auto",
        "docker_image": os.getenv("EDGE_SANDBOX_DOCKER_IMAGE", "python:3.12-alpine").strip() or "python:3.12-alpine",
        "network_enabled": _env_bool("EDGE_SANDBOX_NETWORK_ENABLED", False),
    }
    if defaults["mode"] not in VALID_EXECUTION_MODES:
        defaults["mode"] = "auto"
    if not isinstance(raw, dict):
        return defaults

    normalized = defaults.copy()
    mode = str(raw.get("mode", normalized["mode"])).strip().lower()
    if mode in VALID_EXECUTION_MODES:
        normalized["mode"] = mode
    docker_image = str(raw.get("docker_image", normalized["docker_image"])).strip()
    if docker_image:
        normalized["docker_image"] = docker_image
    network_enabled = raw.get("network_enabled")
    if isinstance(network_enabled, bool):
        normalized["network_enabled"] = network_enabled
    return normalized


def load_execution_config(path: Path | None = None) -> dict[str, Any]:
    file_path = path or get_execution_config_path()
    if not file_path.exists():
        return normalize_execution_config(None)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return normalize_execution_config(None)
    return normalize_execution_config(payload if isinstance(payload, dict) else None)


def save_execution_config(config: dict[str, Any], path: Path | None = None) -> Path:
    normalized = normalize_execution_config(config)
    file_path = path or get_execution_config_path()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(normalized, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return file_path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}
