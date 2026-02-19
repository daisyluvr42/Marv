from __future__ import annotations

from backend.sandbox.runtime import normalize_execution_config


def test_normalize_execution_config_bounds() -> None:
    payload = normalize_execution_config(
        {
            "mode": "sandbox",
            "docker_image": "python:3.12-alpine",
            "network_enabled": True,
        }
    )
    assert payload["mode"] == "sandbox"
    assert payload["docker_image"] == "python:3.12-alpine"
    assert payload["network_enabled"] is True


def test_normalize_execution_config_invalid_mode_fallback() -> None:
    payload = normalize_execution_config({"mode": "unknown", "docker_image": "", "network_enabled": "x"})
    assert payload["mode"] in {"auto", "local", "sandbox"}
    assert payload["docker_image"]
