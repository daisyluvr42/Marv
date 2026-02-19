from __future__ import annotations

from backend.heartbeat.runtime import normalize_heartbeat_config


def test_normalize_heartbeat_config_bounds_and_mode() -> None:
    config = normalize_heartbeat_config(
        {
            "enabled": True,
            "mode": "interval",
            "interval_seconds": 1,
            "cron": "*/2 * * * *",
            "core_health_enabled": False,
            "resume_approved_tools_enabled": True,
            "emit_events": False,
            "memory_decay_enabled": True,
            "memory_decay_half_life_days": 15,
            "memory_decay_min_confidence": 0.1,
        }
    )
    assert config["mode"] == "interval"
    assert config["interval_seconds"] == 5
    assert config["core_health_enabled"] is False
    assert config["emit_events"] is False
    assert config["memory_decay_enabled"] is True
    assert config["memory_decay_half_life_days"] == 15
    assert config["memory_decay_min_confidence"] == 0.1


def test_normalize_heartbeat_config_invalid_mode_falls_back() -> None:
    config = normalize_heartbeat_config({"mode": "unknown", "interval_seconds": 999999})
    assert config["mode"] == "interval"
    assert config["interval_seconds"] == 86400
