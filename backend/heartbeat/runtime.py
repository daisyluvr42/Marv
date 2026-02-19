from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlmodel import select

from backend.agent.state import now_ts
from backend.approvals.service import list_approvals
from backend.core_client.openai_compat import get_core_client
from backend.ledger.events import HeartbeatEvent
from backend.ledger.store import append_event
from backend.storage.db import get_session
from backend.storage.models import ToolCall
from backend.tools.runner import execute_existing_tool_call


VALID_MODES = {"interval", "cron"}
DEFAULT_CRON = "*/1 * * * *"
HEARTBEAT_CONVERSATION_ID = "system:heartbeat"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _default_config_from_env() -> dict[str, Any]:
    return normalize_heartbeat_config(
        {
            "enabled": _env_bool("HEARTBEAT_ENABLED", True),
            "mode": os.getenv("HEARTBEAT_MODE", "interval"),
            "interval_seconds": int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "60")),
            "cron": os.getenv("HEARTBEAT_CRON", DEFAULT_CRON),
            "core_health_enabled": _env_bool("HEARTBEAT_CORE_HEALTH_ENABLED", True),
            "resume_approved_tools_enabled": _env_bool("HEARTBEAT_RESUME_APPROVED_TOOLS_ENABLED", True),
            "emit_events": _env_bool("HEARTBEAT_EMIT_EVENTS", True),
        }
    )


def normalize_heartbeat_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    defaults = {
        "enabled": True,
        "mode": "interval",
        "interval_seconds": 60,
        "cron": DEFAULT_CRON,
        "core_health_enabled": True,
        "resume_approved_tools_enabled": True,
        "emit_events": True,
    }
    if not isinstance(raw, dict):
        return defaults

    normalized = defaults.copy()
    if isinstance(raw.get("enabled"), bool):
        normalized["enabled"] = raw["enabled"]

    mode = str(raw.get("mode", normalized["mode"])).strip().lower()
    if mode in VALID_MODES:
        normalized["mode"] = mode

    try:
        interval_seconds = int(raw.get("interval_seconds", normalized["interval_seconds"]))
    except (TypeError, ValueError):
        interval_seconds = normalized["interval_seconds"]
    normalized["interval_seconds"] = max(5, min(86400, interval_seconds))

    cron = str(raw.get("cron", normalized["cron"])).strip()
    if cron:
        normalized["cron"] = cron

    if isinstance(raw.get("core_health_enabled"), bool):
        normalized["core_health_enabled"] = raw["core_health_enabled"]
    if isinstance(raw.get("resume_approved_tools_enabled"), bool):
        normalized["resume_approved_tools_enabled"] = raw["resume_approved_tools_enabled"]
    if isinstance(raw.get("emit_events"), bool):
        normalized["emit_events"] = raw["emit_events"]
    return normalized


def get_heartbeat_config_path() -> Path:
    env_path = os.getenv("EDGE_HEARTBEAT_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    data_dir = Path(os.getenv("EDGE_DATA_DIR", "./data")).expanduser().resolve()
    return data_dir / "heartbeat-config.json"


def load_heartbeat_config(path: Path | None = None) -> dict[str, Any]:
    file_path = path or get_heartbeat_config_path()
    if not file_path.exists():
        return _default_config_from_env()
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError:
        return _default_config_from_env()
    try:
        payload = __import__("json").loads(raw)
    except ValueError:
        return _default_config_from_env()
    return normalize_heartbeat_config(payload)


def save_heartbeat_config(config: dict[str, Any], path: Path | None = None) -> Path:
    normalized = normalize_heartbeat_config(config)
    file_path = path or get_heartbeat_config_path()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(__import__("json").dumps(normalized, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return file_path


def ensure_heartbeat_config() -> dict[str, Any]:
    config_path = get_heartbeat_config_path()
    if config_path.exists():
        return load_heartbeat_config(config_path)
    defaults = _default_config_from_env()
    save_heartbeat_config(defaults, config_path)
    return defaults


class HeartbeatRuntime:
    def __init__(self) -> None:
        self._scheduler: AsyncIOScheduler | None = None
        self._config: dict[str, Any] = {}
        self._started: bool = False
        self._last_run_ts: int | None = None
        self._last_payload: dict[str, Any] | None = None

    async def start(self) -> None:
        self._config = ensure_heartbeat_config()
        self._started = True
        if not self._config.get("enabled", True):
            return
        self._scheduler = AsyncIOScheduler(timezone="UTC")
        self._scheduler.add_job(
            self._heartbeat_job,
            trigger=self._build_trigger(self._config),
            id="system_heartbeat",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.start()

    async def stop(self) -> None:
        scheduler = self._scheduler
        self._scheduler = None
        self._started = False
        if scheduler is not None:
            scheduler.shutdown(wait=False)

    async def reload(self) -> dict[str, Any]:
        await self.stop()
        await self.start()
        return self.status()

    async def update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        current = load_heartbeat_config()
        merged = current.copy()
        merged.update(updates)
        config = normalize_heartbeat_config(merged)
        save_heartbeat_config(config)
        await self.reload()
        return self.status()

    def status(self) -> dict[str, Any]:
        scheduler_running = bool(self._scheduler and self._scheduler.running)
        jobs = [job.id for job in self._scheduler.get_jobs()] if self._scheduler else []
        return {
            "started": self._started,
            "scheduler_running": scheduler_running,
            "config": self._config or load_heartbeat_config(),
            "jobs": jobs,
            "last_run_ts": self._last_run_ts,
            "last_payload": self._last_payload,
        }

    def _build_trigger(self, config: dict[str, Any]) -> IntervalTrigger | CronTrigger:
        mode = config.get("mode", "interval")
        if mode == "cron":
            cron_expr = str(config.get("cron", DEFAULT_CRON))
            return CronTrigger.from_crontab(cron_expr, timezone="UTC")
        return IntervalTrigger(seconds=int(config.get("interval_seconds", 60)), timezone="UTC")

    async def _heartbeat_job(self) -> None:
        started_ms = now_ts()
        payload: dict[str, Any] = {
            "core": None,
            "resume_approved_tools": None,
        }

        if self._config.get("core_health_enabled", True):
            payload["core"] = await self._check_core_health()
        if self._config.get("resume_approved_tools_enabled", True):
            payload["resume_approved_tools"] = self._resume_approved_tool_calls()

        self._last_run_ts = now_ts()
        self._last_payload = payload

        if self._config.get("emit_events", True):
            status = "ok"
            if isinstance(payload.get("core"), dict) and payload["core"].get("status") == "error":
                status = "degraded"
            append_event(
                HeartbeatEvent(
                    conversation_id=HEARTBEAT_CONVERSATION_ID,
                    ts=self._last_run_ts,
                    actor_id="system",
                    component="scheduler",
                    status=status,
                    latency_ms=self._last_run_ts - started_ms,
                    details=payload,
                )
            )

    async def _check_core_health(self) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            response = await get_core_client().health_check()
            latency_ms = int((time.perf_counter() - start) * 1000)
            return {
                "status": "ok",
                "latency_ms": latency_ms,
                "response": response,
            }
        except Exception as exc:  # pragma: no cover - network defensive path
            latency_ms = int((time.perf_counter() - start) * 1000)
            return {
                "status": "error",
                "latency_ms": latency_ms,
                "error": str(exc),
            }

    def _resume_approved_tool_calls(self) -> dict[str, Any]:
        approvals = list_approvals(status="approved")
        approved_ids = {item.approval_id for item in approvals}
        if not approved_ids:
            return {"checked": 0, "resumed": 0, "errors": []}

        with get_session() as session:
            pending_calls = list(
                session.exec(
                    select(ToolCall).where(
                        ToolCall.status == "pending_approval",
                        ToolCall.approval_id.is_not(None),
                    )
                )
            )

        resumed = 0
        errors: list[dict[str, str]] = []
        checked = 0
        for call in pending_calls:
            if call.approval_id not in approved_ids:
                continue
            checked += 1
            try:
                execute_existing_tool_call(call.tool_call_id, allow_external_write=True)
                resumed += 1
            except Exception as exc:  # pragma: no cover - defensive path
                errors.append({"tool_call_id": call.tool_call_id, "error": str(exc)})

        return {
            "checked": checked,
            "resumed": resumed,
            "errors": errors,
        }


heartbeat_runtime = HeartbeatRuntime()

