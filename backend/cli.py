from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any

import httpx

from backend.approvals.policy import (
    load_approval_policy,
    save_approval_policy,
)
from backend.evolution.cli import best_evolution, run_evolution
from backend.gateway.im_ingress import (
    load_ingress_security_config,
    normalize_ingress_security_config,
)
from backend.permissions.exec_approvals import (
    DEFAULT_MAIN_AGENT,
    VALID_ASK,
    VALID_ASK_FALLBACK,
    VALID_SECURITY,
    evaluate_tool_permission,
    get_agent_policy_with_source,
    get_exec_approvals_path,
    load_exec_approvals,
    normalize_config,
    save_exec_approvals,
)
from backend.sandbox.runtime import (
    load_execution_config,
    save_execution_config,
)
from backend.tools.registry import list_tools, scan_tools


DEFAULT_EDGE_BASE_URL = os.getenv("EDGE_BASE_URL", "http://127.0.0.1:8000")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCTOR_STATUS_ORDER = ("ok", "warn", "fail", "skip")
DOCTOR_HEARTBEAT_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "mode": "interval",
    "interval_seconds": 60,
    "cron": "*/1 * * * *",
    "core_health_enabled": True,
    "resume_approved_tools_enabled": True,
    "emit_events": True,
    "memory_decay_enabled": False,
    "memory_decay_half_life_days": 90,
    "memory_decay_min_confidence": 0.2,
    "skill_distill_enabled": False,
    "skill_distill_window_hours": 24,
    "skill_distill_min_occurrences": 4,
    "skill_distill_max_patterns": 8,
    "skill_distill_max_distill": 3,
}
OPENCLAW_PRESETS: dict[str, dict[str, str]] = {
    "strict": {"security": "allowlist", "ask": "always", "ask_fallback": "deny"},
    "balanced": {"security": "allowlist", "ask": "on-miss", "ask_fallback": "deny"},
    "full": {"security": "full", "ask": "off", "ask_fallback": "full"},
}


def _headers(args: argparse.Namespace) -> dict[str, str]:
    return {
        "X-Actor-Id": args.actor_id,
        "X-Actor-Role": args.actor_role,
    }


def _client(args: argparse.Namespace) -> httpx.Client:
    return httpx.Client(base_url=args.edge_base_url.rstrip("/"), timeout=30.0)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _request_json(
    args: argparse.Namespace,
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        with _client(args) as client:
            response = client.request(
                method,
                path,
                headers=_headers(args),
                json=json_body,
                params=params,
            )
    except httpx.RequestError as exc:
        raise SystemExit(f"request failed: {exc}") from exc
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        raise SystemExit(f"request failed {exc.response.status_code}: {detail}") from exc
    return response.json()


def _parse_json_args(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--args is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("--args must be a JSON object")
    return value


def _split_csv_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _confirm_or_exit(message: str) -> None:
    try:
        answer = input(f"{message}\nType YES to continue: ").strip()
    except EOFError as exc:  # pragma: no cover - interactive guard
        raise SystemExit("confirmation required: input stream is not available") from exc
    if answer != "YES":
        raise SystemExit("operation canceled")


def _validate_choice(value: str | None, valid: set[str], name: str) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered not in valid:
        raise SystemExit(f"invalid {name}: {value}. valid: {sorted(valid)}")
    return lowered


def _maybe_prompt_approval(args: argparse.Namespace, *, approval_id: str) -> None:
    if not sys.stdin.isatty():
        return
    prompt = (
        f"approval pending: {approval_id}\n"
        "Choose action: [a]pprove / [r]eject / [s]kip (default s): "
    )
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return
    if answer in {"", "s", "skip"}:
        return
    if answer in {"a", "approve"}:
        payload = _request_json(
            args,
            "POST",
            f"/v1/approvals/{approval_id}:approve",
            json_body={"actor_id": args.actor_id},
        )
        _print_json({"approval_action": "approved", "approval": payload})
        return
    if answer in {"r", "reject"}:
        payload = _request_json(
            args,
            "POST",
            f"/v1/approvals/{approval_id}:reject",
            json_body={"actor_id": args.actor_id},
        )
        _print_json({"approval_action": "rejected", "approval": payload})
        return
    print("unknown action, skipped")


def _resolve_edge_db_path(project_root: Path) -> Path:
    data_dir = Path(os.getenv("EDGE_DATA_DIR", str(project_root / "data"))).expanduser().resolve()
    default_db_path = data_dir / "edge.db"
    return Path(os.getenv("EDGE_DB_PATH", str(default_db_path))).expanduser().resolve()


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _running_services(project_root: Path) -> list[dict[str, Any]]:
    pid_dir = project_root / ".run"
    services: list[dict[str, Any]] = []
    for name in ("core", "edge", "telegram", "frontend"):
        pid_file = pid_dir / f"{name}.pid"
        if not pid_file.exists():
            continue
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except ValueError:
            continue
        services.append(
            {
                "name": name,
                "pid": pid,
                "alive": _pid_is_alive(pid),
            }
        )
    return services


def _create_migration_archive(project_root: Path, edge_db_path: Path, output_dir: Path) -> dict[str, Any]:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    bundle_name = f"marv_migration_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = (output_dir / f"{bundle_name}.tar.gz").resolve()

    with tempfile.TemporaryDirectory(prefix="marv_migration_") as temp_dir:
        bundle_root = Path(temp_dir) / bundle_name
        (bundle_root / "data").mkdir(parents=True, exist_ok=True)
        copied_db_path = bundle_root / "data" / "edge.db"
        shutil.copy2(edge_db_path, copied_db_path)
        included_files = ["data/edge.db"]

        env_example = project_root / ".env.example"
        if env_example.exists():
            shutil.copy2(env_example, bundle_root / ".env.example")
            included_files.append(".env.example")

        deploy_doc = project_root / "docs" / "DEPLOY_MACBOOK_PRO_M1.md"
        if deploy_doc.exists():
            (bundle_root / "docs").mkdir(parents=True, exist_ok=True)
            shutil.copy2(deploy_doc, bundle_root / "docs" / "DEPLOY_MACBOOK_PRO_M1.md")
            included_files.append("docs/DEPLOY_MACBOOK_PRO_M1.md")

        manifest = {
            "bundle_name": bundle_name,
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "source_host": socket.gethostname(),
            "source_project_root": str(project_root.resolve()),
            "source_edge_db_path": str(edge_db_path),
            "files": included_files,
        }
        manifest_path = bundle_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

        with tarfile.open(archive_path, mode="w:gz") as tar:
            tar.add(bundle_root, arcname=bundle_name)

    return {
        "bundle_name": bundle_name,
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
    }


def cmd_health(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", "/health"))
    return 0


def cmd_chat_send(args: argparse.Namespace) -> int:
    payload = {
        "message": args.message,
        "conversation_id": args.conversation_id,
        "channel": args.channel,
        "channel_id": args.channel_id,
        "user_id": args.user_id,
        "thread_id": args.thread_id,
    }
    response = _request_json(args, "POST", "/v1/agent/messages", json_body=payload)
    _print_json(response)

    if args.follow:
        task_id = response.get("task_id")
        if not task_id:
            return 0
        cmd_task_events(
            argparse.Namespace(
                edge_base_url=args.edge_base_url,
                actor_id=args.actor_id,
                actor_role=args.actor_role,
                task_id=task_id,
            )
        )
    return 0


def cmd_task_get(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", f"/v1/agent/tasks/{args.task_id}"))
    return 0


def cmd_task_events(args: argparse.Namespace) -> int:
    with _client(args) as client:
        with client.stream(
            "GET",
            f"/v1/agent/tasks/{args.task_id}/events",
            headers=_headers(args),
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise SystemExit(f"request failed {exc.response.status_code}: {exc.response.text}") from exc
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    print(line)
                    if line == "event: done":
                        break
                    continue
                if line.startswith("data: "):
                    try:
                        payload = json.loads(line[6:])
                    except json.JSONDecodeError:
                        payload = {"raw": line[6:]}
                    _print_json(payload)
    return 0


def cmd_timeline(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", f"/v1/audit/conversations/{args.conversation_id}/timeline"))
    return 0


def cmd_audit_render(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "POST", "/v1/audit/render", json_body={"task_id": args.task_id}))
    return 0


def cmd_tools_list(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", "/v1/tools"))
    return 0


def cmd_tools_exec(args: argparse.Namespace) -> int:
    payload = {
        "tool": args.tool,
        "args": _parse_json_args(args.args),
        "task_id": args.task_id,
        "tool_call_id": args.tool_call_id,
        "session_id": args.session_id,
        "execution_mode": args.execution_mode,
    }
    response = _request_json(args, "POST", "/v1/tools:execute", json_body=payload)
    _print_json(response)
    if (
        args.prompt_approval
        and response.get("status") == "pending_approval"
        and isinstance(response.get("approval_id"), str)
        and response.get("approval_id")
    ):
        _maybe_prompt_approval(args, approval_id=str(response["approval_id"]))
    return 0


def cmd_approvals_list(args: argparse.Namespace) -> int:
    params = {"status": args.status} if args.status else None
    _print_json(_request_json(args, "GET", "/v1/approvals", params=params))
    return 0


def cmd_approvals_approve(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {"actor_id": args.actor_id}
    if args.grant_scope:
        payload["grant_scope"] = args.grant_scope
    if args.grant_ttl_seconds is not None:
        payload["grant_ttl_seconds"] = args.grant_ttl_seconds
    _print_json(
        _request_json(
            args,
            "POST",
            f"/v1/approvals/{args.approval_id}:approve",
            json_body=payload,
        )
    )
    return 0


def cmd_approvals_reject(args: argparse.Namespace) -> int:
    _print_json(
        _request_json(
            args,
            "POST",
            f"/v1/approvals/{args.approval_id}:reject",
            json_body={"actor_id": args.actor_id},
        )
    )
    return 0


def cmd_approvals_grants_list(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {"limit": args.limit}
    if args.status:
        params["status"] = args.status
    if args.actor:
        params["actor_id"] = args.actor
    if args.session_id:
        params["session_id"] = args.session_id
    _print_json(_request_json(args, "GET", "/v1/system/approvals/grants", params=params))
    return 0


def cmd_approvals_grants_revoke(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "POST", f"/v1/system/approvals/grants/{args.grant_id}:revoke"))
    return 0


def cmd_approvals_policy_show(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", "/v1/system/approvals/policy"))
    return 0


def cmd_approvals_policy_set(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {}
    if args.mode:
        payload["mode"] = args.mode
    if args.risky_risks:
        payload["risky_risks"] = [item.strip() for item in args.risky_risks.split(",") if item.strip()]
    if not payload:
        raise SystemExit("no approval policy fields provided")
    _print_json(_request_json(args, "POST", "/v1/system/approvals/policy", json_body=payload))
    return 0


def cmd_config_propose(args: argparse.Namespace) -> int:
    payload = {
        "natural_language": args.text,
        "scope_type": args.scope_type,
        "scope_id": args.scope_id,
        "actor_id": args.actor_id,
    }
    _print_json(_request_json(args, "POST", "/v1/config/patches:propose", json_body=payload))
    return 0


def cmd_config_commit(args: argparse.Namespace) -> int:
    _print_json(
        _request_json(
            args,
            "POST",
            "/v1/config/patches:commit",
            json_body={"proposal_id": args.proposal_id, "actor_id": args.actor_id},
        )
    )
    return 0


def cmd_config_rollback(args: argparse.Namespace) -> int:
    _print_json(
        _request_json(
            args,
            "POST",
            "/v1/config/revisions:rollback",
            json_body={"revision": args.revision, "actor_id": args.actor_id},
        )
    )
    return 0


def cmd_config_revisions(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    if args.scope_type:
        params["scope_type"] = args.scope_type
    if args.scope_id:
        params["scope_id"] = args.scope_id
    _print_json(_request_json(args, "GET", "/v1/config/revisions", params=params or None))
    return 0


def cmd_config_effective(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {
        "channel": args.channel,
    }
    if args.conversation_id:
        params["conversation_id"] = args.conversation_id
    if args.channel_id:
        params["channel_id"] = args.channel_id
    if args.user_id:
        params["user_id"] = args.user_id
    _print_json(_request_json(args, "GET", "/v1/config/effective", params=params))
    return 0


def cmd_evolve_run(args: argparse.Namespace) -> int:
    payload = run_evolution(config_path=args.config, out_path=args.out, dry_run=args.dry_run)
    _print_json(payload)
    return 0


def cmd_evolve_best(args: argparse.Namespace) -> int:
    payload = best_evolution(run_id=args.run, top_k=args.top_k)
    _print_json(payload)
    return 0


def cmd_memory_write(args: argparse.Namespace) -> int:
    payload = {
        "scope_type": args.scope_type,
        "scope_id": args.scope_id,
        "kind": args.kind,
        "content": args.content,
        "confidence": args.confidence,
        "requires_confirmation": args.requires_confirmation,
    }
    _print_json(_request_json(args, "POST", "/v1/memory/write", json_body=payload))
    return 0


def cmd_memory_query(args: argparse.Namespace) -> int:
    payload = {
        "scope_type": args.scope_type,
        "scope_id": args.scope_id,
        "query": args.query,
        "top_k": args.top_k,
    }
    _print_json(_request_json(args, "POST", "/v1/memory/query", json_body=payload))
    return 0


def cmd_memory_candidates(args: argparse.Namespace) -> int:
    params = {"status": args.status} if args.status else None
    _print_json(_request_json(args, "GET", "/v1/memory/candidates", params=params))
    return 0


def cmd_memory_approve(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "POST", f"/v1/memory/candidates/{args.candidate_id}:approve"))
    return 0


def cmd_memory_reject(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "POST", f"/v1/memory/candidates/{args.candidate_id}:reject"))
    return 0


def cmd_memory_items(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {"limit": args.limit}
    if args.scope_type:
        params["scope_type"] = args.scope_type
    if args.scope_id:
        params["scope_id"] = args.scope_id
    if args.kind:
        params["kind"] = args.kind
    _print_json(_request_json(args, "GET", "/v1/memory/items", params=params))
    return 0


def cmd_memory_update(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {}
    if args.content is not None:
        payload["content"] = args.content
    if args.kind is not None:
        payload["kind"] = args.kind
    if args.confidence is not None:
        payload["confidence"] = args.confidence
    if not payload:
        raise SystemExit("provide at least one of --content/--kind/--confidence")
    _print_json(_request_json(args, "POST", f"/v1/memory/items/{args.item_id}:update", json_body=payload))
    return 0


def cmd_memory_delete(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "POST", f"/v1/memory/items/{args.item_id}:delete"))
    return 0


def cmd_memory_forget(args: argparse.Namespace) -> int:
    payload = {
        "scope_type": args.scope_type,
        "scope_id": args.scope_id,
        "query": args.query,
        "threshold": args.threshold,
        "max_delete": args.max_delete,
    }
    _print_json(_request_json(args, "POST", "/v1/memory/forget", json_body=payload))
    return 0


def cmd_memory_decay(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "half_life_days": args.half_life_days,
        "min_confidence": args.min_confidence,
    }
    if args.scope_type:
        payload["scope_type"] = args.scope_type
    if args.scope_id:
        payload["scope_id"] = args.scope_id
    _print_json(_request_json(args, "POST", "/v1/memory/decay", json_body=payload))
    return 0


def cmd_memory_metrics(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", "/v1/memory/metrics", params={"window_hours": args.window_hours}))
    return 0


def cmd_session_list(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", "/v1/agent/sessions", params={"limit": args.limit}))
    return 0


def cmd_session_get(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", f"/v1/agent/sessions/{args.conversation_id}"))
    return 0


def cmd_session_archive(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "POST", f"/v1/agent/sessions/{args.conversation_id}:archive"))
    return 0


def cmd_session_spawn(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {"name": args.name}
    if args.channel is not None:
        payload["channel"] = args.channel
    if args.channel_id is not None:
        payload["channel_id"] = args.channel_id
    if args.user_id is not None:
        payload["user_id"] = args.user_id
    if args.thread_id is not None:
        payload["thread_id"] = args.thread_id
    _print_json(_request_json(args, "POST", f"/v1/agent/sessions/{args.conversation_id}:spawn", json_body=payload))
    return 0


def cmd_session_send(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {"message": args.message, "actor_id": args.actor_id}
    _print_json(
        _request_json(
            args,
            "POST",
            f"/v1/agent/sessions/{args.conversation_id}:send",
            json_body=payload,
            params={"wait": args.wait, "wait_timeout_seconds": args.wait_timeout_seconds},
        )
    )
    return 0


def cmd_session_history(args: argparse.Namespace) -> int:
    _print_json(
        _request_json(
            args,
            "GET",
            f"/v1/agent/sessions/{args.conversation_id}/history",
            params={"limit": args.limit},
        )
    )
    return 0


def cmd_schedule_list(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {"limit": args.limit}
    if args.status:
        params["status"] = args.status
    _print_json(_request_json(args, "GET", "/v1/scheduled/tasks", params=params))
    return 0


def cmd_schedule_create(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "name": args.name,
        "prompt": args.prompt,
        "cron": args.cron,
        "timezone": args.timezone,
        "channel": args.channel,
        "status": "active" if args.enabled else "paused",
    }
    if args.conversation_id:
        payload["conversation_id"] = args.conversation_id
    if args.channel_id:
        payload["channel_id"] = args.channel_id
    if args.user_id:
        payload["user_id"] = args.user_id
    if args.thread_id:
        payload["thread_id"] = args.thread_id
    _print_json(_request_json(args, "POST", "/v1/scheduled/tasks", json_body=payload))
    return 0


def cmd_schedule_pause(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "POST", f"/v1/scheduled/tasks/{args.schedule_id}:pause"))
    return 0


def cmd_schedule_resume(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "POST", f"/v1/scheduled/tasks/{args.schedule_id}:resume"))
    return 0


def cmd_schedule_run(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "POST", f"/v1/scheduled/tasks/{args.schedule_id}:run"))
    return 0


def cmd_schedule_delete(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "POST", f"/v1/scheduled/tasks/{args.schedule_id}:delete"))
    return 0


def cmd_telegram_pair_code_create(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {"ttl_seconds": args.ttl_seconds}
    if args.chat_id:
        payload["chat_id"] = args.chat_id
    if args.user_id:
        payload["user_id"] = args.user_id
    _print_json(_request_json(args, "POST", "/v1/system/telegram/pairings/codes", json_body=payload))
    return 0


def cmd_telegram_pairings_list(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    if args.chat_id:
        params["chat_id"] = args.chat_id
    if args.user_id:
        params["user_id"] = args.user_id
    _print_json(_request_json(args, "GET", "/v1/system/telegram/pairings", params=params or None))
    return 0


def cmd_telegram_pairings_codes(args: argparse.Namespace) -> int:
    params = {"status": args.status} if args.status else None
    _print_json(_request_json(args, "GET", "/v1/system/telegram/pairings/codes", params=params))
    return 0


def cmd_telegram_pairings_revoke(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "POST", f"/v1/system/telegram/pairings/{args.pairing_id}:revoke"))
    return 0


def cmd_im_channels(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", "/v1/gateway/im/channels"))
    return 0


def cmd_im_ingest(args: argparse.Namespace) -> int:
    payload: dict[str, Any]
    if args.payload_json:
        payload = _parse_json_args(args.payload_json)
    else:
        if not args.message or not args.channel_id or not args.user_id:
            raise SystemExit("when --payload-json is not provided, --message --channel-id --user-id are required")
        payload = {
            "text": args.message,
            "channel_id": args.channel_id,
            "user_id": args.user_id,
        }
        if args.thread_id:
            payload["thread_id"] = args.thread_id
        if args.conversation_id:
            payload["conversation_id"] = args.conversation_id
        if args.actor_id_override:
            payload["actor_id"] = args.actor_id_override

    params: dict[str, Any] = {"wait": args.wait, "wait_timeout_seconds": args.wait_timeout_seconds}
    _print_json(
        _request_json(
            args,
            "POST",
            f"/v1/gateway/im/{args.channel}/inbound",
            json_body=payload,
            params=params,
        )
    )
    return 0


def cmd_im_security_show(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", "/v1/gateway/im/security"))
    return 0


def cmd_im_security_set(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {}
    if args.channel is not None:
        payload["channel"] = args.channel
    if args.dm_policy is not None:
        payload["dm_policy"] = args.dm_policy
    if args.allow_from is not None:
        payload["allow_from"] = _split_csv_values(args.allow_from)
    if args.add_allow_from is not None:
        payload["add_allow_from"] = _split_csv_values(args.add_allow_from)
    if args.remove_allow_from is not None:
        payload["remove_allow_from"] = _split_csv_values(args.remove_allow_from)
    if args.clear_allow_from:
        payload["clear_allow_from"] = True
    if not payload:
        raise SystemExit("no security fields provided")
    _print_json(_request_json(args, "POST", "/v1/gateway/im/security", json_body=payload))
    return 0


def cmd_skills_list(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", "/v1/skills"))
    return 0


def cmd_skills_import(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {}
    if args.source_path:
        payload["source_path"] = args.source_path
    if args.source_name:
        payload["source_name"] = args.source_name
    if args.git_url:
        payload["git_url"] = args.git_url
    if args.git_subdir:
        payload["git_subdir"] = args.git_subdir
    if not payload:
        raise SystemExit("provide --source-path or --git-url")
    _print_json(_request_json(args, "POST", "/v1/skills/import", json_body=payload))
    return 0


def cmd_skills_sync_upstream(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "POST", "/v1/skills/sync-upstream", json_body={}))
    return 0


def cmd_packages_list(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", "/v1/packages"))
    return 0


def cmd_packages_reload(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "POST", "/v1/packages:reload", json_body={}))
    return 0


def cmd_system_core_providers(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", "/v1/system/core/providers"))
    return 0


def cmd_system_core_capabilities(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", "/v1/system/core/capabilities"))
    return 0


def cmd_system_core_models(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", "/v1/system/core/models"))
    return 0


def cmd_system_core_auth(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", "/v1/system/core/auth"))
    return 0


def cmd_system_ipc_reload(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", "/v1/system/ipc-tools"))
    return 0


def cmd_execution_show(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", "/v1/system/execution-mode"))
    return 0


def cmd_execution_set(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {}
    if args.mode:
        payload["mode"] = args.mode
    if args.docker_image:
        payload["docker_image"] = args.docker_image
    if args.network_enabled is not None:
        payload["network_enabled"] = args.network_enabled
    if not payload:
        raise SystemExit("no execution fields provided")
    _print_json(_request_json(args, "POST", "/v1/system/execution-mode", json_body=payload))
    return 0


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        env_key = key.strip()
        if not env_key:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]
        values[env_key] = value
    return values


def _encode_env_value(value: str) -> str:
    cleaned = value.strip()
    if cleaned == "":
        return ""
    if any(ch in cleaned for ch in ['"', "'", " ", "#", ";"]):
        return json.dumps(cleaned, ensure_ascii=True)
    return cleaned


def _write_env_file(*, env_path: Path, values: dict[str, str], template_path: Path) -> None:
    lines: list[str] = []
    consumed: set[str] = set()
    if template_path.exists():
        for line in template_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in line:
                lines.append(line)
                continue
            key, _ = line.split("=", 1)
            env_key = key.strip()
            if env_key in values:
                lines.append(f"{env_key}={_encode_env_value(values[env_key])}")
                consumed.add(env_key)
            else:
                lines.append(line)
    else:
        lines.append("# Generated by marv ops quickstart")

    remaining_keys = sorted([key for key in values.keys() if key not in consumed])
    if remaining_keys and lines and lines[-1].strip():
        lines.append("")
    for key in remaining_keys:
        lines.append(f"{key}={_encode_env_value(values[key])}")

    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _ask_text(*, prompt: str, default: str, assume_yes: bool) -> str:
    if assume_yes or not sys.stdin.isatty():
        return default
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def _ask_bool(*, prompt: str, default: bool, assume_yes: bool) -> bool:
    if assume_yes or not sys.stdin.isatty():
        return default
    suffix = "Y/n" if default else "y/N"
    answer = input(f"{prompt} ({suffix}): ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes", "1", "true", "on"}


def _run_quickstart_wizard(
    *,
    env_path: Path,
    template_path: Path,
    assume_yes: bool,
    with_telegram: bool,
) -> dict[str, str]:
    existing = _load_env_file(env_path)
    defaults = {
        "EDGE_BASE_URL": existing.get("EDGE_BASE_URL", "http://127.0.0.1:8000"),
        "CORE_BASE_URL": existing.get("CORE_BASE_URL", "http://127.0.0.1:9000"),
        "TELEGRAM_BOT_TOKEN": existing.get("TELEGRAM_BOT_TOKEN", ""),
        "TELEGRAM_OWNER_IDS": existing.get("TELEGRAM_OWNER_IDS", ""),
        "TELEGRAM_REQUIRE_PAIRING": existing.get("TELEGRAM_REQUIRE_PAIRING", "false"),
        "EDGE_EXECUTION_MODE": existing.get("EDGE_EXECUTION_MODE", "auto"),
        "CORE_PROVIDER_MATRIX_JSON": existing.get("CORE_PROVIDER_MATRIX_JSON", ""),
    }
    if not assume_yes:
        print("Marv first-run setup")
        print("Press Enter to accept defaults.")

    edge_base_url = _ask_text(
        prompt="Edge API base URL",
        default=defaults["EDGE_BASE_URL"],
        assume_yes=assume_yes,
    )
    core_base_url = _ask_text(
        prompt="Core API base URL",
        default=defaults["CORE_BASE_URL"],
        assume_yes=assume_yes,
    )

    enable_telegram_default = with_telegram or bool(defaults["TELEGRAM_BOT_TOKEN"])
    enable_telegram = _ask_bool(
        prompt="Enable Telegram adapter now",
        default=enable_telegram_default,
        assume_yes=assume_yes,
    )
    telegram_token = defaults["TELEGRAM_BOT_TOKEN"] if enable_telegram else ""
    telegram_owner_ids = defaults["TELEGRAM_OWNER_IDS"] if enable_telegram else ""
    telegram_require_pairing = defaults["TELEGRAM_REQUIRE_PAIRING"] if enable_telegram else "false"
    if enable_telegram:
        telegram_token = _ask_text(
            prompt="Telegram bot token",
            default=telegram_token or "replace_with_bot_token",
            assume_yes=assume_yes,
        )
        telegram_owner_ids = _ask_text(
            prompt="Telegram owner user IDs (comma separated, optional)",
            default=telegram_owner_ids,
            assume_yes=assume_yes,
        )
        telegram_require_pairing = "true" if _ask_bool(
            prompt="Require Telegram pairing",
            default=(str(telegram_require_pairing).strip().lower() in {"1", "true", "yes", "on"}),
            assume_yes=assume_yes,
        ) else "false"

    execution_mode = _ask_text(
        prompt="Execution mode (auto/local/sandbox)",
        default=defaults["EDGE_EXECUTION_MODE"],
        assume_yes=assume_yes,
    ).strip().lower()
    if execution_mode not in {"auto", "local", "sandbox"}:
        execution_mode = "auto"

    configure_matrix = _ask_bool(
        prompt="Configure CORE_PROVIDER_MATRIX_JSON now (advanced)",
        default=bool(defaults["CORE_PROVIDER_MATRIX_JSON"]),
        assume_yes=assume_yes,
    )
    provider_matrix_json = defaults["CORE_PROVIDER_MATRIX_JSON"] if configure_matrix else ""
    if configure_matrix:
        provider_matrix_json = _ask_text(
            prompt="CORE_PROVIDER_MATRIX_JSON (one-line JSON array)",
            default=provider_matrix_json,
            assume_yes=assume_yes,
        )
        if provider_matrix_json:
            try:
                decoded = json.loads(provider_matrix_json)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid CORE_PROVIDER_MATRIX_JSON: {exc}") from exc
            if not isinstance(decoded, list):
                raise SystemExit("invalid CORE_PROVIDER_MATRIX_JSON: must be a JSON array")

    merged = existing.copy()
    merged.update(
        {
            "EDGE_BASE_URL": edge_base_url,
            "CORE_BASE_URL": core_base_url,
            "TELEGRAM_BOT_TOKEN": telegram_token,
            "TELEGRAM_OWNER_IDS": telegram_owner_ids,
            "TELEGRAM_REQUIRE_PAIRING": telegram_require_pairing,
            "EDGE_EXECUTION_MODE": execution_mode,
            "CORE_PROVIDER_MATRIX_JSON": provider_matrix_json,
        }
    )
    _write_env_file(env_path=env_path, values=merged, template_path=template_path)
    return merged


def cmd_ops_quickstart(args: argparse.Namespace) -> int:
    env_path = PROJECT_ROOT / ".env"
    env_example_path = PROJECT_ROOT / ".env.example"
    should_run_wizard = args.wizard or not env_path.exists()
    if should_run_wizard:
        _run_quickstart_wizard(
            env_path=env_path,
            template_path=env_example_path,
            assume_yes=args.yes,
            with_telegram=args.with_telegram,
        )

    start_script = PROJECT_ROOT / "scripts" / "start_all.sh"
    if not start_script.exists():
        raise SystemExit(f"start script not found: {start_script}")
    env = os.environ.copy()
    env["MARV_START_FRONTEND"] = "false" if args.no_frontend else "true"
    env["MARV_INSTALL_FRONTEND_DEPS"] = "false" if args.skip_frontend_install else "true"
    try:
        subprocess.run(["bash", str(start_script)], cwd=PROJECT_ROOT, check=True, env=env)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - shell wrapper
        raise SystemExit(f"failed to start stack: {exc}") from exc

    expected_services = ["core", "edge"]
    if not args.no_frontend:
        expected_services.append("frontend")
    deadline = time.monotonic() + 8.0
    alive_names: set[str] = set()
    while time.monotonic() < deadline:
        alive_names = {
            str(item["name"])
            for item in _running_services(PROJECT_ROOT)
            if bool(item.get("alive"))
        }
        if all(name in alive_names for name in expected_services):
            break
        time.sleep(0.2)
    missing = [name for name in expected_services if name not in alive_names]
    if missing:
        raise SystemExit(
            "quickstart started but some services are not alive: "
            + ",".join(missing)
            + ". Check logs under ./logs/"
        )

    frontend_url = "disabled"
    if "frontend" in alive_names:
        frontend_url = "http://127.0.0.1:3000/chat"
    _print_json(
        {
            "status": "ok",
            "action": "quickstart",
            "wizard_ran": should_run_wizard,
            "env_path": str(env_path),
            "edge_url": "http://127.0.0.1:8000",
            "core_url": "http://127.0.0.1:9000",
            "console_url": frontend_url,
            "alive_services": sorted(alive_names),
            "next_steps": [
                "Run `uv run marv health`",
                "Open console URL and send first message",
                "Stop services with `bash scripts/stop_stack.sh`",
            ],
        }
    )
    return 0


def cmd_ops_stop_services(_: argparse.Namespace) -> int:
    script_path = PROJECT_ROOT / "scripts" / "stop_stack.sh"
    if not script_path.exists():
        raise SystemExit(f"stop script not found: {script_path}")

    services = _running_services(PROJECT_ROOT)
    alive = [item for item in services if item["alive"]]
    if alive:
        service_text = ", ".join(f'{item["name"]}({item["pid"]})' for item in alive)
    else:
        service_text = "no running service detected from pid files"

    _confirm_or_exit(
        "This will stop local services via scripts/stop_stack.sh.\n"
        f"Detected: {service_text}"
    )
    try:
        subprocess.run(["bash", str(script_path)], cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - shell wrapper
        raise SystemExit(f"failed to stop services: {exc}") from exc
    _print_json({"status": "ok", "action": "stop_services", "script": str(script_path)})
    return 0


def cmd_ops_package_migration(args: argparse.Namespace) -> int:
    edge_db_path = _resolve_edge_db_path(PROJECT_ROOT)
    if not edge_db_path.exists():
        raise SystemExit(f"edge db not found: {edge_db_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    _confirm_or_exit(
        "This will package migration bundle from local runtime data.\n"
        f"Source DB: {edge_db_path}\n"
        f"Output dir: {output_dir}"
    )

    archive = _create_migration_archive(PROJECT_ROOT, edge_db_path=edge_db_path, output_dir=output_dir)
    _print_json(
        {
            "status": "ok",
            "action": "package_migration",
            **archive,
            "next_steps": [
                "Copy archive to target machine",
                "Extract archive and replace target data/edge.db",
                "Start stack with scripts/start_stack.sh",
            ],
        }
    )
    return 0


def _extract_task_probe_summary(events: list[dict[str, Any]], task_id: str) -> dict[str, Any]:
    input_ts: int | None = None
    completion_ts: int | None = None
    completion_text: str | None = None
    plan: str | None = None
    route: str | None = None

    for event in events:
        if event.get("task_id") != task_id:
            continue
        event_type = event.get("type")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        if event_type == "InputEvent":
            ts_value = event.get("ts")
            if isinstance(ts_value, int):
                input_ts = ts_value
        elif event_type == "PlanEvent":
            plan_value = payload.get("plan")
            if isinstance(plan_value, str) and plan_value.strip():
                plan = plan_value.strip()
        elif event_type == "RouteEvent":
            route_value = payload.get("route")
            if isinstance(route_value, str) and route_value.strip():
                route = route_value.strip()
        elif event_type == "CompletionEvent":
            ts_value = event.get("ts")
            if isinstance(ts_value, int):
                completion_ts = ts_value
            text_value = payload.get("response_text")
            if isinstance(text_value, str) and text_value.strip():
                completion_text = text_value.strip()

    event_latency_ms = None
    if input_ts is not None and completion_ts is not None:
        event_latency_ms = max(0, completion_ts - input_ts)

    return {
        "input_ts": input_ts,
        "completion_ts": completion_ts,
        "event_latency_ms": event_latency_ms,
        "completion_text": completion_text,
        "plan": plan,
        "route": route,
    }


def cmd_ops_probe(args: argparse.Namespace) -> int:
    payload = {
        "message": args.message,
        "conversation_id": args.conversation_id,
        "channel": args.channel,
        "channel_id": args.channel_id,
        "user_id": args.user_id,
        "thread_id": args.thread_id,
    }
    message_response = _request_json(args, "POST", "/v1/agent/messages", json_body=payload)
    task_id = message_response.get("task_id")
    conversation_id = message_response.get("conversation_id")
    if not isinstance(task_id, str) or not task_id:
        raise SystemExit(f"invalid task_id from edge: {message_response}")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise SystemExit(f"invalid conversation_id from edge: {message_response}")

    started_at = time.monotonic()
    deadline = started_at + args.timeout_seconds
    last_task_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        status_payload = _request_json(args, "GET", f"/v1/agent/tasks/{task_id}")
        last_task_payload = status_payload
        status = status_payload.get("status")
        if status in {"completed", "failed"}:
            break
        time.sleep(args.poll_interval_seconds)

    if not last_task_payload:
        raise SystemExit(f"task status unavailable: {task_id}")
    final_status = last_task_payload.get("status")
    wall_time_ms = int((time.monotonic() - started_at) * 1000)
    if final_status not in {"completed", "failed"}:
        raise SystemExit(f"task timeout after {wall_time_ms}ms: task_id={task_id}")

    timeline_payload = _request_json(args, "GET", f"/v1/audit/conversations/{conversation_id}/timeline")
    timeline_events = timeline_payload.get("events", [])
    if not isinstance(timeline_events, list):
        timeline_events = []
    summary = _extract_task_probe_summary(
        [event for event in timeline_events if isinstance(event, dict)],
        task_id=task_id,
    )

    output: dict[str, Any] = {
        "task_id": task_id,
        "conversation_id": conversation_id,
        "status": final_status,
        "wall_time_ms": wall_time_ms,
        "event_latency_ms": summary["event_latency_ms"],
        "current_stage": last_task_payload.get("current_stage"),
        "last_error": last_task_payload.get("last_error"),
        "plan": summary["plan"],
        "route": summary["route"],
        "completion_text": summary["completion_text"],
    }

    if args.include_effective_config:
        effective_params: dict[str, Any] = {"conversation_id": conversation_id, "channel": args.channel}
        if args.channel_id:
            effective_params["channel_id"] = args.channel_id
        if args.user_id:
            effective_params["user_id"] = args.user_id
        effective_payload = _request_json(args, "GET", "/v1/config/effective", params=effective_params)
        output["channel_scope_id"] = effective_payload.get("channel_scope_id")
        output["effective_config"] = effective_payload.get("effective_config")

    _print_json(output)
    return 0 if final_status == "completed" else 2


def _doctor_add_check(
    checks: list[dict[str, Any]],
    *,
    check_id: str,
    status: str,
    summary: str,
    details: list[str] | None = None,
    fix: str | None = None,
) -> None:
    if status not in DOCTOR_STATUS_ORDER:
        raise ValueError(f"invalid doctor status: {status}")
    payload: dict[str, Any] = {
        "id": check_id,
        "status": status,
        "summary": summary,
    }
    if details:
        payload["details"] = [item for item in details if item]
    if fix:
        payload["fix"] = fix
    checks.append(payload)


def _doctor_read_json_file(path: Path) -> tuple[Any | None, str | None]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"unable to read file: {exc}"
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"


def _doctor_resolve_env_value(*, key: str, env_values: dict[str, str], default: str = "") -> str:
    runtime = os.getenv(key)
    if runtime is not None and runtime.strip():
        return runtime.strip()
    from_file = env_values.get(key)
    if from_file is not None and from_file.strip():
        return from_file.strip()
    return default


def _doctor_resolve_data_dir(*, project_root: Path, env_values: dict[str, str]) -> Path:
    raw = _doctor_resolve_env_value(
        key="EDGE_DATA_DIR",
        env_values=env_values,
        default=str(project_root / "data"),
    )
    return Path(raw).expanduser().resolve()


def _doctor_resolve_path(
    *,
    key: str,
    env_values: dict[str, str],
    default_path: Path,
) -> Path:
    raw = _doctor_resolve_env_value(key=key, env_values=env_values, default=str(default_path))
    return Path(raw).expanduser().resolve()


def _doctor_parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if not lowered:
        return default
    return lowered in {"1", "true", "yes", "y", "on"}


def _doctor_parse_int(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return default


def _doctor_parse_float(value: str | None, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return default


def _doctor_heartbeat_defaults_from_env(env_values: dict[str, str]) -> dict[str, Any]:
    defaults = dict(DOCTOR_HEARTBEAT_DEFAULTS)
    defaults["enabled"] = _doctor_parse_bool(
        _doctor_resolve_env_value(key="HEARTBEAT_ENABLED", env_values=env_values, default="true"),
        default=True,
    )
    defaults["mode"] = _doctor_resolve_env_value(
        key="HEARTBEAT_MODE",
        env_values=env_values,
        default=str(DOCTOR_HEARTBEAT_DEFAULTS["mode"]),
    ).strip().lower() or "interval"
    if defaults["mode"] not in {"interval", "cron"}:
        defaults["mode"] = "interval"
    defaults["interval_seconds"] = max(
        5,
        min(
            86400,
            _doctor_parse_int(
                _doctor_resolve_env_value(key="HEARTBEAT_INTERVAL_SECONDS", env_values=env_values, default="60"),
                default=60,
            ),
        ),
    )
    defaults["cron"] = _doctor_resolve_env_value(
        key="HEARTBEAT_CRON",
        env_values=env_values,
        default=str(DOCTOR_HEARTBEAT_DEFAULTS["cron"]),
    ).strip() or str(DOCTOR_HEARTBEAT_DEFAULTS["cron"])
    defaults["core_health_enabled"] = _doctor_parse_bool(
        _doctor_resolve_env_value(key="HEARTBEAT_CORE_HEALTH_ENABLED", env_values=env_values, default="true"),
        default=True,
    )
    defaults["resume_approved_tools_enabled"] = _doctor_parse_bool(
        _doctor_resolve_env_value(
            key="HEARTBEAT_RESUME_APPROVED_TOOLS_ENABLED",
            env_values=env_values,
            default="true",
        ),
        default=True,
    )
    defaults["emit_events"] = _doctor_parse_bool(
        _doctor_resolve_env_value(key="HEARTBEAT_EMIT_EVENTS", env_values=env_values, default="true"),
        default=True,
    )
    defaults["memory_decay_enabled"] = _doctor_parse_bool(
        _doctor_resolve_env_value(key="HEARTBEAT_MEMORY_DECAY_ENABLED", env_values=env_values, default="false"),
        default=False,
    )
    defaults["memory_decay_half_life_days"] = max(
        1,
        min(
            3650,
            _doctor_parse_int(
                _doctor_resolve_env_value(
                    key="HEARTBEAT_MEMORY_DECAY_HALF_LIFE_DAYS",
                    env_values=env_values,
                    default="90",
                ),
                default=90,
            ),
        ),
    )
    defaults["memory_decay_min_confidence"] = max(
        0.0,
        min(
            1.0,
            _doctor_parse_float(
                _doctor_resolve_env_value(
                    key="HEARTBEAT_MEMORY_DECAY_MIN_CONFIDENCE",
                    env_values=env_values,
                    default="0.2",
                ),
                default=0.2,
            ),
        ),
    )
    defaults["skill_distill_enabled"] = _doctor_parse_bool(
        _doctor_resolve_env_value(
            key="HEARTBEAT_SKILL_DISTILL_ENABLED",
            env_values=env_values,
            default="false",
        ),
        default=False,
    )
    defaults["skill_distill_window_hours"] = max(
        1,
        min(
            168,
            _doctor_parse_int(
                _doctor_resolve_env_value(
                    key="HEARTBEAT_SKILL_DISTILL_WINDOW_HOURS",
                    env_values=env_values,
                    default="24",
                ),
                default=24,
            ),
        ),
    )
    defaults["skill_distill_min_occurrences"] = max(
        2,
        min(
            20,
            _doctor_parse_int(
                _doctor_resolve_env_value(
                    key="HEARTBEAT_SKILL_DISTILL_MIN_OCCURRENCES",
                    env_values=env_values,
                    default="4",
                ),
                default=4,
            ),
        ),
    )
    defaults["skill_distill_max_patterns"] = max(
        1,
        min(
            50,
            _doctor_parse_int(
                _doctor_resolve_env_value(
                    key="HEARTBEAT_SKILL_DISTILL_MAX_PATTERNS",
                    env_values=env_values,
                    default="8",
                ),
                default=8,
            ),
        ),
    )
    defaults["skill_distill_max_distill"] = max(
        1,
        min(
            20,
            _doctor_parse_int(
                _doctor_resolve_env_value(
                    key="HEARTBEAT_SKILL_DISTILL_MAX_DISTILL",
                    env_values=env_values,
                    default="3",
                ),
                default=3,
            ),
        ),
    )
    return defaults


def _doctor_write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _doctor_check_health(base_url: str, *, timeout_seconds: float) -> tuple[bool, str]:
    url = base_url.rstrip("/") + "/health"
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(url)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return False, f"{url}: {exc}"
    if isinstance(payload, dict) and str(payload.get("status", "")).strip().lower() == "ok":
        return True, f"{url}: status=ok"
    return True, f"{url}: reachable"


def _doctor_summarize(checks: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {name: 0 for name in DOCTOR_STATUS_ORDER}
    for item in checks:
        status = str(item.get("status", "")).strip().lower()
        if status in counts:
            counts[status] += 1
    if counts["fail"] > 0:
        overall = "fail"
    elif counts["warn"] > 0:
        overall = "warn"
    else:
        overall = "ok"
    return {
        "overall": overall,
        "counts": counts,
    }


def _doctor_exit_code(*, summary: dict[str, Any], strict: bool) -> int:
    counts = summary.get("counts", {})
    fail_count = int(counts.get("fail", 0))
    warn_count = int(counts.get("warn", 0))
    if fail_count > 0:
        return 2
    if strict and warn_count > 0:
        return 2
    return 0


def _build_doctor_report(
    *,
    project_root: Path,
    edge_base_url: str,
    actor_id: str,
    actor_role: str,
    timeout_seconds: float,
    skip_network: bool,
    apply_fixes: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    env_path = project_root / ".env"
    env_values = _load_env_file(env_path)

    if env_path.exists():
        _doctor_add_check(
            checks,
            check_id="setup.env_file",
            status="ok",
            summary=f"loaded .env ({len(env_values)} keys)",
            details=[str(env_path)],
        )
    else:
        _doctor_add_check(
            checks,
            check_id="setup.env_file",
            status="warn",
            summary=".env not found",
            details=[str(env_path)],
            fix="Run `uv run marv ops quickstart --wizard` to generate local runtime settings.",
        )

    uv_path = shutil.which("uv")
    if uv_path:
        _doctor_add_check(
            checks,
            check_id="setup.runtime_binary",
            status="ok",
            summary="`uv` executable found",
            details=[uv_path],
        )
    else:
        _doctor_add_check(
            checks,
            check_id="setup.runtime_binary",
            status="fail",
            summary="`uv` executable not found in PATH",
            fix="Install `uv` and ensure it is available in PATH.",
        )

    data_dir = _doctor_resolve_data_dir(project_root=project_root, env_values=env_values)
    if data_dir.exists() and not data_dir.is_dir():
        _doctor_add_check(
            checks,
            check_id="runtime.data_dir",
            status="fail",
            summary="runtime data path exists but is not a directory",
            details=[str(data_dir)],
            fix="Set `EDGE_DATA_DIR` to a directory path.",
        )
    elif data_dir.exists():
        _doctor_add_check(
            checks,
            check_id="runtime.data_dir",
            status="ok",
            summary="runtime data directory ready",
            details=[str(data_dir)],
        )
    else:
        if apply_fixes:
            data_dir.mkdir(parents=True, exist_ok=True)
            _doctor_add_check(
                checks,
                check_id="runtime.data_dir",
                status="ok",
                summary="runtime data directory created",
                details=[str(data_dir)],
            )
        else:
            _doctor_add_check(
                checks,
                check_id="runtime.data_dir",
                status="warn",
                summary="runtime data directory missing",
                details=[str(data_dir)],
                fix="Run `uv run marv ops doctor --fix` to create missing runtime files.",
            )

    edge_db_path = _doctor_resolve_path(
        key="EDGE_DB_PATH",
        env_values=env_values,
        default_path=data_dir / "edge.db",
    )
    if edge_db_path.exists():
        _doctor_add_check(
            checks,
            check_id="runtime.edge_db",
            status="ok",
            summary="edge database file found",
            details=[f"path={edge_db_path}", f"size_bytes={edge_db_path.stat().st_size}"],
        )
    else:
        _doctor_add_check(
            checks,
            check_id="runtime.edge_db",
            status="warn",
            summary="edge database not found (will be created on first startup)",
            details=[str(edge_db_path)],
            fix="Start stack once with `uv run marv ops quickstart --yes`.",
        )

    exec_approvals_path = _doctor_resolve_path(
        key="EDGE_EXEC_APPROVALS_PATH",
        env_values=env_values,
        default_path=data_dir / "exec-approvals.json",
    )
    approval_policy_path = _doctor_resolve_path(
        key="EDGE_APPROVAL_POLICY_PATH",
        env_values=env_values,
        default_path=data_dir / "approval-policy.json",
    )
    execution_config_path = _doctor_resolve_path(
        key="EDGE_EXECUTION_CONFIG_PATH",
        env_values=env_values,
        default_path=data_dir / "execution-config.json",
    )
    heartbeat_config_path = _doctor_resolve_path(
        key="EDGE_HEARTBEAT_CONFIG_PATH",
        env_values=env_values,
        default_path=data_dir / "heartbeat-config.json",
    )
    ingress_security_path = _doctor_resolve_path(
        key="EDGE_IM_SECURITY_PATH",
        env_values=env_values,
        default_path=data_dir / "im-security.json",
    )
    ipc_tools_path = _doctor_resolve_path(
        key="EDGE_IPC_TOOLS_PATH",
        env_values=env_values,
        default_path=data_dir / "ipc-tools.json",
    )

    if apply_fixes:
        if not exec_approvals_path.exists():
            save_exec_approvals(load_exec_approvals(path=exec_approvals_path), path=exec_approvals_path)
        if not approval_policy_path.exists():
            save_approval_policy(load_approval_policy(path=approval_policy_path), path=approval_policy_path)
        if not execution_config_path.exists():
            save_execution_config(load_execution_config(path=execution_config_path), path=execution_config_path)
        if not heartbeat_config_path.exists():
            _doctor_write_json_file(heartbeat_config_path, _doctor_heartbeat_defaults_from_env(env_values))

    exec_policy: dict[str, Any] | None = None
    if not exec_approvals_path.exists():
        _doctor_add_check(
            checks,
            check_id="config.exec_approvals",
            status="warn",
            summary="execution approvals file missing",
            details=[str(exec_approvals_path)],
            fix="Run `uv run marv ops doctor --fix` to scaffold default policy.",
        )
    else:
        payload, error = _doctor_read_json_file(exec_approvals_path)
        if error:
            _doctor_add_check(
                checks,
                check_id="config.exec_approvals",
                status="fail",
                summary="execution approvals file is invalid",
                details=[str(exec_approvals_path), error],
                fix="Fix JSON syntax or re-run `uv run marv ops doctor --fix` after removing the broken file.",
            )
        elif not isinstance(payload, dict):
            _doctor_add_check(
                checks,
                check_id="config.exec_approvals",
                status="fail",
                summary="execution approvals file must be a JSON object",
                details=[str(exec_approvals_path)],
            )
        else:
            exec_policy = load_exec_approvals(path=exec_approvals_path)
            defaults = exec_policy.get("defaults", {})
            details = [str(exec_approvals_path)]
            if isinstance(defaults, dict):
                details.append(
                    "defaults="
                    + ",".join(
                        [
                            f"security={defaults.get('security')}",
                            f"ask={defaults.get('ask')}",
                            f"ask_fallback={defaults.get('ask_fallback')}",
                        ]
                    )
                )
            _doctor_add_check(
                checks,
                check_id="config.exec_approvals",
                status="ok",
                summary="execution approvals loaded",
                details=details,
            )

    approval_policy: dict[str, Any] | None = None
    if not approval_policy_path.exists():
        _doctor_add_check(
            checks,
            check_id="config.approval_policy",
            status="warn",
            summary="approval policy file missing",
            details=[str(approval_policy_path)],
            fix="Run `uv run marv ops doctor --fix` to scaffold default policy.",
        )
    else:
        payload, error = _doctor_read_json_file(approval_policy_path)
        if error:
            _doctor_add_check(
                checks,
                check_id="config.approval_policy",
                status="fail",
                summary="approval policy file is invalid",
                details=[str(approval_policy_path), error],
            )
        elif not isinstance(payload, dict):
            _doctor_add_check(
                checks,
                check_id="config.approval_policy",
                status="fail",
                summary="approval policy must be a JSON object",
                details=[str(approval_policy_path)],
            )
        else:
            approval_policy = load_approval_policy(path=approval_policy_path)
            _doctor_add_check(
                checks,
                check_id="config.approval_policy",
                status="ok",
                summary="approval policy loaded",
                details=[str(approval_policy_path), f"mode={approval_policy.get('mode', 'policy')}"],
            )

    execution_policy: dict[str, Any] | None = None
    if not execution_config_path.exists():
        _doctor_add_check(
            checks,
            check_id="config.execution",
            status="warn",
            summary="execution config file missing",
            details=[str(execution_config_path)],
            fix="Run `uv run marv ops doctor --fix` to scaffold execution config.",
        )
    else:
        payload, error = _doctor_read_json_file(execution_config_path)
        if error:
            _doctor_add_check(
                checks,
                check_id="config.execution",
                status="fail",
                summary="execution config file is invalid",
                details=[str(execution_config_path), error],
            )
        elif not isinstance(payload, dict):
            _doctor_add_check(
                checks,
                check_id="config.execution",
                status="fail",
                summary="execution config must be a JSON object",
                details=[str(execution_config_path)],
            )
        else:
            execution_policy = load_execution_config(path=execution_config_path)
            _doctor_add_check(
                checks,
                check_id="config.execution",
                status="ok",
                summary="execution config loaded",
                details=[
                    str(execution_config_path),
                    f"mode={execution_policy.get('mode', 'auto')}",
                    f"docker_image={execution_policy.get('docker_image', '')}",
                    f"network_enabled={execution_policy.get('network_enabled', False)}",
                ],
            )

    if not heartbeat_config_path.exists():
        _doctor_add_check(
            checks,
            check_id="config.heartbeat",
            status="warn",
            summary="heartbeat config file missing",
            details=[str(heartbeat_config_path)],
            fix="Run `uv run marv ops doctor --fix` to scaffold heartbeat config.",
        )
    else:
        payload, error = _doctor_read_json_file(heartbeat_config_path)
        if error:
            _doctor_add_check(
                checks,
                check_id="config.heartbeat",
                status="fail",
                summary="heartbeat config file is invalid",
                details=[str(heartbeat_config_path), error],
            )
        elif not isinstance(payload, dict):
            _doctor_add_check(
                checks,
                check_id="config.heartbeat",
                status="fail",
                summary="heartbeat config must be a JSON object",
                details=[str(heartbeat_config_path)],
            )
        else:
            _doctor_add_check(
                checks,
                check_id="config.heartbeat",
                status="ok",
                summary="heartbeat config loaded",
                details=[str(heartbeat_config_path), f"enabled={payload.get('enabled', True)}"],
            )

    if ipc_tools_path.exists():
        payload, error = _doctor_read_json_file(ipc_tools_path)
        if error:
            _doctor_add_check(
                checks,
                check_id="config.ipc_tools",
                status="fail",
                summary="IPC tools file is invalid",
                details=[str(ipc_tools_path), error],
            )
        elif not isinstance(payload, list):
            _doctor_add_check(
                checks,
                check_id="config.ipc_tools",
                status="fail",
                summary="IPC tools payload must be a JSON array",
                details=[str(ipc_tools_path)],
            )
        else:
            _doctor_add_check(
                checks,
                check_id="config.ipc_tools",
                status="ok",
                summary="IPC tools config loaded",
                details=[str(ipc_tools_path), f"tool_count={len(payload)}"],
            )
    else:
        _doctor_add_check(
            checks,
            check_id="config.ipc_tools",
            status="skip",
            summary="IPC tools file not configured",
            details=[str(ipc_tools_path)],
        )

    if exec_policy is None:
        _doctor_add_check(
            checks,
            check_id="security.exec_defaults",
            status="skip",
            summary="execution policy check skipped (config unavailable)",
        )
    else:
        default_policy, source = get_agent_policy_with_source(exec_policy, DEFAULT_MAIN_AGENT)
        security = str(default_policy.get("security", "full")).strip().lower()
        ask = str(default_policy.get("ask", "off")).strip().lower()
        if security == "full" and ask == "off":
            _doctor_add_check(
                checks,
                check_id="security.exec_defaults",
                status="warn",
                summary="default execution policy is permissive (security=full, ask=off)",
                details=[f"policy_source={source}"],
                fix="Use `uv run marv permissions preset --name balanced` to tighten defaults.",
            )
        else:
            _doctor_add_check(
                checks,
                check_id="security.exec_defaults",
                status="ok",
                summary="execution defaults are not fully permissive",
                details=[f"policy_source={source}", f"security={security}", f"ask={ask}"],
            )

    if approval_policy is None:
        _doctor_add_check(
            checks,
            check_id="security.approval_mode",
            status="skip",
            summary="approval mode check skipped (policy unavailable)",
        )
    else:
        mode = str(approval_policy.get("mode", "policy")).strip().lower()
        if mode == "all":
            _doctor_add_check(
                checks,
                check_id="security.approval_mode",
                status="warn",
                summary="approval mode is set to all (high friction)",
                details=[str(approval_policy_path)],
                fix="Use `uv run marv approvals policy-set --mode policy` for balanced defaults.",
            )
        else:
            _doctor_add_check(
                checks,
                check_id="security.approval_mode",
                status="ok",
                summary="approval mode loaded",
                details=[f"mode={mode}"],
            )

    ingress_security_env = _doctor_resolve_env_value(
        key="IM_INGRESS_SECURITY_JSON",
        env_values=env_values,
        default="",
    )
    ingress_security: dict[str, Any] | None = None
    if ingress_security_env:
        try:
            decoded_security = json.loads(ingress_security_env)
        except json.JSONDecodeError as exc:
            _doctor_add_check(
                checks,
                check_id="config.im_security",
                status="fail",
                summary="IM_INGRESS_SECURITY_JSON is invalid",
                details=[str(exc)],
                fix="Set IM_INGRESS_SECURITY_JSON to a valid JSON object.",
            )
        else:
            if isinstance(decoded_security, dict):
                ingress_security = normalize_ingress_security_config(decoded_security)
                channels = ingress_security.get("channels")
                channel_count = len(channels) if isinstance(channels, dict) else 0
                _doctor_add_check(
                    checks,
                    check_id="config.im_security",
                    status="ok",
                    summary="IM ingress security loaded from env",
                    details=[f"configured_channels={channel_count}"],
                )
            else:
                _doctor_add_check(
                    checks,
                    check_id="config.im_security",
                    status="fail",
                    summary="IM_INGRESS_SECURITY_JSON must be a JSON object",
                )
    elif ingress_security_path.exists():
        payload, error = _doctor_read_json_file(ingress_security_path)
        if error:
            _doctor_add_check(
                checks,
                check_id="config.im_security",
                status="fail",
                summary="IM security policy file is invalid",
                details=[str(ingress_security_path), error],
            )
        elif not isinstance(payload, dict):
            _doctor_add_check(
                checks,
                check_id="config.im_security",
                status="fail",
                summary="IM security policy must be a JSON object",
                details=[str(ingress_security_path)],
            )
        else:
            ingress_security = load_ingress_security_config(path=ingress_security_path)
            channels = ingress_security.get("channels")
            channel_count = len(channels) if isinstance(channels, dict) else 0
            _doctor_add_check(
                checks,
                check_id="config.im_security",
                status="ok",
                summary="IM security policy loaded",
                details=[str(ingress_security_path), f"configured_channels={channel_count}"],
            )
    else:
        _doctor_add_check(
            checks,
            check_id="config.im_security",
            status="skip",
            summary="IM security policy file not configured",
            details=[str(ingress_security_path)],
        )

    if ingress_security is not None:
        channels = ingress_security.get("channels")
        channel_policies = channels if isinstance(channels, dict) else {}
        restrictive = []
        missing_allow = []
        for channel_name, raw_policy in channel_policies.items():
            if not isinstance(channel_name, str) or not isinstance(raw_policy, dict):
                continue
            mode = str(raw_policy.get("dm_policy", "open")).strip().lower()
            allow_from = raw_policy.get("allow_from")
            allow_count = len(allow_from) if isinstance(allow_from, list) else 0
            if mode != "open":
                restrictive.append(f"{channel_name}:{mode}")
            if mode in {"allowlist", "pairing"} and allow_count == 0 and not (channel_name == "telegram" and mode == "pairing"):
                missing_allow.append(channel_name)
        if missing_allow:
            _doctor_add_check(
                checks,
                check_id="security.ingress_policy",
                status="warn",
                summary="some IM channels use restrictive policy but have empty allow_from",
                details=[f"channels={','.join(sorted(missing_allow))}"],
                fix="Use `uv run marv im security-set --channel <name> --add-allow-from <id>`.",
            )
        else:
            _doctor_add_check(
                checks,
                check_id="security.ingress_policy",
                status="ok",
                summary="IM ingress sender policy loaded",
                details=[f"restricted={','.join(sorted(restrictive)) or 'none'}"],
            )
    else:
        _doctor_add_check(
            checks,
            check_id="security.ingress_policy",
            status="skip",
            summary="IM ingress sender policy check skipped",
        )

    ingress_token = _doctor_resolve_env_value(key="IM_INGRESS_TOKEN", env_values=env_values, default="")
    ingress_tokens_json = _doctor_resolve_env_value(key="IM_INGRESS_TOKENS_JSON", env_values=env_values, default="")
    if ingress_tokens_json:
        try:
            parsed_tokens = json.loads(ingress_tokens_json)
        except json.JSONDecodeError as exc:
            _doctor_add_check(
                checks,
                check_id="security.ingress_auth",
                status="fail",
                summary="IM_INGRESS_TOKENS_JSON is invalid",
                details=[str(exc)],
                fix="Set IM_INGRESS_TOKENS_JSON to a valid JSON object.",
            )
        else:
            if isinstance(parsed_tokens, dict) and parsed_tokens:
                _doctor_add_check(
                    checks,
                    check_id="security.ingress_auth",
                    status="ok",
                    summary="IM ingress token map configured",
                    details=[f"channel_rules={len(parsed_tokens)}"],
                )
            else:
                _doctor_add_check(
                    checks,
                    check_id="security.ingress_auth",
                    status="warn",
                    summary="IM_INGRESS_TOKENS_JSON is empty",
                    fix="Set channel tokens or IM_INGRESS_TOKEN before exposing ingress endpoints.",
                )
    elif ingress_token:
        _doctor_add_check(
            checks,
            check_id="security.ingress_auth",
            status="ok",
            summary="global IM ingress token configured",
        )
    else:
        _doctor_add_check(
            checks,
            check_id="security.ingress_auth",
            status="warn",
            summary="IM ingress authentication token is not configured",
            fix="Set IM_INGRESS_TOKEN or IM_INGRESS_TOKENS_JSON before opening ingress to external webhooks.",
        )

    telegram_token = _doctor_resolve_env_value(key="TELEGRAM_BOT_TOKEN", env_values=env_values, default="")
    if telegram_token and telegram_token != "replace_with_bot_token":
        telegram_pairing_enabled = _doctor_parse_bool(
            _doctor_resolve_env_value(
                key="TELEGRAM_REQUIRE_PAIRING",
                env_values=env_values,
                default="false",
            ),
            default=False,
        )
        if telegram_pairing_enabled:
            _doctor_add_check(
                checks,
                check_id="security.telegram_pairing",
                status="ok",
                summary="Telegram pairing protection is enabled",
            )
        else:
            _doctor_add_check(
                checks,
                check_id="security.telegram_pairing",
                status="warn",
                summary="Telegram bot is configured but pairing protection is disabled",
                fix="Set TELEGRAM_REQUIRE_PAIRING=true to enforce first-contact pairing.",
            )
    else:
        _doctor_add_check(
            checks,
            check_id="security.telegram_pairing",
            status="skip",
            summary="Telegram bot token not configured",
        )

    services = _running_services(project_root)
    alive_services = [item for item in services if bool(item.get("alive"))]
    stale_services = [item for item in services if not bool(item.get("alive"))]
    if stale_services:
        _doctor_add_check(
            checks,
            check_id="runtime.services",
            status="warn",
            summary="stale pid files detected",
            details=[f"{item['name']}({item['pid']})" for item in stale_services],
            fix="Run `uv run marv ops stop-services` to clean stale runtime pid files.",
        )
    elif alive_services:
        _doctor_add_check(
            checks,
            check_id="runtime.services",
            status="ok",
            summary="managed runtime services are running",
            details=[f"{item['name']}({item['pid']})" for item in alive_services],
        )
    else:
        _doctor_add_check(
            checks,
            check_id="runtime.services",
            status="warn",
            summary="no managed runtime services are running",
            fix="Start services with `uv run marv ops quickstart --yes`.",
        )

    if execution_policy is None:
        _doctor_add_check(
            checks,
            check_id="runtime.sandbox",
            status="skip",
            summary="sandbox runtime check skipped (execution config unavailable)",
        )
    else:
        mode = str(execution_policy.get("mode", "auto")).strip().lower()
        docker_available = shutil.which("docker") is not None
        if mode == "sandbox" and not docker_available:
            _doctor_add_check(
                checks,
                check_id="runtime.sandbox",
                status="fail",
                summary="execution mode is sandbox but Docker is unavailable",
                fix="Install Docker or switch to local mode via `uv run marv execution set --mode local`.",
            )
        elif mode == "auto" and not docker_available:
            _doctor_add_check(
                checks,
                check_id="runtime.sandbox",
                status="warn",
                summary="execution mode is auto and Docker is unavailable",
                details=["auto mode will fallback to local execution only"],
            )
        elif mode in {"sandbox", "auto"} and docker_available:
            _doctor_add_check(
                checks,
                check_id="runtime.sandbox",
                status="ok",
                summary="Docker runtime available for sandbox execution",
                details=[f"mode={mode}"],
            )
        else:
            _doctor_add_check(
                checks,
                check_id="runtime.sandbox",
                status="skip",
                summary="sandbox runtime not required (mode=local)",
            )

    skills_root = _doctor_resolve_path(
        key="EDGE_SKILLS_ROOT",
        env_values=env_values,
        default_path=project_root / "skill" / "modules",
    )
    if skills_root.exists() and skills_root.is_dir():
        skill_count = sum(1 for _ in skills_root.rglob("SKILL.md"))
        _doctor_add_check(
            checks,
            check_id="workspace.skills_root",
            status="ok",
            summary="skills workspace available",
            details=[str(skills_root), f"skill_count={skill_count}"],
        )
    else:
        if apply_fixes:
            skills_root.mkdir(parents=True, exist_ok=True)
            _doctor_add_check(
                checks,
                check_id="workspace.skills_root",
                status="ok",
                summary="skills workspace created",
                details=[str(skills_root)],
            )
        else:
            _doctor_add_check(
                checks,
                check_id="workspace.skills_root",
                status="warn",
                summary="skills workspace directory missing",
                details=[str(skills_root)],
                fix="Run `uv run marv ops doctor --fix` to create the directory.",
            )

    packages_root = _doctor_resolve_path(
        key="EDGE_PACKAGES_ROOT",
        env_values=env_values,
        default_path=project_root / "packages",
    )
    if packages_root.exists() and packages_root.is_dir():
        package_count = len([item for item in packages_root.iterdir() if item.is_dir()])
        _doctor_add_check(
            checks,
            check_id="workspace.packages_root",
            status="ok",
            summary="packages workspace available",
            details=[str(packages_root), f"package_dirs={package_count}"],
        )
    else:
        _doctor_add_check(
            checks,
            check_id="workspace.packages_root",
            status="skip",
            summary="packages workspace directory missing",
            details=[str(packages_root)],
        )

    edge_health_ok = False
    if skip_network:
        _doctor_add_check(
            checks,
            check_id="network.edge_health",
            status="skip",
            summary="edge health check skipped (--skip-network)",
        )
        _doctor_add_check(
            checks,
            check_id="network.core_health",
            status="skip",
            summary="core health check skipped (--skip-network)",
        )
        _doctor_add_check(
            checks,
            check_id="network.core_auth",
            status="skip",
            summary="core auth check skipped (--skip-network)",
        )
    else:
        edge_ok, edge_message = _doctor_check_health(edge_base_url, timeout_seconds=timeout_seconds)
        edge_health_ok = edge_ok
        if edge_ok:
            _doctor_add_check(
                checks,
                check_id="network.edge_health",
                status="ok",
                summary="edge health endpoint reachable",
                details=[edge_message],
            )
        else:
            _doctor_add_check(
                checks,
                check_id="network.edge_health",
                status="fail",
                summary="edge health endpoint unreachable",
                details=[edge_message],
                fix="Start edge service with `uv run marv ops quickstart --yes`.",
            )

        core_base_url = _doctor_resolve_env_value(
            key="CORE_BASE_URL",
            env_values=env_values,
            default="http://127.0.0.1:9000",
        )
        core_ok, core_message = _doctor_check_health(core_base_url, timeout_seconds=timeout_seconds)
        if core_ok:
            _doctor_add_check(
                checks,
                check_id="network.core_health",
                status="ok",
                summary="core health endpoint reachable",
                details=[core_message],
            )
        else:
            _doctor_add_check(
                checks,
                check_id="network.core_health",
                status="warn",
                summary="core health endpoint unreachable",
                details=[core_message],
            )

        if not edge_health_ok:
            _doctor_add_check(
                checks,
                check_id="network.core_auth",
                status="skip",
                summary="core auth check skipped (edge unavailable)",
            )
        else:
            try:
                with httpx.Client(base_url=edge_base_url.rstrip("/"), timeout=timeout_seconds) as client:
                    response = client.get(
                        "/v1/system/core/auth",
                        headers={
                            "X-Actor-Id": actor_id,
                            "X-Actor-Role": actor_role,
                        },
                    )
                    response.raise_for_status()
                    auth_payload = response.json()
            except Exception as exc:
                _doctor_add_check(
                    checks,
                    check_id="network.core_auth",
                    status="warn",
                    summary="unable to query edge core auth status",
                    details=[str(exc)],
                )
            else:
                provider_count = int(auth_payload.get("count", 0)) if isinstance(auth_payload, dict) else 0
                loaded_count = (
                    int(auth_payload.get("credential_loaded_count", 0)) if isinstance(auth_payload, dict) else 0
                )
                providers = auth_payload.get("providers", []) if isinstance(auth_payload, dict) else []
                unloaded = []
                if isinstance(providers, list):
                    for item in providers:
                        if not isinstance(item, dict):
                            continue
                        if not bool(item.get("credential_loaded")):
                            unloaded.append(str(item.get("name", "unknown")))
                if provider_count <= 0:
                    _doctor_add_check(
                        checks,
                        check_id="network.core_auth",
                        status="warn",
                        summary="no core providers configured",
                        fix="Set CORE_PROVIDER_MATRIX_JSON or provider env variables before production use.",
                    )
                elif loaded_count <= 0:
                    _doctor_add_check(
                        checks,
                        check_id="network.core_auth",
                        status="warn",
                        summary="core providers configured but credentials are not loaded",
                        details=[f"providers={provider_count}"],
                        fix="Set CORE_PROVIDER_<NAME>_API_KEY or CORE_PROVIDER_<NAME>_OAUTH_TOKEN.",
                    )
                elif unloaded:
                    _doctor_add_check(
                        checks,
                        check_id="network.core_auth",
                        status="warn",
                        summary="some core providers are missing credentials",
                        details=[f"loaded={loaded_count}/{provider_count}", f"missing={','.join(unloaded)}"],
                    )
                else:
                    _doctor_add_check(
                        checks,
                        check_id="network.core_auth",
                        status="ok",
                        summary="core provider credentials loaded",
                        details=[f"loaded={loaded_count}/{provider_count}"],
                    )

    summary = _doctor_summarize(checks)
    return {
        "status": summary["overall"],
        "summary": summary["counts"],
        "checks": checks,
        "flags": {
            "strict": False,
            "skip_network": skip_network,
            "fix_applied": apply_fixes,
            "timeout_seconds": timeout_seconds,
        },
        "paths": {
            "project_root": str(project_root),
            "env_path": str(env_path),
            "data_dir": str(data_dir),
            "edge_db_path": str(edge_db_path),
            "exec_approvals_path": str(exec_approvals_path),
            "approval_policy_path": str(approval_policy_path),
            "execution_config_path": str(execution_config_path),
            "heartbeat_config_path": str(heartbeat_config_path),
            "ingress_security_path": str(ingress_security_path),
            "ipc_tools_path": str(ipc_tools_path),
        },
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
    }


def _print_doctor_report(report: dict[str, Any]) -> None:
    print("Marv doctor")
    checks = report.get("checks", [])
    if not isinstance(checks, list):
        checks = []
    labels = {
        "ok": "OK",
        "warn": "WARN",
        "fail": "FAIL",
        "skip": "SKIP",
    }
    for item in checks:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "skip")).strip().lower()
        label = labels.get(status, status.upper())
        check_id = str(item.get("id", "unknown"))
        summary = str(item.get("summary", "")).strip()
        print(f"[{label}] {check_id}: {summary}")
        details = item.get("details", [])
        if isinstance(details, list):
            for detail in details:
                text = str(detail).strip()
                if text:
                    print(f"  - {text}")
        fix = item.get("fix")
        if isinstance(fix, str) and fix.strip():
            print(f"  fix: {fix.strip()}")

    summary = report.get("summary", {})
    if isinstance(summary, dict):
        print(
            "Summary: "
            + ", ".join(
                [
                    f"ok={int(summary.get('ok', 0))}",
                    f"warn={int(summary.get('warn', 0))}",
                    f"fail={int(summary.get('fail', 0))}",
                    f"skip={int(summary.get('skip', 0))}",
                ]
            )
        )
    print(f"Overall: {report.get('status', 'unknown')}")


def cmd_ops_doctor(args: argparse.Namespace) -> int:
    report = _build_doctor_report(
        project_root=PROJECT_ROOT,
        edge_base_url=args.edge_base_url,
        actor_id=args.actor_id,
        actor_role=args.actor_role,
        timeout_seconds=max(0.5, float(args.timeout_seconds)),
        skip_network=bool(args.skip_network),
        apply_fixes=bool(args.fix),
    )
    report_flags = report.get("flags", {})
    if isinstance(report_flags, dict):
        report_flags["strict"] = bool(args.strict)

    if args.json_output:
        _print_json(report)
    else:
        _print_doctor_report(report)

    return _doctor_exit_code(
        summary={"counts": report.get("summary", {})},
        strict=bool(args.strict),
    )


def _update_policy(
    policy: dict[str, Any],
    *,
    security: str | None = None,
    ask: str | None = None,
    ask_fallback: str | None = None,
) -> dict[str, Any]:
    if security is not None:
        policy["security"] = _validate_choice(security, VALID_SECURITY, "security")
    if ask is not None:
        policy["ask"] = _validate_choice(ask, VALID_ASK, "ask")
    if ask_fallback is not None:
        policy["ask_fallback"] = _validate_choice(ask_fallback, VALID_ASK_FALLBACK, "ask_fallback")
    if "allowlist" not in policy or not isinstance(policy["allowlist"], list):
        policy["allowlist"] = []
    return policy


def cmd_permissions_show(args: argparse.Namespace) -> int:
    config = load_exec_approvals()
    if args.agent:
        agent = args.agent.strip()
        if not agent:
            raise SystemExit("agent cannot be empty")
        policies = config.get("agents", {})
        if not isinstance(policies, dict) or agent not in policies:
            raise SystemExit(f"agent policy not found: {agent}")
        _print_json({"path": str(get_exec_approvals_path()), "agent": agent, "policy": policies[agent]})
        return 0
    _print_json({"path": str(get_exec_approvals_path()), "config": config})
    return 0


def cmd_permissions_set_default(args: argparse.Namespace) -> int:
    config = load_exec_approvals()
    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}
    config["defaults"] = _update_policy(
        defaults,
        security=args.security,
        ask=args.ask,
        ask_fallback=args.ask_fallback,
    )
    path = save_exec_approvals(config)
    _print_json({"status": "ok", "path": str(path), "defaults": config["defaults"]})
    return 0


def cmd_permissions_set_agent(args: argparse.Namespace) -> int:
    agent = args.agent.strip()
    if not agent:
        raise SystemExit("agent cannot be empty")
    config = load_exec_approvals()
    agents = config.get("agents", {})
    if not isinstance(agents, dict):
        agents = {}
    policy = agents.get(agent, {})
    if not isinstance(policy, dict):
        policy = {}
    agents[agent] = _update_policy(
        policy,
        security=args.security,
        ask=args.ask,
        ask_fallback=args.ask_fallback,
    )
    config["agents"] = agents
    path = save_exec_approvals(config)
    _print_json({"status": "ok", "path": str(path), "agent": agent, "policy": agents[agent]})
    return 0


def cmd_permissions_unset_agent(args: argparse.Namespace) -> int:
    agent = args.agent.strip()
    if not agent:
        raise SystemExit("agent cannot be empty")
    if agent == DEFAULT_MAIN_AGENT:
        raise SystemExit("cannot unset reserved agent policy: main")
    config = load_exec_approvals()
    agents = config.get("agents", {})
    if not isinstance(agents, dict) or agent not in agents:
        raise SystemExit(f"agent policy not found: {agent}")
    del agents[agent]
    config["agents"] = agents
    path = save_exec_approvals(config)
    _print_json({"status": "ok", "path": str(path), "removed_agent": agent})
    return 0


def _resolve_agent_policy_for_allowlist(config: dict[str, Any], agent: str | None) -> tuple[dict[str, Any], str]:
    agent_key = (agent or DEFAULT_MAIN_AGENT).strip()
    if not agent_key:
        raise SystemExit("agent cannot be empty")
    agents = config.get("agents", {})
    if not isinstance(agents, dict):
        agents = {}
    policy = agents.get(agent_key, {})
    if not isinstance(policy, dict):
        policy = {}
    agents[agent_key] = _update_policy(policy)
    config["agents"] = agents
    return agents[agent_key], agent_key


def cmd_permissions_allowlist_add(args: argparse.Namespace) -> int:
    pattern = args.pattern.strip()
    if not pattern:
        raise SystemExit("pattern cannot be empty")
    config = load_exec_approvals()
    policy, agent_key = _resolve_agent_policy_for_allowlist(config, args.agent)
    allowlist = [str(item).strip() for item in policy.get("allowlist", []) if str(item).strip()]
    if pattern not in allowlist:
        allowlist.append(pattern)
    policy["allowlist"] = sorted(set(allowlist))
    config = normalize_config(config)
    path = save_exec_approvals(config)
    _print_json(
        {
            "status": "ok",
            "path": str(path),
            "agent": agent_key,
            "allowlist": policy["allowlist"],
        }
    )
    return 0


def cmd_permissions_allowlist_remove(args: argparse.Namespace) -> int:
    pattern = args.pattern.strip()
    if not pattern:
        raise SystemExit("pattern cannot be empty")
    config = load_exec_approvals()
    policy, agent_key = _resolve_agent_policy_for_allowlist(config, args.agent)
    allowlist = [str(item).strip() for item in policy.get("allowlist", []) if str(item).strip()]
    if pattern not in allowlist:
        raise SystemExit(f"pattern not found in allowlist: {pattern}")
    allowlist = [item for item in allowlist if item != pattern]
    policy["allowlist"] = allowlist
    config = normalize_config(config)
    path = save_exec_approvals(config)
    _print_json(
        {
            "status": "ok",
            "path": str(path),
            "agent": agent_key,
            "allowlist": policy["allowlist"],
        }
    )
    return 0


def cmd_permissions_eval(args: argparse.Namespace) -> int:
    actor = args.agent.strip()
    tool = args.tool.strip()
    if not actor:
        raise SystemExit("agent cannot be empty")
    if not tool:
        raise SystemExit("tool cannot be empty")
    config = load_exec_approvals()
    policy, source = get_agent_policy_with_source(config, actor)
    decision = evaluate_tool_permission(config, actor_id=actor, tool_name=tool)
    _print_json(
        {
            "path": str(get_exec_approvals_path()),
            "agent": actor,
            "tool": tool,
            "policy_source": source,
            "effective_policy": policy,
            "decision": decision,
        }
    )
    return 0


def _set_policy_preset(target: dict[str, Any], preset: str) -> dict[str, Any]:
    if preset not in OPENCLAW_PRESETS:
        raise SystemExit(f"unknown preset: {preset}. valid={sorted(OPENCLAW_PRESETS)}")
    data = OPENCLAW_PRESETS[preset]
    target["security"] = data["security"]
    target["ask"] = data["ask"]
    target["ask_fallback"] = data["ask_fallback"]
    if "allowlist" not in target or not isinstance(target["allowlist"], list):
        target["allowlist"] = []
    return target


def cmd_permissions_preset(args: argparse.Namespace) -> int:
    preset = args.name.strip().lower()
    config = load_exec_approvals()
    if args.agent:
        agent = args.agent.strip()
        if not agent:
            raise SystemExit("agent cannot be empty")
        agents = config.get("agents", {})
        if not isinstance(agents, dict):
            agents = {}
        policy = agents.get(agent, {})
        if not isinstance(policy, dict):
            policy = {}
        agents[agent] = _set_policy_preset(policy, preset)
        config["agents"] = agents
        path = save_exec_approvals(config)
        _print_json({"status": "ok", "path": str(path), "preset": preset, "agent": agent, "policy": agents[agent]})
        return 0

    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}
    config["defaults"] = _set_policy_preset(defaults, preset)
    path = save_exec_approvals(config)
    _print_json({"status": "ok", "path": str(path), "preset": preset, "defaults": config["defaults"]})
    return 0


def cmd_permissions_allowlist_sync_readonly(args: argparse.Namespace) -> int:
    config = load_exec_approvals()
    policy, agent_key = _resolve_agent_policy_for_allowlist(config, args.agent)
    existing = [str(item).strip() for item in policy.get("allowlist", []) if str(item).strip()]
    scan_tools()
    readonly_names = [
        item["name"]
        for item in list_tools()
        if item.get("risk") == "read_only" and isinstance(item.get("name"), str) and item.get("name")
    ]
    merged = sorted(set(existing + readonly_names))
    policy["allowlist"] = merged
    config = normalize_config(config)
    path = save_exec_approvals(config)
    _print_json(
        {
            "status": "ok",
            "path": str(path),
            "agent": agent_key,
            "added": sorted(set(readonly_names) - set(existing)),
            "allowlist": merged,
        }
    )
    return 0


def cmd_heartbeat_show(args: argparse.Namespace) -> int:
    _print_json(_request_json(args, "GET", "/v1/system/heartbeat"))
    return 0


def cmd_heartbeat_set(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {}
    if args.enabled is not None:
        payload["enabled"] = args.enabled
    if args.mode is not None:
        payload["mode"] = args.mode
    if args.interval_seconds is not None:
        payload["interval_seconds"] = args.interval_seconds
    if args.cron is not None:
        payload["cron"] = args.cron
    if args.core_health_enabled is not None:
        payload["core_health_enabled"] = args.core_health_enabled
    if args.resume_approved_tools_enabled is not None:
        payload["resume_approved_tools_enabled"] = args.resume_approved_tools_enabled
    if args.emit_events is not None:
        payload["emit_events"] = args.emit_events
    if args.memory_decay_enabled is not None:
        payload["memory_decay_enabled"] = args.memory_decay_enabled
    if args.memory_decay_half_life_days is not None:
        payload["memory_decay_half_life_days"] = args.memory_decay_half_life_days
    if args.memory_decay_min_confidence is not None:
        payload["memory_decay_min_confidence"] = args.memory_decay_min_confidence
    if args.skill_distill_enabled is not None:
        payload["skill_distill_enabled"] = args.skill_distill_enabled
    if args.skill_distill_window_hours is not None:
        payload["skill_distill_window_hours"] = args.skill_distill_window_hours
    if args.skill_distill_min_occurrences is not None:
        payload["skill_distill_min_occurrences"] = args.skill_distill_min_occurrences
    if args.skill_distill_max_patterns is not None:
        payload["skill_distill_max_patterns"] = args.skill_distill_max_patterns
    if args.skill_distill_max_distill is not None:
        payload["skill_distill_max_distill"] = args.skill_distill_max_distill
    if not payload:
        raise SystemExit("no heartbeat fields provided")
    _print_json(_request_json(args, "POST", "/v1/system/heartbeat/config", json_body=payload))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marv", description="CLI for Blackbox Edge Runtime")
    parser.add_argument("--edge-base-url", default=DEFAULT_EDGE_BASE_URL)
    parser.add_argument("--actor-id", default=os.getenv("MARV_ACTOR_ID", "cli-user"))
    parser.add_argument("--actor-role", default=os.getenv("MARV_ACTOR_ROLE", "owner"), choices=["owner", "member"])
    sub = parser.add_subparsers(dest="command", required=True)

    health = sub.add_parser("health", help="Check edge health")
    health.set_defaults(func=cmd_health)

    chat = sub.add_parser("chat", help="Chat actions")
    chat_sub = chat.add_subparsers(dest="chat_command", required=True)
    chat_send = chat_sub.add_parser("send", help="Send a message")
    chat_send.add_argument("--message", required=True)
    chat_send.add_argument("--conversation-id")
    chat_send.add_argument("--channel", default="web")
    chat_send.add_argument("--channel-id")
    chat_send.add_argument("--user-id")
    chat_send.add_argument("--thread-id")
    chat_send.add_argument("--follow", action="store_true", help="Stream task events after send")
    chat_send.set_defaults(func=cmd_chat_send)

    task = sub.add_parser("task", help="Task actions")
    task_sub = task.add_subparsers(dest="task_command", required=True)
    task_get = task_sub.add_parser("get", help="Get task status")
    task_get.add_argument("task_id")
    task_get.set_defaults(func=cmd_task_get)
    task_events = task_sub.add_parser("events", help="Stream task SSE events")
    task_events.add_argument("task_id")
    task_events.set_defaults(func=cmd_task_events)

    audit = sub.add_parser("audit", help="Audit actions")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    audit_timeline = audit_sub.add_parser("timeline", help="Get conversation timeline")
    audit_timeline.add_argument("conversation_id")
    audit_timeline.set_defaults(func=cmd_timeline)
    audit_render = audit_sub.add_parser("render", help="Render task audit report")
    audit_render.add_argument("task_id")
    audit_render.set_defaults(func=cmd_audit_render)

    tools = sub.add_parser("tools", help="Tool actions")
    tools_sub = tools.add_subparsers(dest="tools_command", required=True)
    tools_list = tools_sub.add_parser("list", help="List tools")
    tools_list.set_defaults(func=cmd_tools_list)
    tools_exec = tools_sub.add_parser("exec", help="Execute tool")
    tools_exec.add_argument("--tool", required=True)
    tools_exec.add_argument("--args", help='JSON object, e.g. \'{"query":"hello"}\'')
    tools_exec.add_argument("--task-id")
    tools_exec.add_argument("--tool-call-id")
    tools_exec.add_argument("--session-id", help="Conversation/session id for isolated workspace enforcement")
    tools_exec.add_argument("--execution-mode", choices=["auto", "local", "sandbox"], help="IPC tool execution mode override")
    tools_exec.add_argument(
        "--prompt-approval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When tool execution returns pending_approval, prompt to approve/reject immediately",
    )
    tools_exec.set_defaults(func=cmd_tools_exec)

    approvals = sub.add_parser("approvals", help="Approvals")
    approvals_sub = approvals.add_subparsers(dest="approvals_command", required=True)
    approvals_list = approvals_sub.add_parser("list", help="List approvals")
    approvals_list.add_argument("--status")
    approvals_list.set_defaults(func=cmd_approvals_list)
    approvals_approve = approvals_sub.add_parser("approve", help="Approve an approval id")
    approvals_approve.add_argument("approval_id")
    approvals_approve.add_argument("--grant-scope", choices=["one_time", "session", "actor"])
    approvals_approve.add_argument("--grant-ttl-seconds", type=int)
    approvals_approve.set_defaults(func=cmd_approvals_approve)
    approvals_reject = approvals_sub.add_parser("reject", help="Reject an approval id")
    approvals_reject.add_argument("approval_id")
    approvals_reject.set_defaults(func=cmd_approvals_reject)
    approvals_grants = approvals_sub.add_parser("grants", help="List approval grants")
    approvals_grants.add_argument("--status", default="active")
    approvals_grants.add_argument("--actor")
    approvals_grants.add_argument("--session-id")
    approvals_grants.add_argument("--limit", type=int, default=100)
    approvals_grants.set_defaults(func=cmd_approvals_grants_list)
    approvals_revoke_grant = approvals_sub.add_parser("revoke-grant", help="Revoke one approval grant")
    approvals_revoke_grant.add_argument("grant_id")
    approvals_revoke_grant.set_defaults(func=cmd_approvals_grants_revoke)
    approvals_policy_show = approvals_sub.add_parser("policy-show", help="Show approval policy mode")
    approvals_policy_show.set_defaults(func=cmd_approvals_policy_show)
    approvals_policy_set = approvals_sub.add_parser("policy-set", help="Update approval policy mode")
    approvals_policy_set.add_argument("--mode", choices=["policy", "all", "risky"])
    approvals_policy_set.add_argument("--risky-risks", help="Comma separated risk names")
    approvals_policy_set.set_defaults(func=cmd_approvals_policy_set)

    config = sub.add_parser("config", help="Patch config actions")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_propose = config_sub.add_parser("propose", help="Propose a config patch")
    config_propose.add_argument("--text", required=True)
    config_propose.add_argument("--scope-type", default="channel")
    config_propose.add_argument("--scope-id", default="web:default")
    config_propose.set_defaults(func=cmd_config_propose)
    config_commit = config_sub.add_parser("commit", help="Commit proposal")
    config_commit.add_argument("proposal_id")
    config_commit.set_defaults(func=cmd_config_commit)
    config_rollback = config_sub.add_parser("rollback", help="Rollback a revision")
    config_rollback.add_argument("revision")
    config_rollback.set_defaults(func=cmd_config_rollback)
    config_revisions = config_sub.add_parser("revisions", help="List revisions")
    config_revisions.add_argument("--scope-type")
    config_revisions.add_argument("--scope-id")
    config_revisions.set_defaults(func=cmd_config_revisions)
    config_effective = config_sub.add_parser("effective", help="Show effective runtime config for one context")
    config_effective.add_argument("--conversation-id")
    config_effective.add_argument("--channel", default="web")
    config_effective.add_argument("--channel-id")
    config_effective.add_argument("--user-id")
    config_effective.set_defaults(func=cmd_config_effective)

    evolve = sub.add_parser("evolve", help="Patch-native evolution runs")
    evolve_sub = evolve.add_subparsers(dest="evolve_command", required=True)
    evolve_run = evolve_sub.add_parser("run", help="Run offline evolution search")
    evolve_run.add_argument("--config", required=True, help="Path to evolution run config JSON")
    evolve_run.add_argument("--out", help="Optional output path for summarized top individuals")
    evolve_run.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    evolve_run.set_defaults(func=cmd_evolve_run)
    evolve_best = evolve_sub.add_parser("best", help="Show top individuals for one evolution run")
    evolve_best.add_argument("--run", required=True, help="Evolution run id")
    evolve_best.add_argument("--top-k", type=int, default=5)
    evolve_best.set_defaults(func=cmd_evolve_best)

    memory = sub.add_parser("memory", help="Memory actions")
    memory_sub = memory.add_subparsers(dest="memory_command", required=True)
    memory_write = memory_sub.add_parser("write", help="Write memory/candidate")
    memory_write.add_argument("--scope-type", default="user")
    memory_write.add_argument("--scope-id", required=True)
    memory_write.add_argument("--kind", default="preference")
    memory_write.add_argument("--content", required=True)
    memory_write.add_argument("--confidence", type=float, default=0.5)
    memory_write.add_argument("--requires-confirmation", action="store_true")
    memory_write.set_defaults(func=cmd_memory_write)
    memory_query = memory_sub.add_parser("query", help="Query memory")
    memory_query.add_argument("--scope-type", default="user")
    memory_query.add_argument("--scope-id", required=True)
    memory_query.add_argument("--query", required=True)
    memory_query.add_argument("--top-k", type=int, default=5)
    memory_query.set_defaults(func=cmd_memory_query)
    memory_candidates = memory_sub.add_parser("candidates", help="List candidates")
    memory_candidates.add_argument("--status", default="pending")
    memory_candidates.set_defaults(func=cmd_memory_candidates)
    memory_approve = memory_sub.add_parser("approve", help="Approve candidate")
    memory_approve.add_argument("candidate_id")
    memory_approve.set_defaults(func=cmd_memory_approve)
    memory_reject = memory_sub.add_parser("reject", help="Reject candidate")
    memory_reject.add_argument("candidate_id")
    memory_reject.set_defaults(func=cmd_memory_reject)
    memory_items = memory_sub.add_parser("items", help="List memory items")
    memory_items.add_argument("--scope-type")
    memory_items.add_argument("--scope-id")
    memory_items.add_argument("--kind")
    memory_items.add_argument("--limit", type=int, default=100)
    memory_items.set_defaults(func=cmd_memory_items)
    memory_update = memory_sub.add_parser("update", help="Update memory item")
    memory_update.add_argument("item_id")
    memory_update.add_argument("--content")
    memory_update.add_argument("--kind")
    memory_update.add_argument("--confidence", type=float)
    memory_update.set_defaults(func=cmd_memory_update)
    memory_delete = memory_sub.add_parser("delete", help="Delete memory item")
    memory_delete.add_argument("item_id")
    memory_delete.set_defaults(func=cmd_memory_delete)
    memory_forget = memory_sub.add_parser("forget", help="Forget memories by semantic query")
    memory_forget.add_argument("--scope-type", default="user")
    memory_forget.add_argument("--scope-id", required=True)
    memory_forget.add_argument("--query", required=True)
    memory_forget.add_argument("--threshold", type=float, default=0.75)
    memory_forget.add_argument("--max-delete", type=int, default=20)
    memory_forget.set_defaults(func=cmd_memory_forget)
    memory_decay = memory_sub.add_parser("decay", help="Apply confidence decay to memory items")
    memory_decay.add_argument("--half-life-days", type=int, default=90)
    memory_decay.add_argument("--min-confidence", type=float, default=0.2)
    memory_decay.add_argument("--scope-type")
    memory_decay.add_argument("--scope-id")
    memory_decay.set_defaults(func=cmd_memory_decay)
    memory_metrics = memory_sub.add_parser("metrics", help="Show memory metrics")
    memory_metrics.add_argument("--window-hours", type=int, default=24)
    memory_metrics.set_defaults(func=cmd_memory_metrics)

    sessions = sub.add_parser("sessions", help="Session workspace actions")
    sessions_sub = sessions.add_subparsers(dest="sessions_command", required=True)
    sessions_list = sessions_sub.add_parser("list", help="List conversation workspaces")
    sessions_list.add_argument("--limit", type=int, default=100)
    sessions_list.set_defaults(func=cmd_session_list)
    sessions_get = sessions_sub.add_parser("get", help="Get one conversation workspace")
    sessions_get.add_argument("conversation_id")
    sessions_get.set_defaults(func=cmd_session_get)
    sessions_archive = sessions_sub.add_parser("archive", help="Archive one conversation workspace")
    sessions_archive.add_argument("conversation_id")
    sessions_archive.set_defaults(func=cmd_session_archive)
    sessions_spawn = sessions_sub.add_parser("spawn", help="Spawn one subagent session from parent conversation")
    sessions_spawn.add_argument("conversation_id", help="Parent conversation id")
    sessions_spawn.add_argument("--name", default="worker")
    sessions_spawn.add_argument("--channel")
    sessions_spawn.add_argument("--channel-id")
    sessions_spawn.add_argument("--user-id")
    sessions_spawn.add_argument("--thread-id")
    sessions_spawn.set_defaults(func=cmd_session_spawn)
    sessions_send = sessions_sub.add_parser("send", help="Send message to one existing session")
    sessions_send.add_argument("conversation_id")
    sessions_send.add_argument("--message", required=True)
    sessions_send.add_argument("--wait", action=argparse.BooleanOptionalAction, default=False)
    sessions_send.add_argument("--wait-timeout-seconds", type=float, default=120.0)
    sessions_send.set_defaults(func=cmd_session_send)
    sessions_history = sessions_sub.add_parser("history", help="Read one session conversation history")
    sessions_history.add_argument("conversation_id")
    sessions_history.add_argument("--limit", type=int, default=100)
    sessions_history.set_defaults(func=cmd_session_history)

    schedule = sub.add_parser("schedule", help="Cron scheduled task management")
    schedule_sub = schedule.add_subparsers(dest="schedule_command", required=True)
    schedule_list = schedule_sub.add_parser("list", help="List scheduled tasks")
    schedule_list.add_argument("--status")
    schedule_list.add_argument("--limit", type=int, default=200)
    schedule_list.set_defaults(func=cmd_schedule_list)
    schedule_create = schedule_sub.add_parser("create", help="Create one scheduled task")
    schedule_create.add_argument("--name", required=True)
    schedule_create.add_argument("--prompt", required=True)
    schedule_create.add_argument("--cron", required=True, help='crontab format, e.g. "*/10 * * * *"')
    schedule_create.add_argument("--timezone", default="UTC")
    schedule_create.add_argument("--conversation-id")
    schedule_create.add_argument("--channel", default="web")
    schedule_create.add_argument("--channel-id")
    schedule_create.add_argument("--user-id")
    schedule_create.add_argument("--thread-id")
    schedule_create.add_argument("--enabled", action=argparse.BooleanOptionalAction, default=True)
    schedule_create.set_defaults(func=cmd_schedule_create)
    schedule_pause = schedule_sub.add_parser("pause", help="Pause one scheduled task")
    schedule_pause.add_argument("schedule_id")
    schedule_pause.set_defaults(func=cmd_schedule_pause)
    schedule_resume = schedule_sub.add_parser("resume", help="Resume one scheduled task")
    schedule_resume.add_argument("schedule_id")
    schedule_resume.set_defaults(func=cmd_schedule_resume)
    schedule_run = schedule_sub.add_parser("run", help="Run one scheduled task immediately")
    schedule_run.add_argument("schedule_id")
    schedule_run.set_defaults(func=cmd_schedule_run)
    schedule_delete = schedule_sub.add_parser("delete", help="Delete one scheduled task")
    schedule_delete.add_argument("schedule_id")
    schedule_delete.set_defaults(func=cmd_schedule_delete)

    skills = sub.add_parser("skills", help="Skill ecosystem management")
    skills_sub = skills.add_subparsers(dest="skills_command", required=True)
    skills_list = skills_sub.add_parser("list", help="List installed skills")
    skills_list.set_defaults(func=cmd_skills_list)
    skills_import = skills_sub.add_parser("import", help="Import skills from local directory or git repo")
    skills_import.add_argument("--source-path")
    skills_import.add_argument("--source-name")
    skills_import.add_argument("--git-url")
    skills_import.add_argument("--git-subdir", default="")
    skills_import.set_defaults(func=cmd_skills_import)
    skills_sync = skills_sub.add_parser("sync-upstream", help="Sync skills from OpenClaw and LobsterAI")
    skills_sync.set_defaults(func=cmd_skills_sync_upstream)

    packages = sub.add_parser("packages", help="Package contract management")
    packages_sub = packages.add_subparsers(dest="packages_command", required=True)
    packages_list = packages_sub.add_parser("list", help="List installed runtime packages")
    packages_list.set_defaults(func=cmd_packages_list)
    packages_reload = packages_sub.add_parser("reload", help="Reload runtime package hooks")
    packages_reload.set_defaults(func=cmd_packages_reload)

    im = sub.add_parser("im", help="Multi-channel IM ingress")
    im_sub = im.add_subparsers(dest="im_command", required=True)
    im_channels = im_sub.add_parser("channels", help="List supported IM channels")
    im_channels.set_defaults(func=cmd_im_channels)
    im_ingest = im_sub.add_parser("ingest", help="Ingest one inbound message into runtime")
    im_ingest.add_argument("--channel", required=True, choices=["telegram", "discord", "slack", "dingtalk", "feishu", "webchat"])
    im_ingest.add_argument("--message")
    im_ingest.add_argument("--channel-id")
    im_ingest.add_argument("--user-id")
    im_ingest.add_argument("--thread-id")
    im_ingest.add_argument("--conversation-id")
    im_ingest.add_argument("--actor-id-override")
    im_ingest.add_argument("--payload-json", help='Raw JSON payload object (overrides --message/--channel-id/--user-id)')
    im_ingest.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    im_ingest.add_argument("--wait-timeout-seconds", type=float, default=120.0)
    im_ingest.set_defaults(func=cmd_im_ingest)
    im_security_show = im_sub.add_parser("security-show", help="Show IM ingress sender security policy")
    im_security_show.set_defaults(func=cmd_im_security_show)
    im_security_set = im_sub.add_parser("security-set", help="Update IM ingress sender security policy")
    im_security_set.add_argument("--channel", help="Channel id or defaults/*")
    im_security_set.add_argument("--dm-policy", choices=["open", "allowlist", "pairing"])
    im_security_set.add_argument("--allow-from", help="Replace allow_from with comma-separated entries")
    im_security_set.add_argument("--add-allow-from", help="Add comma-separated allow_from entries")
    im_security_set.add_argument("--remove-allow-from", help="Remove comma-separated allow_from entries")
    im_security_set.add_argument("--clear-allow-from", action="store_true")
    im_security_set.set_defaults(func=cmd_im_security_set)

    telegram = sub.add_parser("telegram", help="Telegram pairing operations")
    telegram_sub = telegram.add_subparsers(dest="telegram_command", required=True)
    telegram_pair = telegram_sub.add_parser("pair", help="Manage pairing codes")
    telegram_pair_sub = telegram_pair.add_subparsers(dest="telegram_pair_command", required=True)
    telegram_pair_create = telegram_pair_sub.add_parser("create-code", help="Create a pairing code")
    telegram_pair_create.add_argument("--chat-id")
    telegram_pair_create.add_argument("--user-id")
    telegram_pair_create.add_argument("--ttl-seconds", type=int, default=900)
    telegram_pair_create.set_defaults(func=cmd_telegram_pair_code_create)
    telegram_pair_codes = telegram_pair_sub.add_parser("codes", help="List pairing codes")
    telegram_pair_codes.add_argument("--status")
    telegram_pair_codes.set_defaults(func=cmd_telegram_pairings_codes)
    telegram_pair_list = telegram_pair_sub.add_parser("list", help="List active pairings")
    telegram_pair_list.add_argument("--chat-id")
    telegram_pair_list.add_argument("--user-id")
    telegram_pair_list.set_defaults(func=cmd_telegram_pairings_list)
    telegram_pair_revoke = telegram_pair_sub.add_parser("revoke", help="Revoke one pairing")
    telegram_pair_revoke.add_argument("pairing_id")
    telegram_pair_revoke.set_defaults(func=cmd_telegram_pairings_revoke)

    system = sub.add_parser("system", help="System diagnostics")
    system_sub = system.add_subparsers(dest="system_command", required=True)
    system_core = system_sub.add_parser("core-providers", help="Show provider fallback matrix")
    system_core.set_defaults(func=cmd_system_core_providers)
    system_core_caps = system_sub.add_parser("core-capabilities", help="Show provider capability summary")
    system_core_caps.set_defaults(func=cmd_system_core_capabilities)
    system_core_models = system_sub.add_parser("core-models", help="Show model catalog across providers")
    system_core_models.set_defaults(func=cmd_system_core_models)
    system_core_auth = system_sub.add_parser("core-auth", help="Show provider auth credential loading status")
    system_core_auth.set_defaults(func=cmd_system_core_auth)
    system_ipc = system_sub.add_parser("ipc-reload", help="Reload IPC tools from config")
    system_ipc.set_defaults(func=cmd_system_ipc_reload)

    ops = sub.add_parser("ops", help="Local operations")
    ops_sub = ops.add_subparsers(dest="ops_command", required=True)
    ops_pack = ops_sub.add_parser("package-migration", help="Create migration tarball from local edge db")
    ops_pack.add_argument("--output-dir", default=str(PROJECT_ROOT / "dist" / "migrations"))
    ops_pack.set_defaults(func=cmd_ops_package_migration)
    ops_quickstart = ops_sub.add_parser(
        "quickstart",
        help="First-run setup wizard + one-command start for core/edge/frontend",
    )
    ops_quickstart.add_argument("--wizard", action="store_true")
    ops_quickstart.add_argument("--yes", action="store_true")
    ops_quickstart.add_argument("--with-telegram", action="store_true")
    ops_quickstart.add_argument("--no-frontend", action="store_true")
    ops_quickstart.add_argument("--skip-frontend-install", action="store_true")
    ops_quickstart.set_defaults(func=cmd_ops_quickstart)
    ops_stop = ops_sub.add_parser("stop-services", help="Stop local core/edge/telegram/frontend services")
    ops_stop.set_defaults(func=cmd_ops_stop_services)
    ops_probe = ops_sub.add_parser("probe", help="Run one end-to-end runtime probe and print latency summary")
    ops_probe.add_argument("--message", required=True)
    ops_probe.add_argument("--conversation-id")
    ops_probe.add_argument("--channel", default="web")
    ops_probe.add_argument("--channel-id")
    ops_probe.add_argument("--user-id")
    ops_probe.add_argument("--thread-id")
    ops_probe.add_argument("--timeout-seconds", type=float, default=60.0)
    ops_probe.add_argument("--poll-interval-seconds", type=float, default=0.5)
    ops_probe.add_argument("--include-effective-config", action=argparse.BooleanOptionalAction, default=True)
    ops_probe.set_defaults(func=cmd_ops_probe)
    ops_doctor = ops_sub.add_parser(
        "doctor",
        help="Run OpenClaw-style runtime diagnostics (config, security, health)",
    )
    ops_doctor.add_argument(
        "--json-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit machine-readable JSON report",
    )
    ops_doctor.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit with non-zero status when warnings are present",
    )
    ops_doctor.add_argument(
        "--fix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Create missing runtime config files with safe defaults",
    )
    ops_doctor.add_argument(
        "--skip-network",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip edge/core HTTP health and auth checks",
    )
    ops_doctor.add_argument("--timeout-seconds", type=float, default=3.0, help="HTTP timeout for network checks")
    ops_doctor.set_defaults(func=cmd_ops_doctor)

    permissions = sub.add_parser("permissions", help="OpenClaw-like execution permission policies")
    permissions_sub = permissions.add_subparsers(dest="permissions_command", required=True)

    perm_show = permissions_sub.add_parser("show", help="Show permission config or one agent policy")
    perm_show.add_argument("--agent")
    perm_show.set_defaults(func=cmd_permissions_show)

    perm_eval = permissions_sub.add_parser("eval", help="Evaluate effective decision for one agent/tool pair")
    perm_eval.add_argument("--agent", required=True)
    perm_eval.add_argument("--tool", required=True)
    perm_eval.set_defaults(func=cmd_permissions_eval)

    perm_default = permissions_sub.add_parser("set-default", help="Update default execution policy")
    perm_default.add_argument("--security", choices=sorted(VALID_SECURITY))
    perm_default.add_argument("--ask", choices=sorted(VALID_ASK))
    perm_default.add_argument("--ask-fallback", choices=sorted(VALID_ASK_FALLBACK))
    perm_default.set_defaults(func=cmd_permissions_set_default)

    perm_agent = permissions_sub.add_parser("set-agent", help="Create/update actor-specific policy")
    perm_agent.add_argument("--agent", required=True)
    perm_agent.add_argument("--security", choices=sorted(VALID_SECURITY))
    perm_agent.add_argument("--ask", choices=sorted(VALID_ASK))
    perm_agent.add_argument("--ask-fallback", choices=sorted(VALID_ASK_FALLBACK))
    perm_agent.set_defaults(func=cmd_permissions_set_agent)

    perm_unset = permissions_sub.add_parser("unset-agent", help="Remove actor-specific policy")
    perm_unset.add_argument("--agent", required=True)
    perm_unset.set_defaults(func=cmd_permissions_unset_agent)

    perm_allowlist = permissions_sub.add_parser("allowlist", help="Manage allowlist patterns")
    perm_allowlist_sub = perm_allowlist.add_subparsers(dest="permissions_allowlist_command", required=True)
    perm_allowlist_add = perm_allowlist_sub.add_parser("add", help="Add allowlist pattern")
    perm_allowlist_add.add_argument("--pattern", required=True)
    perm_allowlist_add.add_argument("--agent", default=DEFAULT_MAIN_AGENT)
    perm_allowlist_add.set_defaults(func=cmd_permissions_allowlist_add)
    perm_allowlist_remove = perm_allowlist_sub.add_parser("remove", help="Remove allowlist pattern")
    perm_allowlist_remove.add_argument("--pattern", required=True)
    perm_allowlist_remove.add_argument("--agent", default=DEFAULT_MAIN_AGENT)
    perm_allowlist_remove.set_defaults(func=cmd_permissions_allowlist_remove)
    perm_allowlist_sync = perm_allowlist_sub.add_parser(
        "sync-readonly",
        help="Add all read_only tools into allowlist for selected agent",
    )
    perm_allowlist_sync.add_argument("--agent", default=DEFAULT_MAIN_AGENT)
    perm_allowlist_sync.set_defaults(func=cmd_permissions_allowlist_sync_readonly)

    perm_preset = permissions_sub.add_parser("preset", help="Apply OpenClaw-like preset policy")
    perm_preset.add_argument("--name", required=True, choices=sorted(OPENCLAW_PRESETS.keys()))
    perm_preset.add_argument("--agent", help="Apply preset to one agent; omit to apply to defaults")
    perm_preset.set_defaults(func=cmd_permissions_preset)

    heartbeat = sub.add_parser("heartbeat", help="APScheduler heartbeat runtime configuration")
    heartbeat_sub = heartbeat.add_subparsers(dest="heartbeat_command", required=True)
    heartbeat_show = heartbeat_sub.add_parser("show", help="Show heartbeat runtime status and config")
    heartbeat_show.set_defaults(func=cmd_heartbeat_show)
    heartbeat_set = heartbeat_sub.add_parser("set", help="Update heartbeat config and reload scheduler")
    heartbeat_set.add_argument("--enabled", action=argparse.BooleanOptionalAction, default=None)
    heartbeat_set.add_argument("--mode", choices=["interval", "cron"])
    heartbeat_set.add_argument("--interval-seconds", type=int)
    heartbeat_set.add_argument("--cron")
    heartbeat_set.add_argument("--core-health-enabled", action=argparse.BooleanOptionalAction, default=None)
    heartbeat_set.add_argument("--resume-approved-tools-enabled", action=argparse.BooleanOptionalAction, default=None)
    heartbeat_set.add_argument("--emit-events", action=argparse.BooleanOptionalAction, default=None)
    heartbeat_set.add_argument("--memory-decay-enabled", action=argparse.BooleanOptionalAction, default=None)
    heartbeat_set.add_argument("--memory-decay-half-life-days", type=int)
    heartbeat_set.add_argument("--memory-decay-min-confidence", type=float)
    heartbeat_set.add_argument("--skill-distill-enabled", action=argparse.BooleanOptionalAction, default=None)
    heartbeat_set.add_argument("--skill-distill-window-hours", type=int)
    heartbeat_set.add_argument("--skill-distill-min-occurrences", type=int)
    heartbeat_set.add_argument("--skill-distill-max-patterns", type=int)
    heartbeat_set.add_argument("--skill-distill-max-distill", type=int)
    heartbeat_set.set_defaults(func=cmd_heartbeat_set)

    execution = sub.add_parser("execution", help="Execution mode / sandbox runtime configuration")
    execution_sub = execution.add_subparsers(dest="execution_command", required=True)
    execution_show = execution_sub.add_parser("show", help="Show current execution mode config")
    execution_show.set_defaults(func=cmd_execution_show)
    execution_set = execution_sub.add_parser("set", help="Update execution mode config")
    execution_set.add_argument("--mode", choices=["auto", "local", "sandbox"])
    execution_set.add_argument("--docker-image")
    execution_set.add_argument("--network-enabled", action=argparse.BooleanOptionalAction, default=None)
    execution_set.set_defaults(func=cmd_execution_set)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
