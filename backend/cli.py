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
from typing import Any

import httpx

from backend.permissions.exec_approvals import (
    DEFAULT_MAIN_AGENT,
    VALID_ASK,
    VALID_ASK_FALLBACK,
    VALID_SECURITY,
    get_exec_approvals_path,
    load_exec_approvals,
    normalize_config,
    save_exec_approvals,
)


DEFAULT_EDGE_BASE_URL = os.getenv("EDGE_BASE_URL", "http://127.0.0.1:8000")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    with _client(args) as client:
        response = client.request(
            method,
            path,
            headers=_headers(args),
            json=json_body,
            params=params,
        )
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
    for name in ("core", "edge", "telegram"):
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
    }
    _print_json(_request_json(args, "POST", "/v1/tools:execute", json_body=payload))
    return 0


def cmd_approvals_list(args: argparse.Namespace) -> int:
    params = {"status": args.status} if args.status else None
    _print_json(_request_json(args, "GET", "/v1/approvals", params=params))
    return 0


def cmd_approvals_approve(args: argparse.Namespace) -> int:
    _print_json(
        _request_json(
            args,
            "POST",
            f"/v1/approvals/{args.approval_id}:approve",
            json_body={"actor_id": args.actor_id},
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
    tools_exec.set_defaults(func=cmd_tools_exec)

    approvals = sub.add_parser("approvals", help="Approvals")
    approvals_sub = approvals.add_subparsers(dest="approvals_command", required=True)
    approvals_list = approvals_sub.add_parser("list", help="List approvals")
    approvals_list.add_argument("--status")
    approvals_list.set_defaults(func=cmd_approvals_list)
    approvals_approve = approvals_sub.add_parser("approve", help="Approve an approval id")
    approvals_approve.add_argument("approval_id")
    approvals_approve.set_defaults(func=cmd_approvals_approve)
    approvals_reject = approvals_sub.add_parser("reject", help="Reject an approval id")
    approvals_reject.add_argument("approval_id")
    approvals_reject.set_defaults(func=cmd_approvals_reject)

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

    ops = sub.add_parser("ops", help="Local operations")
    ops_sub = ops.add_subparsers(dest="ops_command", required=True)
    ops_pack = ops_sub.add_parser("package-migration", help="Create migration tarball from local edge db")
    ops_pack.add_argument("--output-dir", default=str(PROJECT_ROOT / "dist" / "migrations"))
    ops_pack.set_defaults(func=cmd_ops_package_migration)
    ops_stop = ops_sub.add_parser("stop-services", help="Stop local core/edge/telegram services")
    ops_stop.set_defaults(func=cmd_ops_stop_services)

    permissions = sub.add_parser("permissions", help="OpenClaw-like execution permission policies")
    permissions_sub = permissions.add_subparsers(dest="permissions_command", required=True)

    perm_show = permissions_sub.add_parser("show", help="Show permission config or one agent policy")
    perm_show.add_argument("--agent")
    perm_show.set_defaults(func=cmd_permissions_show)

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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
