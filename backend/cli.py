from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import httpx


DEFAULT_EDGE_BASE_URL = os.getenv("EDGE_BASE_URL", "http://127.0.0.1:8000")


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
