from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.approvals.service import (
    create_approval,
    create_approval_grant,
    find_matching_approval_grant,
    get_tool_call_by_approval_id,
    list_approval_grants,
    list_approvals,
    revoke_approval_grant,
    update_approval_status,
)
from backend.approvals.policy import (
    VALID_APPROVAL_MODES,
    decide_approval_mode,
    get_approval_policy_path,
    load_approval_policy,
    save_approval_policy,
)
from backend.audit.render import render_task_audit
from backend.agent.auth import get_actor_context, require_owner
from backend.agent.config import get_settings
from backend.agent.processor import process_task
from backend.agent.queue import task_queue
from backend.agent.session_runtime import (
    ensure_session_workspace,
    get_session_workspace,
    list_session_workspaces,
    mark_session_archived,
)
from backend.core_client.openai_compat import get_core_client
from backend.gateway.im_ingress import (
    SUPPORTED_IM_CHANNELS,
    IngressAuthError,
    IngressError,
    IngressIgnored,
    parse_ingress_payload,
    parse_slack_url_verification,
    verify_ingress_auth,
)
from backend.gateway.pairing import (
    create_pair_code,
    list_pair_codes,
    list_pairings,
    revoke_pairing,
    serialize_pair_code,
    serialize_pairing,
)
from backend.agent.state import create_task, get_conversation, get_task, now_ts, upsert_conversation
from backend.heartbeat.runtime import get_heartbeat_config_path, heartbeat_runtime
from backend.ledger.events import InputEvent, PatchCommittedEvent, PatchRolledBackEvent
from backend.ledger.store import append_event
from backend.ledger.store import query_events
from backend.memory.store import (
    apply_memory_confidence_decay,
    approve_candidate,
    delete_memory_item,
    forget_memory_by_query,
    get_memory_metrics,
    list_candidates,
    list_memory_items,
    query_memory,
    reject_candidate,
    update_memory_item,
    write_memory,
)
from backend.permissions.exec_approvals import evaluate_tool_permission, load_exec_approvals
from backend.patch.proposals import create_patch_proposal, get_patch_proposal, update_patch_proposal_status
from backend.patch.state import (
    create_revision,
    ensure_seed,
    get_effective_config,
    get_effective_config_for_runtime,
    get_revision,
    list_revisions,
    update_revision_status,
)
from backend.packages.manager import get_packages_root, list_installed_packages, load_runtime_packages
from backend.sandbox.runtime import (
    get_execution_config_path,
    load_execution_config,
    save_execution_config,
)
from backend.scheduler.runtime import scheduled_task_runtime
from backend.scheduler.store import (
    create_scheduled_task,
    delete_scheduled_task,
    get_scheduled_task,
    list_scheduled_tasks,
    set_scheduled_task_status,
)
from backend.skills.manager import (
    get_skills_root,
    import_skills_from_directory,
    import_skills_from_git,
    list_installed_skills,
)
from backend.storage.db import init_db
from backend.tools.registry import get_tool_spec, list_tools, scan_tools, sync_tools_registry
from backend.tools.ipc_bridge import get_ipc_tools_path, load_ipc_tools
from backend.tools.runner import create_tool_call, execute_existing_tool_call, execute_tool_call, get_tool_call


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    ensure_seed()
    scan_tools()
    ipc_loaded = load_ipc_tools()
    packages_runtime = load_runtime_packages()
    sync_tools_registry()
    task_queue.set_processor(process_task)
    await task_queue.start()
    await scheduled_task_runtime.start()
    await heartbeat_runtime.start()
    if ipc_loaded:
        print(f"loaded ipc tools: {','.join(sorted(ipc_loaded))}")
    if int(packages_runtime.get("loaded_count", 0)) > 0:
        print(
            "loaded packages: "
            + ",".join(str(item.get("name", "")) for item in packages_runtime.get("loaded", []) if isinstance(item, dict))
        )
    try:
        yield
    finally:
        await heartbeat_runtime.stop()
        await scheduled_task_runtime.stop()
        await task_queue.stop()


app = FastAPI(title="Blackbox Edge Runtime", version="0.1.0", lifespan=lifespan)


class MessageRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    channel: str = "web"
    channel_id: str | None = None
    user_id: str | None = None
    thread_id: str | None = None
    actor_id: str | None = None


class ToolExecuteRequest(BaseModel):
    tool: str
    args: dict[str, object] = Field(default_factory=dict)
    task_id: str | None = None
    actor_id: str | None = None
    tool_call_id: str | None = None
    session_id: str | None = None
    execution_mode: str | None = None  # auto | local | sandbox (for ipc tools)


class ApprovalDecisionRequest(BaseModel):
    actor_id: str | None = None
    grant_scope: str | None = None  # one_time | session | actor
    grant_ttl_seconds: int | None = None


class PatchProposeRequest(BaseModel):
    natural_language: str = Field(min_length=1)
    scope_type: str = "channel"
    scope_id: str = "web:default"
    actor_id: str | None = None


class PatchCommitRequest(BaseModel):
    proposal_id: str
    actor_id: str | None = None


class ConfigRollbackRequest(BaseModel):
    revision: str
    actor_id: str | None = None


class MemoryWriteRequest(BaseModel):
    scope_type: str = "user"
    scope_id: str
    kind: str = "preference"
    content: str = Field(min_length=1)
    confidence: float = 0.5
    requires_confirmation: bool = True


class MemoryQueryRequest(BaseModel):
    scope_type: str = "user"
    scope_id: str
    query: str = Field(min_length=1)
    top_k: int = 5


class MemoryUpdateRequest(BaseModel):
    content: str | None = None
    kind: str | None = None
    confidence: float | None = None


class MemoryForgetRequest(BaseModel):
    scope_type: str = "user"
    scope_id: str
    query: str = Field(min_length=1)
    threshold: float = 0.75
    max_delete: int = 20


class MemoryDecayRequest(BaseModel):
    half_life_days: int = 90
    min_confidence: float = 0.2
    scope_type: str | None = None
    scope_id: str | None = None


class AuditRenderRequest(BaseModel):
    task_id: str


class HeartbeatConfigUpdateRequest(BaseModel):
    enabled: bool | None = None
    mode: str | None = None
    interval_seconds: int | None = None
    cron: str | None = None
    core_health_enabled: bool | None = None
    resume_approved_tools_enabled: bool | None = None
    emit_events: bool | None = None
    memory_decay_enabled: bool | None = None
    memory_decay_half_life_days: int | None = None
    memory_decay_min_confidence: float | None = None


class TelegramPairCodeCreateRequest(BaseModel):
    chat_id: str | None = None
    user_id: str | None = None
    ttl_seconds: int = 900


class ApprovalPolicyUpdateRequest(BaseModel):
    mode: str | None = None
    risky_risks: list[str] | None = None


class ExecutionModeUpdateRequest(BaseModel):
    mode: str | None = None
    docker_image: str | None = None
    network_enabled: bool | None = None


class ScheduledTaskCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    cron: str = Field(min_length=1)
    timezone: str = "UTC"
    conversation_id: str | None = None
    channel: str = "web"
    channel_id: str | None = None
    user_id: str | None = None
    thread_id: str | None = None
    status: str = "active"
    actor_id: str | None = None


class SessionSpawnRequest(BaseModel):
    name: str = "worker"
    channel: str | None = None
    channel_id: str | None = None
    user_id: str | None = None
    thread_id: str | None = None
    actor_id: str | None = None


class SessionSendRequest(BaseModel):
    message: str = Field(min_length=1)
    actor_id: str | None = None


class SkillImportRequest(BaseModel):
    source_path: str | None = None
    source_name: str | None = None
    git_url: str | None = None
    git_subdir: str = ""


def _extract_completion_text(audit_payload: dict[str, object]) -> str | None:
    timeline = audit_payload.get("timeline")
    if not isinstance(timeline, list):
        return None
    for item in reversed(timeline):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "CompletionEvent":
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        text = str(payload.get("response_text", "")).strip()
        if text:
            return text
    return None


async def _enqueue_agent_message(
    *,
    message: str,
    conversation_id: str | None,
    channel: str,
    channel_id: str | None,
    user_id: str | None,
    thread_id: str | None,
    actor_id: str,
) -> dict[str, object]:
    resolved_conversation_id = conversation_id or f"conv_{uuid4().hex}"
    upsert_conversation(
        conversation_id=resolved_conversation_id,
        channel=channel,
        channel_id=channel_id,
        user_id=user_id,
        thread_id=thread_id,
    )
    session_workspace = ensure_session_workspace(conversation_id=resolved_conversation_id, actor_id=actor_id)
    task = create_task(conversation_id=resolved_conversation_id, status="queued", stage="plan")
    append_event(
        InputEvent(
            conversation_id=resolved_conversation_id,
            task_id=task.id,
            ts=now_ts(),
            actor_id=actor_id,
            message=message,
        )
    )
    await task_queue.enqueue_task(task.id)
    return {
        "conversation_id": resolved_conversation_id,
        "session_workspace": session_workspace.workspace_path,
        "task_id": task.id,
        "status": task.status,
    }


async def _wait_task_terminal(task_id: str, *, timeout_seconds: float) -> dict[str, object]:
    timeout = max(1.0, min(900.0, timeout_seconds))
    deadline = asyncio.get_running_loop().time() + timeout
    last_payload: dict[str, object] = {}
    while asyncio.get_running_loop().time() < deadline:
        task = get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        last_payload = {
            "id": task.id,
            "conversation_id": task.conversation_id,
            "status": task.status,
            "last_error": task.last_error,
            "current_stage": task.current_stage,
            "updated_at": task.updated_at,
        }
        if task.status in {"completed", "failed"}:
            return last_payload
        await asyncio.sleep(0.2)
    raise HTTPException(status_code=504, detail=f"task timeout: {task_id}, last={last_payload}")


def _serialize_scheduled_task(item: object) -> dict[str, object]:
    return {
        "schedule_id": getattr(item, "schedule_id"),
        "name": getattr(item, "name"),
        "prompt": getattr(item, "prompt"),
        "cron": getattr(item, "cron"),
        "timezone": getattr(item, "timezone"),
        "status": getattr(item, "status"),
        "conversation_id": getattr(item, "conversation_id"),
        "channel": getattr(item, "channel"),
        "channel_id": getattr(item, "channel_id"),
        "user_id": getattr(item, "user_id"),
        "thread_id": getattr(item, "thread_id"),
        "created_at": getattr(item, "created_at"),
        "updated_at": getattr(item, "updated_at"),
        "created_by": getattr(item, "created_by"),
        "last_run_at": getattr(item, "last_run_at"),
        "last_task_id": getattr(item, "last_task_id"),
        "last_error": getattr(item, "last_error"),
        "next_run_at": getattr(item, "next_run_at"),
    }


def _safe_subagent_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.").lower()
    return cleaned or "worker"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/system/heartbeat")
async def system_heartbeat_status() -> dict[str, object]:
    status = heartbeat_runtime.status()
    status["config_path"] = str(get_heartbeat_config_path())
    return status


@app.post("/v1/system/heartbeat/config")
async def system_heartbeat_update(body: HeartbeatConfigUpdateRequest, request: Request) -> dict[str, object]:
    require_owner(request)
    updates = body.model_dump(exclude_none=True)
    status = await heartbeat_runtime.update_config(updates)
    status["config_path"] = str(get_heartbeat_config_path())
    return status


@app.get("/v1/system/core/providers")
async def system_core_providers() -> dict[str, object]:
    return get_core_client().provider_status()


@app.get("/v1/system/core/capabilities")
async def system_core_capabilities() -> dict[str, object]:
    return get_core_client().provider_capabilities()


@app.get("/v1/system/core/models")
async def system_core_models() -> dict[str, object]:
    return get_core_client().model_catalog()


@app.get("/v1/system/core/auth")
async def system_core_auth() -> dict[str, object]:
    return get_core_client().provider_auth_status()


@app.get("/v1/system/ipc-tools")
async def system_ipc_tools() -> dict[str, object]:
    loaded = load_ipc_tools()
    sync_tools_registry()
    return {
        "path": str(get_ipc_tools_path()),
        "loaded": loaded,
        "count": len(loaded),
    }


@app.get("/v1/system/approvals/policy")
async def system_approvals_policy() -> dict[str, object]:
    policy = load_approval_policy()
    return {
        "path": str(get_approval_policy_path()),
        "policy": policy,
        "valid_modes": sorted(VALID_APPROVAL_MODES),
    }


@app.post("/v1/system/approvals/policy")
async def system_approvals_policy_update(body: ApprovalPolicyUpdateRequest, request: Request) -> dict[str, object]:
    require_owner(request)
    current = load_approval_policy()
    updates = body.model_dump(exclude_none=True)
    if "mode" in updates:
        updates["mode"] = str(updates["mode"]).strip().lower()
    merged = current.copy()
    merged.update(updates)
    save_approval_policy(merged)
    policy = load_approval_policy()
    return {
        "path": str(get_approval_policy_path()),
        "policy": policy,
    }


@app.get("/v1/system/execution-mode")
async def system_execution_mode() -> dict[str, object]:
    return {
        "path": str(get_execution_config_path()),
        "config": load_execution_config(),
    }


@app.post("/v1/system/execution-mode")
async def system_execution_mode_update(body: ExecutionModeUpdateRequest, request: Request) -> dict[str, object]:
    require_owner(request)
    current = load_execution_config()
    updates = body.model_dump(exclude_none=True)
    if "mode" in updates:
        updates["mode"] = str(updates["mode"]).strip().lower()
    current.update(updates)
    save_execution_config(current)
    return {
        "path": str(get_execution_config_path()),
        "config": load_execution_config(),
    }


@app.get("/v1/system/approvals/grants")
async def system_approval_grants(
    request: Request,
    actor_id: str | None = None,
    session_id: str | None = None,
    status: str | None = "active",
    limit: int = 100,
) -> dict[str, object]:
    require_owner(request)
    items = list_approval_grants(actor_id=actor_id, session_id=session_id, status=status, limit=limit)
    return {
        "count": len(items),
        "grants": [
            {
                "grant_id": item.grant_id,
                "actor_id": item.actor_id,
                "tool_pattern": item.tool_pattern,
                "session_id": item.session_id,
                "status": item.status,
                "created_at": item.created_at,
                "expires_at": item.expires_at,
                "created_by": item.created_by,
                "source_approval_id": item.source_approval_id,
            }
            for item in items
        ],
    }


@app.post("/v1/system/approvals/grants/{grant_id}:revoke")
async def system_approval_grants_revoke(grant_id: str, request: Request) -> dict[str, object]:
    require_owner(request)
    item = revoke_approval_grant(grant_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Grant not found")
    return {
        "grant_id": item.grant_id,
        "status": item.status,
    }


@app.post("/v1/system/telegram/pairings/codes")
async def system_create_telegram_pair_code(
    body: TelegramPairCodeCreateRequest,
    request: Request,
) -> dict[str, object]:
    actor = require_owner(request)
    item = create_pair_code(
        created_by=actor.actor_id,
        ttl_seconds=body.ttl_seconds,
        chat_id=body.chat_id,
        user_id=body.user_id,
    )
    return serialize_pair_code(item)


@app.get("/v1/system/telegram/pairings")
async def system_list_telegram_pairings(
    request: Request,
    chat_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, object]:
    require_owner(request)
    pairings = list_pairings(chat_id=chat_id, user_id=user_id)
    return {
        "count": len(pairings),
        "pairings": [serialize_pairing(item) for item in pairings],
    }


@app.get("/v1/system/telegram/pairings/codes")
async def system_list_telegram_pair_codes(request: Request, status: str | None = None) -> dict[str, object]:
    require_owner(request)
    codes = list_pair_codes(status=status)
    return {
        "count": len(codes),
        "codes": [serialize_pair_code(item) for item in codes],
    }


@app.post("/v1/system/telegram/pairings/{pairing_id}:revoke")
async def system_revoke_telegram_pairing(pairing_id: str, request: Request) -> dict[str, object]:
    require_owner(request)
    item = revoke_pairing(pairing_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Pairing not found")
    return serialize_pairing(item)


@app.get("/v1/audit/conversations/{conversation_id}/timeline")
async def timeline(conversation_id: str) -> dict[str, object]:
    events = query_events(conversation_id=conversation_id)
    return {
        "conversation_id": conversation_id,
        "count": len(events),
        "events": [
            {
                "event_id": event.event_id,
                "task_id": event.task_id,
                "type": event.type,
                "ts": event.ts,
                "actor_id": event.actor_id,
                "payload": json.loads(event.payload_json),
            }
            for event in events
        ],
    }


@app.get("/v1/agent/tasks/{task_id}")
async def task_status(task_id: str) -> dict[str, object]:
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "id": task.id,
        "conversation_id": task.conversation_id,
        "status": task.status,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "last_error": task.last_error,
        "current_stage": task.current_stage,
    }


@app.get("/v1/agent/tasks/{task_id}/events")
async def task_events(task_id: str) -> StreamingResponse:
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_stream() -> object:
        seen_event_ids: set[str] = set()
        while True:
            current_task = get_task(task_id)
            events = query_events(conversation_id=task.conversation_id, task_id=task_id)
            for event in events:
                if event.event_id in seen_event_ids:
                    continue
                seen_event_ids.add(event.event_id)
                payload = {
                    "event_id": event.event_id,
                    "type": event.type,
                    "ts": event.ts,
                    "payload": json.loads(event.payload_json),
                }
                yield f"data: {json.dumps(payload, ensure_ascii=True)}\\n\\n"

            if current_task and current_task.status in {"completed", "failed"}:
                yield "event: done\\ndata: {}\\n\\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/v1/agent/sessions")
async def agent_sessions(limit: int = 100) -> dict[str, object]:
    sessions = list_session_workspaces(limit=limit)
    return {
        "count": len(sessions),
        "sessions": [
            {
                "conversation_id": item.conversation_id,
                "workspace_path": item.workspace_path,
                "status": item.status,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
                "actor_id": item.actor_id,
            }
            for item in sessions
        ],
    }


@app.get("/v1/agent/sessions/{conversation_id}")
async def agent_session_get(conversation_id: str) -> dict[str, object]:
    item = get_session_workspace(conversation_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "conversation_id": item.conversation_id,
        "workspace_path": item.workspace_path,
        "status": item.status,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "actor_id": item.actor_id,
    }


@app.post("/v1/agent/sessions/{conversation_id}:archive")
async def agent_session_archive(conversation_id: str, request: Request) -> dict[str, object]:
    require_owner(request)
    item = mark_session_archived(conversation_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "conversation_id": item.conversation_id,
        "status": item.status,
        "updated_at": item.updated_at,
    }


@app.post("/v1/agent/sessions/{conversation_id}:spawn")
async def agent_session_spawn(
    conversation_id: str,
    body: SessionSpawnRequest,
    request: Request,
) -> dict[str, object]:
    actor = get_actor_context(request)
    parent = get_conversation(conversation_id)
    if parent is None:
        raise HTTPException(status_code=404, detail="Parent conversation not found")
    child_name = _safe_subagent_name(body.name)
    child_conversation_id = f"subagent:{conversation_id}:{child_name}:{uuid4().hex[:8]}"
    upsert_conversation(
        conversation_id=child_conversation_id,
        channel=body.channel or parent.channel,
        channel_id=body.channel_id if body.channel_id is not None else parent.channel_id,
        user_id=body.user_id if body.user_id is not None else parent.user_id,
        thread_id=body.thread_id if body.thread_id is not None else parent.thread_id,
    )
    session = ensure_session_workspace(conversation_id=child_conversation_id, actor_id=body.actor_id or actor.actor_id)
    return {
        "parent_conversation_id": conversation_id,
        "conversation_id": child_conversation_id,
        "name": child_name,
        "workspace_path": session.workspace_path,
        "status": session.status,
    }


@app.post("/v1/agent/sessions/{conversation_id}:send")
async def agent_session_send(
    conversation_id: str,
    body: SessionSendRequest,
    request: Request,
    wait: bool = False,
    wait_timeout_seconds: float = 120.0,
) -> dict[str, object]:
    actor = get_actor_context(request)
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Session conversation not found")
    accepted = await _enqueue_agent_message(
        message=body.message,
        conversation_id=conversation_id,
        channel=conversation.channel,
        channel_id=conversation.channel_id,
        user_id=conversation.user_id,
        thread_id=conversation.thread_id,
        actor_id=body.actor_id or actor.actor_id,
    )
    if not wait:
        return accepted
    terminal = await _wait_task_terminal(str(accepted["task_id"]), timeout_seconds=wait_timeout_seconds)
    audit = render_task_audit(
        task_id=str(accepted["task_id"]),
        conversation_id=str(accepted["conversation_id"]),
    )
    return {
        **accepted,
        "terminal_status": terminal["status"],
        "last_error": terminal.get("last_error"),
        "completion_text": _extract_completion_text(audit),
    }


@app.get("/v1/agent/sessions/{conversation_id}/history")
async def agent_session_history(conversation_id: str, limit: int = 100) -> dict[str, object]:
    events = query_events(conversation_id=conversation_id)
    if limit > 0:
        events = events[-min(limit, 1000) :]
    return {
        "conversation_id": conversation_id,
        "count": len(events),
        "events": [
            {
                "event_id": event.event_id,
                "task_id": event.task_id,
                "type": event.type,
                "ts": event.ts,
                "actor_id": event.actor_id,
                "payload": json.loads(event.payload_json),
            }
            for event in events
        ],
    }


@app.get("/v1/scheduled/tasks")
async def scheduled_tasks_list(status: str | None = None, limit: int = 200) -> dict[str, object]:
    items = list_scheduled_tasks(status=status, limit=limit)
    return {
        "count": len(items),
        "runtime": scheduled_task_runtime.status(),
        "tasks": [_serialize_scheduled_task(item) for item in items],
    }


@app.post("/v1/scheduled/tasks")
async def scheduled_tasks_create(body: ScheduledTaskCreateRequest, request: Request) -> dict[str, object]:
    actor = require_owner(request)
    item = create_scheduled_task(
        name=body.name.strip(),
        prompt=body.prompt.strip(),
        cron=body.cron.strip(),
        timezone=body.timezone.strip() or "UTC",
        created_by=body.actor_id or actor.actor_id,
        conversation_id=body.conversation_id,
        channel=body.channel,
        channel_id=body.channel_id,
        user_id=body.user_id,
        thread_id=body.thread_id,
        status=body.status,
    )
    runtime = await scheduled_task_runtime.reload()
    return {
        "task": _serialize_scheduled_task(item),
        "runtime": runtime,
    }


@app.post("/v1/scheduled/tasks/{schedule_id}:pause")
async def scheduled_tasks_pause(schedule_id: str, request: Request) -> dict[str, object]:
    require_owner(request)
    item = set_scheduled_task_status(schedule_id, status="paused")
    if item is None:
        raise HTTPException(status_code=404, detail="Scheduled task not found")
    runtime = await scheduled_task_runtime.reload()
    return {
        "task": _serialize_scheduled_task(item),
        "runtime": runtime,
    }


@app.post("/v1/scheduled/tasks/{schedule_id}:resume")
async def scheduled_tasks_resume(schedule_id: str, request: Request) -> dict[str, object]:
    require_owner(request)
    item = set_scheduled_task_status(schedule_id, status="active")
    if item is None:
        raise HTTPException(status_code=404, detail="Scheduled task not found")
    runtime = await scheduled_task_runtime.reload()
    return {
        "task": _serialize_scheduled_task(item),
        "runtime": runtime,
    }


@app.post("/v1/scheduled/tasks/{schedule_id}:run")
async def scheduled_tasks_run_once(schedule_id: str, request: Request) -> dict[str, object]:
    require_owner(request)
    if get_scheduled_task(schedule_id) is None:
        raise HTTPException(status_code=404, detail="Scheduled task not found")
    return await scheduled_task_runtime.run_once(schedule_id)


@app.post("/v1/scheduled/tasks/{schedule_id}:delete")
async def scheduled_tasks_delete(schedule_id: str, request: Request) -> dict[str, object]:
    require_owner(request)
    deleted = delete_scheduled_task(schedule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Scheduled task not found")
    runtime = await scheduled_task_runtime.reload()
    return {
        "schedule_id": schedule_id,
        "status": "deleted",
        "runtime": runtime,
    }


@app.get("/v1/packages")
async def packages_list() -> dict[str, object]:
    items = list_installed_packages()
    return {
        "root": str(get_packages_root()),
        "count": len(items),
        "packages": items,
    }


@app.post("/v1/packages:reload")
async def packages_reload(request: Request) -> dict[str, object]:
    require_owner(request)
    report = load_runtime_packages()
    sync_tools_registry()
    return report


@app.get("/v1/skills")
async def skills_list() -> dict[str, object]:
    items = list_installed_skills()
    return {
        "root": str(get_skills_root()),
        "count": len(items),
        "skills": items,
    }


@app.post("/v1/skills/import")
async def skills_import(body: SkillImportRequest, request: Request) -> dict[str, object]:
    require_owner(request)
    if body.source_path:
        report = import_skills_from_directory(source_dir=body.source_path, source_name=body.source_name)
        return {"mode": "directory", "report": report}
    if body.git_url:
        report = import_skills_from_git(
            git_url=body.git_url,
            subdir=body.git_subdir,
            source_name=body.source_name,
        )
        return {"mode": "git", "report": report}
    raise HTTPException(status_code=400, detail="either source_path or git_url is required")


@app.post("/v1/skills/sync-upstream")
async def skills_sync_upstream(request: Request) -> dict[str, object]:
    require_owner(request)
    sources = [
        {
            "git_url": "https://github.com/openclaw/openclaw.git",
            "subdir": "skills",
            "source_name": "openclaw",
        },
        {
            "git_url": "https://github.com/netease-youdao/LobsterAI.git",
            "subdir": "SKILLs",
            "source_name": "lobsterai",
        },
    ]
    reports: list[dict[str, object]] = []
    for source in sources:
        report = import_skills_from_git(
            git_url=str(source["git_url"]),
            subdir=str(source["subdir"]),
            source_name=str(source["source_name"]),
        )
        reports.append({"source": source, "report": report})
    imported_total = sum(int(item["report"]["imported_count"]) for item in reports)
    blocked_total = sum(int(item["report"]["blocked_count"]) for item in reports)
    return {
        "sources": reports,
        "imported_count": imported_total,
        "blocked_count": blocked_total,
        "root": str(get_skills_root()),
    }


@app.get("/v1/gateway/im/channels")
async def gateway_im_channels() -> dict[str, object]:
    return {
        "count": len(SUPPORTED_IM_CHANNELS),
        "channels": sorted(SUPPORTED_IM_CHANNELS),
    }


@app.post("/v1/gateway/im/{channel}/inbound")
async def gateway_im_inbound(
    channel: str,
    request: Request,
    wait: bool = True,
    wait_timeout_seconds: float = 120.0,
) -> dict[str, object]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be a JSON object")

    normalized_channel = channel.strip().lower()
    if normalized_channel == "slack":
        challenge = parse_slack_url_verification(payload)
        if challenge is not None:
            return {"type": "url_verification", "challenge": challenge}

    try:
        verify_ingress_auth(
            channel=normalized_channel,
            headers={key.lower(): value for key, value in request.headers.items()},
        )
        message = parse_ingress_payload(normalized_channel, payload)
    except IngressIgnored as exc:
        return {"status": "ignored", "reason": str(exc), "channel": normalized_channel}
    except IngressAuthError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except IngressError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    accepted = await _enqueue_agent_message(
        message=message.text,
        conversation_id=message.resolve_conversation_id(),
        channel=message.channel,
        channel_id=message.channel_id,
        user_id=message.user_id,
        thread_id=message.thread_id,
        actor_id=message.resolve_actor_id(),
    )
    accepted["channel"] = message.channel
    accepted["actor_id"] = message.resolve_actor_id()

    if not wait:
        return accepted

    terminal = await _wait_task_terminal(accepted["task_id"], timeout_seconds=wait_timeout_seconds)  # type: ignore[arg-type]
    audit = render_task_audit(
        task_id=str(accepted["task_id"]),
        conversation_id=str(accepted["conversation_id"]),
    )
    return {
        **accepted,
        "terminal_status": terminal["status"],
        "last_error": terminal.get("last_error"),
        "completion_text": _extract_completion_text(audit),
    }


@app.post("/v1/agent/messages")
async def agent_messages(body: MessageRequest, request: Request) -> dict[str, object]:
    actor = get_actor_context(request)
    return await _enqueue_agent_message(
        message=body.message,
        conversation_id=body.conversation_id,
        channel=body.channel,
        channel_id=body.channel_id,
        user_id=body.user_id,
        thread_id=body.thread_id,
        actor_id=body.actor_id or actor.actor_id,
    )


@app.get("/v1/tools")
async def tools_list() -> dict[str, object]:
    tools = list_tools()
    return {
        "count": len(tools),
        "tools": tools,
    }


@app.post("/v1/tools:execute")
async def tools_execute(body: ToolExecuteRequest, request: Request) -> dict[str, object]:
    actor = get_actor_context(request)
    if body.tool_call_id:
        existing = get_tool_call(body.tool_call_id)
        if existing is not None:
            return {
                "tool_call_id": existing.tool_call_id,
                "status": existing.status,
                "tool": existing.tool,
                "approval_id": existing.approval_id,
                "result": json.loads(existing.result_json) if existing.result_json else None,
                "error": existing.error,
                "idempotent_hit": True,
            }

    spec = get_tool_spec(body.tool)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Tool not found: {body.tool}")

    session_workspace_path: str | None = None
    resolved_session_id: str | None = None
    if body.session_id:
        session = get_session_workspace(body.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if session.status != "active":
            raise HTTPException(status_code=403, detail="Session is not active")
        session_workspace_path = session.workspace_path
        resolved_session_id = body.session_id
    elif body.task_id:
        task = get_task(body.task_id)
        if task is not None:
            session = ensure_session_workspace(conversation_id=task.conversation_id, actor_id=body.actor_id or actor.actor_id)
            session_workspace_path = session.workspace_path
            resolved_session_id = task.conversation_id

    if spec.risk == "external_write" and session_workspace_path:
        target = str(body.args.get("target", ""))
        if target.startswith("file://"):
            import os
            from pathlib import Path

            raw_path = target[len("file://") :].strip()
            candidate = Path(raw_path).expanduser().resolve()
            workspace = Path(session_workspace_path).expanduser().resolve()
            try:
                candidate.relative_to(workspace)
            except ValueError:
                raise HTTPException(
                    status_code=403,
                    detail=f"external_write target must stay inside session workspace: {workspace}",
                )

    if spec.risk == "external_write":
        if actor.role != "owner":
            raise HTTPException(status_code=403, detail="Member cannot execute external_write tools")

    resolved_tool_args: dict[str, object] = dict(body.args)
    if spec.version.startswith("ipc-"):
        if session_workspace_path:
            resolved_tool_args["__marv_session_workspace"] = session_workspace_path
        if body.execution_mode:
            resolved_tool_args["__marv_execution_mode"] = body.execution_mode

    requester_actor_id = body.actor_id or actor.actor_id
    perm_cfg = load_exec_approvals()
    perm = evaluate_tool_permission(perm_cfg, actor_id=requester_actor_id, tool_name=body.tool)
    if perm["decision"] == "deny":
        raise HTTPException(status_code=403, detail=f"Tool blocked by exec policy: {perm['reason']}")

    grant = find_matching_approval_grant(
        actor_id=requester_actor_id,
        tool_name=body.tool,
        session_id=resolved_session_id,
    )

    approval_policy = load_approval_policy()
    require_approval, policy_reason = decide_approval_mode(
        policy=approval_policy,
        tool_risk=spec.risk,
        policy_decision=str(perm["decision"]),
    )
    if (
        str(approval_policy.get("mode", "policy")).strip().lower() == "policy"
        and str(perm["decision"]) == "ask"
    ):
        policy_reason = str(perm["reason"])
    if grant is not None:
        require_approval = False
        policy_reason = f"approval_grant:{grant.grant_id}"

    if require_approval:
        summary = f"{body.tool} requires approval ({policy_reason})"
        approval = create_approval(
            approval_type="tool_execute",
            summary=summary,
            actor_id=requester_actor_id,
            constraints={
                "tool": body.tool,
                "one_time": True,
                "policy_reason": policy_reason,
                "requester_actor_id": requester_actor_id,
                "session_id": resolved_session_id,
            },
        )
        tool_call = create_tool_call(
            task_id=body.task_id,
            tool_name=body.tool,
            args=resolved_tool_args,
            status="pending_approval",
            approval_id=approval.approval_id,
            tool_call_id=body.tool_call_id,
        )
        return {
            "tool_call_id": tool_call.tool_call_id,
            "status": "pending_approval",
            "approval_id": approval.approval_id,
            "tool": body.tool,
            "policy_reason": policy_reason,
            "session_workspace": session_workspace_path,
            "session_id": resolved_session_id,
        }

    try:
        tool_call = execute_tool_call(
            task_id=body.task_id,
            tool_name=body.tool,
            args=resolved_tool_args,
            tool_call_id=body.tool_call_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    result = json.loads(tool_call.result_json) if tool_call.result_json else None
    return {
        "tool_call_id": tool_call.tool_call_id,
        "status": tool_call.status,
        "tool": tool_call.tool,
        "result": result,
        "error": tool_call.error,
        "session_workspace": session_workspace_path,
        "session_id": resolved_session_id,
        "approval_grant_id": grant.grant_id if grant is not None else None,
    }


@app.get("/v1/approvals")
async def approvals_list(status: str | None = None) -> dict[str, object]:
    approvals = list_approvals(status=status)
    return {
        "count": len(approvals),
        "approvals": [
            {
                "approval_id": item.approval_id,
                "type": item.type,
                "status": item.status,
                "summary": item.summary,
                "constraints": json.loads(item.constraints_json),
                "created_at": item.created_at,
                "updated_at": item.updated_at,
                "actor_id": item.actor_id,
                "decided_by": item.decided_by,
            }
            for item in approvals
        ],
    }


@app.post("/v1/approvals/{approval_id}:approve")
async def approvals_approve(approval_id: str, body: ApprovalDecisionRequest, request: Request) -> dict[str, object]:
    actor = require_owner(request)
    grant_scope = (body.grant_scope or "one_time").strip().lower()
    if grant_scope not in {"one_time", "session", "actor"}:
        raise HTTPException(status_code=400, detail="grant_scope must be one_time|session|actor")

    approval = update_approval_status(
        approval_id=approval_id,
        status="approved",
        decided_by=body.actor_id or actor.actor_id,
    )
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    tool_call = get_tool_call_by_approval_id(approval_id)
    approval_constraints = json.loads(approval.constraints_json)
    grant = None
    if grant_scope in {"session", "actor"}:
        if not isinstance(approval_constraints, dict):
            approval_constraints = {}
        tool_name = str(approval_constraints.get("tool", "")).strip()
        requester_actor = str(approval_constraints.get("requester_actor_id", "")).strip() or (approval.actor_id or "")
        if not tool_name or not requester_actor:
            raise HTTPException(status_code=400, detail="approval constraints missing tool/requester_actor_id")
        session_scope = None
        if grant_scope == "session":
            session_scope = str(approval_constraints.get("session_id", "")).strip() or None
            if not session_scope:
                raise HTTPException(status_code=400, detail="session grant requires session_id in approval constraints")
        grant = create_approval_grant(
            actor_id=requester_actor,
            tool_pattern=tool_name,
            session_id=session_scope,
            ttl_seconds=body.grant_ttl_seconds or 900,
            created_by=body.actor_id or actor.actor_id,
            source_approval_id=approval_id,
        )

    if tool_call is not None:
        tool_call = execute_existing_tool_call(tool_call.tool_call_id, allow_external_write=True)
    return {
        "approval_id": approval.approval_id,
        "status": approval.status,
        "tool_call_id": tool_call.tool_call_id if tool_call else None,
        "tool_call_status": tool_call.status if tool_call else None,
        "grant": (
            {
                "grant_id": grant.grant_id,
                "actor_id": grant.actor_id,
                "tool_pattern": grant.tool_pattern,
                "session_id": grant.session_id,
                "expires_at": grant.expires_at,
            }
            if grant is not None
            else None
        ),
    }


@app.post("/v1/approvals/{approval_id}:reject")
async def approvals_reject(approval_id: str, body: ApprovalDecisionRequest, request: Request) -> dict[str, object]:
    actor = require_owner(request)
    approval = update_approval_status(
        approval_id=approval_id,
        status="rejected",
        decided_by=body.actor_id or actor.actor_id,
    )
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    return {
        "approval_id": approval.approval_id,
        "status": approval.status,
    }


@app.get("/v1/config/revisions")
async def config_revisions(scope_type: str | None = None, scope_id: str | None = None) -> dict[str, object]:
    revisions = list_revisions(scope_type=scope_type, scope_id=scope_id)
    effective = None
    if scope_type and scope_id:
        effective = get_effective_config(scope_type=scope_type, scope_id=scope_id)
    return {
        "count": len(revisions),
        "revisions": [
            {
                "revision": item.revision,
                "scope_type": item.scope_type,
                "scope_id": item.scope_id,
                "created_at": item.created_at,
                "actor_id": item.actor_id,
                "patch": json.loads(item.patch_json),
                "explanation": item.explanation,
                "risk_level": item.risk_level,
                "status": item.status,
            }
            for item in revisions
        ],
        "effective_config": effective,
    }


@app.get("/v1/config/effective")
async def config_effective_runtime(
    conversation_id: str | None = None,
    channel: str = "web",
    channel_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, object]:
    resolved_conversation_id = conversation_id or f"preview:{channel}:{channel_id or 'default'}:{user_id or 'anonymous'}"
    effective = get_effective_config_for_runtime(
        conversation_id=resolved_conversation_id,
        channel=channel,
        channel_id=channel_id,
        user_id=user_id,
    )
    return {
        "conversation_id": resolved_conversation_id,
        "channel": channel,
        "channel_scope_id": f"{channel}:{channel_id or 'default'}",
        "user_id": user_id,
        "effective_config": effective,
    }


@app.post("/v1/config/patches:propose")
async def config_patches_propose(body: PatchProposeRequest, request: Request) -> dict[str, object]:
    actor = get_actor_context(request)
    proposal = create_patch_proposal(
        scope_type=body.scope_type,
        scope_id=body.scope_id,
        natural_language=body.natural_language,
        actor_id=body.actor_id or actor.actor_id,
    )
    return {
        "proposal_id": proposal.proposal_id,
        "scope_type": proposal.scope_type,
        "scope_id": proposal.scope_id,
        "risk_level": proposal.risk_level,
        "needs_approval": proposal.needs_approval,
        "patch": json.loads(proposal.patch_json),
        "explanation": proposal.explanation,
        "status": proposal.status,
    }


@app.post("/v1/config/patches:commit")
async def config_patches_commit(body: PatchCommitRequest, request: Request) -> dict[str, object]:
    actor = get_actor_context(request)
    proposal = get_patch_proposal(body.proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="Proposal not found")
    if proposal.status != "open":
        raise HTTPException(status_code=400, detail="Proposal is not open")
    if proposal.risk_level in {"L2", "L3"} and actor.role != "owner":
        raise HTTPException(status_code=403, detail="Owner role required for L2/L3 commit")

    patch = json.loads(proposal.patch_json)
    revision = create_revision(
        scope_type=proposal.scope_type,
        scope_id=proposal.scope_id,
        actor_id=body.actor_id or actor.actor_id,
        patch=patch,
        explanation=proposal.explanation,
        risk_level=proposal.risk_level,
        status="committed",
    )
    update_patch_proposal_status(proposal.proposal_id, status="committed")

    append_event(
        PatchCommittedEvent(
            conversation_id=f"config:{proposal.scope_type}:{proposal.scope_id}",
            ts=now_ts(),
            actor_id=body.actor_id or actor.actor_id,
            proposal_id=proposal.proposal_id,
            revision=revision.revision,
            risk_level=proposal.risk_level,
            patch=patch,
        )
    )

    return {
        "proposal_id": proposal.proposal_id,
        "revision": revision.revision,
        "status": revision.status,
        "risk_level": revision.risk_level,
        "effective_config": get_effective_config(proposal.scope_type, proposal.scope_id),
    }


@app.post("/v1/config/revisions:rollback")
async def config_revisions_rollback(body: ConfigRollbackRequest, request: Request) -> dict[str, object]:
    actor = require_owner(request)
    target = get_revision(body.revision)
    if target is None:
        raise HTTPException(status_code=404, detail="Revision not found")
    if target.status != "committed":
        raise HTTPException(status_code=400, detail="Only committed revision can be rolled back")

    update_revision_status(target.revision, status="rolled_back")
    rollback_revision = create_revision(
        scope_type=target.scope_type,
        scope_id=target.scope_id,
        actor_id=body.actor_id or actor.actor_id,
        patch={"rolled_back_revision": target.revision},
        explanation=f"Rollback {target.revision}",
        risk_level="L3",
        status="rolled_back",
    )

    append_event(
        PatchRolledBackEvent(
            conversation_id=f"config:{target.scope_type}:{target.scope_id}",
            ts=now_ts(),
            actor_id=body.actor_id or actor.actor_id,
            revision=target.revision,
            rollback_revision=rollback_revision.revision,
        )
    )

    return {
        "rolled_back": target.revision,
        "rollback_revision": rollback_revision.revision,
        "effective_config": get_effective_config(target.scope_type, target.scope_id),
    }


@app.post("/v1/memory/write")
async def memory_write(body: MemoryWriteRequest) -> dict[str, object]:
    outcome = await write_memory(
        scope_type=body.scope_type,
        scope_id=body.scope_id,
        kind=body.kind,
        content=body.content,
        confidence=body.confidence,
        requires_confirmation=body.requires_confirmation,
    )
    item = outcome["item"]
    return {
        "target": outcome["target"],
        "id": item.id,
        "status": getattr(item, "status", "active"),
        "scope_type": item.scope_type,
        "scope_id": item.scope_id,
        "kind": item.kind,
        "content": item.content,
        "confidence": item.confidence,
    }


@app.post("/v1/memory/query")
async def memory_query(body: MemoryQueryRequest) -> dict[str, object]:
    results = await query_memory(
        scope_type=body.scope_type,
        scope_id=body.scope_id,
        query=body.query,
        top_k=body.top_k,
    )
    return {
        "count": len(results),
        "results": results,
    }


@app.get("/v1/memory/items")
async def memory_items(
    scope_type: str | None = None,
    scope_id: str | None = None,
    kind: str | None = None,
    limit: int = 100,
) -> dict[str, object]:
    items = list_memory_items(scope_type=scope_type, scope_id=scope_id, kind=kind, limit=limit)
    return {
        "count": len(items),
        "items": [
            {
                "id": item.id,
                "scope_type": item.scope_type,
                "scope_id": item.scope_id,
                "kind": item.kind,
                "content": item.content,
                "confidence": item.confidence,
                "created_at": item.created_at,
            }
            for item in items
        ],
    }


@app.post("/v1/memory/items/{item_id}:update")
async def memory_item_update(item_id: str, body: MemoryUpdateRequest, request: Request) -> dict[str, object]:
    require_owner(request)
    item = await update_memory_item(
        item_id=item_id,
        content=body.content,
        kind=body.kind,
        confidence=body.confidence,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Memory item not found")
    return {
        "id": item.id,
        "scope_type": item.scope_type,
        "scope_id": item.scope_id,
        "kind": item.kind,
        "content": item.content,
        "confidence": item.confidence,
        "created_at": item.created_at,
    }


@app.post("/v1/memory/items/{item_id}:delete")
async def memory_item_delete(item_id: str, request: Request) -> dict[str, object]:
    require_owner(request)
    deleted = delete_memory_item(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory item not found")
    return {"id": item_id, "status": "deleted"}


@app.post("/v1/memory/forget")
async def memory_forget(body: MemoryForgetRequest, request: Request) -> dict[str, object]:
    require_owner(request)
    return await forget_memory_by_query(
        scope_type=body.scope_type,
        scope_id=body.scope_id,
        query=body.query,
        threshold=body.threshold,
        max_delete=body.max_delete,
    )


@app.post("/v1/memory/decay")
async def memory_decay(body: MemoryDecayRequest, request: Request) -> dict[str, object]:
    require_owner(request)
    return apply_memory_confidence_decay(
        half_life_days=body.half_life_days,
        min_confidence=body.min_confidence,
        scope_type=body.scope_type,
        scope_id=body.scope_id,
    )


@app.get("/v1/memory/metrics")
async def memory_metrics(window_hours: int = 24) -> dict[str, object]:
    return get_memory_metrics(lookback_hours=window_hours)


@app.get("/v1/memory/candidates")
async def memory_candidates(status: str = "pending") -> dict[str, object]:
    candidates = list_candidates(status=status)
    return {
        "count": len(candidates),
        "candidates": [
            {
                "id": item.id,
                "scope_type": item.scope_type,
                "scope_id": item.scope_id,
                "kind": item.kind,
                "content": item.content,
                "confidence": item.confidence,
                "status": item.status,
                "created_at": item.created_at,
            }
            for item in candidates
        ],
    }


@app.post("/v1/memory/candidates/{candidate_id}:approve")
async def memory_candidate_approve(candidate_id: str, request: Request) -> dict[str, object]:
    actor = require_owner(request)
    candidate, memory_item = approve_candidate(candidate_id=candidate_id, actor_id=actor.actor_id)
    if candidate is None or memory_item is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return {
        "candidate_id": candidate.id,
        "status": candidate.status,
        "memory_item_id": memory_item.id,
    }


@app.post("/v1/memory/candidates/{candidate_id}:reject")
async def memory_candidate_reject(candidate_id: str, request: Request) -> dict[str, object]:
    actor = require_owner(request)
    candidate = reject_candidate(candidate_id=candidate_id, actor_id=actor.actor_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return {
        "candidate_id": candidate.id,
        "status": candidate.status,
    }


@app.post("/v1/audit/render")
async def audit_render(body: AuditRenderRequest) -> dict[str, object]:
    task = get_task(body.task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return render_task_audit(task_id=task.id, conversation_id=task.conversation_id)
