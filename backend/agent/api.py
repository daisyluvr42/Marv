from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.approvals.service import (
    create_approval,
    get_tool_call_by_approval_id,
    list_approvals,
    update_approval_status,
)
from backend.audit.render import render_task_audit
from backend.agent.auth import get_actor_context, require_owner
from backend.agent.config import get_settings
from backend.agent.processor import process_task
from backend.agent.queue import task_queue
from backend.agent.state import create_task, get_task, now_ts, upsert_conversation
from backend.heartbeat.runtime import get_heartbeat_config_path, heartbeat_runtime
from backend.ledger.events import InputEvent, PatchCommittedEvent, PatchRolledBackEvent
from backend.ledger.store import append_event
from backend.ledger.store import query_events
from backend.memory.store import approve_candidate, list_candidates, query_memory, reject_candidate, write_memory
from backend.permissions.exec_approvals import evaluate_tool_permission, load_exec_approvals
from backend.patch.proposals import create_patch_proposal, get_patch_proposal, update_patch_proposal_status
from backend.patch.state import (
    create_revision,
    ensure_seed,
    get_effective_config,
    get_revision,
    list_revisions,
    update_revision_status,
)
from backend.storage.db import init_db
from backend.tools.registry import get_tool_spec, list_tools, scan_tools, sync_tools_registry
from backend.tools.runner import create_tool_call, execute_existing_tool_call, execute_tool_call, get_tool_call


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    ensure_seed()
    scan_tools()
    sync_tools_registry()
    task_queue.set_processor(process_task)
    await task_queue.start()
    await heartbeat_runtime.start()
    try:
        yield
    finally:
        await heartbeat_runtime.stop()
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


class ApprovalDecisionRequest(BaseModel):
    actor_id: str | None = None


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


@app.post("/v1/agent/messages")
async def agent_messages(body: MessageRequest, request: Request) -> dict[str, object]:
    actor = get_actor_context(request)
    conversation_id = body.conversation_id or f"conv_{uuid4().hex}"
    upsert_conversation(
        conversation_id=conversation_id,
        channel=body.channel,
        channel_id=body.channel_id,
        user_id=body.user_id,
        thread_id=body.thread_id,
    )
    task = create_task(conversation_id=conversation_id, status="queued", stage="plan")
    append_event(
        InputEvent(
            conversation_id=conversation_id,
            task_id=task.id,
            ts=now_ts(),
            actor_id=body.actor_id or actor.actor_id,
            message=body.message,
        )
    )
    await task_queue.enqueue_task(task.id)
    return {
        "conversation_id": conversation_id,
        "task_id": task.id,
        "status": task.status,
    }


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

    if spec.risk == "external_write":
        if actor.role != "owner":
            raise HTTPException(status_code=403, detail="Member cannot execute external_write tools")

    perm_cfg = load_exec_approvals()
    perm = evaluate_tool_permission(perm_cfg, actor_id=body.actor_id or actor.actor_id, tool_name=body.tool)
    if perm["decision"] == "deny":
        raise HTTPException(status_code=403, detail=f"Tool blocked by exec policy: {perm['reason']}")

    require_approval = spec.risk == "external_write" or perm["decision"] == "ask"
    if require_approval:
        summary = f"{body.tool} requires approval ({perm['reason']})"
        approval = create_approval(
            approval_type="tool_execute",
            summary=summary,
            actor_id=body.actor_id or actor.actor_id,
            constraints={"tool": body.tool, "one_time": True, "policy_reason": perm["reason"]},
        )
        tool_call = create_tool_call(
            task_id=body.task_id,
            tool_name=body.tool,
            args=body.args,
            status="pending_approval",
            approval_id=approval.approval_id,
            tool_call_id=body.tool_call_id,
        )
        return {
            "tool_call_id": tool_call.tool_call_id,
            "status": "pending_approval",
            "approval_id": approval.approval_id,
            "tool": body.tool,
            "policy_reason": perm["reason"],
        }

    try:
        tool_call = execute_tool_call(
            task_id=body.task_id,
            tool_name=body.tool,
            args=body.args,
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
    approval = update_approval_status(
        approval_id=approval_id,
        status="approved",
        decided_by=body.actor_id or actor.actor_id,
    )
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    tool_call = get_tool_call_by_approval_id(approval_id)
    if tool_call is not None:
        tool_call = execute_existing_tool_call(tool_call.tool_call_id, allow_external_write=True)
    return {
        "approval_id": approval.approval_id,
        "status": approval.status,
        "tool_call_id": tool_call.tool_call_id if tool_call else None,
        "tool_call_status": tool_call.status if tool_call else None,
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
