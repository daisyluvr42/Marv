from __future__ import annotations

import json
from uuid import uuid4

from sqlmodel import select

from backend.agent.state import now_ts
from backend.storage.db import get_session
from backend.storage.models import ToolCall
from backend.tools.registry import get_tool_function, get_tool_spec


def create_tool_call(
    task_id: str | None,
    tool_name: str,
    args: dict[str, object],
    status: str,
    approval_id: str | None = None,
    tool_call_id: str | None = None,
) -> ToolCall:
    if tool_call_id:
        with get_session() as session:
            existing = session.exec(select(ToolCall).where(ToolCall.tool_call_id == tool_call_id)).first()
            if existing is not None:
                return existing

    tool_call = ToolCall(
        tool_call_id=tool_call_id or f"tc_{uuid4().hex}",
        task_id=task_id,
        tool=tool_name,
        args_json=json.dumps(args, ensure_ascii=True),
        status=status,
        approval_id=approval_id,
        created_at=now_ts(),
        updated_at=now_ts(),
    )
    with get_session() as session:
        session.add(tool_call)
        session.commit()
        session.refresh(tool_call)
    return tool_call


def execute_tool_call(
    task_id: str | None,
    tool_name: str,
    args: dict[str, object],
    tool_call_id: str | None = None,
) -> ToolCall:
    spec = get_tool_spec(tool_name)
    if spec is None:
        raise ValueError(f"Tool not found: {tool_name}")
    if spec.risk != "read_only":
        raise PermissionError(f"Tool requires approval: {spec.risk}")

    tool_call = create_tool_call(
        task_id=task_id,
        tool_name=tool_name,
        args=args,
        status="running",
        tool_call_id=tool_call_id,
    )
    if tool_call.status in {"ok", "error"}:
        return tool_call
    return execute_existing_tool_call(tool_call.tool_call_id, allow_external_write=False)


def get_tool_call(tool_call_id: str) -> ToolCall | None:
    with get_session() as session:
        return session.exec(select(ToolCall).where(ToolCall.tool_call_id == tool_call_id)).first()


def execute_existing_tool_call(tool_call_id: str, allow_external_write: bool) -> ToolCall:
    with get_session() as session:
        tool_call = session.exec(select(ToolCall).where(ToolCall.tool_call_id == tool_call_id)).first()
        if tool_call is None:
            raise ValueError(f"Tool call not found: {tool_call_id}")

    spec = get_tool_spec(tool_call.tool)
    func = get_tool_function(tool_call.tool)
    if spec is None or func is None:
        raise ValueError(f"Tool not found: {tool_call.tool}")
    if spec.risk == "external_write" and not allow_external_write:
        raise PermissionError("external_write requires approval")

    args = json.loads(tool_call.args_json)
    try:
        tool_call.status = "running"
        result = func(**args)
        tool_call.status = "ok"
        tool_call.result_json = json.dumps(result, ensure_ascii=True)
        tool_call.error = None
    except Exception as exc:  # pragma: no cover - defensive path
        tool_call.status = "error"
        tool_call.error = str(exc)
    finally:
        tool_call.updated_at = now_ts()

    with get_session() as session:
        existing = session.exec(select(ToolCall).where(ToolCall.tool_call_id == tool_call.tool_call_id)).first()
        if existing is None:
            raise ValueError(f"Tool call not found while saving: {tool_call_id}")
        existing.status = tool_call.status
        existing.result_json = tool_call.result_json
        existing.error = tool_call.error
        existing.updated_at = tool_call.updated_at
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing
