from __future__ import annotations

import json

from sqlmodel import select

from backend.ledger.store import query_events
from backend.storage.db import get_session
from backend.storage.models import Approval, ToolCall


def render_task_audit(task_id: str, conversation_id: str) -> dict[str, object]:
    events = query_events(conversation_id=conversation_id, task_id=task_id)
    timeline = [
        {
            "event_id": event.event_id,
            "type": event.type,
            "ts": event.ts,
            "payload": json.loads(event.payload_json),
        }
        for event in events
    ]

    with get_session() as session:
        tool_calls = list(session.exec(select(ToolCall).where(ToolCall.task_id == task_id).order_by(ToolCall.created_at.asc())))
        approval_ids = [call.approval_id for call in tool_calls if call.approval_id]
        approvals: list[Approval] = []
        if approval_ids:
            approvals = list(
                session.exec(select(Approval).where(Approval.approval_id.in_(approval_ids)).order_by(Approval.created_at.asc()))
            )

    return {
        "task_id": task_id,
        "conversation_id": conversation_id,
        "timeline": timeline,
        "summary": {
            "event_count": len(timeline),
            "tool_call_count": len(tool_calls),
            "approval_count": len(approvals),
        },
        "tool_calls": [
            {
                "tool_call_id": item.tool_call_id,
                "tool": item.tool,
                "status": item.status,
                "approval_id": item.approval_id,
                "args": json.loads(item.args_json),
                "result": json.loads(item.result_json) if item.result_json else None,
                "error": item.error,
            }
            for item in tool_calls
        ],
        "approvals": [
            {
                "approval_id": item.approval_id,
                "status": item.status,
                "summary": item.summary,
                "decided_by": item.decided_by,
                "constraints": json.loads(item.constraints_json),
            }
            for item in approvals
        ],
    }
