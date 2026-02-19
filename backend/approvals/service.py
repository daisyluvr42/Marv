from __future__ import annotations

import json
from uuid import uuid4

from sqlmodel import select

from backend.agent.state import now_ts
from backend.storage.db import get_session
from backend.storage.models import Approval, ToolCall


def create_approval(
    approval_type: str,
    summary: str,
    actor_id: str | None = None,
    constraints: dict[str, object] | None = None,
) -> Approval:
    approval = Approval(
        approval_id=f"ap_{uuid4().hex}",
        type=approval_type,
        status="pending",
        summary=summary,
        constraints_json=json.dumps(constraints or {"one_time": True}, ensure_ascii=True),
        created_at=now_ts(),
        updated_at=now_ts(),
        actor_id=actor_id,
    )
    with get_session() as session:
        session.add(approval)
        session.commit()
        session.refresh(approval)
        return approval


def list_approvals(status: str | None = None) -> list[Approval]:
    with get_session() as session:
        stmt = select(Approval)
        if status:
            stmt = stmt.where(Approval.status == status)
        stmt = stmt.order_by(Approval.created_at.desc())
        return list(session.exec(stmt))


def update_approval_status(approval_id: str, status: str, decided_by: str | None = None) -> Approval | None:
    with get_session() as session:
        approval = session.exec(select(Approval).where(Approval.approval_id == approval_id)).first()
        if approval is None:
            return None
        approval.status = status
        approval.decided_by = decided_by
        approval.updated_at = now_ts()
        session.add(approval)
        session.commit()
        session.refresh(approval)
        return approval


def get_tool_call_by_approval_id(approval_id: str) -> ToolCall | None:
    with get_session() as session:
        return session.exec(select(ToolCall).where(ToolCall.approval_id == approval_id)).first()
