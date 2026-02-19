from __future__ import annotations

from fnmatch import fnmatchcase
import json
from uuid import uuid4

from sqlmodel import select

from backend.agent.state import now_ts
from backend.storage.db import get_session
from backend.storage.models import Approval, ApprovalGrant, ToolCall


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


def create_approval_grant(
    *,
    actor_id: str,
    tool_pattern: str,
    session_id: str | None,
    ttl_seconds: int,
    created_by: str,
    source_approval_id: str | None = None,
) -> ApprovalGrant:
    ttl = max(60, min(ttl_seconds, 24 * 3600))
    ts = now_ts()
    grant = ApprovalGrant(
        grant_id=f"ag_{uuid4().hex}",
        actor_id=actor_id,
        tool_pattern=tool_pattern,
        session_id=session_id,
        status="active",
        created_at=ts,
        expires_at=ts + ttl * 1000,
        created_by=created_by,
        source_approval_id=source_approval_id,
    )
    with get_session() as session:
        session.add(grant)
        session.commit()
        session.refresh(grant)
        return grant


def list_approval_grants(
    *,
    actor_id: str | None = None,
    session_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[ApprovalGrant]:
    _expire_approval_grants()
    with get_session() as session:
        stmt = select(ApprovalGrant).order_by(ApprovalGrant.created_at.desc()).limit(max(1, min(limit, 500)))
        if actor_id:
            stmt = stmt.where(ApprovalGrant.actor_id == actor_id)
        if session_id:
            stmt = stmt.where(ApprovalGrant.session_id == session_id)
        if status:
            stmt = stmt.where(ApprovalGrant.status == status)
        return list(session.exec(stmt))


def revoke_approval_grant(grant_id: str) -> ApprovalGrant | None:
    with get_session() as session:
        grant = session.exec(select(ApprovalGrant).where(ApprovalGrant.grant_id == grant_id)).first()
        if grant is None:
            return None
        grant.status = "revoked"
        session.add(grant)
        session.commit()
        session.refresh(grant)
        return grant


def find_matching_approval_grant(
    *,
    actor_id: str,
    tool_name: str,
    session_id: str | None,
) -> ApprovalGrant | None:
    _expire_approval_grants()
    with get_session() as session:
        stmt = select(ApprovalGrant).where(
            ApprovalGrant.actor_id == actor_id,
            ApprovalGrant.status == "active",
        )
        candidates = list(session.exec(stmt))
    lowered_tool = tool_name.strip().lower()
    for grant in candidates:
        if grant.session_id and session_id and grant.session_id != session_id:
            continue
        if grant.session_id and not session_id:
            continue
        pattern = grant.tool_pattern.strip().lower()
        if pattern and fnmatchcase(lowered_tool, pattern):
            return grant
    return None


def _expire_approval_grants() -> None:
    ts = now_ts()
    with get_session() as session:
        stmt = select(ApprovalGrant).where(
            ApprovalGrant.status == "active",
            ApprovalGrant.expires_at <= ts,
        )
        expired = list(session.exec(stmt))
        if not expired:
            return
        for grant in expired:
            grant.status = "expired"
            session.add(grant)
        session.commit()
