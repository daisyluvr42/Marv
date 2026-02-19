from __future__ import annotations

import re
from pathlib import Path

from sqlmodel import select

from backend.agent.config import get_settings
from backend.agent.state import now_ts
from backend.storage.db import get_session
from backend.storage.models import SessionWorkspace


def ensure_session_workspace(conversation_id: str, actor_id: str | None = None) -> SessionWorkspace:
    safe_id = _safe_name(conversation_id)
    root = get_settings().data_dir / "sessions"
    path = root / safe_id
    path.mkdir(parents=True, exist_ok=True)

    with get_session() as session:
        existing = session.exec(select(SessionWorkspace).where(SessionWorkspace.conversation_id == conversation_id)).first()
        ts = now_ts()
        if existing is None:
            existing = SessionWorkspace(
                conversation_id=conversation_id,
                workspace_path=str(path.resolve()),
                status="active",
                created_at=ts,
                updated_at=ts,
                actor_id=actor_id,
            )
        else:
            existing.workspace_path = str(path.resolve())
            existing.updated_at = ts
            if actor_id:
                existing.actor_id = actor_id
            if existing.status != "active":
                existing.status = "active"
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing


def get_session_workspace(conversation_id: str) -> SessionWorkspace | None:
    with get_session() as session:
        return session.exec(select(SessionWorkspace).where(SessionWorkspace.conversation_id == conversation_id)).first()


def list_session_workspaces(limit: int = 100) -> list[SessionWorkspace]:
    with get_session() as session:
        stmt = select(SessionWorkspace).order_by(SessionWorkspace.updated_at.desc()).limit(max(1, min(limit, 500)))
        return list(session.exec(stmt))


def mark_session_archived(conversation_id: str) -> SessionWorkspace | None:
    with get_session() as session:
        item = session.exec(select(SessionWorkspace).where(SessionWorkspace.conversation_id == conversation_id)).first()
        if item is None:
            return None
        item.status = "archived"
        item.updated_at = now_ts()
        session.add(item)
        session.commit()
        session.refresh(item)
        return item


def _safe_name(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return cleaned or "session"
