from __future__ import annotations

import json
from uuid import uuid4

from sqlmodel import select

from backend.ledger.events import BaseEvent
from backend.storage.db import get_session
from backend.storage.models import LedgerEvent


def append_event(event: BaseEvent) -> LedgerEvent:
    payload = event.payload()
    event_id = event.event_id or f"evt_{uuid4().hex}"

    record = LedgerEvent(
        event_id=event_id,
        task_id=event.task_id,
        conversation_id=event.conversation_id,
        type=event.type,
        ts=event.ts,
        actor_id=event.actor_id,
        payload_json=json.dumps(payload, ensure_ascii=True),
    )
    with get_session() as session:
        session.add(record)
        session.commit()
        session.refresh(record)
    return record


def query_events(conversation_id: str, task_id: str | None = None) -> list[LedgerEvent]:
    with get_session() as session:
        stmt = select(LedgerEvent).where(LedgerEvent.conversation_id == conversation_id)
        if task_id:
            stmt = stmt.where(LedgerEvent.task_id == task_id)
        stmt = stmt.order_by(LedgerEvent.ts.asc(), LedgerEvent.id.asc())
        return list(session.exec(stmt))
