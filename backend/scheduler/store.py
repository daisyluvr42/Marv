from __future__ import annotations

from uuid import uuid4

from sqlmodel import select

from backend.agent.state import now_ts
from backend.storage.db import get_session
from backend.storage.models import ScheduledTask


def create_scheduled_task(
    *,
    name: str,
    prompt: str,
    cron: str,
    timezone: str,
    created_by: str,
    conversation_id: str | None,
    channel: str,
    channel_id: str | None,
    user_id: str | None,
    thread_id: str | None,
    status: str = "active",
) -> ScheduledTask:
    ts = now_ts()
    item = ScheduledTask(
        schedule_id=f"st_{uuid4().hex}",
        name=name,
        prompt=prompt,
        cron=cron,
        timezone=timezone or "UTC",
        status=status if status in {"active", "paused"} else "active",
        conversation_id=conversation_id,
        channel=channel,
        channel_id=channel_id,
        user_id=user_id,
        thread_id=thread_id,
        created_at=ts,
        updated_at=ts,
        created_by=created_by,
    )
    with get_session() as session:
        session.add(item)
        session.commit()
        session.refresh(item)
        return item


def list_scheduled_tasks(*, status: str | None = None, limit: int = 200) -> list[ScheduledTask]:
    with get_session() as session:
        stmt = select(ScheduledTask).order_by(ScheduledTask.created_at.desc()).limit(max(1, min(limit, 1000)))
        if status:
            stmt = stmt.where(ScheduledTask.status == status)
        return list(session.exec(stmt))


def get_scheduled_task(schedule_id: str) -> ScheduledTask | None:
    with get_session() as session:
        return session.exec(select(ScheduledTask).where(ScheduledTask.schedule_id == schedule_id)).first()


def set_scheduled_task_status(schedule_id: str, status: str) -> ScheduledTask | None:
    with get_session() as session:
        item = session.exec(select(ScheduledTask).where(ScheduledTask.schedule_id == schedule_id)).first()
        if item is None:
            return None
        item.status = status
        item.updated_at = now_ts()
        session.add(item)
        session.commit()
        session.refresh(item)
        return item


def delete_scheduled_task(schedule_id: str) -> bool:
    with get_session() as session:
        item = session.exec(select(ScheduledTask).where(ScheduledTask.schedule_id == schedule_id)).first()
        if item is None:
            return False
        session.delete(item)
        session.commit()
        return True


def touch_scheduled_task_run(
    *,
    schedule_id: str,
    task_id: str | None,
    error: str | None,
    next_run_at: int | None = None,
) -> ScheduledTask | None:
    with get_session() as session:
        item = session.exec(select(ScheduledTask).where(ScheduledTask.schedule_id == schedule_id)).first()
        if item is None:
            return None
        item.last_run_at = now_ts()
        item.last_task_id = task_id
        item.last_error = error
        item.next_run_at = next_run_at
        item.updated_at = now_ts()
        session.add(item)
        session.commit()
        session.refresh(item)
        return item


def set_scheduled_task_next_run(schedule_id: str, next_run_at: int | None) -> ScheduledTask | None:
    with get_session() as session:
        item = session.exec(select(ScheduledTask).where(ScheduledTask.schedule_id == schedule_id)).first()
        if item is None:
            return None
        item.next_run_at = next_run_at
        item.updated_at = now_ts()
        session.add(item)
        session.commit()
        session.refresh(item)
        return item
