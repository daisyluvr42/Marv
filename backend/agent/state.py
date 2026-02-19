from __future__ import annotations

import time
from uuid import uuid4

from sqlmodel import select

from backend.storage.db import get_session
from backend.storage.models import Conversation, Task


def now_ts() -> int:
    return int(time.time() * 1000)


def create_task(conversation_id: str, status: str = "queued", stage: str = "plan") -> Task:
    task = Task(
        id=f"task_{uuid4().hex}",
        conversation_id=conversation_id,
        status=status,
        created_at=now_ts(),
        updated_at=now_ts(),
        current_stage=stage,
    )
    with get_session() as session:
        session.add(task)
        session.commit()
        session.refresh(task)
    return task


def upsert_conversation(
    conversation_id: str,
    channel: str = "web",
    channel_id: str | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
) -> Conversation:
    with get_session() as session:
        conversation = session.exec(select(Conversation).where(Conversation.id == conversation_id)).first()
        if conversation is None:
            conversation = Conversation(
                id=conversation_id,
                channel=channel,
                channel_id=channel_id,
                user_id=user_id,
                thread_id=thread_id,
                created_at=now_ts(),
                updated_at=now_ts(),
            )
        else:
            conversation.channel = channel
            conversation.channel_id = channel_id
            conversation.user_id = user_id
            conversation.thread_id = thread_id
            conversation.updated_at = now_ts()
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
        return conversation


def update_task_status(
    task_id: str,
    status: str,
    stage: str | None = None,
    last_error: str | None = None,
) -> Task | None:
    with get_session() as session:
        task = session.exec(select(Task).where(Task.id == task_id)).first()
        if not task:
            return None
        task.status = status
        task.updated_at = now_ts()
        if stage is not None:
            task.current_stage = stage
        if last_error is not None:
            task.last_error = last_error
        session.add(task)
        session.commit()
        session.refresh(task)
        return task


def get_task(task_id: str) -> Task | None:
    with get_session() as session:
        return session.exec(select(Task).where(Task.id == task_id)).first()
