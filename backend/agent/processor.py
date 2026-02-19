from __future__ import annotations

import json

from backend.agent.state import get_task, now_ts, update_task_status
from backend.core_client.openai_compat import get_core_client
from backend.ledger.events import CompletionEvent, InputEvent, PlanEvent, RouteEvent
from backend.ledger.store import append_event, query_events


async def process_task(task_id: str) -> None:
    task = get_task(task_id)
    if task is None:
        return

    update_task_status(task_id=task_id, status="running", stage="plan")
    append_event(
        PlanEvent(
            conversation_id=task.conversation_id,
            task_id=task_id,
            ts=now_ts(),
            plan="Generate response via core chat completion",
        )
    )

    append_event(
        RouteEvent(
            conversation_id=task.conversation_id,
            task_id=task_id,
            ts=now_ts(),
            route="core:/v1/chat/completions",
        )
    )

    user_input = _get_task_input(conversation_id=task.conversation_id, task_id=task_id)
    try:
        core_client = get_core_client()
        await core_client.health_check()
        response = await core_client.chat_completions(
            messages=[{"role": "user", "content": user_input}],
            stream=False,
        )
        response_text = (
            response.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or "[empty-response]"
        )
        append_event(
            CompletionEvent(
                conversation_id=task.conversation_id,
                task_id=task_id,
                ts=now_ts(),
                response_text=response_text,
            )
        )
        update_task_status(task_id=task_id, status="completed", stage="answer")
    except Exception as exc:  # pragma: no cover - guarded fallback path
        append_event(
            RouteEvent(
                conversation_id=task.conversation_id,
                task_id=task_id,
                ts=now_ts(),
                route=f"fallback:core_unavailable:{exc}",
            )
        )
        fallback_text = "Core 当前不可用，已降级为本地回退回复。"
        append_event(
            CompletionEvent(
                conversation_id=task.conversation_id,
                task_id=task_id,
                ts=now_ts(),
                response_text=f"{fallback_text} 原始输入: {user_input}",
            )
        )
        update_task_status(task_id=task_id, status="completed", stage="answer", last_error=str(exc))


def _get_task_input(conversation_id: str, task_id: str) -> str:
    events = query_events(conversation_id=conversation_id, task_id=task_id)
    for event in reversed(events):
        if event.type != "InputEvent":
            continue
        payload = json.loads(event.payload_json)
        return str(payload.get("message", ""))
    return ""
