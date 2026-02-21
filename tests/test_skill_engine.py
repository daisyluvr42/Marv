from __future__ import annotations

from uuid import uuid4

import pytest

from backend.agent.state import create_task, now_ts, upsert_conversation
from backend.ledger.events import CompletionEvent, InputEvent
from backend.ledger.store import append_event, query_events
from backend.storage.db import init_db
from backend.tools.registry import scan_tools
from backend.tools.runner import create_tool_call
from marv.engine.reflection import SkillEngine


@pytest.mark.asyncio
async def test_skill_engine_distill_and_match(monkeypatch, tmp_path) -> None:
    async def _fake_embed_text(text: str, model: str = "mock-embedding") -> list[float]:
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr("marv.engine.reflection.embed_text", _fake_embed_text)
    init_db()
    scan_tools()

    conversation_id = "conv_skill_engine_distill"
    unique_query = f"帮我查一下上海天气_{uuid4().hex}"
    upsert_conversation(conversation_id=conversation_id, channel="web")
    created_tasks: list[str] = []
    for index in range(4):
        task = create_task(conversation_id=conversation_id, status="completed", stage="answer")
        append_event(
            InputEvent(
                    conversation_id=conversation_id,
                    task_id=task.id,
                    ts=now_ts() + index,
                    actor_id="owner",
                    message=unique_query,
                )
            )
        create_tool_call(
            task_id=task.id,
            tool_name="mock_web_search",
            args={"query": unique_query},
            status="ok",
        )
        append_event(
            CompletionEvent(
                conversation_id=conversation_id,
                task_id=task.id,
                ts=now_ts() + index + 1,
                response_text="已完成查询。",
            )
        )
        created_tasks.append(task.id)

    engine = SkillEngine(data_dir=tmp_path / "data")
    candidates = await engine.analyze_ledger(window_hours=48, min_occurrences=3, max_patterns=10)
    assert candidates
    target = next((item for item in candidates if str(item.get("sample_message", "")) == unique_query), None)
    assert target is not None
    assert target["occurrences"] >= 4
    assert any(task_id in created_tasks for task_id in target["task_ids"])

    events = query_events(conversation_id=conversation_id, task_id=created_tasks[0])
    blueprint = await engine.distill_trajectory(events)
    assert blueprint is not None
    assert blueprint.steps
    assert blueprint.min_permission_level == 1

    stored = await engine.solidify_skill(blueprint, status="candidate", occurrences=5)
    assert stored["status"] == "production"

    matched = await engine.match_blueprint(intent_text=unique_query, min_score=0.9)
    assert matched is not None
    assert matched["name"] == stored["name"]
