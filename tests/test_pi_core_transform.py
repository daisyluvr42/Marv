from __future__ import annotations

from backend.pi_core.schema import PiMessage, PiTurnContext
from backend.pi_core.transform import build_pi_turn_context, compact_turn_context, to_openai_messages


def test_build_pi_turn_context_includes_memory_prompt() -> None:
    context = build_pi_turn_context(
        system_prompt="system-a",
        memory_prompt="memory-a",
        user_input="hello",
        metadata={"task_id": "t1"},
    )
    assert context.system_prompts == ["system-a", "memory-a"]
    assert len(context.messages) == 1
    assert context.messages[0].role == "user"
    assert context.messages[0].content == "hello"
    assert context.metadata["task_id"] == "t1"


def test_compact_turn_context_keeps_tail_messages() -> None:
    context = PiTurnContext(
        system_prompts=["system-a"],
        messages=[PiMessage(role="user", content=f"m{i}") for i in range(5)],
    )
    compacted = compact_turn_context(context, max_messages=2)
    assert len(compacted.messages) == 2
    assert compacted.messages[0].content == "m3"
    assert compacted.messages[1].content == "m4"


def test_to_openai_messages_maps_tool_role() -> None:
    context = PiTurnContext(
        system_prompts=["system-a"],
        messages=[
            PiMessage(role="assistant", content="ok"),
            PiMessage(role="tool", content="search => done"),
        ],
    )
    converted = to_openai_messages(context)
    assert converted[0] == {"role": "system", "content": "system-a"}
    assert converted[1] == {"role": "assistant", "content": "ok"}
    assert converted[2]["role"] == "user"
    assert converted[2]["content"] == "TOOL_RESULT search => done"

