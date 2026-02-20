from __future__ import annotations

from typing import Any

from backend.pi_core.schema import PiMessage, PiTurnContext


def build_pi_turn_context(
    *,
    system_prompt: str,
    user_input: str,
    memory_prompt: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> PiTurnContext:
    prompts: list[str] = []
    if system_prompt.strip():
        prompts.append(system_prompt.strip())
    if isinstance(memory_prompt, str) and memory_prompt.strip():
        prompts.append(memory_prompt.strip())

    messages: list[PiMessage] = []
    if user_input.strip():
        messages.append(PiMessage(role="user", content=user_input.strip()))

    return PiTurnContext(
        system_prompts=prompts,
        messages=messages,
        metadata=dict(metadata or {}),
    )


def compact_turn_context(context: PiTurnContext, *, max_messages: int = 64) -> PiTurnContext:
    if max_messages <= 0:
        return PiTurnContext(
            system_prompts=list(context.system_prompts),
            messages=[],
            metadata=dict(context.metadata),
        )

    if len(context.messages) <= max_messages:
        return PiTurnContext(
            system_prompts=list(context.system_prompts),
            messages=list(context.messages),
            metadata=dict(context.metadata),
        )

    kept = context.messages[-max_messages:]
    return PiTurnContext(
        system_prompts=list(context.system_prompts),
        messages=kept,
        metadata=dict(context.metadata),
    )


def to_openai_messages(context: PiTurnContext) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for prompt in context.system_prompts:
        cleaned = prompt.strip()
        if not cleaned:
            continue
        messages.append({"role": "system", "content": cleaned})

    for item in context.messages:
        if item.role == "system":
            messages.append({"role": "system", "content": item.content})
            continue
        if item.role == "tool":
            messages.append({"role": "user", "content": f"TOOL_RESULT {item.content}"})
            continue
        messages.append({"role": item.role, "content": item.content})
    return messages

