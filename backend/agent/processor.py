from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from typing import Any
from uuid import uuid4

from backend.agent.session_runtime import ensure_session_workspace, get_session_workspace, mark_session_archived
from backend.agent.state import create_task, get_conversation, get_task, now_ts, update_task_status, upsert_conversation
from backend.approvals.policy import decide_approval_mode, load_approval_policy
from backend.approvals.service import create_approval, find_matching_approval_grant
from backend.core_client.openai_compat import get_core_client
from backend.ledger.events import CompletionEvent, InputEvent, PiTurnEvent, PlanEvent, RouteEvent
from backend.ledger.store import append_event, query_events
from backend.memory.store import (
    extract_memory_candidates,
    list_memory_items,
    query_memory_multi,
    record_memory_retrieval,
    write_memory,
)
from backend.patch.state import get_effective_config_for_runtime
from backend.pi_core import build_pi_turn_context, compact_turn_context, to_openai_messages
from backend.permissions.exec_approvals import evaluate_tool_permission, load_exec_approvals
from backend.tools.registry import get_tool_spec, list_tools
from backend.tools.runner import create_tool_call, execute_tool_call
from marv.engine.reflection import Blueprint, get_skill_engine


async def process_task(
    task_id: str,
    *,
    effective_config_override: dict[str, object] | None = None,
    core_client_override: Any | None = None,
    exec_approvals_override: dict[str, object] | None = None,
    approval_policy_override: dict[str, object] | None = None,
) -> None:
    task = get_task(task_id)
    if task is None:
        return
    conversation = get_conversation(task.conversation_id)
    channel = conversation.channel if conversation else "web"
    channel_id = conversation.channel_id if conversation else None
    user_id = conversation.user_id if conversation else None
    thread_id = conversation.thread_id if conversation else None

    if effective_config_override is None:
        effective_config = get_effective_config_for_runtime(
            conversation_id=task.conversation_id,
            channel=channel,
            channel_id=channel_id,
            user_id=user_id,
        )
    else:
        effective_config = dict(effective_config_override)
    persona_prompt = _build_persona_system_prompt(effective_config)
    response_style = str(effective_config.get("response_style", "balanced"))
    user_input, requester_actor_id = _get_task_input_context(conversation_id=task.conversation_id, task_id=task_id)
    memory_config = _resolve_memory_runtime_config(effective_config)
    loop_config = _resolve_tool_loop_config(effective_config)
    routing_config = _resolve_model_routing_config(effective_config)
    skill_router_config = _resolve_skill_router_config(effective_config)
    auto_subagents_config = _resolve_auto_subagents_config(effective_config)

    memory_entries: list[dict[str, object]] = []
    memory_scopes: list[tuple[str, str, float]] = []
    memory_lookup_latency_ms = 0
    memory_lookup_error: str | None = None
    try:
        memory_entries, memory_scopes, memory_lookup_latency_ms = await _load_runtime_memories(
            conversation_id=task.conversation_id,
            channel=channel,
            channel_id=channel_id,
            user_id=user_id,
            query_text=user_input,
            config=memory_config,
        )
    except Exception as exc:  # pragma: no cover - best effort retrieval
        memory_lookup_error = str(exc)
    memory_prompt = _build_memory_system_prompt(memory_entries)
    routing_state = _build_routing_state(
        user_input=user_input,
        memory_entries=memory_entries,
        loop_config=loop_config,
        routing_config=routing_config,
    )

    update_task_status(task_id=task_id, status="running", stage="plan")
    memory_plan_suffix = (
        f", memory_hits={len(memory_entries)}, memory_lookup_ms={memory_lookup_latency_ms}, memory_policy={memory_config['strictness']}"
    )
    if memory_lookup_error:
        memory_plan_suffix += f", memory_lookup_error={memory_lookup_error}"
    loop_plan_suffix = (
        f", loop_enabled={loop_config['enabled']}, max_steps={loop_config['max_steps']}, max_tool_calls={loop_config['max_tool_calls']}"
    )
    routing_plan_suffix = (
        f", routing_enabled={routing_state['enabled']}, intent={routing_state['intent']},"
        f" initial_tier={routing_state['base_tier']}, initial_locality={routing_state['base_locality']}"
    )
    skill_plan_suffix = (
        f", skill_router_enabled={skill_router_config['enabled']},"
        f" skill_router_min_score={skill_router_config['min_score']},"
        f" skill_router_max_steps={skill_router_config['max_steps']},"
        f" skill_router_semantic_match={skill_router_config['semantic_match']}"
    )
    auto_subagents_plan_suffix = (
        f", auto_subagents_enabled={auto_subagents_config['enabled']},"
        f" auto_subagents_threshold={auto_subagents_config['complexity_threshold']},"
        f" auto_subagents_child_timeout={auto_subagents_config['child_timeout_seconds']}"
    )
    append_event(
        PlanEvent(
            conversation_id=task.conversation_id,
            task_id=task_id,
            ts=now_ts(),
            plan=(
                "Generate response via core chat completion with "
                f"persona(response_style={response_style})"
                f"{memory_plan_suffix}{loop_plan_suffix}{routing_plan_suffix}{skill_plan_suffix}{auto_subagents_plan_suffix}"
            ),
        )
    )

    try:
        if memory_scopes:
            record_memory_retrieval(
                task_id=task_id,
                conversation_id=task.conversation_id,
                query_text=user_input,
                scopes=memory_scopes,
                results=memory_entries,
                latency_ms=memory_lookup_latency_ms,
            )

        core_client = core_client_override or get_core_client()
        await core_client.health_check()
        loop_prompt = _build_tool_loop_system_prompt(loop_config=loop_config)
        system_prompt = persona_prompt if not loop_prompt else f"{persona_prompt}\n\n{loop_prompt}"
        pi_context = build_pi_turn_context(
            system_prompt=system_prompt,
            memory_prompt=memory_prompt,
            user_input=user_input,
            metadata={
                "conversation_id": task.conversation_id,
                "task_id": task_id,
                "engine": "pi_compatible",
            },
        )
        pi_context = compact_turn_context(
            pi_context,
            max_messages=max(16, int(loop_config["max_steps"]) * 6),
        )
        messages = to_openai_messages(pi_context)
        append_event(
            PiTurnEvent(
                conversation_id=task.conversation_id,
                task_id=task_id,
                ts=now_ts(),
                stage="context_ready",
                details={
                    "system_prompt_count": len(pi_context.system_prompts),
                    "message_count": len(pi_context.messages),
                    "openai_message_count": len(messages),
                    "engine": str(pi_context.metadata.get("engine", "pi_compatible")),
                },
            )
        )
        response_text, skill_mode_used = await _maybe_run_skill_sop(
            core_client=core_client,
            user_input=user_input,
            task_id=task_id,
            conversation_id=task.conversation_id,
            requester_actor_id=requester_actor_id,
            routing_state=routing_state,
            skill_router_config=skill_router_config,
            exec_approvals_override=exec_approvals_override,
            approval_policy_override=approval_policy_override,
        )
        if response_text is None:
            response_text = await _maybe_run_auto_subagents(
                core_client=core_client,
                task_id=task_id,
                conversation_id=task.conversation_id,
                channel=channel,
                channel_id=channel_id,
                user_id=user_id,
                thread_id=thread_id,
                requester_actor_id=requester_actor_id,
                user_input=user_input,
                routing_state=routing_state,
                effective_config=effective_config,
                auto_subagents_config=auto_subagents_config,
                exec_approvals_override=exec_approvals_override,
                approval_policy_override=approval_policy_override,
            )
        if response_text is None:
            response_text = await _run_agent_loop(
                core_client=core_client,
                messages=messages,
                task_id=task_id,
                conversation_id=task.conversation_id,
                requester_actor_id=requester_actor_id,
                loop_config=loop_config,
                routing_state=routing_state,
                exec_approvals_override=exec_approvals_override,
                approval_policy_override=approval_policy_override,
            )
        elif skill_mode_used:
            append_event(
                RouteEvent(
                    conversation_id=task.conversation_id,
                    task_id=task_id,
                    ts=now_ts(),
                    route=f"skill_router:completed:{skill_mode_used}",
                )
            )
        append_event(
            CompletionEvent(
                conversation_id=task.conversation_id,
                task_id=task_id,
                ts=now_ts(),
                response_text=response_text,
            )
        )
        await _persist_extracted_user_memories(
            user_id=user_id,
            conversation_id=task.conversation_id,
            user_input=user_input,
        )
        await _maybe_flush_conversation_summary_memory(
            conversation_id=task.conversation_id,
            response_text=response_text,
            config=memory_config,
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


def _get_task_input_context(conversation_id: str, task_id: str) -> tuple[str, str]:
    events = query_events(conversation_id=conversation_id, task_id=task_id)
    for event in reversed(events):
        if event.type != "InputEvent":
            continue
        payload = json.loads(event.payload_json)
        return str(payload.get("message", "")), str(payload.get("actor_id") or "main")
    return "", "main"


def _build_persona_system_prompt(effective_config: dict[str, object]) -> str:
    identity = str(effective_config.get("identity", "blackbox-agent"))
    response_style = str(effective_config.get("response_style", "balanced"))
    safety = effective_config.get("safety", {})
    safety_text = json.dumps(safety, ensure_ascii=True, sort_keys=True) if isinstance(safety, dict) else "{}"
    return (
        "You are the runtime persona configured by system seed + committed patches.\n"
        f"identity={identity}\n"
        f"response_style={response_style}\n"
        f"safety={safety_text}\n"
        "Follow the configured persona and safety settings while answering the user."
    )


def _resolve_tool_loop_config(effective_config: dict[str, object]) -> dict[str, object]:
    raw = effective_config.get("tool_loop")
    if not isinstance(raw, dict):
        raw = {}

    enabled = bool(raw.get("enabled", True))
    max_steps = raw.get("max_steps", 6)
    max_tool_calls = raw.get("max_tool_calls", 4)
    max_observation_chars = raw.get("max_observation_chars", 1800)
    strict_json_actions = bool(raw.get("strict_json_actions", True))
    max_protocol_repairs = raw.get("max_protocol_repairs", 2)

    if not isinstance(max_steps, int):
        max_steps = 6
    if not isinstance(max_tool_calls, int):
        max_tool_calls = 4
    if not isinstance(max_observation_chars, int):
        max_observation_chars = 1800
    if not isinstance(max_protocol_repairs, int):
        max_protocol_repairs = 2

    return {
        "enabled": enabled,
        "max_steps": max(1, min(20, max_steps)),
        "max_tool_calls": max(0, min(20, max_tool_calls)),
        "max_observation_chars": max(200, min(8000, max_observation_chars)),
        "strict_json_actions": strict_json_actions,
        "max_protocol_repairs": max(0, min(8, max_protocol_repairs)),
    }


def _resolve_model_routing_config(effective_config: dict[str, object]) -> dict[str, object]:
    raw = effective_config.get("model_routing")
    if not isinstance(raw, dict):
        raw = {}

    prefer_locality = _normalize_route_locality(raw.get("prefer_locality"), default="local")
    simple_tier = _normalize_route_tier(raw.get("simple_tier"), default="local_light")
    standard_tier = _normalize_route_tier(raw.get("standard_tier"), default="local_main")
    cloud_tier = _normalize_route_tier(raw.get("cloud_tier"), default="cloud_high")

    return {
        "enabled": bool(raw.get("enabled", True)),
        "prefer_locality": prefer_locality,
        "allow_cloud_fallback": bool(raw.get("allow_cloud_fallback", True)),
        "simple_tier": simple_tier,
        "standard_tier": standard_tier,
        "cloud_tier": cloud_tier,
        "direct_cloud_on_extreme": bool(raw.get("direct_cloud_on_extreme", False)),
        "extreme_score_threshold": _clamp_int(raw.get("extreme_score_threshold"), default=8, min_value=4, max_value=20),
        "moderate_context_chars": _clamp_int(raw.get("moderate_context_chars"), default=480, min_value=120, max_value=4000),
        "high_context_chars": _clamp_int(raw.get("high_context_chars"), default=1200, min_value=200, max_value=12000),
        "escalate_after_steps": _clamp_int(raw.get("escalate_after_steps"), default=3, min_value=1, max_value=12),
        "escalate_after_protocol_repairs": _clamp_int(
            raw.get("escalate_after_protocol_repairs"),
            default=2,
            min_value=0,
            max_value=8,
        ),
        "escalate_after_reflect_rounds": _clamp_int(
            raw.get("escalate_after_reflect_rounds"),
            default=2,
            min_value=0,
            max_value=8,
        ),
        "escalate_after_local_failures": _clamp_int(
            raw.get("escalate_after_local_failures"),
            default=1,
            min_value=1,
            max_value=5,
        ),
        "complexity_keywords": _resolve_keyword_list(
            raw.get("complexity_keywords"),
            default=[
                "design",
                "architecture",
                "refactor",
                "optimize",
                "algorithm",
                "benchmark",
                "pipeline",
                "workflow",
                "integrate",
                "multi-step",
                "step by step",
                "调试",
                "架构",
                "优化",
                "多步",
                "工具调用",
            ],
        ),
        "context_keywords": _resolve_keyword_list(
            raw.get("context_keywords"),
            default=[
                "context",
                "history",
                "previous",
                "conversation",
                "memory",
                "document",
                "across",
                "结合",
                "上下文",
                "历史",
                "多轮",
                "全文",
            ],
        ),
    }


def _resolve_skill_router_config(effective_config: dict[str, object]) -> dict[str, object]:
    raw = effective_config.get("skill_router")
    if not isinstance(raw, dict):
        raw = {}
    min_score = raw.get("min_score", 0.9)
    if not isinstance(min_score, (int, float)):
        min_score = 0.9
    max_steps = raw.get("max_steps", 8)
    if not isinstance(max_steps, int):
        max_steps = 8
    return {
        "enabled": bool(raw.get("enabled", True)),
        "min_score": max(0.5, min(0.99, float(min_score))),
        "max_steps": max(1, min(20, max_steps)),
        "semantic_match": bool(raw.get("semantic_match", False)),
    }


def _resolve_auto_subagents_config(effective_config: dict[str, object]) -> dict[str, object]:
    raw = effective_config.get("auto_subagents")
    if not isinstance(raw, dict):
        raw = {}

    complexity_threshold = raw.get("complexity_threshold", 7)
    if not isinstance(complexity_threshold, int):
        complexity_threshold = 7

    min_input_chars = raw.get("min_input_chars", 180)
    if not isinstance(min_input_chars, int):
        min_input_chars = 180

    max_result_chars = raw.get("max_result_chars", 3200)
    if not isinstance(max_result_chars, int):
        max_result_chars = 3200

    child_timeout_seconds = raw.get("child_timeout_seconds", 120.0)
    if isinstance(child_timeout_seconds, bool) or not isinstance(child_timeout_seconds, (int, float)):
        child_timeout_seconds = 120.0

    return {
        "enabled": bool(raw.get("enabled", False)),
        "complexity_threshold": max(1, min(20, complexity_threshold)),
        "min_input_chars": max(0, min(8000, min_input_chars)),
        "trigger_intents": _resolve_intent_list(raw.get("trigger_intents"), default=["complex", "extreme"]),
        "max_result_chars": max(400, min(12000, max_result_chars)),
        "child_timeout_seconds": max(10.0, min(1800.0, float(child_timeout_seconds))),
        "auto_archive_children": bool(raw.get("auto_archive_children", True)),
    }


def _should_trigger_auto_subagents(
    *,
    user_input: str,
    routing_state: dict[str, object],
    auto_subagents_config: dict[str, object],
) -> tuple[bool, str]:
    raw = user_input.strip()
    if not raw:
        return False, "empty_input"

    intent = str(routing_state.get("intent", "simple")).strip().lower() or "simple"
    allowed_intents = auto_subagents_config.get("trigger_intents")
    if isinstance(allowed_intents, list) and intent not in allowed_intents:
        return False, f"intent_filtered:{intent}"

    complexity_score = int(routing_state.get("complexity_score", 0))
    context_score = int(routing_state.get("context_score", 0))
    total_score = complexity_score + context_score
    threshold = int(auto_subagents_config.get("complexity_threshold", 7))
    if total_score < threshold:
        return False, f"score_below_threshold:{total_score}<{threshold}"

    min_chars = int(auto_subagents_config.get("min_input_chars", 180))
    if len(raw) < min_chars:
        return False, f"input_too_short:{len(raw)}<{min_chars}"

    return True, f"intent={intent},score={total_score},len={len(raw)}"


def _build_routing_state(
    *,
    user_input: str,
    memory_entries: list[dict[str, object]],
    loop_config: dict[str, object],
    routing_config: dict[str, object],
) -> dict[str, object]:
    if not bool(routing_config["enabled"]):
        return {
            "enabled": False,
            "intent": "disabled",
            "base_tier": "local_main",
            "base_locality": "local",
            "cloud_tier": "cloud_high",
            "allow_cloud_fallback": True,
            "cloud_escalated": False,
            "escalation_reason": "",
            "local_failures": 0,
            "complexity_score": 0,
            "context_score": 0,
            "escalate_after_steps": 3,
            "escalate_after_protocol_repairs": 2,
            "escalate_after_reflect_rounds": 2,
            "escalate_after_local_failures": 1,
        }

    intent = _classify_routing_intent(
        user_input=user_input,
        memory_entries_count=len(memory_entries),
        loop_enabled=bool(loop_config.get("enabled")),
        routing_config=routing_config,
    )
    return {
        "enabled": True,
        "intent": intent["intent"],
        "base_tier": intent["initial_tier"],
        "base_locality": intent["initial_locality"],
        "cloud_tier": str(routing_config["cloud_tier"]),
        "allow_cloud_fallback": bool(routing_config["allow_cloud_fallback"]),
        "cloud_escalated": bool(intent.get("initial_locality") == "cloud"),
        "escalation_reason": "",
        "local_failures": 0,
        "complexity_score": int(intent["complexity_score"]),
        "context_score": int(intent["context_score"]),
        "escalate_after_steps": int(routing_config["escalate_after_steps"]),
        "escalate_after_protocol_repairs": int(routing_config["escalate_after_protocol_repairs"]),
        "escalate_after_reflect_rounds": int(routing_config["escalate_after_reflect_rounds"]),
        "escalate_after_local_failures": int(routing_config["escalate_after_local_failures"]),
    }


def _classify_routing_intent(
    *,
    user_input: str,
    memory_entries_count: int,
    loop_enabled: bool,
    routing_config: dict[str, object],
) -> dict[str, object]:
    raw = user_input.strip()
    lowered = raw.lower()
    char_count = len(raw)
    word_count = len(re.findall(r"[A-Za-z0-9_]+", raw))

    complexity_hits = _count_keyword_hits(lowered, routing_config["complexity_keywords"])
    context_hits = _count_keyword_hits(lowered, routing_config["context_keywords"])
    chain_markers = bool(
        re.search(r"\b(first|then|finally|after that|step)\b|然后|最后|接着|同时", lowered)
    )

    complexity_score = 0
    context_score = 0
    if char_count >= int(routing_config["moderate_context_chars"]):
        complexity_score += 1
    if char_count >= int(routing_config["high_context_chars"]):
        complexity_score += 2
        context_score += 1
    if word_count >= 80:
        complexity_score += 1
    if complexity_hits >= 3:
        complexity_score += 2
    elif complexity_hits > 0:
        complexity_score += 1
    if context_hits >= 2:
        context_score += 2
    elif context_hits > 0:
        context_score += 1
    if memory_entries_count >= 4:
        context_score += 1
    if chain_markers:
        complexity_score += 1
    if loop_enabled:
        complexity_score += 1

    total_score = complexity_score + context_score
    if total_score <= 2:
        intent = "simple"
        initial_tier = str(routing_config["simple_tier"])
    elif total_score <= 6:
        intent = "standard"
        initial_tier = str(routing_config["standard_tier"])
    else:
        intent = "complex"
        initial_tier = str(routing_config["standard_tier"])

    initial_locality = str(routing_config["prefer_locality"])
    if bool(routing_config["direct_cloud_on_extreme"]) and total_score >= int(routing_config["extreme_score_threshold"]):
        initial_tier = str(routing_config["cloud_tier"])
        initial_locality = "cloud"
        intent = "extreme"

    return {
        "intent": intent,
        "initial_tier": initial_tier,
        "initial_locality": initial_locality,
        "complexity_score": complexity_score,
        "context_score": context_score,
    }


def _normalize_route_tier(value: object, *, default: str) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"local_light", "local_main", "cloud_high"}:
            return lowered
    return default


def _normalize_route_locality(value: object, *, default: str) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"local", "cloud"}:
            return lowered
    return default


def _clamp_int(value: object, *, default: int, min_value: int, max_value: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        return default
    return max(min_value, min(max_value, parsed))


def _resolve_keyword_list(value: object, *, default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(default)
    results: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        keyword = item.strip().lower()
        if not keyword or keyword in seen:
            continue
        seen.add(keyword)
        results.append(keyword)
    return results or list(default)


def _resolve_intent_list(value: object, *, default: list[str]) -> list[str]:
    allowed = {"simple", "standard", "complex", "extreme"}
    if not isinstance(value, list):
        return list(default)
    values: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        intent = item.strip().lower()
        if not intent or intent not in allowed or intent in seen:
            continue
        seen.add(intent)
        values.append(intent)
    return values or list(default)


def _count_keyword_hits(text: str, keywords: object) -> int:
    if not isinstance(keywords, list):
        return 0
    hits = 0
    for item in keywords:
        if not isinstance(item, str):
            continue
        if item and item in text:
            hits += 1
    return hits


def _build_tool_loop_system_prompt(*, loop_config: dict[str, object]) -> str:
    if not bool(loop_config["enabled"]):
        return ""

    tools = list_tools()
    if not tools:
        return ""

    max_tool_calls = int(loop_config["max_tool_calls"])
    tool_lines: list[str] = []
    for item in tools:
        schema = item.get("schema", {})
        required = schema.get("required", []) if isinstance(schema, dict) else []
        tool_lines.append(
            f"- {item['name']} (risk={item['risk']}, required={required if isinstance(required, list) else []})"
        )

    return (
        "You can solve the request via iterative tool use.\n"
        "When you need tools, respond with STRICT JSON only (no markdown):\n"
        '{"action":"tool_call","tool_name":"<name>","arguments":{...},"reflection":"why this step"}\n'
        "When done, respond with STRICT JSON only:\n"
        '{"action":"final","final_response":"<answer>"}\n'
        "If you want to think without tool execution, respond with STRICT JSON:\n"
        '{"action":"reflect","reflection":"next reasoning step"}\n'
        "If your response is rejected for protocol format, resend strict JSON only.\n"
        f"Tool call budget: {max_tool_calls}. Available tools:\n"
        + "\n".join(tool_lines)
    )


async def _maybe_run_skill_sop(
    *,
    core_client: Any,
    user_input: str,
    task_id: str,
    conversation_id: str,
    requester_actor_id: str,
    routing_state: dict[str, object],
    skill_router_config: dict[str, object],
    exec_approvals_override: dict[str, object] | None = None,
    approval_policy_override: dict[str, object] | None = None,
) -> tuple[str | None, str | None]:
    if not bool(skill_router_config.get("enabled", True)):
        return None, None
    try:
        matched = await get_skill_engine().match_blueprint(
            intent_text=user_input,
            min_score=float(skill_router_config.get("min_score", 0.9)),
            allow_semantic=bool(skill_router_config.get("semantic_match", False)),
        )
    except Exception:
        matched = None
    if matched is None:
        append_event(
            RouteEvent(
                conversation_id=conversation_id,
                task_id=task_id,
                ts=now_ts(),
                route="skill_router:miss",
            )
        )
        return None, None

    blueprint = matched.get("blueprint")
    if not isinstance(blueprint, Blueprint):
        return None, None
    blueprint_id = str(matched.get("blueprint_id", "")).strip()
    score = float(matched.get("score", 0.0))
    append_event(
        RouteEvent(
            conversation_id=conversation_id,
            task_id=task_id,
            ts=now_ts(),
            route=f"skill_router:hit:{blueprint.name}:score={score:.3f}",
        )
    )
    result_text, success = await _execute_skill_blueprint(
        core_client=core_client,
        blueprint=blueprint,
        user_input=user_input,
        task_id=task_id,
        conversation_id=conversation_id,
        requester_actor_id=requester_actor_id,
        routing_state=routing_state,
        max_steps=int(skill_router_config.get("max_steps", 8)),
        exec_approvals_override=exec_approvals_override,
        approval_policy_override=approval_policy_override,
    )
    if blueprint_id:
        await get_skill_engine().record_execution_result(blueprint_id, success=success)
    if result_text is None:
        return None, None
    return result_text, blueprint_id


async def _maybe_run_auto_subagents(
    *,
    core_client: Any,
    task_id: str,
    conversation_id: str,
    channel: str,
    channel_id: str | None,
    user_id: str | None,
    thread_id: str | None,
    requester_actor_id: str,
    user_input: str,
    routing_state: dict[str, object],
    effective_config: dict[str, object],
    auto_subagents_config: dict[str, object],
    exec_approvals_override: dict[str, object] | None = None,
    approval_policy_override: dict[str, object] | None = None,
) -> str | None:
    if not bool(auto_subagents_config.get("enabled", False)):
        return None

    should_run, reason = _should_trigger_auto_subagents(
        user_input=user_input,
        routing_state=routing_state,
        auto_subagents_config=auto_subagents_config,
    )
    if not should_run:
        append_event(
            RouteEvent(
                conversation_id=conversation_id,
                task_id=task_id,
                ts=now_ts(),
                route=f"auto_subagents:skip:{reason}",
            )
        )
        return None

    append_event(
        RouteEvent(
            conversation_id=conversation_id,
            task_id=task_id,
            ts=now_ts(),
            route=f"auto_subagents:trigger:{reason}",
        )
    )
    append_event(
        PlanEvent(
            conversation_id=conversation_id,
            task_id=task_id,
            ts=now_ts(),
            plan="auto_subagents flow: blueprint -> executor -> supervisor",
        )
    )

    child_config = _build_subagent_child_config(effective_config)
    timeout_seconds = float(auto_subagents_config.get("child_timeout_seconds", 120.0))
    max_result_chars = int(auto_subagents_config.get("max_result_chars", 3200))
    spawned_children: list[tuple[str, str]] = []

    try:
        blueprint_result = await _run_auto_subagent_role(
            role="blueprint",
            prompt=_build_blueprint_prompt(user_input),
            parent_task_id=task_id,
            parent_conversation_id=conversation_id,
            channel=channel,
            channel_id=channel_id,
            user_id=user_id,
            thread_id=thread_id,
            requester_actor_id=requester_actor_id,
            child_effective_config=child_config,
            timeout_seconds=timeout_seconds,
            core_client=core_client,
            exec_approvals_override=exec_approvals_override,
            approval_policy_override=approval_policy_override,
        )
        spawned_children.append(("blueprint", str(blueprint_result["conversation_id"])))
        blueprint_text = _truncate_text(str(blueprint_result.get("completion_text", "")).strip(), max_chars=max_result_chars)
        if not blueprint_text:
            append_event(
                RouteEvent(
                    conversation_id=conversation_id,
                    task_id=task_id,
                    ts=now_ts(),
                    route="auto_subagents:blueprint_empty",
                )
            )
            return None

        executor_result = await _run_auto_subagent_role(
            role="executor",
            prompt=_build_executor_prompt(user_input=user_input, blueprint_text=blueprint_text),
            parent_task_id=task_id,
            parent_conversation_id=conversation_id,
            channel=channel,
            channel_id=channel_id,
            user_id=user_id,
            thread_id=thread_id,
            requester_actor_id=requester_actor_id,
            child_effective_config=child_config,
            timeout_seconds=timeout_seconds,
            core_client=core_client,
            exec_approvals_override=exec_approvals_override,
            approval_policy_override=approval_policy_override,
        )
        spawned_children.append(("executor", str(executor_result["conversation_id"])))
        executor_text = _truncate_text(str(executor_result.get("completion_text", "")).strip(), max_chars=max_result_chars)
        if not executor_text:
            append_event(
                RouteEvent(
                    conversation_id=conversation_id,
                    task_id=task_id,
                    ts=now_ts(),
                    route="auto_subagents:executor_empty",
                )
            )
            return None

        supervisor_result = await _run_auto_subagent_role(
            role="supervisor",
            prompt=_build_supervisor_prompt(
                user_input=user_input,
                blueprint_text=blueprint_text,
                executor_text=executor_text,
            ),
            parent_task_id=task_id,
            parent_conversation_id=conversation_id,
            channel=channel,
            channel_id=channel_id,
            user_id=user_id,
            thread_id=thread_id,
            requester_actor_id=requester_actor_id,
            child_effective_config=child_config,
            timeout_seconds=timeout_seconds,
            core_client=core_client,
            exec_approvals_override=exec_approvals_override,
            approval_policy_override=approval_policy_override,
        )
        spawned_children.append(("supervisor", str(supervisor_result["conversation_id"])))
        supervisor_text = _truncate_text(str(supervisor_result.get("completion_text", "")).strip(), max_chars=max_result_chars)

        final_text = supervisor_text or executor_text or blueprint_text
        if not final_text:
            return None
        append_event(
            RouteEvent(
                conversation_id=conversation_id,
                task_id=task_id,
                ts=now_ts(),
                route="auto_subagents:completed",
            )
        )
        return final_text
    except Exception as exc:
        append_event(
            RouteEvent(
                conversation_id=conversation_id,
                task_id=task_id,
                ts=now_ts(),
                route=f"auto_subagents:error:{type(exc).__name__}",
            )
        )
        return None
    finally:
        if bool(auto_subagents_config.get("auto_archive_children", True)):
            for role, child_conversation_id in spawned_children:
                archived = mark_session_archived(child_conversation_id)
                archive_status = "ok" if archived is not None else "missing"
                append_event(
                    RouteEvent(
                        conversation_id=conversation_id,
                        task_id=task_id,
                        ts=now_ts(),
                        route=f"auto_subagents:recalled:{role}:{child_conversation_id}:{archive_status}",
                    )
                )


async def _run_auto_subagent_role(
    *,
    role: str,
    prompt: str,
    parent_task_id: str,
    parent_conversation_id: str,
    channel: str,
    channel_id: str | None,
    user_id: str | None,
    thread_id: str | None,
    requester_actor_id: str,
    child_effective_config: dict[str, object],
    timeout_seconds: float,
    core_client: Any,
    exec_approvals_override: dict[str, object] | None = None,
    approval_policy_override: dict[str, object] | None = None,
) -> dict[str, object]:
    child_conversation_id = _spawn_subagent_conversation(
        parent_conversation_id=parent_conversation_id,
        role=role,
        channel=channel,
        channel_id=channel_id,
        user_id=user_id,
        thread_id=thread_id,
        actor_id=requester_actor_id,
    )
    append_event(
        RouteEvent(
            conversation_id=parent_conversation_id,
            task_id=parent_task_id,
            ts=now_ts(),
            route=f"auto_subagents:spawn:{role}:{child_conversation_id}",
        )
    )

    child_task = create_task(conversation_id=child_conversation_id, status="queued", stage="plan")
    append_event(
        InputEvent(
            conversation_id=child_conversation_id,
            task_id=child_task.id,
            ts=now_ts(),
            actor_id=f"subagent::{role}",
            message=prompt,
        )
    )

    timeout_happened = False
    try:
        await asyncio.wait_for(
            process_task(
                child_task.id,
                effective_config_override=child_effective_config,
                core_client_override=core_client,
                exec_approvals_override=exec_approvals_override,
                approval_policy_override=approval_policy_override,
            ),
            timeout=max(10.0, timeout_seconds),
        )
    except asyncio.TimeoutError:
        timeout_happened = True
        update_task_status(
            task_id=child_task.id,
            status="failed",
            stage="answer",
            last_error=f"auto_subagents timeout for role={role}",
        )
        append_event(
            RouteEvent(
                conversation_id=parent_conversation_id,
                task_id=parent_task_id,
                ts=now_ts(),
                route=f"auto_subagents:timeout:{role}:{child_conversation_id}",
            )
        )
    except Exception as exc:
        update_task_status(
            task_id=child_task.id,
            status="failed",
            stage="answer",
            last_error=f"auto_subagents role={role} failed: {exc}",
        )
        append_event(
            RouteEvent(
                conversation_id=parent_conversation_id,
                task_id=parent_task_id,
                ts=now_ts(),
                route=f"auto_subagents:role_error:{role}:{type(exc).__name__}",
            )
        )

    child_task_state = get_task(child_task.id)
    child_status = str(child_task_state.status) if child_task_state is not None else ("failed" if timeout_happened else "unknown")
    child_last_error = str(child_task_state.last_error) if child_task_state and child_task_state.last_error else None
    completion_text = _extract_completion_text_for_task(conversation_id=child_conversation_id, task_id=child_task.id)
    append_event(
        RouteEvent(
            conversation_id=parent_conversation_id,
            task_id=parent_task_id,
            ts=now_ts(),
            route=f"auto_subagents:role_done:{role}:{child_status}",
        )
    )
    return {
        "role": role,
        "conversation_id": child_conversation_id,
        "task_id": child_task.id,
        "status": child_status,
        "last_error": child_last_error,
        "completion_text": completion_text,
    }


def _build_subagent_child_config(effective_config: dict[str, object]) -> dict[str, object]:
    child = copy.deepcopy(effective_config)

    raw_auto = child.get("auto_subagents")
    auto_cfg = dict(raw_auto) if isinstance(raw_auto, dict) else {}
    auto_cfg["enabled"] = False
    child["auto_subagents"] = auto_cfg

    raw_loop = child.get("tool_loop")
    loop_cfg = dict(raw_loop) if isinstance(raw_loop, dict) else {}
    loop_cfg["enabled"] = False
    child["tool_loop"] = loop_cfg

    raw_skill_router = child.get("skill_router")
    skill_cfg = dict(raw_skill_router) if isinstance(raw_skill_router, dict) else {}
    skill_cfg["enabled"] = False
    child["skill_router"] = skill_cfg

    return child


def _spawn_subagent_conversation(
    *,
    parent_conversation_id: str,
    role: str,
    channel: str,
    channel_id: str | None,
    user_id: str | None,
    thread_id: str | None,
    actor_id: str,
) -> str:
    role_name = _safe_subagent_name(role)
    child_conversation_id = f"subagent:{parent_conversation_id}:{role_name}:{uuid4().hex[:8]}"
    upsert_conversation(
        conversation_id=child_conversation_id,
        channel=channel,
        channel_id=channel_id,
        user_id=user_id,
        thread_id=thread_id,
    )
    ensure_session_workspace(conversation_id=child_conversation_id, actor_id=actor_id)
    return child_conversation_id


def _extract_completion_text_for_task(*, conversation_id: str, task_id: str) -> str:
    events = query_events(conversation_id=conversation_id, task_id=task_id)
    for event in reversed(events):
        if event.type != "CompletionEvent":
            continue
        try:
            payload = json.loads(event.payload_json)
        except json.JSONDecodeError:
            continue
        text = str(payload.get("response_text", "")).strip()
        if text:
            return text
    return ""


def _build_blueprint_prompt(user_input: str) -> str:
    return (
        "ROLE: blueprint\n"
        "你是规划分身。你的职责是把需求拆解为可执行蓝图，不做实现。\n"
        "请输出：目标、里程碑、验收标准、关键风险。\n\n"
        f"用户任务：\n{user_input}"
    )


def _build_executor_prompt(*, user_input: str, blueprint_text: str) -> str:
    return (
        "ROLE: executor\n"
        "你是执行分身。请严格按照蓝图完成任务，并给出执行结果与证据。\n"
        "请输出：执行摘要、关键步骤、证据、未完成项。\n\n"
        f"用户任务：\n{user_input}\n\n"
        f"蓝图输入：\n{blueprint_text}"
    )


def _build_supervisor_prompt(*, user_input: str, blueprint_text: str, executor_text: str) -> str:
    return (
        "ROLE: supervisor\n"
        "你是监督分身。请对蓝图与执行结果做一致性审查，指出缺口并给出最终答复。\n"
        "如果证据不足，请明确说明不确定点。\n"
        "请输出用户可直接阅读的最终结果。\n\n"
        f"用户任务：\n{user_input}\n\n"
        f"蓝图：\n{blueprint_text}\n\n"
        f"执行结果：\n{executor_text}"
    )


def _safe_subagent_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.").lower()
    return cleaned or "worker"


def _truncate_text(text: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


async def _execute_skill_blueprint(
    *,
    core_client: Any,
    blueprint: Blueprint,
    user_input: str,
    task_id: str,
    conversation_id: str,
    requester_actor_id: str,
    routing_state: dict[str, object],
    max_steps: int,
    exec_approvals_override: dict[str, object] | None = None,
    approval_policy_override: dict[str, object] | None = None,
) -> tuple[str | None, bool]:
    digests: list[dict[str, object]] = []
    steps = list(blueprint.steps)[: max(1, max_steps)]
    for index, step in enumerate(steps):
        tool_name = str(step.get("tool_name", "")).strip()
        if not tool_name:
            return None, False
        spec = get_tool_spec(tool_name)
        step_risk = str(step.get("risk", spec.risk if spec is not None else "unknown"))
        if _permission_level_from_risk(step_risk) > int(blueprint.min_permission_level):
            append_event(
                RouteEvent(
                    conversation_id=conversation_id,
                    task_id=task_id,
                    ts=now_ts(),
                    route=f"skill_router:blocked_permission:{tool_name}",
                )
            )
            return None, False

        raw_args = step.get("arguments", {})
        args = _render_blueprint_args(raw_args, user_input=user_input)
        if not isinstance(args, dict):
            args = {}
        outcome = _execute_tool_in_loop(
            task_id=task_id,
            conversation_id=conversation_id,
            requester_actor_id=requester_actor_id,
            tool_name=tool_name,
            args=args,
            exec_approvals_override=exec_approvals_override,
            approval_policy_override=approval_policy_override,
        )
        append_event(
            RouteEvent(
                conversation_id=conversation_id,
                task_id=task_id,
                ts=now_ts(),
                route=f"skill_router:step:{index + 1}:{tool_name}:{outcome.get('status', 'unknown')}",
            )
        )
        if str(outcome.get("status")) == "pending_approval":
            approval_id = str(outcome.get("approval_id", "")).strip()
            reason = str(outcome.get("policy_reason", "")).strip()
            message = (
                f"命中技能路径 `{blueprint.name}`，但步骤 `{tool_name}` 需要审批（{approval_id}）。"
                f"原因：{reason}。"
            )
            return message, False
        if str(outcome.get("status")) != "ok":
            return None, False
        digests.append(_tool_outcome_digest(tool_name=tool_name, outcome=outcome))

    route_tier, preferred_locality, allow_cloud_fallback = _route_hints_from_state(routing_state)
    digest_text = json.dumps(digests, ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": "You are operating in fast SOP mode. Produce final answer from structured tool digests only.",
        },
        {
            "role": "user",
            "content": (
                f"User request:\n{user_input}\n\n"
                f"SOP name: {blueprint.name}\n"
                f"Tool digests (no raw third-party content):\n{digest_text}\n\n"
                "Return the final answer for the user."
            ),
        },
    ]
    try:
        response = await _invoke_chat_completions_with_route(
            core_client=core_client,
            messages=messages,
            route_tier=route_tier,
            preferred_locality=preferred_locality,
            allow_cloud_fallback=allow_cloud_fallback,
        )
        text_value = (
            response.get("choices", [{}])[0].get("message", {}).get("content", "").strip() if isinstance(response, dict) else ""
        )
        if text_value:
            return text_value, True
    except Exception:
        pass
    return "已按历史成功路径执行完成。", True


async def _run_agent_loop(
    *,
    core_client: Any,
    messages: list[dict[str, str]],
    task_id: str,
    conversation_id: str,
    requester_actor_id: str,
    loop_config: dict[str, object],
    routing_state: dict[str, object],
    exec_approvals_override: dict[str, object] | None = None,
    approval_policy_override: dict[str, object] | None = None,
) -> str:
    max_steps = int(loop_config["max_steps"])
    max_tool_calls = int(loop_config["max_tool_calls"])
    max_observation_chars = int(loop_config["max_observation_chars"])
    strict_json_actions = bool(loop_config["strict_json_actions"])
    max_protocol_repairs = int(loop_config["max_protocol_repairs"])

    working_messages = list(messages)
    tool_calls_used = 0
    latest_assistant_text = ""
    protocol_repairs_used = 0
    expecting_protocol_json = False
    consecutive_reflects = 0

    for step in range(1, max_steps + 1):
        escalation_reason = _maybe_escalate_routing_state(
            routing_state=routing_state,
            step=step,
            protocol_repairs_used=protocol_repairs_used,
            consecutive_reflects=consecutive_reflects,
        )
        if escalation_reason:
            append_event(
                RouteEvent(
                    conversation_id=conversation_id,
                    task_id=task_id,
                    ts=now_ts(),
                    route=f"core:routing_escalation:{escalation_reason}",
                )
            )
        route_tier, preferred_locality, allow_cloud_fallback = _route_hints_from_state(routing_state)
        append_event(
            RouteEvent(
                conversation_id=conversation_id,
                task_id=task_id,
                ts=now_ts(),
                route=(
                    f"core:/v1/chat/completions#step-{step}:tier={route_tier}:"
                    f"locality={preferred_locality}:cloud_fallback={allow_cloud_fallback}"
                ),
            )
        )
        response = await _request_core_chat_completion(
            core_client=core_client,
            messages=working_messages,
            route_tier=route_tier,
            preferred_locality=preferred_locality,
            allow_cloud_fallback=allow_cloud_fallback,
            routing_state=routing_state,
            conversation_id=conversation_id,
            task_id=task_id,
        )
        _emit_provider_route_event(response=response, conversation_id=conversation_id, task_id=task_id)
        assistant_text = (
            response.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or "[empty-response]"
        )
        latest_assistant_text = assistant_text
        visible_assistant_text = _sanitize_visible_assistant_text(assistant_text)

        directive = _parse_loop_directive(assistant_text)
        if directive is None:
            if not bool(loop_config["enabled"]):
                return visible_assistant_text or assistant_text
            should_attempt_protocol_repair = (
                strict_json_actions
                and protocol_repairs_used < max_protocol_repairs
                and (expecting_protocol_json or _looks_like_protocol_attempt(assistant_text))
            )
            if should_attempt_protocol_repair:
                protocol_repairs_used += 1
                append_event(
                    RouteEvent(
                        conversation_id=conversation_id,
                        task_id=task_id,
                        ts=now_ts(),
                        route=f"core:protocol_repair:{protocol_repairs_used}",
                    )
                )
                working_messages.append({"role": "assistant", "content": _sanitize_assistant_for_history(assistant_text)})
                working_messages.append(
                    {
                        "role": "user",
                        "content": _build_protocol_repair_prompt(
                            raw_response=assistant_text,
                            remaining_repairs=max_protocol_repairs - protocol_repairs_used,
                            reason="response is not valid JSON action protocol",
                        ),
                    }
                )
                expecting_protocol_json = True
                continue
            return visible_assistant_text or assistant_text

        action = str(directive.get("action", "")).strip().lower()
        reflection = str(directive.get("reflection", "")).strip()
        expecting_protocol_json = False
        if reflection:
            append_event(
                PlanEvent(
                    conversation_id=conversation_id,
                    task_id=task_id,
                    ts=now_ts(),
                    plan=f"loop_step={step} reflection={reflection[:240]}",
                )
            )

        if action == "final":
            final_response = str(directive.get("final_response", "")).strip()
            return final_response or visible_assistant_text or assistant_text

        if action == "reflect":
            consecutive_reflects += 1
            working_messages.append({"role": "assistant", "content": _sanitize_assistant_for_history(assistant_text)})
            working_messages.append(
                {
                    "role": "user",
                    "content": "Continue reasoning. Return strict JSON action only.",
                }
            )
            expecting_protocol_json = True
            continue

        consecutive_reflects = 0
        if action != "tool_call":
            return visible_assistant_text or assistant_text
        if tool_calls_used >= max_tool_calls:
            return "已达到本轮工具调用上限，请缩小问题范围后重试。"

        tool_name = str(directive.get("tool_name", "")).strip()
        arguments = directive.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        if not tool_name:
            should_attempt_protocol_repair = strict_json_actions and protocol_repairs_used < max_protocol_repairs
            if should_attempt_protocol_repair:
                protocol_repairs_used += 1
                append_event(
                    RouteEvent(
                        conversation_id=conversation_id,
                        task_id=task_id,
                        ts=now_ts(),
                        route=f"core:protocol_repair:{protocol_repairs_used}:missing_tool_name",
                    )
                )
                working_messages.append({"role": "assistant", "content": _sanitize_assistant_for_history(assistant_text)})
                working_messages.append(
                    {
                        "role": "user",
                        "content": _build_protocol_repair_prompt(
                            raw_response=assistant_text,
                            remaining_repairs=max_protocol_repairs - protocol_repairs_used,
                            reason="tool_call action missing tool_name",
                        ),
                    }
                )
                expecting_protocol_json = True
                continue
            return visible_assistant_text or "模型返回了无效的工具调用指令。"
        tool_calls_used += 1

        tool_outcome = _execute_tool_in_loop(
            task_id=task_id,
            conversation_id=conversation_id,
            requester_actor_id=requester_actor_id,
            tool_name=tool_name,
            args=arguments,
            exec_approvals_override=exec_approvals_override,
            approval_policy_override=approval_policy_override,
        )
        append_event(
            RouteEvent(
                conversation_id=conversation_id,
                task_id=task_id,
                ts=now_ts(),
                route=f"tool:{tool_name}:{tool_outcome['status']}",
            )
        )

        if tool_outcome["status"] == "pending_approval":
            approval_id = str(tool_outcome.get("approval_id", "")).strip()
            policy_reason = str(tool_outcome.get("policy_reason", "")).strip()
            return (
                f"我已识别需要调用 `{tool_name}`，并创建审批请求 `{approval_id}`。"
                f"审批原因：{policy_reason}。请审批后重试本任务。"
            )

        observation = json.dumps(tool_outcome, ensure_ascii=False)
        if len(observation) > max_observation_chars:
            observation = observation[:max_observation_chars] + "...(truncated)"
        working_messages.append({"role": "assistant", "content": _sanitize_assistant_for_history(assistant_text)})
        working_messages.append(
            {
                "role": "user",
                "content": (
                    f"TOOL_RESULT {tool_name} => {observation}\n"
                    "Reflect and decide next step. Return strict JSON action only."
                ),
            }
        )
        expecting_protocol_json = True

    return _sanitize_visible_assistant_text(latest_assistant_text) or latest_assistant_text or "任务已完成。"


async def _request_core_chat_completion(
    *,
    core_client: Any,
    messages: list[dict[str, str]],
    route_tier: str,
    preferred_locality: str,
    allow_cloud_fallback: bool,
    routing_state: dict[str, object],
    conversation_id: str,
    task_id: str,
) -> dict[str, object]:
    try:
        return await _invoke_chat_completions_with_route(
            core_client=core_client,
            messages=messages,
            route_tier=route_tier,
            preferred_locality=preferred_locality,
            allow_cloud_fallback=allow_cloud_fallback,
        )
    except Exception as exc:
        local_failure_threshold = int(routing_state.get("escalate_after_local_failures", 1))
        if not bool(routing_state.get("enabled")) or bool(routing_state.get("cloud_escalated")):
            raise
        routing_state["local_failures"] = int(routing_state.get("local_failures", 0)) + 1
        if int(routing_state["local_failures"]) < max(1, local_failure_threshold):
            raise

        _escalate_routing_state(
            routing_state=routing_state,
            reason=f"local_failure:{type(exc).__name__}",
        )
        append_event(
            RouteEvent(
                conversation_id=conversation_id,
                task_id=task_id,
                ts=now_ts(),
                route=f"core:routing_escalation:local_failure:{type(exc).__name__}",
            )
        )
        cloud_tier, cloud_locality, cloud_fallback = _route_hints_from_state(routing_state)
        return await _invoke_chat_completions_with_route(
            core_client=core_client,
            messages=messages,
            route_tier=cloud_tier,
            preferred_locality=cloud_locality,
            allow_cloud_fallback=cloud_fallback,
        )


async def _invoke_chat_completions_with_route(
    *,
    core_client: Any,
    messages: list[dict[str, str]],
    route_tier: str,
    preferred_locality: str,
    allow_cloud_fallback: bool,
) -> dict[str, object]:
    try:
        return await core_client.chat_completions(
            messages=messages,
            stream=False,
            route_tier=route_tier,
            preferred_locality=preferred_locality,
            allow_cloud_fallback=allow_cloud_fallback,
        )
    except TypeError as exc:
        error_text = str(exc)
        unsupported_kwargs = (
            "unexpected keyword argument" in error_text
            or "got an unexpected keyword argument" in error_text
        )
        if not unsupported_kwargs:
            raise
        return await core_client.chat_completions(messages=messages, stream=False)


def _route_hints_from_state(routing_state: dict[str, object]) -> tuple[str, str, bool]:
    if not bool(routing_state.get("enabled")):
        return (
            str(routing_state.get("base_tier", "local_main")),
            str(routing_state.get("base_locality", "local")),
            bool(routing_state.get("allow_cloud_fallback", True)),
        )
    if bool(routing_state.get("cloud_escalated")):
        return (
            str(routing_state.get("cloud_tier", "cloud_high")),
            "cloud",
            bool(routing_state.get("allow_cloud_fallback", True)),
        )
    return (
        str(routing_state.get("base_tier", "local_main")),
        str(routing_state.get("base_locality", "local")),
        False,
    )


def _maybe_escalate_routing_state(
    *,
    routing_state: dict[str, object],
    step: int,
    protocol_repairs_used: int,
    consecutive_reflects: int,
) -> str | None:
    if not bool(routing_state.get("enabled")) or bool(routing_state.get("cloud_escalated")):
        return None

    step_threshold = int(routing_state.get("escalate_after_steps", 3))
    repair_threshold = int(routing_state.get("escalate_after_protocol_repairs", 2))
    reflect_threshold = int(routing_state.get("escalate_after_reflect_rounds", 2))

    if step > max(1, step_threshold):
        return _escalate_routing_state(routing_state=routing_state, reason=f"multi_round_timeout:step_{step}")
    if repair_threshold > 0 and protocol_repairs_used >= repair_threshold:
        return _escalate_routing_state(
            routing_state=routing_state,
            reason=f"protocol_repairs:{protocol_repairs_used}",
        )
    if reflect_threshold > 0 and consecutive_reflects >= reflect_threshold:
        return _escalate_routing_state(
            routing_state=routing_state,
            reason=f"reflect_stall:{consecutive_reflects}",
        )
    return None


def _escalate_routing_state(*, routing_state: dict[str, object], reason: str) -> str:
    routing_state["cloud_escalated"] = True
    routing_state["escalation_reason"] = reason
    return reason


def _emit_provider_route_event(*, response: dict[str, object], conversation_id: str, task_id: str) -> None:
    provider = str(response.get("_provider", "")).strip()
    if not provider:
        return
    tier = str(response.get("_provider_tier", "")).strip() or "unknown"
    locality = str(response.get("_provider_locality", "")).strip() or "unknown"
    auth_mode = str(response.get("_provider_auth_mode", "")).strip() or "unknown"
    model = str(response.get("_provider_model", "")).strip() or "unknown"
    append_event(
        RouteEvent(
            conversation_id=conversation_id,
            task_id=task_id,
            ts=now_ts(),
            route=f"core_provider:{provider}:tier={tier}:locality={locality}:auth={auth_mode}:model={model}",
        )
    )


def _parse_loop_directive(text: str) -> dict[str, object] | None:
    raw = text.strip()
    if not raw:
        return None

    candidates = _collect_json_object_candidates(raw)

    for candidate in candidates:
        parsed: Any
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue

        action = _normalize_loop_action(parsed.get("action") or parsed.get("type"))
        if action not in {"final", "tool_call", "reflect"}:
            continue

        tool_name = str(parsed.get("tool_name") or parsed.get("tool") or "").strip()
        arguments = _decode_tool_arguments(parsed.get("arguments") if parsed.get("arguments") is not None else parsed.get("args", {}))

        final_response = str(parsed.get("final_response") or parsed.get("content") or "").strip()
        reflection = str(parsed.get("reflection") or parsed.get("reason") or "").strip()
        return {
            "action": action,
            "tool_name": tool_name,
            "arguments": arguments,
            "final_response": final_response,
            "reflection": reflection,
        }

    downgraded_tool_call = _extract_tool_call_from_downgraded_text(raw)
    if downgraded_tool_call is not None:
        return downgraded_tool_call
    return None


def _collect_json_object_candidates(raw: str) -> list[str]:
    candidates: list[str] = [raw]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE))
    cursor = 0
    while cursor < len(raw):
        extracted = _extract_first_json_object(raw, start_index=cursor)
        if extracted is None:
            break
        block, next_cursor = extracted
        candidates.append(block)
        cursor = next_cursor

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        item = candidate.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _extract_first_json_object(text: str, *, start_index: int = 0) -> tuple[str, int] | None:
    start = text.find("{", max(0, start_index))
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1], idx + 1
        start = text.find("{", start + 1)
    return None


def _normalize_loop_action(value: object) -> str:
    action = str(value or "").strip().lower()
    if action in {"done", "answer"}:
        return "final"
    if action in {"tool", "call_tool"}:
        return "tool_call"
    if action in {"think"}:
        return "reflect"
    return action


def _decode_tool_arguments(value: object) -> dict[str, object]:
    arguments = value
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
            arguments = decoded if isinstance(decoded, dict) else {}
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        return {}
    return dict(arguments)


def _extract_tool_call_from_downgraded_text(raw: str) -> dict[str, object] | None:
    marker = re.search(r"\[Tool\s+Call:\s*([^\]\n]+)\]", raw, flags=re.IGNORECASE)
    if marker is None:
        return None
    tool_label = marker.group(1).strip()
    tool_name = re.sub(r"\s*\(ID:.*$", "", tool_label, flags=re.IGNORECASE).strip()
    if not tool_name:
        return None

    arguments: dict[str, object] = {}
    args_anchor = re.search(r"Arguments\s*:", raw, flags=re.IGNORECASE)
    if args_anchor is not None:
        extracted = _extract_first_json_object(raw, start_index=args_anchor.end())
        if extracted is not None:
            candidate, _ = extracted
            try:
                decoded = json.loads(candidate)
                if isinstance(decoded, dict):
                    arguments = decoded
            except json.JSONDecodeError:
                arguments = {}

    return {
        "action": "tool_call",
        "tool_name": tool_name,
        "arguments": arguments,
        "final_response": "",
        "reflection": "Recovered tool call from downgraded text output.",
    }


def _looks_like_protocol_attempt(text: str) -> bool:
    raw = text.strip()
    if not raw:
        return False
    lowered = raw.lower()
    markers = (
        "```json",
        '"action"',
        "action:",
        "tool_call",
        "final_response",
        "[tool call:",
        "arguments:",
        "<invoke",
        "minimax:tool_call",
    )
    return raw.startswith("{") or any(marker in lowered for marker in markers)


def _sanitize_assistant_for_history(text: str, max_chars: int = 4000) -> str:
    cleaned = _strip_protocol_leak_text(text).strip()
    if not cleaned:
        cleaned = "[assistant-response-empty-after-sanitize]"
    return cleaned[:max_chars]


def _sanitize_visible_assistant_text(text: str) -> str:
    return _strip_protocol_leak_text(text).strip()


def _strip_protocol_leak_text(text: str) -> str:
    raw = text or ""
    if not raw:
        return raw

    cleaned = raw
    if "minimax:tool_call" in cleaned.lower() or "<invoke" in cleaned.lower():
        cleaned = re.sub(r"<invoke\b[^>]*>[\s\S]*?</invoke>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?minimax:tool_call>", "", cleaned, flags=re.IGNORECASE)

    cleaned = _strip_downgraded_tool_call_text(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _strip_downgraded_tool_call_text(text: str) -> str:
    marker_re = re.compile(r"\[Tool\s+Call:[^\]]*\]", flags=re.IGNORECASE)
    result: list[str] = []
    cursor = 0
    for match in marker_re.finditer(text):
        start = match.start()
        if start < cursor:
            continue
        result.append(text[cursor:start])
        idx = match.end()
        while idx < len(text) and text[idx] in {" ", "\t", "\r", "\n"}:
            idx += 1
        if text[idx : idx + 9].lower() == "arguments":
            idx += 9
            if idx < len(text) and text[idx] == ":":
                idx += 1
            while idx < len(text) and text[idx] in {" ", "\t", "\r", "\n"}:
                idx += 1
            extracted = _extract_first_json_object(text, start_index=idx)
            if extracted is not None:
                _, idx = extracted
            else:
                while idx < len(text) and text[idx] not in {"\r", "\n"}:
                    idx += 1
        cursor = idx
    result.append(text[cursor:])
    cleaned = "".join(result)
    cleaned = re.sub(
        r"\[Tool\s+Result\s+for\s+ID[^\]]*\][\s\S]*?(?=\n\s*\[Tool|\n\s*\[Historical\s+context:|$)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\[Historical\s+context:[^\]]*\]\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _build_protocol_repair_prompt(*, raw_response: str, remaining_repairs: int, reason: str) -> str:
    snippet = _sanitize_visible_assistant_text(raw_response) or raw_response.strip()
    if len(snippet) > 240:
        snippet = snippet[:240] + "...(truncated)"
    return (
        "Your previous response violated the JSON action protocol.\n"
        f"Reason: {reason}.\n"
        "Resend exactly ONE JSON object (no markdown, no prose) using one of:\n"
        '{"action":"tool_call","tool_name":"<name>","arguments":{...},"reflection":"..."}\n'
        '{"action":"reflect","reflection":"..."}\n'
        '{"action":"final","final_response":"..."}\n'
        f"Remaining protocol-repair retries for this task: {max(0, remaining_repairs)}.\n"
        f"Previous response excerpt: {snippet}"
    )


def _render_blueprint_args(value: object, *, user_input: str) -> object:
    if isinstance(value, dict):
        return {str(key): _render_blueprint_args(item, user_input=user_input) for key, item in value.items()}
    if isinstance(value, list):
        return [_render_blueprint_args(item, user_input=user_input) for item in value]
    if isinstance(value, str):
        return value.replace("{{user_input}}", user_input)
    return value


def _permission_level_from_risk(risk: str) -> int:
    return 1 if risk.strip().lower() == "read_only" else 2


def _tool_outcome_digest(*, tool_name: str, outcome: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {
        "tool": tool_name,
        "status": str(outcome.get("status", "")),
    }
    result = outcome.get("result")
    if isinstance(result, dict):
        payload["result_shape"] = {"type": "object", "keys": sorted([str(key) for key in result.keys()])[:24]}
    elif isinstance(result, list):
        payload["result_shape"] = {"type": "array", "length": len(result)}
    elif result is None:
        payload["result_shape"] = {"type": "null"}
    else:
        payload["result_shape"] = {"type": type(result).__name__}
    if outcome.get("approval_id"):
        payload["approval_id"] = str(outcome.get("approval_id"))
    return payload


def _execute_tool_in_loop(
    *,
    task_id: str,
    conversation_id: str,
    requester_actor_id: str,
    tool_name: str,
    args: dict[str, object],
    exec_approvals_override: dict[str, object] | None = None,
    approval_policy_override: dict[str, object] | None = None,
) -> dict[str, object]:
    spec = get_tool_spec(tool_name)
    if spec is None:
        return {"status": "error", "error": f"Tool not found: {tool_name}"}

    resolved_args: dict[str, object] = dict(args)
    if spec.version.startswith("ipc-"):
        workspace = get_session_workspace(conversation_id)
        if workspace is not None and workspace.workspace_path:
            resolved_args["__marv_session_workspace"] = workspace.workspace_path

    perm_cfg = exec_approvals_override if isinstance(exec_approvals_override, dict) else load_exec_approvals()
    perm = evaluate_tool_permission(perm_cfg, actor_id=requester_actor_id, tool_name=tool_name)
    if perm["decision"] == "deny":
        return {
            "status": "blocked",
            "error": f"Tool blocked by exec policy: {perm['reason']}",
            "policy_reason": str(perm["reason"]),
        }

    grant = find_matching_approval_grant(
        actor_id=requester_actor_id,
        tool_name=tool_name,
        session_id=conversation_id,
    )
    approval_policy = approval_policy_override if isinstance(approval_policy_override, dict) else load_approval_policy()
    require_approval, policy_reason = decide_approval_mode(
        policy=approval_policy,
        tool_risk=spec.risk,
        policy_decision=str(perm["decision"]),
    )
    if str(approval_policy.get("mode", "policy")).strip().lower() == "policy" and str(perm["decision"]) == "ask":
        policy_reason = str(perm["reason"])

    # Keep autonomous execution safe: risky tools are always gated by approval flow.
    if spec.risk != "read_only":
        require_approval = True
        if "external_write" in spec.risk:
            policy_reason = "autonomous_loop requires approval for external_write"
        elif "approval_mode" not in policy_reason:
            policy_reason = f"autonomous_loop requires approval for risk={spec.risk}"

    if grant is not None and spec.risk == "read_only":
        require_approval = False
        policy_reason = f"approval_grant:{grant.grant_id}"

    if require_approval:
        approval = create_approval(
            approval_type="tool_execute",
            summary=f"{tool_name} requires approval ({policy_reason})",
            actor_id=requester_actor_id,
            constraints={
                "tool": tool_name,
                "one_time": True,
                "policy_reason": policy_reason,
                "requester_actor_id": requester_actor_id,
                "session_id": conversation_id,
            },
        )
        tool_call = create_tool_call(
            task_id=task_id,
            tool_name=tool_name,
            args=resolved_args,
            status="pending_approval",
            approval_id=approval.approval_id,
        )
        return {
            "status": "pending_approval",
            "approval_id": approval.approval_id,
            "tool_call_id": tool_call.tool_call_id,
            "policy_reason": policy_reason,
        }

    try:
        tool_call = execute_tool_call(
            task_id=task_id,
            tool_name=tool_name,
            args=resolved_args,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    return {
        "status": tool_call.status,
        "tool_call_id": tool_call.tool_call_id,
        "result": json.loads(tool_call.result_json) if tool_call.result_json else None,
        "error": tool_call.error,
        "approval_grant_id": grant.grant_id if grant is not None else None,
    }


def _resolve_memory_runtime_config(effective_config: dict[str, object]) -> dict[str, object]:
    memory = effective_config.get("memory")
    if not isinstance(memory, dict):
        memory = {}

    strictness = str(memory.get("strictness", "balanced")).strip().lower() or "balanced"
    min_score_default = {"strict": 0.42, "balanced": 0.27, "loose": 0.18}.get(strictness, 0.27)

    top_k = memory.get("top_k", 5)
    if not isinstance(top_k, int):
        top_k = 5

    max_turn = memory.get("max_memories_per_turn", 6)
    if not isinstance(max_turn, int):
        max_turn = 6

    min_score = memory.get("min_score", min_score_default)
    if not isinstance(min_score, (float, int)):
        min_score = min_score_default

    ttl_days = memory.get("ttl_days", 365)
    if not isinstance(ttl_days, int):
        ttl_days = 365

    half_life_days = memory.get("half_life_days", 120)
    if not isinstance(half_life_days, int):
        half_life_days = 120

    always_include_user_memory = bool(memory.get("always_include_user_memory", True))
    compaction_enabled = bool(memory.get("compaction_enabled", True))

    compaction_threshold_turns = memory.get("compaction_threshold_turns", 12)
    if not isinstance(compaction_threshold_turns, int):
        compaction_threshold_turns = 12

    compaction_window_turns = memory.get("compaction_window_turns", 8)
    if not isinstance(compaction_window_turns, int):
        compaction_window_turns = 8

    return {
        "strictness": strictness,
        "top_k": max(1, min(16, top_k)),
        "max_memories_per_turn": max(1, min(20, max_turn)),
        "min_score": max(0.0, min(1.5, float(min_score))),
        "ttl_days": max(0, min(3650, ttl_days)),
        "half_life_days": max(1, min(3650, half_life_days)),
        "always_include_user_memory": always_include_user_memory,
        "compaction_enabled": compaction_enabled,
        "compaction_threshold_turns": max(2, min(100, compaction_threshold_turns)),
        "compaction_window_turns": max(2, min(40, compaction_window_turns)),
    }


async def _load_runtime_memories(
    *,
    conversation_id: str,
    channel: str,
    channel_id: str | None,
    user_id: str | None,
    query_text: str,
    config: dict[str, object],
) -> tuple[list[dict[str, object]], list[tuple[str, str, float]], int]:
    start = time.perf_counter()
    if not query_text.strip():
        return [], [], 0

    scopes: list[tuple[str, str, float]] = [("conversation", conversation_id, 0.95)]
    if user_id:
        scopes.insert(0, ("user", user_id, 1.10))
    scopes.append(("channel", f"{channel}:{channel_id or 'default'}", 0.75))

    entries = await query_memory_multi(
        scopes=scopes,
        query=query_text,
        top_k=int(config["top_k"]),
        min_score=float(config["min_score"]),
        ttl_days=int(config["ttl_days"]),
        half_life_days=int(config["half_life_days"]),
    )

    if user_id and bool(config["always_include_user_memory"]):
        # Ensure durable user preferences are present even when lexical similarity is low.
        user_items = list_memory_items(scope_type="user", scope_id=user_id, kind="preference", limit=3)
        for item in user_items:
            if any(str(entry.get("id")) == item.id for entry in entries):
                continue
            entries.append(
                {
                    "id": item.id,
                    "scope_type": item.scope_type,
                    "scope_id": item.scope_id,
                    "kind": item.kind,
                    "content": item.content,
                    "confidence": item.confidence,
                    "score": 0.26,
                    "vector_score": 0.0,
                    "lexical_score": 0.0,
                    "age_days": 0.0,
                }
            )

    entries = sorted(entries, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    entries = entries[: int(config["max_memories_per_turn"])]
    latency_ms = int((time.perf_counter() - start) * 1000)
    return entries, scopes, latency_ms


def _build_memory_system_prompt(entries: list[dict[str, object]]) -> str | None:
    if not entries:
        return None
    lines = [
        "Runtime memory (structured and scoped). Use it only when relevant to the current request.",
        "If current user instruction conflicts with memory, prioritize the current user instruction.",
    ]
    for idx, item in enumerate(entries, 1):
        scope_type = str(item.get("scope_type", "unknown"))
        scope_id = str(item.get("scope_id", "unknown"))
        score = float(item.get("score", 0.0))
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{idx}. [{scope_type}:{scope_id} score={score:.3f}] {content[:220]}")
    if len(lines) <= 2:
        return None
    return "\n".join(lines)


async def _persist_extracted_user_memories(
    *,
    user_id: str | None,
    conversation_id: str,
    user_input: str,
) -> None:
    candidates = extract_memory_candidates(user_input, max_items=3)
    if not candidates:
        return
    scope_type = "user" if user_id else "conversation"
    scope_id = user_id or conversation_id
    if not scope_id:
        return
    for candidate in candidates:
        try:
            await write_memory(
                scope_type=scope_type,
                scope_id=scope_id,
                kind=str(candidate.get("kind", "preference")),
                content=str(candidate.get("content", "")).strip(),
                confidence=float(candidate.get("confidence", 0.6)),
                requires_confirmation=bool(candidate.get("requires_confirmation", True)),
            )
        except Exception:
            continue


async def _maybe_flush_conversation_summary_memory(
    *,
    conversation_id: str,
    response_text: str,
    config: dict[str, object],
) -> None:
    if not bool(config["compaction_enabled"]):
        return

    events = query_events(conversation_id=conversation_id)
    turn_inputs = [item for item in events if item.type == "InputEvent"]
    threshold = int(config["compaction_threshold_turns"])
    if len(turn_inputs) == 0 or len(turn_inputs) % threshold != 0:
        return

    window_turns = int(config["compaction_window_turns"])
    recent_inputs: list[str] = []
    recent_answers: list[str] = []
    for item in reversed(events):
        payload = json.loads(item.payload_json)
        if item.type == "InputEvent":
            text = str(payload.get("message", "")).strip()
            if text:
                recent_inputs.append(text)
        elif item.type == "CompletionEvent":
            text = str(payload.get("response_text", "")).strip()
            if text:
                recent_answers.append(text)
        if len(recent_inputs) >= window_turns and len(recent_answers) >= window_turns:
            break

    if not recent_inputs:
        return

    summary_lines = [
        f"conversation={conversation_id}",
        f"turns_observed={len(turn_inputs)}",
        "recent_user_points:",
    ]
    for idx, text in enumerate(reversed(recent_inputs[:window_turns]), 1):
        summary_lines.append(f"- user[{idx}]: {text[:180]}")
    summary_lines.append("recent_agent_points:")
    for idx, text in enumerate(reversed(recent_answers[:window_turns]), 1):
        summary_lines.append(f"- agent[{idx}]: {text[:180]}")
    summary_lines.append(f"latest_answer={response_text[:200]}")

    summary = "\n".join(summary_lines)
    try:
        await write_memory(
            scope_type="conversation",
            scope_id=conversation_id,
            kind="conversation_summary",
            content=summary,
            confidence=0.78,
            requires_confirmation=False,
        )
    except Exception:
        return
