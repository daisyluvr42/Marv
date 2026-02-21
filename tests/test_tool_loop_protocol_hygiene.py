from __future__ import annotations

from backend.agent.processor import (
    _resolve_auto_subagents_config,
    _build_routing_state,
    _classify_routing_intent,
    _maybe_escalate_routing_state,
    _parse_loop_directive,
    _resolve_model_routing_config,
    _sanitize_visible_assistant_text,
)


def test_parse_loop_directive_recovers_downgraded_tool_call_text() -> None:
    text = (
        "[Tool Call: mock_web_search (ID: toolu_123)]\n"
        'Arguments: {"query":"Marv runtime status"}'
    )
    directive = _parse_loop_directive(text)
    assert directive is not None
    assert directive["action"] == "tool_call"
    assert directive["tool_name"] == "mock_web_search"
    assert directive["arguments"] == {"query": "Marv runtime status"}


def test_sanitize_visible_assistant_text_strips_protocol_leakage() -> None:
    text = (
        "Before.\n"
        '[Tool Call: read (ID: toolu_1)]\nArguments: {"path":"/tmp/a.txt"}\n'
        "[Historical context: tool call info]\n"
        "After."
    )
    cleaned = _sanitize_visible_assistant_text(text)
    assert "Before." in cleaned
    assert "After." in cleaned
    assert "[Tool Call:" not in cleaned
    assert "[Historical context:" not in cleaned


def test_classify_routing_intent_prefers_local_light_for_simple_prompt() -> None:
    routing_config = _resolve_model_routing_config({})
    intent = _classify_routing_intent(
        user_input="Translate this sentence to Chinese.",
        memory_entries_count=0,
        loop_enabled=False,
        routing_config=routing_config,
    )
    assert intent["intent"] == "simple"
    assert intent["initial_tier"] == "local_light"
    assert intent["initial_locality"] == "local"


def test_routing_state_escalates_after_step_threshold() -> None:
    routing_config = _resolve_model_routing_config(
        {"model_routing": {"escalate_after_steps": 2, "escalate_after_protocol_repairs": 0, "escalate_after_reflect_rounds": 0}}
    )
    routing_state = _build_routing_state(
        user_input="Summarize this long architecture review and optimize deployment workflow.",
        memory_entries=[],
        loop_config={"enabled": True},
        routing_config=routing_config,
    )
    reason = _maybe_escalate_routing_state(
        routing_state=routing_state,
        step=3,
        protocol_repairs_used=0,
        consecutive_reflects=0,
    )
    assert reason is not None
    assert reason.startswith("multi_round_timeout")
    assert routing_state["cloud_escalated"] is True


def test_resolve_auto_subagents_defaults_to_disabled() -> None:
    config = _resolve_auto_subagents_config({})
    assert config["enabled"] is False
    assert config["complexity_threshold"] >= 1
    assert "complex" in config["trigger_intents"]
