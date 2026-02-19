from __future__ import annotations

from backend.permissions.exec_approvals import evaluate_tool_permission


def test_evaluate_allowlist_on_miss_requires_approval() -> None:
    config = {
        "version": 1,
        "defaults": {"security": "allowlist", "ask": "on-miss", "ask_fallback": "deny", "allowlist": ["mock_*"]},
        "agents": {},
    }
    decision = evaluate_tool_permission(config, actor_id="u1", tool_name="other_tool")
    assert decision["decision"] == "ask"


def test_evaluate_allowlist_off_denies_on_miss() -> None:
    config = {
        "version": 1,
        "defaults": {"security": "allowlist", "ask": "off", "ask_fallback": "deny", "allowlist": ["mock_*"]},
        "agents": {},
    }
    decision = evaluate_tool_permission(config, actor_id="u1", tool_name="other_tool")
    assert decision["decision"] == "deny"


def test_evaluate_agent_override() -> None:
    config = {
        "version": 1,
        "defaults": {"security": "deny", "ask": "off", "ask_fallback": "deny", "allowlist": []},
        "agents": {
            "u_owner": {"security": "full", "ask": "off", "ask_fallback": "deny", "allowlist": []},
        },
    }
    owner_decision = evaluate_tool_permission(config, actor_id="u_owner", tool_name="mock_web_search")
    assert owner_decision["decision"] == "allow"
    member_decision = evaluate_tool_permission(config, actor_id="u_member", tool_name="mock_web_search")
    assert member_decision["decision"] == "deny"

