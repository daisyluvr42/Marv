from __future__ import annotations

import json
import time
from uuid import uuid4

from fastapi.testclient import TestClient


def _headers(role: str = "owner", actor_id: str = "test-owner") -> dict[str, str]:
    return {
        "X-Actor-Role": role,
        "X-Actor-Id": actor_id,
    }


def _poll_task_completed(client: TestClient, task_id: str, timeout_seconds: float = 5.0) -> dict[str, object]:
    deadline = time.time() + timeout_seconds
    last_payload: dict[str, object] = {}
    while time.time() < deadline:
        response = client.get(f"/v1/agent/tasks/{task_id}")
        assert response.status_code == 200
        payload = response.json()
        last_payload = payload
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.1)
    raise AssertionError(f"task did not complete in time, last={last_payload}")


class _DummyCoreClient:
    async def health_check(self) -> dict[str, str]:
        return {"status": "ok"}

    async def chat_completions(self, messages: list[dict[str, str]], stream: bool = False, model: str = "mock") -> dict[str, object]:
        text = messages[-1]["content"] if messages else ""
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"test-echo:{text}",
                    }
                }
            ]
        }


def test_message_pipeline_records_completion_event(monkeypatch) -> None:
    monkeypatch.setattr("backend.agent.processor.get_core_client", lambda: _DummyCoreClient())
    from backend.agent.api import app

    conv_id = f"conv_test_{uuid4().hex}"
    with TestClient(app) as client:
        response = client.post(
            "/v1/agent/messages",
            json={"message": "hello-test", "conversation_id": conv_id, "channel": "web"},
            headers=_headers(),
        )
        assert response.status_code == 200
        task_id = response.json()["task_id"]

        task = _poll_task_completed(client, task_id)
        assert task["status"] == "completed"

        timeline = client.get(f"/v1/audit/conversations/{conv_id}/timeline")
        assert timeline.status_code == 200
        payload = timeline.json()
        types = [event["type"] for event in payload["events"]]
        assert "InputEvent" in types
        assert "PlanEvent" in types
        assert "RouteEvent" in types
        assert "CompletionEvent" in types
        completion = [event for event in payload["events"] if event["type"] == "CompletionEvent"][-1]
        assert completion["payload"]["response_text"] == "test-echo:hello-test"


def test_external_write_requires_owner_and_approval_flow() -> None:
    from backend.agent.api import app

    with TestClient(app) as client:
        member_response = client.post(
            "/v1/tools:execute",
            json={"tool": "mock_external_write", "args": {"target": "file://x", "content": "abc"}},
            headers=_headers(role="member", actor_id="member-1"),
        )
        assert member_response.status_code == 403

        pending = client.post(
            "/v1/tools:execute",
            json={"tool": "mock_external_write", "args": {"target": "file://x", "content": "abc"}},
            headers=_headers(role="owner", actor_id="owner-1"),
        )
        assert pending.status_code == 200
        pending_payload = pending.json()
        assert pending_payload["status"] == "pending_approval"
        approval_id = pending_payload["approval_id"]

        approve_member = client.post(
            f"/v1/approvals/{approval_id}:approve",
            json={},
            headers=_headers(role="member", actor_id="member-1"),
        )
        assert approve_member.status_code == 403

        approve_owner = client.post(
            f"/v1/approvals/{approval_id}:approve",
            json={},
            headers=_headers(role="owner", actor_id="owner-1"),
        )
        assert approve_owner.status_code == 200
        approved_payload = approve_owner.json()
        assert approved_payload["status"] == "approved"
        assert approved_payload["tool_call_status"] == "ok"


def test_config_propose_commit_rollback_flow() -> None:
    from backend.agent.api import app

    scope_id = f"scope_{uuid4().hex}"
    with TestClient(app) as client:
        proposal = client.post(
            "/v1/config/patches:propose",
            json={"natural_language": "更简洁", "scope_type": "channel", "scope_id": scope_id},
            headers=_headers(role="member", actor_id="member-1"),
        )
        assert proposal.status_code == 200
        proposal_id = proposal.json()["proposal_id"]

        committed = client.post(
            "/v1/config/patches:commit",
            json={"proposal_id": proposal_id},
            headers=_headers(role="member", actor_id="member-1"),
        )
        assert committed.status_code == 200
        committed_payload = committed.json()
        assert committed_payload["effective_config"]["response_style"] == "concise"
        revision = committed_payload["revision"]

        rollback_member = client.post(
            "/v1/config/revisions:rollback",
            json={"revision": revision},
            headers=_headers(role="member", actor_id="member-1"),
        )
        assert rollback_member.status_code == 403

        rollback_owner = client.post(
            "/v1/config/revisions:rollback",
            json={"revision": revision},
            headers=_headers(role="owner", actor_id="owner-1"),
        )
        assert rollback_owner.status_code == 200
        rollback_payload = rollback_owner.json()
        assert rollback_payload["effective_config"]["response_style"] == "balanced"


def test_memory_candidate_approve_and_query(monkeypatch) -> None:
    async def _fake_embed_text(text: str, model: str = "mock-embedding") -> list[float]:
        base = float((len(text) % 7) + 1)
        return [base, base / 2.0, base / 3.0, base / 4.0]

    monkeypatch.setattr("backend.memory.store.embed_text", _fake_embed_text)
    from backend.agent.api import app

    scope_id = f"user_{uuid4().hex}"
    with TestClient(app) as client:
        written = client.post(
            "/v1/memory/write",
            json={
                "scope_type": "user",
                "scope_id": scope_id,
                "kind": "preference",
                "content": "我偏好简洁回答",
                "confidence": 0.2,
                "requires_confirmation": True,
            },
            headers=_headers(role="member", actor_id="member-1"),
        )
        assert written.status_code == 200
        payload = written.json()
        assert payload["target"] == "candidate"
        candidate_id = payload["id"]

        approved = client.post(
            f"/v1/memory/candidates/{candidate_id}:approve",
            headers=_headers(role="owner", actor_id="owner-1"),
        )
        assert approved.status_code == 200

        queried = client.post(
            "/v1/memory/query",
            json={"scope_type": "user", "scope_id": scope_id, "query": "简洁回答", "top_k": 3},
            headers=_headers(role="member", actor_id="member-1"),
        )
        assert queried.status_code == 200
        queried_payload = queried.json()
        assert queried_payload["count"] >= 1
        assert any(item["content"] == "我偏好简洁回答" for item in queried_payload["results"])


def test_tool_execute_idempotent_hit() -> None:
    from backend.agent.api import app

    fixed_tool_call_id = f"tc_id_{uuid4().hex}"
    with TestClient(app) as client:
        first = client.post(
            "/v1/tools:execute",
            json={
                "tool": "mock_web_search",
                "args": {"query": "same"},
                "tool_call_id": fixed_tool_call_id,
            },
            headers=_headers(role="owner", actor_id="owner-1"),
        )
        assert first.status_code == 200
        assert first.json()["tool_call_id"] == fixed_tool_call_id

        second = client.post(
            "/v1/tools:execute",
            json={
                "tool": "mock_web_search",
                "args": {"query": "same"},
                "tool_call_id": fixed_tool_call_id,
            },
            headers=_headers(role="owner", actor_id="owner-1"),
        )
        assert second.status_code == 200
        second_payload = second.json()
        assert second_payload["tool_call_id"] == fixed_tool_call_id
        assert second_payload["idempotent_hit"] is True


def test_tool_permission_allowlist_on_miss_requires_approval(tmp_path, monkeypatch) -> None:
    policy_file = tmp_path / "exec-approvals.json"
    policy_file.write_text(
        json.dumps(
            {
                "version": 1,
                "defaults": {
                    "security": "allowlist",
                    "ask": "on-miss",
                    "ask_fallback": "deny",
                    "allowlist": ["mock_web_search"],
                },
                "agents": {},
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EDGE_EXEC_APPROVALS_PATH", str(policy_file))
    from backend.agent.api import app

    with TestClient(app) as client:
        blocked_to_approval = client.post(
            "/v1/tools:execute",
            json={"tool": "mock_external_write", "args": {"target": "file://x", "content": "abc"}},
            headers=_headers(role="owner", actor_id="owner-1"),
        )
        assert blocked_to_approval.status_code == 200
        payload = blocked_to_approval.json()
        assert payload["status"] == "pending_approval"
        assert payload["policy_reason"] == "allowlist_miss"


def test_tool_permission_allowlist_off_denies(tmp_path, monkeypatch) -> None:
    policy_file = tmp_path / "exec-approvals.json"
    policy_file.write_text(
        json.dumps(
            {
                "version": 1,
                "defaults": {
                    "security": "allowlist",
                    "ask": "off",
                    "ask_fallback": "deny",
                    "allowlist": ["mock_web_search"],
                },
                "agents": {},
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EDGE_EXEC_APPROVALS_PATH", str(policy_file))
    from backend.agent.api import app

    with TestClient(app) as client:
        denied = client.post(
            "/v1/tools:execute",
            json={"tool": "mock_external_write", "args": {"target": "file://x", "content": "abc"}},
            headers=_headers(role="owner", actor_id="owner-1"),
        )
        assert denied.status_code == 403
        assert "Tool blocked by exec policy" in denied.text


def test_heartbeat_config_owner_update(tmp_path, monkeypatch) -> None:
    heartbeat_path = tmp_path / "heartbeat-config.json"
    monkeypatch.setenv("EDGE_HEARTBEAT_CONFIG_PATH", str(heartbeat_path))
    monkeypatch.setenv("HEARTBEAT_ENABLED", "false")
    from backend.agent.api import app

    with TestClient(app) as client:
        status = client.get("/v1/system/heartbeat")
        assert status.status_code == 200
        assert status.json()["config_path"] == str(heartbeat_path)

        member_update = client.post(
            "/v1/system/heartbeat/config",
            json={"enabled": True, "mode": "interval", "interval_seconds": 30},
            headers=_headers(role="member", actor_id="member-1"),
        )
        assert member_update.status_code == 403

        owner_update = client.post(
            "/v1/system/heartbeat/config",
            json={
                "enabled": True,
                "mode": "interval",
                "interval_seconds": 30,
                "core_health_enabled": True,
                "resume_approved_tools_enabled": False,
                "emit_events": False,
            },
            headers=_headers(role="owner", actor_id="owner-1"),
        )
        assert owner_update.status_code == 200
        payload = owner_update.json()
        assert payload["config"]["enabled"] is True
        assert payload["config"]["mode"] == "interval"
        assert payload["config"]["interval_seconds"] == 30
        assert payload["config"]["resume_approved_tools_enabled"] is False
        assert payload["config"]["emit_events"] is False
