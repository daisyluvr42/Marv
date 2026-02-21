"""Offline suite evaluator using existing agent task execution path."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlmodel import select

from backend.agent.processor import process_task
from backend.agent.state import create_task, get_task, now_ts, upsert_conversation
from backend.approvals.policy import DEFAULT_RISKY_RISKS
from backend.ledger.events import InputEvent
from backend.ledger.store import append_event, query_events
from backend.permissions.exec_approvals import normalize_config as normalize_exec_approvals
from backend.storage.db import get_session
from backend.storage.models import ToolCall


class OfflineCase(BaseModel):
    """One offline testcase for the EA evaluator."""

    name: str
    input: str
    expected_contains: str | list[str] | None = None
    assertions: list[dict[str, Any]] = Field(default_factory=list)
    tooling_allowed: bool = False


class _OfflineCoreClient:
    """Deterministic local core stub to keep offline suite self-contained."""

    async def health_check(self) -> dict[str, str]:
        return {"status": "ok"}

    async def chat_completions(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        model: str = "mock",
        route_tier: str | None = None,
        preferred_locality: str | None = None,
        allow_cloud_fallback: bool = True,
    ) -> dict[str, Any]:
        _ = (stream, route_tier, preferred_locality, allow_cloud_fallback, model)
        user_message = ""
        for item in reversed(messages):
            if str(item.get("role", "")) == "user":
                user_message = str(item.get("content", ""))
                break
        if user_message.startswith("json:"):
            json_payload = {"echo": user_message.removeprefix("json:").strip(), "ok": True}
            content = f"json-payload:{json.dumps(json_payload, ensure_ascii=True)}"
        else:
            content = f"offline-echo:{user_message}"
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                    }
                }
            ]
        }


class OfflineSuiteEvaluator:
    """Offline evaluator that runs testcases through the existing task processor."""

    def __init__(
        self,
        suite_path: str | Path,
        *,
        core_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.suite_path = Path(suite_path).expanduser().resolve()
        self.core_client_factory = core_client_factory or _OfflineCoreClient
        self.cases = self._load_suite(self.suite_path)
        if not self.cases:
            raise ValueError(f"offline suite is empty: {self.suite_path}")

    async def evaluate_individual(
        self,
        *,
        run_id: str,
        individual_id: str,
        candidate_config: dict[str, Any],
        actor_id: str,
        cost_norm_cap: float,
        risk_norm_cap: float,
    ) -> tuple[float, dict[str, Any], str | None]:
        """Run all testcases and return (fitness, metrics, error)."""

        try:
            suite_metrics: list[dict[str, Any]] = []
            pass_count = 0
            total_cost = 0.0
            total_risk = 0.0

            exec_approvals_override = self._build_exec_approvals_override(candidate_config)
            approval_policy_override = self._build_approval_policy_override(candidate_config)
            task_config = self._to_task_runtime_config(candidate_config)

            for index, case in enumerate(self.cases):
                case_result = await self._run_case(
                    run_id=run_id,
                    individual_id=individual_id,
                    case=case,
                    case_index=index,
                    actor_id=actor_id,
                    effective_config=task_config,
                    exec_approvals_override=exec_approvals_override,
                    approval_policy_override=approval_policy_override,
                )
                suite_metrics.append(case_result)
                if bool(case_result.get("success")):
                    pass_count += 1
                total_cost += float(case_result.get("cost", 0.0))
                total_risk += float(case_result.get("risk", 0.0))

            total_cases = max(1, len(self.cases))
            success_rate = pass_count / total_cases
            normalized_cost = min(1.0, total_cost / max(1.0, cost_norm_cap))
            normalized_risk = min(1.0, total_risk / max(1.0, risk_norm_cap))
            fitness = success_rate - 0.05 * normalized_cost - 0.1 * normalized_risk

            metrics: dict[str, Any] = {
                "suite_path": str(self.suite_path),
                "total_cases": len(self.cases),
                "passed_cases": pass_count,
                "success_rate": success_rate,
                "cost": total_cost,
                "risk": total_risk,
                "normalized_cost": normalized_cost,
                "normalized_risk": normalized_risk,
                "fitness_formula": "success_rate - 0.05*normalized_cost - 0.1*normalized_risk",
                "fitness": fitness,
                "cases": suite_metrics,
            }
            return fitness, metrics, None
        except Exception as exc:
            metrics = {
                "suite_path": str(self.suite_path),
                "fitness": -1.0,
                "error": str(exc),
            }
            return -1.0, metrics, str(exc)

    async def _run_case(
        self,
        *,
        run_id: str,
        individual_id: str,
        case: OfflineCase,
        case_index: int,
        actor_id: str,
        effective_config: dict[str, Any],
        exec_approvals_override: dict[str, Any],
        approval_policy_override: dict[str, Any],
    ) -> dict[str, Any]:
        conversation_id = f"evo:{run_id}:{individual_id}:{case_index}:{uuid4().hex[:8]}"
        upsert_conversation(conversation_id=conversation_id, channel="web")
        task = create_task(conversation_id=conversation_id, status="queued", stage="plan")
        append_event(
            InputEvent(
                conversation_id=conversation_id,
                task_id=task.id,
                ts=now_ts(),
                actor_id=actor_id,
                message=case.input,
            )
        )

        await process_task(
            task.id,
            effective_config_override=copy.deepcopy(effective_config),
            core_client_override=self.core_client_factory(),
            exec_approvals_override=copy.deepcopy(exec_approvals_override),
            approval_policy_override=copy.deepcopy(approval_policy_override),
        )

        task_state = get_task(task.id)
        events = query_events(conversation_id=conversation_id, task_id=task.id)
        completion_text = self._extract_completion_text(events)
        assertions_ok, assertion_reasons = self._evaluate_assertions(case=case, response_text=completion_text)
        tool_calls = self._load_tool_calls(task.id)

        tool_call_count = len(tool_calls)
        approvals_triggered_count = sum(1 for item in tool_calls if item.approval_id)
        sandbox_violations = sum(1 for item in tool_calls if (item.error and "sandbox" in item.error.lower()))
        if task_state is not None and task_state.last_error and "sandbox" in task_state.last_error.lower():
            sandbox_violations += 1

        tooling_violation = (not case.tooling_allowed) and tool_call_count > 0
        success = assertions_ok and not tooling_violation and bool(task_state and task_state.status == "completed")

        event_count = len(events)
        cost = float(event_count + tool_call_count)
        risk = float(approvals_triggered_count + sandbox_violations)

        return {
            "name": case.name,
            "task_id": task.id,
            "status": task_state.status if task_state else "missing",
            "success": success,
            "tooling_allowed": case.tooling_allowed,
            "tooling_violation": tooling_violation,
            "assertion_reasons": assertion_reasons,
            "response_text": completion_text,
            "event_count": event_count,
            "tool_call_count": tool_call_count,
            "approvals_triggered_count": approvals_triggered_count,
            "sandbox_violations": sandbox_violations,
            "cost": cost,
            "risk": risk,
            "error": task_state.last_error if task_state else "task-not-found",
        }

    def _evaluate_assertions(self, *, case: OfflineCase, response_text: str) -> tuple[bool, list[str]]:
        reasons: list[str] = []

        contains = case.expected_contains
        expected_items: list[str] = []
        if isinstance(contains, str):
            expected_items = [contains]
        elif isinstance(contains, list):
            expected_items = [str(item) for item in contains]

        for needle in expected_items:
            if needle and needle not in response_text:
                reasons.append(f"missing expected_contains: {needle}")

        for assertion in case.assertions:
            kind = str(assertion.get("type", "contains")).strip().lower()
            if kind == "contains":
                value = str(assertion.get("value", ""))
                if value and value not in response_text:
                    reasons.append(f"contains assertion failed: {value}")
            elif kind == "regex":
                pattern = str(assertion.get("pattern", ""))
                if pattern and not re.search(pattern, response_text):
                    reasons.append(f"regex assertion failed: {pattern}")
            elif kind == "json_schema":
                schema = assertion.get("schema")
                ok, detail = self._validate_json_schema(response_text, schema if isinstance(schema, dict) else {})
                if not ok:
                    reasons.append(f"json_schema assertion failed: {detail}")
            else:
                reasons.append(f"unsupported assertion type: {kind}")

        return len(reasons) == 0, reasons

    def _validate_json_schema(self, payload_text: str, schema: dict[str, Any]) -> tuple[bool, str]:
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            candidate = self._extract_first_json_object(payload_text)
            if candidate is None:
                return False, "response is not valid JSON"
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                return False, "response is not valid JSON"
        if not isinstance(payload, dict):
            return False, "response JSON is not an object"

        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                name = str(key)
                if name not in payload:
                    return False, f"missing required field: {name}"

        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, spec in properties.items():
                if key not in payload or not isinstance(spec, dict):
                    continue
                expected_type = str(spec.get("type", "")).strip().lower()
                if not expected_type:
                    continue
                if not self._matches_json_type(payload[key], expected_type):
                    return False, f"field {key} type mismatch: expected {expected_type}"

        return True, "ok"

    def _extract_first_json_object(self, text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None
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
                    return text[start : idx + 1]
        return None

    def _matches_json_type(self, value: Any, expected: str) -> bool:
        if expected == "string":
            return isinstance(value, str)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "object":
            return isinstance(value, dict)
        if expected == "array":
            return isinstance(value, list)
        if expected == "null":
            return value is None
        return True

    def _extract_completion_text(self, events: list[Any]) -> str:
        for event in reversed(events):
            if event.type != "CompletionEvent":
                continue
            payload = json.loads(event.payload_json)
            value = str(payload.get("response_text", ""))
            if value:
                return value
        return ""

    def _load_tool_calls(self, task_id: str) -> list[ToolCall]:
        with get_session() as session:
            stmt = select(ToolCall).where(ToolCall.task_id == task_id)
            return list(session.exec(stmt))

    def _to_task_runtime_config(self, candidate_config: dict[str, Any]) -> dict[str, Any]:
        runtime = copy.deepcopy(candidate_config)
        memory = runtime.get("memory")
        if isinstance(memory, dict) and "score_threshold" in memory:
            value = memory.get("score_threshold")
            if isinstance(value, (int, float)):
                memory["min_score"] = float(value)
        return runtime

    def _build_exec_approvals_override(self, candidate_config: dict[str, Any]) -> dict[str, Any]:
        allowlist: list[str] = []
        tools = candidate_config.get("tools")
        if isinstance(tools, dict):
            allowlist_value = tools.get("allowlist")
            if isinstance(allowlist_value, list):
                allowlist = [str(item).strip() for item in allowlist_value if str(item).strip()]

        raw = {
            "version": 1,
            "defaults": {
                "security": "allowlist",
                "ask": "on-miss",
                "ask_fallback": "deny",
                "allowlist": allowlist,
            },
            "agents": {},
        }
        return normalize_exec_approvals(raw)

    def _build_approval_policy_override(self, candidate_config: dict[str, Any]) -> dict[str, Any]:
        mode = "risky"
        approvals = candidate_config.get("approvals")
        if isinstance(approvals, dict):
            candidate_mode = str(approvals.get("mode", "")).strip().lower()
            if candidate_mode in {"risky", "policy", "all"}:
                mode = candidate_mode
        return {
            "mode": mode,
            "risky_risks": sorted(DEFAULT_RISKY_RISKS),
        }

    def _load_suite(self, suite_path: Path) -> list[OfflineCase]:
        if not suite_path.exists():
            raise FileNotFoundError(f"offline suite not found: {suite_path}")

        payload: Any
        suffix = suite_path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("YAML suite requested but PyYAML is not installed") from exc
            payload = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(suite_path.read_text(encoding="utf-8"))

        if not isinstance(payload, dict):
            raise ValueError("offline suite file must be a JSON/YAML object")
        testcases = payload.get("testcases")
        if not isinstance(testcases, list):
            raise ValueError("offline suite must contain a 'testcases' list")

        cases: list[OfflineCase] = []
        for item in testcases:
            if not isinstance(item, dict):
                continue
            cases.append(OfflineCase.model_validate(item))
        return cases
