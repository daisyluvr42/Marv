from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlmodel import select

from backend.agent.config import get_settings
from backend.agent.state import now_ts
from backend.ledger.store import query_events
from backend.memory.embeddings import embed_text
from backend.storage.db import get_session
from backend.storage.models import SkillBlueprint, ToolCall
from backend.tools.registry import get_tool_spec


class Blueprint(BaseModel):
    name: str
    version: str
    intent_hash: str = ""
    intent_vector: list[float] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    variables: dict[str, str] = Field(default_factory=dict)
    min_permission_level: int = 1
    source_tool_risks: list[str] = Field(default_factory=list)


class SkillEngine:
    def __init__(self, *, data_dir: Path | None = None) -> None:
        root = data_dir or get_settings().data_dir
        self._skills_dir = root / "skills"

    async def analyze_ledger(
        self,
        *,
        window_hours: int = 24,
        min_occurrences: int = 4,
        max_scan_tasks: int = 1200,
        max_patterns: int = 20,
    ) -> list[dict[str, Any]]:
        since_ts = now_ts() - max(1, window_hours) * 3600 * 1000
        rows: list[dict[str, Any]] = []
        stmt = text(
            """
            SELECT
              t.id AS task_id,
              t.conversation_id AS conversation_id,
              COALESCE(json_extract(inp.payload_json, '$.message'), '') AS message,
              COALESCE(
                (
                  SELECT group_concat(x.tool, '>')
                  FROM (
                    SELECT tc.tool AS tool
                    FROM tool_calls tc
                    WHERE tc.task_id = t.id AND tc.status = 'ok'
                    ORDER BY tc.created_at ASC
                  ) AS x
                ),
                ''
              ) AS tool_seq
            FROM tasks t
            JOIN ledger_events inp ON inp.task_id = t.id AND inp.type = 'InputEvent'
            WHERE t.status = 'completed' AND t.updated_at >= :since_ts
            ORDER BY t.updated_at DESC
            LIMIT :max_scan_tasks
            """
        )
        with get_session() as session:
            records = session.execute(
                stmt,
                {"since_ts": since_ts, "max_scan_tasks": max(50, min(max_scan_tasks, 20000))},
            ).mappings()
            for row in records:
                message = str(row.get("message") or "").strip()
                if not message:
                    continue
                tool_seq = str(row.get("tool_seq") or "").strip()
                rows.append(
                    {
                        "task_id": str(row.get("task_id") or ""),
                        "conversation_id": str(row.get("conversation_id") or ""),
                        "message": message,
                        "tool_seq": tool_seq,
                        "intent_hash": _intent_hash(message),
                        "tool_seq_hash": _sha1(tool_seq),
                    }
                )

        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (row["intent_hash"], row["tool_seq_hash"])
            item = grouped.get(key)
            if item is None:
                grouped[key] = {
                    "intent_hash": row["intent_hash"],
                    "tool_seq_hash": row["tool_seq_hash"],
                    "occurrences": 1,
                    "task_ids": [row["task_id"]],
                    "conversation_ids": [row["conversation_id"]],
                    "sample_message": row["message"],
                    "tool_seq": row["tool_seq"],
                }
                continue
            item["occurrences"] = int(item["occurrences"]) + 1
            item["task_ids"].append(row["task_id"])
            item["conversation_ids"].append(row["conversation_id"])

        candidates = [
            value
            for value in grouped.values()
            if int(value["occurrences"]) >= max(2, min_occurrences) and value["tool_seq"].strip()
        ]
        candidates.sort(key=lambda item: int(item["occurrences"]), reverse=True)
        candidates = candidates[: max(1, max_patterns)]

        output: list[dict[str, Any]] = []
        for item in candidates:
            task_ids = [str(task_id) for task_id in item["task_ids"][:3]]
            event_chains: dict[str, list[str]] = {}
            for task_id in task_ids:
                chain = self._event_chain_for_task(task_id)
                if chain:
                    event_chains[task_id] = chain
            output.append(
                {
                    "intent_hash": item["intent_hash"],
                    "tool_seq_hash": item["tool_seq_hash"],
                    "occurrences": int(item["occurrences"]),
                    "task_ids": task_ids,
                    "conversation_ids": [str(cid) for cid in item["conversation_ids"][:3]],
                    "sample_message": str(item["sample_message"]),
                    "tool_seq": str(item["tool_seq"]),
                    "event_id_chains": event_chains,
                }
            )
        return output

    async def distill_trajectory(self, events: list[Any]) -> Blueprint | None:
        if not events:
            return None
        task_id = ""
        conversation_id = ""
        input_text = ""
        completion_seen = False
        for event in events:
            task_id = task_id or str(getattr(event, "task_id", "") or "")
            conversation_id = conversation_id or str(getattr(event, "conversation_id", "") or "")
            if getattr(event, "type", "") == "InputEvent":
                try:
                    payload = json.loads(str(getattr(event, "payload_json", "{}")))
                except json.JSONDecodeError:
                    payload = {}
                input_text = str(payload.get("message", "")).strip() or input_text
            elif getattr(event, "type", "") == "CompletionEvent":
                completion_seen = True
        if not task_id or not input_text or not completion_seen:
            return None

        tool_calls = self._load_success_tool_calls(task_id)
        if not tool_calls:
            return None

        steps: list[dict[str, Any]] = []
        source_tool_risks: list[str] = []
        max_permission_level = 1
        prev_signature = ""
        for call in tool_calls:
            tool_name = str(call.tool).strip()
            if not tool_name:
                continue
            try:
                args = json.loads(call.args_json) if call.args_json else {}
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            signature = f"{tool_name}:{_sha1(json.dumps(args, ensure_ascii=True, sort_keys=True))}"
            if signature == prev_signature:
                continue
            prev_signature = signature

            args_template = _parameterize_args(args, input_text=input_text)
            spec = get_tool_spec(tool_name)
            risk = str(spec.risk if spec is not None else "unknown")
            source_tool_risks.append(risk)
            permission = _permission_level_from_risk(risk)
            if permission > max_permission_level:
                max_permission_level = permission
            steps.append(
                {
                    "tool_name": tool_name,
                    "arguments": args_template,
                    "risk": risk,
                }
            )
        if not steps:
            return None

        intent_vector = await embed_text(input_text)
        normalized_tools = "_".join(step["tool_name"] for step in steps[:2])
        name = f"skill_{normalized_tools or 'flow'}_{_intent_hash(input_text)[:8]}"
        return Blueprint(
            name=name,
            version="1",
            intent_hash=_intent_hash(input_text),
            intent_vector=[float(v) for v in intent_vector],
            steps=steps,
            success_criteria=["completion_event_present", f"tool_steps>={len(steps)}"],
            variables={"user_input": "latest user message"},
            min_permission_level=max_permission_level,
            source_tool_risks=sorted(set(source_tool_risks)),
        )

    async def solidify_skill(
        self,
        blueprint: Blueprint,
        *,
        status: str = "candidate",
        occurrences: int = 1,
        source_event_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_status = status.strip().lower()
        if normalized_status not in {"candidate", "production", "archived"}:
            normalized_status = "candidate"
        safe_name = _slug(blueprint.name) or "skill"
        with get_session() as session:
            existing = list(session.exec(select(SkillBlueprint).where(SkillBlueprint.name == safe_name)))
            max_version = max([int(item.version) for item in existing], default=0)
            next_version = max_version + 1
            if blueprint.min_permission_level <= 1 and occurrences >= 5:
                normalized_status = "production"
            record = SkillBlueprint(
                blueprint_id=f"sb_{uuid4().hex}",
                name=safe_name,
                version=next_version,
                status=normalized_status,
                intent_hash=blueprint.intent_hash or _sha1(safe_name),
                intent_embedding_json=json.dumps(blueprint.intent_vector, ensure_ascii=True),
                blueprint_json=json.dumps(blueprint.model_dump(mode="json"), ensure_ascii=True),
                success_criteria_json=json.dumps(blueprint.success_criteria, ensure_ascii=True),
                variables_json=json.dumps(blueprint.variables, ensure_ascii=True),
                min_permission_level=int(blueprint.min_permission_level),
                source_tool_risks_json=json.dumps(blueprint.source_tool_risks, ensure_ascii=True),
                source_event_ids_json=json.dumps(source_event_ids or [], ensure_ascii=True),
                hit_count=0,
                success_count=0,
                created_at=now_ts(),
                updated_at=now_ts(),
            )
            session.add(record)
            session.commit()
            session.refresh(record)

        skill_dir = self._skills_dir / safe_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        path = skill_dir / f"v{next_version}.json"
        path.write_text(json.dumps(blueprint.model_dump(mode="json"), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        return {
            "blueprint_id": record.blueprint_id,
            "name": safe_name,
            "version": next_version,
            "status": record.status,
            "path": str(path),
        }

    async def match_blueprint(
        self,
        *,
        intent_text: str,
        min_score: float = 0.9,
        allow_semantic: bool = False,
        max_candidates: int = 120,
    ) -> dict[str, Any] | None:
        query = intent_text.strip()
        if not query:
            return None
        query_hash = _intent_hash(query)
        with get_session() as session:
            exact = list(
                session.exec(
                    select(SkillBlueprint)
                    .where(
                        SkillBlueprint.status == "production",
                        SkillBlueprint.intent_hash == query_hash,
                    )
                    .order_by(SkillBlueprint.success_count.desc(), SkillBlueprint.hit_count.desc(), SkillBlueprint.updated_at.desc())
                    .limit(10)
                )
            )
            pool = exact
            if not pool and allow_semantic:
                pool = list(
                    session.exec(
                        select(SkillBlueprint)
                        .where(SkillBlueprint.status == "production")
                        .order_by(SkillBlueprint.success_count.desc(), SkillBlueprint.hit_count.desc(), SkillBlueprint.updated_at.desc())
                        .limit(max(10, min(max_candidates, 500)))
                    )
                )
        best_row: SkillBlueprint | None = None
        best_score = -1.0
        if not pool:
            return None

        if exact:
            query_vec: list[float] = []
        else:
            query_vec = await embed_text(query)
        for row in pool:
            if exact:
                score = 1.0
            else:
                try:
                    row_vec = [float(v) for v in json.loads(row.intent_embedding_json)]
                except json.JSONDecodeError:
                    row_vec = []
                score = _cosine(query_vec, row_vec)
            if score > best_score:
                best_score = score
                best_row = row
        if best_row is None or best_score < max(0.5, min_score):
            return None

        payload = json.loads(best_row.blueprint_json)
        blueprint = Blueprint.model_validate(payload)
        await self.increment_hit(best_row.blueprint_id)
        return {
            "blueprint_id": best_row.blueprint_id,
            "score": best_score,
            "name": best_row.name,
            "version": best_row.version,
            "blueprint": blueprint,
        }

    async def increment_hit(self, blueprint_id: str) -> None:
        with get_session() as session:
            row = session.exec(select(SkillBlueprint).where(SkillBlueprint.blueprint_id == blueprint_id)).first()
            if row is None:
                return
            row.hit_count = int(row.hit_count) + 1
            row.updated_at = now_ts()
            session.add(row)
            session.commit()

    async def record_execution_result(self, blueprint_id: str, *, success: bool) -> None:
        with get_session() as session:
            row = session.exec(select(SkillBlueprint).where(SkillBlueprint.blueprint_id == blueprint_id)).first()
            if row is None:
                return
            if success:
                row.success_count = int(row.success_count) + 1
            row.updated_at = now_ts()
            session.add(row)
            session.commit()

    async def run_cycle(
        self,
        *,
        window_hours: int = 24,
        min_occurrences: int = 4,
        max_patterns: int = 8,
        max_distill: int = 4,
    ) -> dict[str, Any]:
        scanned = await self.analyze_ledger(
            window_hours=window_hours,
            min_occurrences=min_occurrences,
            max_patterns=max_patterns,
        )
        distilled = 0
        stored: list[dict[str, Any]] = []
        skipped = 0
        for item in scanned:
            if distilled >= max(1, max_distill):
                break
            task_ids = [str(value) for value in item.get("task_ids", [])]
            if not task_ids:
                skipped += 1
                continue
            task_id = task_ids[0]
            if self._has_recent_blueprint_for_intent(str(item.get("intent_hash", ""))):
                skipped += 1
                continue
            events = query_events(conversation_id=str(item.get("conversation_ids", [""])[0]), task_id=task_id)
            blueprint = await self.distill_trajectory(events)
            if blueprint is None:
                skipped += 1
                continue
            blueprint.intent_hash = str(item.get("intent_hash") or blueprint.intent_hash)
            source_event_ids = list((item.get("event_id_chains", {}) or {}).get(task_id, []))
            stored_item = await self.solidify_skill(
                blueprint,
                status="candidate",
                occurrences=int(item.get("occurrences", 1)),
                source_event_ids=source_event_ids,
            )
            stored.append(stored_item)
            distilled += 1
        return {
            "scanned_patterns": len(scanned),
            "distilled_count": distilled,
            "skipped_count": skipped,
            "stored": stored,
        }

    def _event_chain_for_task(self, task_id: str) -> list[str]:
        with get_session() as session:
            stmt = text(
                """
                SELECT event_id
                FROM ledger_events
                WHERE task_id = :task_id
                ORDER BY ts ASC, id ASC
                """
            )
            rows = session.execute(stmt, {"task_id": task_id}).mappings()
            return [str(row.get("event_id")) for row in rows if str(row.get("event_id") or "").strip()]

    def _load_success_tool_calls(self, task_id: str) -> list[ToolCall]:
        with get_session() as session:
            return list(
                session.exec(
                    select(ToolCall)
                    .where(ToolCall.task_id == task_id, ToolCall.status == "ok")
                    .order_by(ToolCall.created_at.asc(), ToolCall.tool_call_id.asc())
                )
            )

    def _has_recent_blueprint_for_intent(self, intent_hash: str, *, recent_hours: int = 24) -> bool:
        if not intent_hash:
            return False
        since_ts = now_ts() - max(1, recent_hours) * 3600 * 1000
        with get_session() as session:
            existing = session.exec(
                select(SkillBlueprint).where(
                    SkillBlueprint.intent_hash == intent_hash,
                    SkillBlueprint.updated_at >= since_ts,
                )
            ).first()
            return existing is not None


def _permission_level_from_risk(risk: str) -> int:
    lowered = risk.strip().lower()
    if lowered == "read_only":
        return 1
    return 2


def _parameterize_args(args: dict[str, Any], *, input_text: str) -> dict[str, Any]:
    return {str(key): _parameterize_value(value, input_text=input_text) for key, value in args.items()}


def _parameterize_value(value: Any, *, input_text: str) -> Any:
    if isinstance(value, str):
        raw = value.strip()
        if raw and _is_same_text(raw, input_text):
            return "{{user_input}}"
        if raw and len(input_text) >= 16 and input_text.strip() in raw:
            return raw.replace(input_text.strip(), "{{user_input}}")
        return value
    if isinstance(value, dict):
        return {str(key): _parameterize_value(item, input_text=input_text) for key, item in value.items()}
    if isinstance(value, list):
        return [_parameterize_value(item, input_text=input_text) for item in value]
    return value


def _is_same_text(left: str, right: str) -> bool:
    return _normalize_text(left) == _normalize_text(right)


def _normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _intent_hash(text: str) -> str:
    return _sha1(_normalize_text(text))


def _sha1(text_value: str) -> str:
    return hashlib.sha1(text_value.encode("utf-8")).hexdigest()


def _slug(name: str) -> str:
    lowered = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_")
    return lowered[:96]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    if length == 0:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for idx in range(length):
        a = float(left[idx])
        b = float(right[idx])
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / ((left_norm ** 0.5) * (right_norm ** 0.5))


_skill_engine = SkillEngine()


def get_skill_engine() -> SkillEngine:
    return _skill_engine
