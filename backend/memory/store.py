from __future__ import annotations

import hashlib
import json
import re
from math import sqrt
from uuid import uuid4

from sqlalchemy import and_, or_
from sqlmodel import Session, select

from backend.agent.state import now_ts
from backend.memory.embeddings import embed_text
from backend.storage.db import get_session
from backend.storage.models import MemoryCandidate, MemoryItem, MemoryRetrievalLog


MILLIS_PER_DAY = 24 * 60 * 60 * 1000


async def write_memory(
    scope_type: str,
    scope_id: str,
    kind: str,
    content: str,
    confidence: float,
    requires_confirmation: bool,
) -> dict[str, object]:
    normalized_content = _normalize_text(content)
    with get_session() as session:
        existing_item = _find_existing_memory_item(
            session=session,
            scope_type=scope_type,
            scope_id=scope_id,
            kind=kind,
            normalized_content=normalized_content,
        )
        if existing_item is not None:
            if confidence > existing_item.confidence:
                existing_item.confidence = confidence
                session.add(existing_item)
                session.commit()
                session.refresh(existing_item)
            return {"target": "memory_item", "item": existing_item}

        if requires_confirmation or confidence < 0.6:
            existing_candidate = _find_pending_candidate(
                session=session,
                scope_type=scope_type,
                scope_id=scope_id,
                kind=kind,
                normalized_content=normalized_content,
            )
            if existing_candidate is not None:
                if confidence > existing_candidate.confidence:
                    existing_candidate.confidence = confidence
                    session.add(existing_candidate)
                    session.commit()
                    session.refresh(existing_candidate)
                return {"target": "candidate", "item": existing_candidate}

    embedding = await embed_text(content)
    payload = {
        "scope_type": scope_type,
        "scope_id": scope_id,
        "kind": kind,
        "content": content,
        "embedding_json": json.dumps(embedding, ensure_ascii=True),
        "confidence": confidence,
    }
    with get_session() as session:
        if requires_confirmation or confidence < 0.6:
            candidate = MemoryCandidate(id=f"cand_{uuid4().hex}", status="pending", created_at=now_ts(), **payload)
            session.add(candidate)
            session.commit()
            session.refresh(candidate)
            return {"target": "candidate", "item": candidate}

        item = MemoryItem(id=f"mem_{uuid4().hex}", created_at=now_ts(), **payload)
        session.add(item)
        session.commit()
        session.refresh(item)
        return {"target": "memory_item", "item": item}


def list_candidates(status: str = "pending") -> list[MemoryCandidate]:
    with get_session() as session:
        stmt = select(MemoryCandidate).where(MemoryCandidate.status == status).order_by(MemoryCandidate.created_at.desc())
        return list(session.exec(stmt))


def list_memory_items(
    *,
    scope_type: str | None = None,
    scope_id: str | None = None,
    kind: str | None = None,
    limit: int = 100,
) -> list[MemoryItem]:
    with get_session() as session:
        stmt = select(MemoryItem).order_by(MemoryItem.created_at.desc())
        if scope_type:
            stmt = stmt.where(MemoryItem.scope_type == scope_type)
        if scope_id:
            stmt = stmt.where(MemoryItem.scope_id == scope_id)
        if kind:
            stmt = stmt.where(MemoryItem.kind == kind)
        if limit > 0:
            stmt = stmt.limit(min(limit, 500))
        return list(session.exec(stmt))


def get_memory_item(item_id: str) -> MemoryItem | None:
    with get_session() as session:
        return session.exec(select(MemoryItem).where(MemoryItem.id == item_id)).first()


async def update_memory_item(
    item_id: str,
    *,
    content: str | None = None,
    kind: str | None = None,
    confidence: float | None = None,
) -> MemoryItem | None:
    with get_session() as session:
        item = session.exec(select(MemoryItem).where(MemoryItem.id == item_id)).first()
        if item is None:
            return None

        content_changed = False
        if content is not None and content.strip() and content.strip() != item.content:
            item.content = content.strip()
            content_changed = True
        if kind is not None and kind.strip():
            item.kind = kind.strip()
        if confidence is not None:
            item.confidence = max(0.0, min(1.0, float(confidence)))

        if content_changed:
            vector = await embed_text(item.content)
            item.embedding_json = json.dumps(vector, ensure_ascii=True)

        session.add(item)
        session.commit()
        session.refresh(item)
        return item


def delete_memory_item(item_id: str) -> bool:
    with get_session() as session:
        item = session.exec(select(MemoryItem).where(MemoryItem.id == item_id)).first()
        if item is None:
            return False
        session.delete(item)
        session.commit()
        return True


def approve_candidate(candidate_id: str, actor_id: str) -> tuple[MemoryCandidate | None, MemoryItem | None]:
    with get_session() as session:
        candidate = session.exec(select(MemoryCandidate).where(MemoryCandidate.id == candidate_id)).first()
        if candidate is None:
            return None, None
        candidate.status = "approved"
        candidate.decided_at = now_ts()
        candidate.decided_by = actor_id
        existing_item = _find_existing_memory_item(
            session=session,
            scope_type=candidate.scope_type,
            scope_id=candidate.scope_id,
            kind=candidate.kind,
            normalized_content=_normalize_text(candidate.content),
        )
        if existing_item is not None:
            if candidate.confidence > existing_item.confidence:
                existing_item.confidence = candidate.confidence
            session.add(candidate)
            session.add(existing_item)
            session.commit()
            session.refresh(candidate)
            session.refresh(existing_item)
            return candidate, existing_item

        memory_item = MemoryItem(
            id=f"mem_{uuid4().hex}",
            scope_type=candidate.scope_type,
            scope_id=candidate.scope_id,
            kind=candidate.kind,
            content=candidate.content,
            embedding_json=candidate.embedding_json,
            confidence=candidate.confidence,
            created_at=now_ts(),
        )

        session.add(candidate)
        session.add(memory_item)
        session.commit()
        session.refresh(candidate)
        session.refresh(memory_item)
        return candidate, memory_item


def reject_candidate(candidate_id: str, actor_id: str) -> MemoryCandidate | None:
    with get_session() as session:
        candidate = session.exec(select(MemoryCandidate).where(MemoryCandidate.id == candidate_id)).first()
        if candidate is None:
            return None
        candidate.status = "rejected"
        candidate.decided_at = now_ts()
        candidate.decided_by = actor_id
        session.add(candidate)
        session.commit()
        session.refresh(candidate)
        return candidate


async def query_memory(scope_type: str, scope_id: str, query: str, top_k: int) -> list[dict[str, object]]:
    results = await query_memory_multi(
        scopes=[(scope_type, scope_id, 1.0)],
        query=query,
        top_k=top_k,
    )
    return [
        {
            "id": item["id"],
            "kind": item["kind"],
            "content": item["content"],
            "confidence": item["confidence"],
            "score": item["score"],
        }
        for item in results
    ]


async def query_memory_multi(
    *,
    scopes: list[tuple[str, str, float]],
    query: str,
    top_k: int,
    min_score: float = 0.25,
    ttl_days: int = 365,
    half_life_days: int = 120,
) -> list[dict[str, object]]:
    if top_k <= 0:
        return []
    clauses = [and_(MemoryItem.scope_type == scope_type, MemoryItem.scope_id == scope_id) for scope_type, scope_id, _ in scopes]
    if not clauses:
        return []
    with get_session() as session:
        stmt = select(MemoryItem).where(or_(*clauses))
        items = list(session.exec(stmt))
    if not items:
        return []

    query_vec = await embed_text(query)
    scope_weights = {(scope_type, scope_id): max(0.2, min(1.5, float(weight))) for scope_type, scope_id, weight in scopes}
    dedup: dict[str, dict[str, object]] = {}
    now_ms = now_ts()

    for item in items:
        age_days = max(0.0, (now_ms - item.created_at) / MILLIS_PER_DAY)
        if ttl_days > 0 and age_days > ttl_days:
            continue

        item_vec = [float(v) for v in json.loads(item.embedding_json)]
        vector_score = _cosine_similarity(query_vec, item_vec)
        lexical_score = _lexical_overlap(query, item.content)
        confidence_score = max(0.0, min(1.0, float(item.confidence)))

        decay_factor = 1.0
        if half_life_days > 0:
            decay_factor = 0.5 ** (age_days / half_life_days)

        weighted_score = (
            ((vector_score * 0.72) + (lexical_score * 0.23) + (confidence_score * 0.05))
            * scope_weights.get((item.scope_type, item.scope_id), 1.0)
            * decay_factor
        )
        if weighted_score < min_score:
            continue
        payload = {
            "id": item.id,
            "scope_type": item.scope_type,
            "scope_id": item.scope_id,
            "kind": item.kind,
            "content": item.content,
            "confidence": item.confidence,
            "score": weighted_score,
            "vector_score": vector_score,
            "lexical_score": lexical_score,
            "age_days": age_days,
        }
        dedup_key = _normalize_text(item.content)
        existing = dedup.get(dedup_key)
        if existing is None or float(payload["score"]) > float(existing["score"]):
            dedup[dedup_key] = payload
    ranked = sorted(dedup.values(), key=lambda item: float(item["score"]), reverse=True)
    return ranked[:top_k]


async def forget_memory_by_query(
    *,
    scope_type: str,
    scope_id: str,
    query: str,
    threshold: float = 0.75,
    max_delete: int = 20,
) -> dict[str, object]:
    threshold = max(0.0, min(1.0, threshold))
    candidates = await query_memory_multi(
        scopes=[(scope_type, scope_id, 1.0)],
        query=query,
        top_k=max_delete,
        min_score=threshold,
        ttl_days=0,
    )
    deleted_ids: list[str] = []
    with get_session() as session:
        for item in candidates:
            memory_item = session.exec(select(MemoryItem).where(MemoryItem.id == str(item["id"]))).first()
            if memory_item is None:
                continue
            session.delete(memory_item)
            deleted_ids.append(memory_item.id)
        session.commit()
    return {
        "scope_type": scope_type,
        "scope_id": scope_id,
        "query": query,
        "deleted_count": len(deleted_ids),
        "deleted_ids": deleted_ids,
    }


def apply_memory_confidence_decay(
    *,
    half_life_days: int = 90,
    min_confidence: float = 0.2,
    scope_type: str | None = None,
    scope_id: str | None = None,
) -> dict[str, object]:
    half_life_days = max(1, half_life_days)
    min_confidence = max(0.0, min(1.0, min_confidence))
    with get_session() as session:
        stmt = select(MemoryItem)
        if scope_type:
            stmt = stmt.where(MemoryItem.scope_type == scope_type)
        if scope_id:
            stmt = stmt.where(MemoryItem.scope_id == scope_id)
        items = list(session.exec(stmt))

        now_ms = now_ts()
        updated = 0
        for item in items:
            age_days = max(0.0, (now_ms - item.created_at) / MILLIS_PER_DAY)
            decay_factor = 0.5 ** (age_days / half_life_days)
            new_conf = max(min_confidence, min(1.0, float(item.confidence) * decay_factor))
            if abs(new_conf - float(item.confidence)) < 1e-6:
                continue
            item.confidence = new_conf
            session.add(item)
            updated += 1
        session.commit()

    return {
        "updated": updated,
        "half_life_days": half_life_days,
        "min_confidence": min_confidence,
        "scope_type": scope_type,
        "scope_id": scope_id,
    }


def record_memory_retrieval(
    *,
    task_id: str | None,
    conversation_id: str,
    query_text: str,
    scopes: list[tuple[str, str, float]],
    results: list[dict[str, object]],
    latency_ms: int,
) -> MemoryRetrievalLog:
    scores = [float(item.get("score", 0.0)) for item in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    top_score = max(scores) if scores else 0.0
    scope_summary = ",".join(f"{scope_type}:{scope_id}:{weight:.2f}" for scope_type, scope_id, weight in scopes)
    query_hash = hashlib.sha256(_normalize_text(query_text).encode("utf-8")).hexdigest()[:16]
    log = MemoryRetrievalLog(
        id=f"mrl_{uuid4().hex}",
        task_id=task_id,
        conversation_id=conversation_id,
        query_hash=query_hash,
        scope_summary=scope_summary,
        hit_count=len(results),
        avg_score=avg_score,
        top_score=top_score,
        latency_ms=max(0, int(latency_ms)),
        created_at=now_ts(),
    )
    with get_session() as session:
        session.add(log)
        session.commit()
        session.refresh(log)
    return log


def get_memory_metrics(*, lookback_hours: int = 24) -> dict[str, object]:
    lookback_hours = max(1, min(24 * 30, lookback_hours))
    since_ts = now_ts() - lookback_hours * 60 * 60 * 1000

    with get_session() as session:
        items = list(session.exec(select(MemoryItem)))
        pending_candidates = list(
            session.exec(select(MemoryCandidate).where(MemoryCandidate.status == "pending"))
        )
        logs = list(session.exec(select(MemoryRetrievalLog).where(MemoryRetrievalLog.created_at >= since_ts)))

    total_items = len(items)
    avg_conf = (sum(float(item.confidence) for item in items) / total_items) if total_items else 0.0
    duplicate_ratio = _estimate_duplicate_ratio(items)
    scope_count = len({(item.scope_type, item.scope_id) for item in items})

    query_count = len(logs)
    hit_logs = [item for item in logs if item.hit_count > 0]
    avg_hits = (sum(item.hit_count for item in logs) / query_count) if query_count else 0.0
    hit_rate = (len(hit_logs) / query_count) if query_count else 0.0
    avg_latency_ms = (sum(item.latency_ms for item in logs) / query_count) if query_count else 0.0

    return {
        "window_hours": lookback_hours,
        "memory_items": {
            "count": total_items,
            "avg_confidence": avg_conf,
            "scope_count": scope_count,
            "duplicate_ratio": duplicate_ratio,
        },
        "candidates": {
            "pending_count": len(pending_candidates),
        },
        "retrieval": {
            "query_count": query_count,
            "hit_rate": hit_rate,
            "avg_hits": avg_hits,
            "avg_latency_ms": avg_latency_ms,
        },
    }


def extract_memory_candidates(text: str, *, max_items: int = 3) -> list[dict[str, object]]:
    if max_items <= 0:
        return []
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return []

    candidates: list[dict[str, object]] = []
    lowered = cleaned.casefold()

    for prefix in ("请记住", "帮我记住", "记住", "remember that", "please remember"):
        if lowered.startswith(prefix):
            raw = cleaned[len(prefix) :].strip(" :：,，。.!?？")
            if len(raw) >= 4:
                candidates.append(
                    {
                        "kind": "preference",
                        "content": raw,
                        "confidence": 0.92,
                        "requires_confirmation": False,
                    }
                )
            break

    patterns = [
        (r"我(喜欢|偏好|习惯|常用)([^。！？\n]{1,80})", "preference", 0.72),
        (r"我不喜欢([^。！？\n]{1,80})", "dislike", 0.72),
        (r"我叫([A-Za-z0-9_\-\u4e00-\u9fff]{1,32})", "profile", 0.68),
        (r"我的(时区|城市|公司|职业|工作)是([^。！？\n]{1,80})", "profile", 0.66),
    ]
    for pattern, kind, confidence in patterns:
        for match in re.finditer(pattern, cleaned):
            if pattern.startswith("我("):
                content = f"我{match.group(1)}{match.group(2)}".strip()
            elif pattern.startswith("我不喜欢"):
                content = f"我不喜欢{match.group(1)}".strip()
            elif pattern.startswith("我叫"):
                content = f"我叫{match.group(1)}".strip()
            else:
                content = f"我的{match.group(1)}是{match.group(2)}".strip()
            if len(content) < 4:
                continue
            candidates.append(
                {
                    "kind": kind,
                    "content": content,
                    "confidence": confidence,
                    "requires_confirmation": True,
                }
            )
            if len(candidates) >= max_items * 2:
                break

    dedup: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        key = _normalize_text(str(candidate["content"]))
        existing = dedup.get(key)
        if existing is None or float(candidate["confidence"]) > float(existing["confidence"]):
            dedup[key] = candidate
    ranked = sorted(dedup.values(), key=lambda item: float(item["confidence"]), reverse=True)
    return ranked[:max_items]


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split()).casefold()


def _find_existing_memory_item(
    *,
    session: Session,
    scope_type: str,
    scope_id: str,
    kind: str,
    normalized_content: str,
) -> MemoryItem | None:
    stmt = select(MemoryItem).where(
        MemoryItem.scope_type == scope_type,
        MemoryItem.scope_id == scope_id,
        MemoryItem.kind == kind,
    )
    for item in session.exec(stmt):
        if _normalize_text(item.content) == normalized_content:
            return item
    return None


def _find_pending_candidate(
    *,
    session: Session,
    scope_type: str,
    scope_id: str,
    kind: str,
    normalized_content: str,
) -> MemoryCandidate | None:
    stmt = select(MemoryCandidate).where(
        MemoryCandidate.scope_type == scope_type,
        MemoryCandidate.scope_id == scope_id,
        MemoryCandidate.kind == kind,
        MemoryCandidate.status == "pending",
    )
    for item in session.exec(stmt):
        if _normalize_text(item.content) == normalized_content:
            return item
    return None


def _estimate_duplicate_ratio(items: list[MemoryItem]) -> float:
    if not items:
        return 0.0
    seen: set[str] = set()
    dup = 0
    for item in items:
        key = f"{item.scope_type}:{item.scope_id}:{item.kind}:{_normalize_text(item.content)}"
        if key in seen:
            dup += 1
        else:
            seen.add(key)
    return dup / len(items)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text.casefold()) if token}


def _lexical_overlap(a: str, b: str) -> float:
    a_tokens = _tokenize(a)
    b_tokens = _tokenize(b)
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    if union == 0:
        return 0.0
    return inter / union
