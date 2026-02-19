from __future__ import annotations

import json
from math import sqrt
from uuid import uuid4

from sqlmodel import select

from backend.agent.state import now_ts
from backend.memory.embeddings import embed_text
from backend.storage.db import get_session
from backend.storage.models import MemoryCandidate, MemoryItem


async def write_memory(
    scope_type: str,
    scope_id: str,
    kind: str,
    content: str,
    confidence: float,
    requires_confirmation: bool,
) -> dict[str, object]:
    embedding = await embed_text(content)
    payload = {
        "scope_type": scope_type,
        "scope_id": scope_id,
        "kind": kind,
        "content": content,
        "embedding_json": json.dumps(embedding, ensure_ascii=True),
        "confidence": confidence,
    }

    if requires_confirmation or confidence < 0.6:
        candidate = MemoryCandidate(
            id=f"cand_{uuid4().hex}",
            status="pending",
            created_at=now_ts(),
            **payload,
        )
        with get_session() as session:
            session.add(candidate)
            session.commit()
            session.refresh(candidate)
        return {"target": "candidate", "item": candidate}

    item = MemoryItem(
        id=f"mem_{uuid4().hex}",
        created_at=now_ts(),
        **payload,
    )
    with get_session() as session:
        session.add(item)
        session.commit()
        session.refresh(item)
    return {"target": "memory_item", "item": item}


def list_candidates(status: str = "pending") -> list[MemoryCandidate]:
    with get_session() as session:
        stmt = select(MemoryCandidate).where(MemoryCandidate.status == status).order_by(MemoryCandidate.created_at.desc())
        return list(session.exec(stmt))


def approve_candidate(candidate_id: str, actor_id: str) -> tuple[MemoryCandidate | None, MemoryItem | None]:
    with get_session() as session:
        candidate = session.exec(select(MemoryCandidate).where(MemoryCandidate.id == candidate_id)).first()
        if candidate is None:
            return None, None
        candidate.status = "approved"
        candidate.decided_at = now_ts()
        candidate.decided_by = actor_id

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
    query_vec = await embed_text(query)
    with get_session() as session:
        stmt = select(MemoryItem).where(MemoryItem.scope_type == scope_type, MemoryItem.scope_id == scope_id)
        items = list(session.exec(stmt))

    scored: list[tuple[float, MemoryItem]] = []
    for item in items:
        item_vec = [float(v) for v in json.loads(item.embedding_json)]
        score = _cosine_similarity(query_vec, item_vec)
        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "id": item.id,
            "kind": item.kind,
            "content": item.content,
            "confidence": item.confidence,
            "score": score,
        }
        for score, item in scored[:top_k]
    ]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
