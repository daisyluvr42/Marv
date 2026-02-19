from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, Request


@dataclass
class ActorContext:
    actor_id: str
    role: str


def get_actor_context(request: Request) -> ActorContext:
    actor_id = request.headers.get("X-Actor-Id", "anonymous")
    role = request.headers.get("X-Actor-Role", "member").strip().lower()
    if role not in {"owner", "member"}:
        raise HTTPException(status_code=400, detail="Invalid X-Actor-Role; expected owner|member")
    return ActorContext(actor_id=actor_id, role=role)


def require_owner(request: Request) -> ActorContext:
    actor = get_actor_context(request)
    if actor.role != "owner":
        raise HTTPException(status_code=403, detail="Owner role required")
    return actor
