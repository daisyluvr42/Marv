from __future__ import annotations

import copy
import json
from uuid import uuid4

from sqlmodel import select

from backend.agent.state import now_ts
from backend.storage.db import get_session
from backend.storage.models import ConfigRevision, ConfigSeed


DEFAULT_SEED = {
    "identity": "blackbox-agent",
    "response_style": "balanced",
    "safety": {
        "allow_external_write": False,
        "strict_mode": True,
    },
}


def ensure_seed() -> ConfigSeed:
    with get_session() as session:
        seed = session.exec(select(ConfigSeed).where(ConfigSeed.id == "seed")).first()
        if seed is None:
            seed = ConfigSeed(id="seed", seed_json=json.dumps(DEFAULT_SEED, ensure_ascii=True), created_at=now_ts())
            session.add(seed)
            session.commit()
            session.refresh(seed)
        return seed


def read_seed() -> dict[str, object]:
    seed = ensure_seed()
    return json.loads(seed.seed_json)


def list_revisions(scope_type: str | None = None, scope_id: str | None = None) -> list[ConfigRevision]:
    with get_session() as session:
        stmt = select(ConfigRevision).order_by(ConfigRevision.created_at.desc())
        if scope_type:
            stmt = stmt.where(ConfigRevision.scope_type == scope_type)
        if scope_id:
            stmt = stmt.where(ConfigRevision.scope_id == scope_id)
        return list(session.exec(stmt))


def get_revision(revision_id: str) -> ConfigRevision | None:
    with get_session() as session:
        return session.exec(select(ConfigRevision).where(ConfigRevision.revision == revision_id)).first()


def update_revision_status(revision_id: str, status: str) -> ConfigRevision | None:
    with get_session() as session:
        revision = session.exec(select(ConfigRevision).where(ConfigRevision.revision == revision_id)).first()
        if revision is None:
            return None
        revision.status = status
        session.add(revision)
        session.commit()
        session.refresh(revision)
        return revision


def create_revision(
    scope_type: str,
    scope_id: str,
    actor_id: str,
    patch: dict[str, object],
    explanation: str,
    risk_level: str,
    status: str = "committed",
) -> ConfigRevision:
    revision = ConfigRevision(
        revision=f"rev_{uuid4().hex}",
        scope_type=scope_type,
        scope_id=scope_id,
        created_at=now_ts(),
        actor_id=actor_id,
        patch_json=json.dumps(patch, ensure_ascii=True),
        explanation=explanation,
        risk_level=risk_level,
        status=status,
    )
    with get_session() as session:
        session.add(revision)
        session.commit()
        session.refresh(revision)
        return revision


def get_effective_config(scope_type: str, scope_id: str) -> dict[str, object]:
    effective = copy.deepcopy(read_seed())
    revisions = list_revisions()
    for revision in sorted(revisions, key=lambda item: item.created_at):
        if revision.status != "committed":
            continue
        if revision.scope_type == "global":
            _merge_patch(effective, json.loads(revision.patch_json))
        if revision.scope_type == scope_type and revision.scope_id == scope_id:
            _merge_patch(effective, json.loads(revision.patch_json))
    return effective


def _merge_patch(target: dict[str, object], patch: dict[str, object]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_patch(target[key], value)  # type: ignore[index]
        else:
            target[key] = value
