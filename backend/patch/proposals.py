from __future__ import annotations

import json
from uuid import uuid4

from sqlmodel import select

from backend.agent.state import now_ts
from backend.storage.db import get_session
from backend.storage.models import PatchProposal


def compile_patch(natural_language: str) -> tuple[dict[str, object], str, str, bool]:
    text = natural_language.strip()
    if "更简洁" in text or "简洁" in text:
        return (
            {"response_style": "concise"},
            "L1",
            "将回复风格调整为简洁模式。",
            False,
        )
    if "外部写" in text or "external_write" in text:
        return (
            {"safety": {"allow_external_write": True}},
            "L3",
            "该变更会放宽外部写权限，属于高风险操作。",
            True,
        )
    return (
        {"response_style": "balanced"},
        "L2",
        "未命中特定规则，采用通用风格调整。",
        True,
    )


def create_patch_proposal(
    scope_type: str,
    scope_id: str,
    natural_language: str,
    actor_id: str,
) -> PatchProposal:
    patch, risk_level, explanation, needs_approval = compile_patch(natural_language)
    proposal = PatchProposal(
        proposal_id=f"pp_{uuid4().hex}",
        scope_type=scope_type,
        scope_id=scope_id,
        natural_language=natural_language,
        patch_json=json.dumps(patch, ensure_ascii=True),
        risk_level=risk_level,
        explanation=explanation,
        needs_approval=needs_approval,
        created_at=now_ts(),
        actor_id=actor_id,
        status="open",
    )
    with get_session() as session:
        session.add(proposal)
        session.commit()
        session.refresh(proposal)
        return proposal


def get_patch_proposal(proposal_id: str) -> PatchProposal | None:
    with get_session() as session:
        return session.exec(select(PatchProposal).where(PatchProposal.proposal_id == proposal_id)).first()


def update_patch_proposal_status(proposal_id: str, status: str) -> PatchProposal | None:
    with get_session() as session:
        proposal = session.exec(select(PatchProposal).where(PatchProposal.proposal_id == proposal_id)).first()
        if proposal is None:
            return None
        proposal.status = status
        session.add(proposal)
        session.commit()
        session.refresh(proposal)
        return proposal
