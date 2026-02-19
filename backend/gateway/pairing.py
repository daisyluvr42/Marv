from __future__ import annotations

import secrets
import string
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from sqlmodel import select

from backend.agent.state import now_ts
from backend.storage.db import get_session
from backend.storage.models import TelegramPairCode, TelegramPairing


@dataclass
class PairVerifyResult:
    ok: bool
    reason: str
    pairing: TelegramPairing | None = None


def create_pair_code(
    *,
    created_by: str,
    ttl_seconds: int = 900,
    chat_id: str | None = None,
    user_id: str | None = None,
) -> TelegramPairCode:
    ttl_seconds = max(60, min(ttl_seconds, 24 * 3600))
    ts = now_ts()
    code = _generate_code()
    item = TelegramPairCode(
        code_id=f"tpc_{uuid4().hex}",
        code=code,
        chat_id=chat_id,
        user_id=user_id,
        status="open",
        created_at=ts,
        expires_at=ts + ttl_seconds * 1000,
        created_by=created_by,
    )
    with get_session() as session:
        session.add(item)
        session.commit()
        session.refresh(item)
        return item


def list_pair_codes(*, status: str | None = None, limit: int = 100) -> list[TelegramPairCode]:
    with get_session() as session:
        stmt = select(TelegramPairCode).order_by(TelegramPairCode.created_at.desc()).limit(max(1, min(limit, 500)))
        if status:
            stmt = stmt.where(TelegramPairCode.status == status)
        return list(session.exec(stmt))


def list_pairings(*, chat_id: str | None = None, user_id: str | None = None, limit: int = 100) -> list[TelegramPairing]:
    with get_session() as session:
        stmt = select(TelegramPairing).order_by(TelegramPairing.last_seen_at.desc()).limit(max(1, min(limit, 500)))
        if chat_id:
            stmt = stmt.where(TelegramPairing.chat_id == chat_id)
        if user_id:
            stmt = stmt.where(TelegramPairing.user_id == user_id)
        return list(session.exec(stmt))


def verify_pair_code(*, code: str, chat_id: str, user_id: str) -> PairVerifyResult:
    text = code.strip().upper()
    if not text:
        return PairVerifyResult(ok=False, reason="empty_code")

    ts = now_ts()
    with get_session() as session:
        code_row = session.exec(select(TelegramPairCode).where(TelegramPairCode.code == text)).first()
        if code_row is None:
            return PairVerifyResult(ok=False, reason="code_not_found")
        if code_row.status != "open":
            return PairVerifyResult(ok=False, reason=f"code_status={code_row.status}")
        if code_row.expires_at < ts:
            code_row.status = "expired"
            session.add(code_row)
            session.commit()
            return PairVerifyResult(ok=False, reason="code_expired")
        if code_row.chat_id and code_row.chat_id != chat_id:
            return PairVerifyResult(ok=False, reason="chat_id_mismatch")
        if code_row.user_id and code_row.user_id != user_id:
            return PairVerifyResult(ok=False, reason="user_id_mismatch")

        existing = session.exec(
            select(TelegramPairing).where(
                TelegramPairing.chat_id == chat_id,
                TelegramPairing.user_id == user_id,
                TelegramPairing.status == "active",
            )
        ).first()
        if existing is None:
            existing = TelegramPairing(
                pairing_id=f"tp_{uuid4().hex}",
                chat_id=chat_id,
                user_id=user_id,
                status="active",
                paired_at=ts,
                expires_at=None,
                last_seen_at=ts,
                created_from_code=code_row.code_id,
            )
        else:
            existing.last_seen_at = ts

        code_row.status = "consumed"
        code_row.consumed_at = ts
        code_row.consumed_by = f"telegram:{user_id}@{chat_id}"

        session.add(existing)
        session.add(code_row)
        session.commit()
        session.refresh(existing)
        return PairVerifyResult(ok=True, reason="paired", pairing=existing)


def touch_pairing(*, chat_id: str, user_id: str) -> bool:
    ts = now_ts()
    with get_session() as session:
        item = session.exec(
            select(TelegramPairing).where(
                TelegramPairing.chat_id == chat_id,
                TelegramPairing.user_id == user_id,
                TelegramPairing.status == "active",
            )
        ).first()
        if item is None:
            return False
        if item.expires_at is not None and item.expires_at < ts:
            item.status = "expired"
            session.add(item)
            session.commit()
            return False
        item.last_seen_at = ts
        session.add(item)
        session.commit()
        return True


def revoke_pairing(pairing_id: str) -> TelegramPairing | None:
    with get_session() as session:
        item = session.exec(select(TelegramPairing).where(TelegramPairing.pairing_id == pairing_id)).first()
        if item is None:
            return None
        item.status = "revoked"
        item.last_seen_at = now_ts()
        session.add(item)
        session.commit()
        session.refresh(item)
        return item


def pairing_required() -> bool:
    from os import getenv

    value = getenv("TELEGRAM_REQUIRE_PAIRING", "false").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def parse_pair_command(text: str) -> str | None:
    message = text.strip()
    if not message:
        return None
    lowered = message.lower()
    if not lowered.startswith("/pair"):
        return None
    parts = message.split()
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def serialize_pair_code(item: TelegramPairCode) -> dict[str, Any]:
    return {
        "code_id": item.code_id,
        "code": item.code,
        "chat_id": item.chat_id,
        "user_id": item.user_id,
        "status": item.status,
        "created_at": item.created_at,
        "expires_at": item.expires_at,
        "created_by": item.created_by,
        "consumed_at": item.consumed_at,
        "consumed_by": item.consumed_by,
    }


def serialize_pairing(item: TelegramPairing) -> dict[str, Any]:
    return {
        "pairing_id": item.pairing_id,
        "chat_id": item.chat_id,
        "user_id": item.user_id,
        "status": item.status,
        "paired_at": item.paired_at,
        "expires_at": item.expires_at,
        "last_seen_at": item.last_seen_at,
        "created_from_code": item.created_from_code,
    }


def _generate_code(length: int = 8) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))
