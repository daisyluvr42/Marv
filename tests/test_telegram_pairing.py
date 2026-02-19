from __future__ import annotations

from uuid import uuid4

from backend.gateway.pairing import (
    create_pair_code,
    list_pairings,
    revoke_pairing,
    touch_pairing,
    verify_pair_code,
)
from backend.storage.db import init_db


def test_pairing_code_verify_and_revoke_flow() -> None:
    init_db()
    chat_id = str(-1000000000000 - int(uuid4().int % 10000))
    user_id = str(100000 + int(uuid4().int % 10000))
    code = create_pair_code(created_by="owner-1", ttl_seconds=300, chat_id=chat_id, user_id=user_id)

    verified = verify_pair_code(code=code.code, chat_id=chat_id, user_id=user_id)
    assert verified.ok is True
    assert verified.pairing is not None

    active = touch_pairing(chat_id=chat_id, user_id=user_id)
    assert active is True

    existing = list_pairings(chat_id=chat_id, user_id=user_id)
    assert len(existing) >= 1

    revoked = revoke_pairing(existing[0].pairing_id)
    assert revoked is not None
    assert revoked.status == "revoked"
