from __future__ import annotations

from backend.gateway.telegram import (
    TelegramGateway,
    TelegramSettings,
    _conversation_id,
    _extract_completion_text,
    _parse_int_set,
)


def test_parse_int_set() -> None:
    assert _parse_int_set(None) == set()
    assert _parse_int_set("") == set()
    assert _parse_int_set("1, 2,-3") == {1, 2, -3}


def test_conversation_id_includes_chat_and_topic() -> None:
    assert _conversation_id(chat_id=123, thread_id=None) == "telegram:123:0"
    assert _conversation_id(chat_id=-456, thread_id=12) == "telegram:-456:12"


def test_extract_completion_text_from_timeline() -> None:
    payload = {
        "timeline": [
            {"type": "InputEvent", "payload": {"message": "hi"}},
            {"type": "CompletionEvent", "payload": {"response_text": "done-1"}},
            {"type": "CompletionEvent", "payload": {"response_text": "done-2"}},
        ]
    }
    assert _extract_completion_text(payload) == "done-2"


def test_split_text_respects_chunk_size() -> None:
    settings = TelegramSettings(
        bot_token="t",
        edge_base_url="http://127.0.0.1:8000",
        telegram_api_base_url="https://api.telegram.org",
        poll_timeout_seconds=20,
        task_wait_timeout_seconds=120,
        poll_interval_seconds=0.2,
        error_backoff_seconds=2.0,
        owner_user_ids=set(),
        allowed_chat_ids=set(),
        drop_pending_updates_on_start=True,
    )
    gateway = TelegramGateway(settings=settings)
    text = "x" * 8500
    chunks = gateway._split_text(text, limit=3900)
    assert len(chunks) == 3
    assert sum(len(item) for item in chunks) == len(text)
    assert all(len(item) <= 3900 for item in chunks)

