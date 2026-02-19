from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from backend.gateway.pairing import parse_pair_command, pairing_required, touch_pairing, verify_pair_code


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int_set(value: str | None) -> set[int]:
    if not value:
        return set()
    result: set[int] = set()
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            result.add(int(item))
        except ValueError as exc:
            raise SystemExit(f"invalid integer id: {item}") from exc
    return result


def _conversation_id(chat_id: int, thread_id: int | None) -> str:
    topic = thread_id if thread_id is not None else 0
    return f"telegram:{chat_id}:{topic}"


def _extract_completion_text(audit_payload: dict[str, Any]) -> str | None:
    timeline = audit_payload.get("timeline")
    if not isinstance(timeline, list):
        return None
    for item in reversed(timeline):
        if not isinstance(item, dict) or item.get("type") != "CompletionEvent":
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        text = str(payload.get("response_text", "")).strip()
        if text:
            return text
    return None


@dataclass
class TelegramSettings:
    bot_token: str
    edge_base_url: str
    telegram_api_base_url: str
    poll_timeout_seconds: int
    task_wait_timeout_seconds: int
    poll_interval_seconds: float
    error_backoff_seconds: float
    owner_user_ids: set[int]
    allowed_chat_ids: set[int]
    drop_pending_updates_on_start: bool
    require_pairing: bool

    @classmethod
    def from_env(cls) -> "TelegramSettings":
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise SystemExit("TELEGRAM_BOT_TOKEN is required")
        return cls(
            bot_token=token,
            edge_base_url=os.getenv("EDGE_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
            telegram_api_base_url=os.getenv("TELEGRAM_API_BASE_URL", "https://api.telegram.org").rstrip("/"),
            poll_timeout_seconds=int(os.getenv("TELEGRAM_POLL_TIMEOUT_SECONDS", "20")),
            task_wait_timeout_seconds=int(os.getenv("TELEGRAM_TASK_WAIT_TIMEOUT_SECONDS", "120")),
            poll_interval_seconds=float(os.getenv("TELEGRAM_POLL_INTERVAL_SECONDS", "0.2")),
            error_backoff_seconds=float(os.getenv("TELEGRAM_ERROR_BACKOFF_SECONDS", "2")),
            owner_user_ids=_parse_int_set(os.getenv("TELEGRAM_OWNER_IDS")),
            allowed_chat_ids=_parse_int_set(os.getenv("TELEGRAM_ALLOWED_CHAT_IDS")),
            drop_pending_updates_on_start=_parse_bool(os.getenv("TELEGRAM_DROP_PENDING_ON_START"), True),
            require_pairing=pairing_required(),
        )


class TelegramGateway:
    def __init__(self, settings: TelegramSettings):
        self.settings = settings
        self.offset: int | None = None

    def run(self) -> None:
        print("telegram gateway started")
        with httpx.Client(base_url=self.settings.telegram_api_base_url, timeout=30.0) as tg_client:
            with httpx.Client(base_url=self.settings.edge_base_url, timeout=30.0) as edge_client:
                if self.settings.drop_pending_updates_on_start:
                    updates = self._get_updates(tg_client, timeout=0, limit=100)
                    if updates:
                        self.offset = max(int(item["update_id"]) for item in updates) + 1
                while True:
                    try:
                        updates = self._get_updates(
                            tg_client,
                            timeout=self.settings.poll_timeout_seconds,
                            offset=self.offset,
                        )
                        for update in updates:
                            self.offset = int(update["update_id"]) + 1
                            self._handle_update(update, tg_client, edge_client)
                        time.sleep(self.settings.poll_interval_seconds)
                    except Exception as exc:  # pragma: no cover - network guarded loop
                        print(f"telegram gateway error: {exc}")
                        time.sleep(self.settings.error_backoff_seconds)

    def _telegram_path(self, method: str) -> str:
        return f"/bot{self.settings.bot_token}/{method}"

    def _get_updates(
        self,
        tg_client: httpx.Client,
        *,
        timeout: int,
        limit: int = 100,
        offset: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "timeout": timeout,
            "limit": limit,
            "allowed_updates": json.dumps(["message"]),
        }
        if offset is not None:
            params["offset"] = offset
        response = tg_client.get(self._telegram_path("getUpdates"), params=params)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(f"telegram getUpdates failed: {payload}")
        result = payload.get("result", [])
        if not isinstance(result, list):
            raise RuntimeError("telegram getUpdates returned non-list result")
        return [item for item in result if isinstance(item, dict)]

    def _send_message(
        self,
        tg_client: httpx.Client,
        *,
        chat_id: int,
        text: str,
        reply_to_message_id: int | None = None,
    ) -> None:
        chunks = self._split_text(text)
        for index, chunk in enumerate(chunks):
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "text": chunk,
            }
            if index == 0 and reply_to_message_id is not None:
                payload["reply_to_message_id"] = reply_to_message_id
                payload["allow_sending_without_reply"] = True
            response = tg_client.post(self._telegram_path("sendMessage"), json=payload)
            response.raise_for_status()
            body = response.json()
            if not body.get("ok"):
                raise RuntimeError(f"telegram sendMessage failed: {body}")

    def _split_text(self, text: str, limit: int = 3900) -> list[str]:
        content = text.strip() or "[empty-response]"
        if len(content) <= limit:
            return [content]
        chunks: list[str] = []
        current = content
        while len(current) > limit:
            split_at = current.rfind("\n", 0, limit)
            if split_at < 0:
                split_at = limit
            chunks.append(current[:split_at].strip())
            current = current[split_at:].strip()
        if current:
            chunks.append(current)
        return chunks

    def _handle_update(
        self,
        update: dict[str, Any],
        tg_client: httpx.Client,
        edge_client: httpx.Client,
    ) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return
        from_user = message.get("from")
        if isinstance(from_user, dict) and bool(from_user.get("is_bot")):
            return

        chat = message.get("chat")
        if not isinstance(chat, dict) or "id" not in chat:
            return
        chat_id = int(chat["id"])
        if self.settings.allowed_chat_ids and chat_id not in self.settings.allowed_chat_ids:
            self._send_message(
                tg_client,
                chat_id=chat_id,
                text="This chat is not allowed to use this bot.",
                reply_to_message_id=int(message["message_id"]) if "message_id" in message else None,
            )
            return

        user_id = int(from_user["id"]) if isinstance(from_user, dict) and "id" in from_user else 0
        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            self._send_message(
                tg_client,
                chat_id=chat_id,
                text="Only text messages are supported right now.",
                reply_to_message_id=int(message["message_id"]) if "message_id" in message else None,
            )
            return

        role = "owner" if user_id in self.settings.owner_user_ids else "member"
        thread_id = message.get("message_thread_id")
        if thread_id is not None:
            thread_id = int(thread_id)

        if self.settings.require_pairing:
            paired = touch_pairing(chat_id=str(chat_id), user_id=str(user_id))
            if not paired:
                pair_code = parse_pair_command(text)
                if pair_code is None:
                    self._send_message(
                        tg_client,
                        chat_id=chat_id,
                        text="Pairing required. Send /pair <code> first.",
                        reply_to_message_id=int(message["message_id"]) if "message_id" in message else None,
                    )
                    return
                if not pair_code:
                    self._send_message(
                        tg_client,
                        chat_id=chat_id,
                        text="Usage: /pair <code>",
                        reply_to_message_id=int(message["message_id"]) if "message_id" in message else None,
                    )
                    return
                result = verify_pair_code(code=pair_code, chat_id=str(chat_id), user_id=str(user_id))
                if not result.ok:
                    self._send_message(
                        tg_client,
                        chat_id=chat_id,
                        text=f"Pairing failed: {result.reason}",
                        reply_to_message_id=int(message["message_id"]) if "message_id" in message else None,
                    )
                    return
                self._send_message(
                    tg_client,
                    chat_id=chat_id,
                    text="Pairing succeeded. You can talk to the agent now.",
                    reply_to_message_id=int(message["message_id"]) if "message_id" in message else None,
                )
                return

        edge_headers = {
            "X-Actor-Id": f"telegram:{user_id}",
            "X-Actor-Role": role,
        }
        payload = {
            "message": text.strip(),
            "conversation_id": _conversation_id(chat_id, thread_id),
            "channel": "telegram",
            "channel_id": str(chat_id),
            "user_id": str(user_id),
            "thread_id": str(thread_id) if thread_id is not None else None,
            "actor_id": f"telegram:{user_id}",
        }
        try:
            message_response = edge_client.post("/v1/agent/messages", json=payload, headers=edge_headers)
            message_response.raise_for_status()
            task_id = message_response.json().get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise RuntimeError(f"invalid task_id from edge: {message_response.text}")

            completion_text = self._await_task_completion(edge_client, task_id=task_id, headers=edge_headers)
            self._send_message(
                tg_client,
                chat_id=chat_id,
                text=completion_text,
                reply_to_message_id=int(message["message_id"]) if "message_id" in message else None,
            )
        except Exception as exc:
            self._send_message(
                tg_client,
                chat_id=chat_id,
                text=f"Agent request failed: {exc}",
                reply_to_message_id=int(message["message_id"]) if "message_id" in message else None,
            )

    def _await_task_completion(self, edge_client: httpx.Client, *, task_id: str, headers: dict[str, str]) -> str:
        deadline = time.time() + self.settings.task_wait_timeout_seconds
        last_status: dict[str, Any] | None = None
        while time.time() < deadline:
            response = edge_client.get(f"/v1/agent/tasks/{task_id}", headers=headers)
            response.raise_for_status()
            status_payload = response.json()
            last_status = status_payload
            status = status_payload.get("status")
            if status in {"completed", "failed"}:
                break
            time.sleep(0.5)

        if not last_status:
            raise RuntimeError("task status unavailable")
        if last_status.get("status") not in {"completed", "failed"}:
            raise RuntimeError(f"task timeout: {task_id}")

        audit_response = edge_client.post("/v1/audit/render", json={"task_id": task_id}, headers=headers)
        audit_response.raise_for_status()
        completion_text = _extract_completion_text(audit_response.json())
        if completion_text:
            return completion_text
        status = last_status.get("status")
        error = last_status.get("last_error") or "unknown"
        return f"Task finished with status={status}, error={error}"


def main() -> int:
    settings = TelegramSettings.from_env()
    gateway = TelegramGateway(settings=settings)
    gateway.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
