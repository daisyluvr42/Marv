from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


SUPPORTED_IM_CHANNELS = {"telegram", "discord", "slack", "dingtalk", "feishu", "webchat"}


class IngressError(RuntimeError):
    pass


class IngressIgnored(IngressError):
    pass


class IngressAuthError(IngressError):
    pass


@dataclass(frozen=True)
class IngressMessage:
    channel: str
    channel_id: str
    user_id: str
    text: str
    thread_id: str | None = None
    actor_id: str | None = None
    actor_role: str = "member"
    conversation_id: str | None = None
    metadata: dict[str, Any] | None = None

    def resolve_actor_id(self) -> str:
        return self.actor_id or f"{self.channel}:{self.user_id}"

    def resolve_conversation_id(self) -> str:
        if self.conversation_id:
            return self.conversation_id
        thread = self.thread_id or "0"
        return f"{self.channel}:{self.channel_id}:{thread}"


def verify_ingress_auth(*, channel: str, headers: dict[str, str]) -> None:
    tokens = _load_channel_tokens()
    expected = tokens.get(channel) or tokens.get("*")
    if not expected:
        return

    provided = headers.get("x-marv-token") or _extract_bearer_token(headers.get("authorization"))
    if provided == expected:
        return
    raise IngressAuthError("invalid or missing ingress token")


def parse_ingress_payload(channel: str, payload: dict[str, Any]) -> IngressMessage:
    normalized_channel = channel.strip().lower()
    if normalized_channel not in SUPPORTED_IM_CHANNELS:
        raise IngressError(f"unsupported channel: {channel}")
    if not isinstance(payload, dict):
        raise IngressError("payload must be a JSON object")

    if normalized_channel == "slack":
        return _parse_slack_payload(payload)
    if normalized_channel == "discord":
        return _parse_discord_payload(payload)
    if normalized_channel == "dingtalk":
        return _parse_dingtalk_payload(payload)
    if normalized_channel == "feishu":
        return _parse_feishu_payload(payload)
    if normalized_channel == "telegram":
        return _parse_telegram_payload(payload)
    return _parse_generic_payload(normalized_channel, payload)


def parse_slack_url_verification(payload: dict[str, Any]) -> str | None:
    if not isinstance(payload, dict):
        return None
    if str(payload.get("type", "")).strip().lower() != "url_verification":
        return None
    challenge = str(payload.get("challenge", "")).strip()
    return challenge or None


def _parse_slack_payload(payload: dict[str, Any]) -> IngressMessage:
    event = payload.get("event")
    if not isinstance(event, dict):
        return _parse_generic_payload("slack", payload)
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        raise IngressIgnored("ignore bot message")
    text = str(event.get("text", "")).strip()
    channel_id = str(event.get("channel", "")).strip()
    user_id = str(event.get("user", "")).strip()
    if not text or not channel_id or not user_id:
        raise IngressError("slack event missing required fields")
    thread = str(event.get("thread_ts") or event.get("ts") or "").strip() or None
    return IngressMessage(
        channel="slack",
        channel_id=channel_id,
        user_id=user_id,
        text=text,
        thread_id=thread,
        metadata={"raw_type": str(payload.get("type", ""))},
    )


def _parse_discord_payload(payload: dict[str, Any]) -> IngressMessage:
    content = str(payload.get("content", "")).strip()
    channel_id = str(payload.get("channel_id", "")).strip()
    author = payload.get("author")
    if not isinstance(author, dict):
        return _parse_generic_payload("discord", payload)
    if bool(author.get("bot")):
        raise IngressIgnored("ignore bot message")
    user_id = str(author.get("id", "")).strip()
    thread_id = str(payload.get("thread_id", "")).strip() or None
    if not content or not channel_id or not user_id:
        raise IngressError("discord payload missing required fields")
    return IngressMessage(
        channel="discord",
        channel_id=channel_id,
        user_id=user_id,
        text=content,
        thread_id=thread_id,
    )


def _parse_dingtalk_payload(payload: dict[str, Any]) -> IngressMessage:
    text = ""
    text_obj = payload.get("text")
    if isinstance(text_obj, dict):
        text = str(text_obj.get("content", "")).strip()
    if not text:
        text = str(payload.get("content", "")).strip()
    conversation = str(payload.get("conversationId", "")).strip() or str(payload.get("chat_id", "")).strip()
    user_id = str(payload.get("senderStaffId", "")).strip() or str(payload.get("user_id", "")).strip()
    if not text or not conversation or not user_id:
        return _parse_generic_payload("dingtalk", payload)
    return IngressMessage(
        channel="dingtalk",
        channel_id=conversation,
        user_id=user_id,
        text=text,
        thread_id=str(payload.get("sessionWebhookExpiredTime", "")).strip() or None,
    )


def _parse_feishu_payload(payload: dict[str, Any]) -> IngressMessage:
    event = payload.get("event")
    if not isinstance(event, dict):
        return _parse_generic_payload("feishu", payload)
    sender = event.get("sender")
    if not isinstance(sender, dict):
        return _parse_generic_payload("feishu", payload)
    sender_id = sender.get("sender_id")
    if not isinstance(sender_id, dict):
        return _parse_generic_payload("feishu", payload)
    user_id = str(sender_id.get("open_id", "")).strip() or str(sender_id.get("user_id", "")).strip()

    message = event.get("message")
    if not isinstance(message, dict):
        return _parse_generic_payload("feishu", payload)
    chat_id = str(message.get("chat_id", "")).strip()
    content_raw = message.get("content", "")
    text = ""
    if isinstance(content_raw, str):
        try:
            decoded = json.loads(content_raw)
        except json.JSONDecodeError:
            text = content_raw.strip()
        else:
            if isinstance(decoded, dict):
                text = str(decoded.get("text", "")).strip()
    elif isinstance(content_raw, dict):
        text = str(content_raw.get("text", "")).strip()

    if not text or not chat_id or not user_id:
        return _parse_generic_payload("feishu", payload)
    thread_id = str(message.get("thread_id", "")).strip() or None
    return IngressMessage(
        channel="feishu",
        channel_id=chat_id,
        user_id=user_id,
        text=text,
        thread_id=thread_id,
    )


def _parse_telegram_payload(payload: dict[str, Any]) -> IngressMessage:
    message = payload.get("message")
    if isinstance(message, dict):
        chat = message.get("chat")
        from_user = message.get("from")
        if not isinstance(chat, dict) or not isinstance(from_user, dict):
            raise IngressError("telegram message missing chat/from")
        text = str(message.get("text", "")).strip()
        channel_id = str(chat.get("id", "")).strip()
        user_id = str(from_user.get("id", "")).strip()
        thread_id = str(message.get("message_thread_id", "")).strip() or None
    else:
        text = str(payload.get("text", "")).strip()
        channel_id = str(payload.get("chat_id", "")).strip() or str(payload.get("channel_id", "")).strip()
        user_id = str(payload.get("user_id", "")).strip()
        thread_id = str(payload.get("thread_id", "")).strip() or None

    if not text or not channel_id or not user_id:
        raise IngressError("telegram payload missing required fields")
    return IngressMessage(
        channel="telegram",
        channel_id=channel_id,
        user_id=user_id,
        text=text,
        thread_id=thread_id,
    )


def _parse_generic_payload(channel: str, payload: dict[str, Any]) -> IngressMessage:
    text = str(payload.get("text", "")).strip() or str(payload.get("message", "")).strip()
    channel_id = str(payload.get("channel_id", "")).strip() or str(payload.get("chat_id", "")).strip()
    user_id = str(payload.get("user_id", "")).strip() or str(payload.get("sender_id", "")).strip()
    if not text or not channel_id or not user_id:
        raise IngressError("payload missing required fields: text/channel_id/user_id")
    return IngressMessage(
        channel=channel,
        channel_id=channel_id,
        user_id=user_id,
        text=text,
        thread_id=str(payload.get("thread_id", "")).strip() or None,
        actor_id=str(payload.get("actor_id", "")).strip() or None,
        actor_role=str(payload.get("actor_role", "member")).strip() or "member",
        conversation_id=str(payload.get("conversation_id", "")).strip() or None,
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
    )


def _load_channel_tokens() -> dict[str, str]:
    raw = os.getenv("IM_INGRESS_TOKENS_JSON", "").strip()
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            result: dict[str, str] = {}
            for key, value in payload.items():
                token = str(value).strip()
                if token:
                    result[str(key).strip().lower()] = token
            if result:
                return result

    fallback = os.getenv("IM_INGRESS_TOKEN", "").strip()
    if fallback:
        return {"*": fallback}
    return {}


def _extract_bearer_token(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.lower().startswith("bearer "):
        token = text[7:].strip()
        return token or None
    return None
