from __future__ import annotations

from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from backend.agent.config import get_settings


class CoreClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.core_base_url
        self._timeout = settings.request_timeout_seconds
        self._max_retries = settings.max_retries

    async def health_check(self) -> dict[str, Any]:
        return await self._request("GET", "/health")

    async def chat_completions(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        model: str = "mock",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        return await self._request("POST", "/v1/chat/completions", json=payload)

    async def embeddings(
        self,
        input_text: str | list[str],
        model: str = "mock-embedding",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "input": input_text,
        }
        return await self._request("POST", "/v1/embeddings", json=payload)

    async def _request(self, method: str, path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=0.2, min=0.2, max=2),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)),
            reraise=True,
        ):
            with attempt:
                async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
                    response = await client.request(method, path, json=json)
                    response.raise_for_status()
                    return response.json()
        raise RuntimeError("unreachable")


_core_client = CoreClient()


def get_core_client() -> CoreClient:
    return _core_client
