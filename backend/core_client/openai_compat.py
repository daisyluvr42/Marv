from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from backend.agent.config import get_settings


@dataclass(frozen=True)
class ProviderEndpoint:
    name: str
    base_url: str
    timeout_seconds: float
    max_retries: int
    tier: str = "local_main"  # local_light | local_main | cloud_high
    locality: str = "local"  # local | cloud
    auth_mode: str = "api"  # api | oauth | none
    priority: int = 100
    model_default: str | None = None
    models_by_tier: dict[str, str] = field(default_factory=dict)


class CoreClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._providers = _load_providers(settings)

    async def health_check(self) -> dict[str, Any]:
        return await self._request("GET", "/health")

    async def chat_completions(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        model: str = "mock",
        route_tier: str | None = None,
        preferred_locality: str | None = None,
        allow_cloud_fallback: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        return await self._request(
            "POST",
            "/v1/chat/completions",
            json=payload,
            route_tier=route_tier,
            preferred_locality=preferred_locality,
            allow_cloud_fallback=allow_cloud_fallback,
        )

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

    def provider_status(self) -> dict[str, object]:
        return {
            "count": len(self._providers),
            "providers": [
                {
                    "name": item.name,
                    "base_url": item.base_url,
                    "timeout_seconds": item.timeout_seconds,
                    "max_retries": item.max_retries,
                    "tier": item.tier,
                    "locality": item.locality,
                    "auth_mode": item.auth_mode,
                    "priority": item.priority,
                    "model_default": item.model_default,
                    "models_by_tier": item.models_by_tier,
                }
                for item in self._providers
            ],
        }

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        route_tier: str | None = None,
        preferred_locality: str | None = None,
        allow_cloud_fallback: bool = True,
    ) -> dict[str, Any]:
        errors: list[dict[str, str]] = []
        candidates = _order_providers(
            providers=self._providers,
            route_tier=route_tier,
            preferred_locality=preferred_locality,
            allow_cloud_fallback=allow_cloud_fallback,
        )
        for provider in candidates:
            provider_payload = _build_payload_for_provider(
                path=path,
                payload=json,
                provider=provider,
                route_tier=route_tier,
            )
            try:
                return await self._request_with_provider(provider, method, path, provider_payload)
            except Exception as exc:  # pragma: no cover - integration guarded
                errors.append({"provider": provider.name, "error": str(exc)})
                continue
        raise RuntimeError(f"all providers failed: {errors}")

    async def _request_with_provider(
        self,
        provider: ProviderEndpoint,
        method: str,
        path: str,
        json: dict[str, Any] | None,
    ) -> dict[str, Any]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(provider.max_retries + 1),
            wait=wait_exponential(multiplier=0.2, min=0.2, max=2),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)),
            reraise=True,
        ):
            with attempt:
                async with httpx.AsyncClient(base_url=provider.base_url, timeout=provider.timeout_seconds) as client:
                    response = await client.request(method, path, json=json)
                    if response.status_code >= 500 or response.status_code == 429:
                        response.raise_for_status()
                    elif response.status_code >= 400:
                        # do not retry hard client failures, but let fallback provider try
                        raise RuntimeError(f"provider={provider.name} non-retryable status={response.status_code}: {response.text}")
                    payload = response.json()
                    if isinstance(payload, dict):
                        payload.setdefault("_provider", provider.name)
                        payload.setdefault("_provider_tier", provider.tier)
                        payload.setdefault("_provider_locality", provider.locality)
                        payload.setdefault("_provider_auth_mode", provider.auth_mode)
                        model = json.get("model") if isinstance(json, dict) else None
                        if isinstance(model, str) and model.strip():
                            payload.setdefault("_provider_model", model)
                    return payload
        raise RuntimeError("unreachable")


def _load_providers(settings: Any) -> list[ProviderEndpoint]:
    raw = settings.core_provider_matrix_json
    providers: list[ProviderEndpoint] = []
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = []
        if isinstance(payload, list):
            for idx, item in enumerate(payload):
                if not isinstance(item, dict):
                    continue
                base_url = str(item.get("base_url", "")).strip().rstrip("/")
                if not base_url:
                    continue
                locality = _normalize_locality(item.get("locality")) or _infer_locality(base_url)
                tier = _normalize_tier(item.get("tier")) or ("local_main" if locality == "local" else "cloud_high")
                auth_mode = _normalize_auth_mode(item.get("auth_mode"))
                if auth_mode is None:
                    auth_mode = "oauth" if "oauth" in str(item.get("name", "")).lower() else "api"
                priority = item.get("priority", idx * 10)
                if not isinstance(priority, int):
                    priority = idx * 10
                model_default = item.get("model")
                if not isinstance(model_default, str):
                    model_default = None
                models_by_tier = _parse_models_by_tier(item)
                providers.append(
                    ProviderEndpoint(
                        name=str(item.get("name", f"provider-{idx + 1}")).strip() or f"provider-{idx + 1}",
                        base_url=base_url,
                        timeout_seconds=max(0.5, float(item.get("timeout_seconds", settings.request_timeout_seconds))),
                        max_retries=max(0, int(item.get("max_retries", settings.max_retries))),
                        tier=tier,
                        locality=locality,
                        auth_mode=auth_mode,
                        priority=priority,
                        model_default=model_default,
                        models_by_tier=models_by_tier,
                    )
                )
    if providers:
        return providers
    default_locality = _infer_locality(settings.core_base_url)
    default_tier = "local_main" if default_locality == "local" else "cloud_high"
    return [
        ProviderEndpoint(
            name="default",
            base_url=settings.core_base_url,
            timeout_seconds=settings.request_timeout_seconds,
            max_retries=settings.max_retries,
            locality=default_locality,
            tier=default_tier,
            auth_mode="api",
            priority=0,
        )
    ]


def _normalize_tier(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    return lowered if lowered in {"local_light", "local_main", "cloud_high"} else None


def _normalize_locality(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    return lowered if lowered in {"local", "cloud"} else None


def _normalize_auth_mode(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    return lowered if lowered in {"api", "oauth", "none"} else None


def _infer_locality(base_url: str) -> str:
    host = (urlparse(base_url).hostname or "").strip().lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return "local"
    if host.startswith("127."):
        return "local"
    return "cloud"


def _parse_models_by_tier(item: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    raw_models = item.get("models")
    if isinstance(raw_models, dict):
        for key, value in raw_models.items():
            tier = _normalize_tier(key)
            if tier is None or not isinstance(value, str):
                continue
            model = value.strip()
            if model:
                result[tier] = model

    legacy_keys = {
        "local_light": item.get("model_local_light"),
        "local_main": item.get("model_local_main"),
        "cloud_high": item.get("model_cloud_high"),
    }
    for tier, value in legacy_keys.items():
        if isinstance(value, str) and value.strip():
            result[tier] = value.strip()
    return result


def _build_payload_for_provider(
    *,
    path: str,
    payload: dict[str, Any] | None,
    provider: ProviderEndpoint,
    route_tier: str | None,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    copied = dict(payload)
    if path != "/v1/chat/completions":
        return copied
    requested_model = copied.get("model")
    if not isinstance(requested_model, str):
        requested_model = ""
    model = requested_model.strip()
    if model and model not in {"auto", "mock"}:
        return copied
    tier = _normalize_tier(route_tier) or provider.tier
    selected = provider.models_by_tier.get(tier) or provider.model_default
    if selected:
        copied["model"] = selected
    return copied


def _order_providers(
    *,
    providers: list[ProviderEndpoint],
    route_tier: str | None,
    preferred_locality: str | None,
    allow_cloud_fallback: bool,
) -> list[ProviderEndpoint]:
    if not providers:
        return providers
    desired_tier = _normalize_tier(route_tier) or "local_main"
    desired_locality = _normalize_locality(preferred_locality)

    candidates = [item for item in providers if allow_cloud_fallback or item.locality == "local"]
    if not candidates:
        candidates = list(providers)

    fallback_preference = {
        "local_light": ("local_light", "local_main", "cloud_high"),
        "local_main": ("local_main", "local_light", "cloud_high"),
        "cloud_high": ("cloud_high", "local_main", "local_light"),
    }[desired_tier]

    tier_rank = {value: idx for idx, value in enumerate(fallback_preference)}

    def score(provider: ProviderEndpoint) -> tuple[int, int, int, str]:
        tier_score = tier_rank.get(provider.tier, len(tier_rank) + 1)
        locality_penalty = 0
        if desired_locality and provider.locality != desired_locality:
            locality_penalty = 2
        return (tier_score, locality_penalty, provider.priority, provider.name)

    return sorted(candidates, key=score)


_core_client = CoreClient()


def get_core_client() -> CoreClient:
    return _core_client
