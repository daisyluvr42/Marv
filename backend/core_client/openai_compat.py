from __future__ import annotations

import json
import os
import re
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
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    auth_env: str | None = None
    api_key_env: str | None = None
    oauth_token_env: str | None = None
    static_headers: dict[str, str] = field(default_factory=dict)


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
                    "models": _collect_provider_models(item),
                    "route_tiers": _collect_provider_route_tiers(item),
                    "auth_header": item.auth_header,
                    "auth_scheme": item.auth_scheme,
                    "auth_env": item.auth_env,
                    "api_key_env": item.api_key_env,
                    "oauth_token_env": item.oauth_token_env,
                    "static_headers": item.static_headers,
                }
                for item in self._providers
            ],
        }

    def provider_capabilities(self) -> dict[str, object]:
        providers: list[dict[str, object]] = []
        locality_counts: dict[str, int] = {"local": 0, "cloud": 0}
        auth_counts: dict[str, int] = {"api": 0, "oauth": 0, "none": 0}
        tier_counts: dict[str, int] = {"local_light": 0, "local_main": 0, "cloud_high": 0}

        for item in self._providers:
            models = _collect_provider_models(item)
            route_tiers = _collect_provider_route_tiers(item)
            locality_counts[item.locality] = locality_counts.get(item.locality, 0) + 1
            auth_counts[item.auth_mode] = auth_counts.get(item.auth_mode, 0) + 1
            tier_counts[item.tier] = tier_counts.get(item.tier, 0) + 1
            providers.append(
                {
                    "name": item.name,
                    "base_url": item.base_url,
                    "tier": item.tier,
                    "locality": item.locality,
                    "auth_mode": item.auth_mode,
                    "priority": item.priority,
                    "route_tiers": route_tiers,
                    "models": models,
                    "supports": {
                        "chat_completions": True,
                        "embeddings": True,
                    },
                }
            )

        return {
            "count": len(providers),
            "providers": providers,
            "summary": {
                "locality": locality_counts,
                "auth_mode": auth_counts,
                "tier": tier_counts,
            },
        }

    def model_catalog(self) -> dict[str, object]:
        models: list[dict[str, object]] = []
        for item in sorted(self._providers, key=lambda entry: (entry.priority, entry.name)):
            route_map: dict[str, str] = {}
            for tier, model in item.models_by_tier.items():
                route_map[tier] = model
            if item.model_default:
                route_map.setdefault(item.tier, item.model_default)

            by_model: dict[str, list[str]] = {}
            for tier, model in route_map.items():
                by_model.setdefault(model, []).append(tier)
            if item.model_default:
                by_model.setdefault(item.model_default, [])

            for model, tiers in sorted(by_model.items()):
                models.append(
                    {
                        "provider": item.name,
                        "model": model,
                        "tiers": sorted(set(tiers)),
                        "default_model": bool(item.model_default and model == item.model_default),
                        "locality": item.locality,
                        "auth_mode": item.auth_mode,
                        "priority": item.priority,
                    }
                )

        return {
            "count": len(models),
            "models": models,
        }

    def provider_auth_status(self) -> dict[str, object]:
        providers: list[dict[str, object]] = []
        loaded_count = 0
        for item in self._providers:
            token, loaded_from = _resolve_provider_token(item)
            loaded = bool(token)
            if loaded:
                loaded_count += 1
            providers.append(
                {
                    "name": item.name,
                    "auth_mode": item.auth_mode,
                    "auth_header": item.auth_header,
                    "auth_scheme": item.auth_scheme,
                    "auth_env_candidates": _auth_env_candidates(item),
                    "credential_loaded": loaded,
                    "loaded_from_env": loaded_from,
                }
            )
        return {
            "count": len(providers),
            "credential_loaded_count": loaded_count,
            "providers": providers,
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
                    response = await client.request(
                        method,
                        path,
                        json=json,
                        headers=_build_provider_headers(provider),
                    )
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
                auth_header_raw = item.get("auth_header")
                auth_header = str(auth_header_raw).strip() if isinstance(auth_header_raw, str) else "Authorization"
                if not auth_header:
                    auth_header = "Authorization"
                auth_scheme_raw = item.get("auth_scheme")
                if isinstance(auth_scheme_raw, str):
                    auth_scheme = auth_scheme_raw.strip()
                else:
                    auth_scheme = "Bearer" if auth_header.lower() == "authorization" and auth_mode != "none" else ""
                auth_env = str(item.get("auth_env", "")).strip() or None
                api_key_env = str(item.get("api_key_env", "")).strip() or None
                oauth_token_env = str(item.get("oauth_token_env", "")).strip() or None
                static_headers = _normalize_static_headers(item.get("headers"))
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
                        auth_header=auth_header,
                        auth_scheme=auth_scheme,
                        auth_env=auth_env,
                        api_key_env=api_key_env,
                        oauth_token_env=oauth_token_env,
                        static_headers=static_headers,
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


def _collect_provider_models(provider: ProviderEndpoint) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    if provider.model_default and provider.model_default not in seen:
        models.append(provider.model_default)
        seen.add(provider.model_default)
    for model in provider.models_by_tier.values():
        if model not in seen:
            models.append(model)
            seen.add(model)
    return models


def _collect_provider_route_tiers(provider: ProviderEndpoint) -> list[str]:
    tiers = {provider.tier}
    tiers.update(provider.models_by_tier.keys())
    normalized = [item for item in tiers if item in {"local_light", "local_main", "cloud_high"}]
    return sorted(normalized)


def _normalize_static_headers(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    headers: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            continue
        header_key = key.strip()
        header_value = item.strip()
        if not header_key or not header_value:
            continue
        headers[header_key] = header_value
    return headers


def _auth_env_candidates(provider: ProviderEndpoint) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for key in [
        provider.auth_env,
        provider.oauth_token_env if provider.auth_mode == "oauth" else None,
        provider.api_key_env if provider.auth_mode == "api" else None,
        _default_provider_auth_env(provider),
    ]:
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        candidates.append(key)
    return candidates


def _default_provider_auth_env(provider: ProviderEndpoint) -> str | None:
    slug = _provider_env_slug(provider.name)
    if not slug:
        return None
    if provider.auth_mode == "oauth":
        return f"CORE_PROVIDER_{slug}_OAUTH_TOKEN"
    if provider.auth_mode == "api":
        return f"CORE_PROVIDER_{slug}_API_KEY"
    return None


def _provider_env_slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return cleaned.upper()


def _resolve_provider_token(provider: ProviderEndpoint) -> tuple[str | None, str | None]:
    if provider.auth_mode == "none":
        return None, None
    for env_key in _auth_env_candidates(provider):
        token = os.getenv(env_key, "").strip()
        if token:
            return token, env_key
    return None, None


def _build_provider_headers(provider: ProviderEndpoint) -> dict[str, str] | None:
    headers = dict(provider.static_headers)
    token, _ = _resolve_provider_token(provider)
    if token:
        header_name = provider.auth_header.strip() or "Authorization"
        scheme = provider.auth_scheme.strip()
        if header_name.lower() == "authorization":
            header_value = f"{scheme} {token}" if scheme else token
        else:
            header_value = token if not scheme else f"{scheme} {token}"
        headers[header_name] = header_value
    return headers or None


_core_client = CoreClient()


def get_core_client() -> CoreClient:
    return _core_client
