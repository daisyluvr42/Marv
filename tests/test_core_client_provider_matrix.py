from __future__ import annotations

from types import SimpleNamespace

from backend.core_client.openai_compat import (
    CoreClient,
    ProviderEndpoint,
    _build_payload_for_provider,
    _build_provider_headers,
    _load_providers,
    _order_providers,
)


def test_load_providers_from_matrix_json() -> None:
    settings = SimpleNamespace(
        core_provider_matrix_json='[{"name":"p1","base_url":"http://127.0.0.1:9000","timeout_seconds":3,"max_retries":1,"tier":"local_light","models":{"local_light":"phi-mini","local_main":"phi"}},{"name":"p2","base_url":"https://api.example.com","auth_mode":"oauth","priority":3,"model":"gpt-4.1"}]',
        core_base_url="http://127.0.0.1:9000",
        request_timeout_seconds=5.0,
        max_retries=2,
    )
    providers = _load_providers(settings)
    assert len(providers) == 2
    assert providers[0].name == "p1"
    assert providers[1].name == "p2"
    assert providers[0].tier == "local_light"
    assert providers[0].locality == "local"
    assert providers[0].models_by_tier["local_light"] == "phi-mini"
    assert providers[1].tier == "cloud_high"
    assert providers[1].locality == "cloud"
    assert providers[1].auth_mode == "oauth"
    assert providers[1].model_default == "gpt-4.1"


def test_load_providers_parses_auth_header_and_env() -> None:
    settings = SimpleNamespace(
        core_provider_matrix_json='[{"name":"secure-provider","base_url":"https://api.example.com","auth_mode":"api","auth_header":"X-API-Key","auth_scheme":"","auth_env":"MY_PROVIDER_TOKEN","headers":{"X-Tenant":"marv"}}]',
        core_base_url="http://127.0.0.1:9000",
        request_timeout_seconds=5.0,
        max_retries=2,
    )
    providers = _load_providers(settings)
    assert len(providers) == 1
    assert providers[0].auth_mode == "api"
    assert providers[0].auth_header == "X-API-Key"
    assert providers[0].auth_scheme == ""
    assert providers[0].auth_env == "MY_PROVIDER_TOKEN"
    assert providers[0].static_headers["X-Tenant"] == "marv"


def test_load_providers_falls_back_to_default() -> None:
    settings = SimpleNamespace(
        core_provider_matrix_json="",
        core_base_url="http://127.0.0.1:9000",
        request_timeout_seconds=5.0,
        max_retries=2,
    )
    providers = _load_providers(settings)
    assert len(providers) == 1
    assert providers[0].name == "default"
    assert providers[0].tier == "local_main"
    assert providers[0].locality == "local"


def test_order_providers_prefers_local_tier_and_can_filter_cloud() -> None:
    providers = [
        ProviderEndpoint(name="cloud", base_url="https://api.example.com", timeout_seconds=5, max_retries=1, tier="cloud_high", locality="cloud", priority=1),
        ProviderEndpoint(name="local-main", base_url="http://127.0.0.1:9000", timeout_seconds=5, max_retries=1, tier="local_main", locality="local", priority=2),
        ProviderEndpoint(name="local-light", base_url="http://127.0.0.1:9001", timeout_seconds=5, max_retries=1, tier="local_light", locality="local", priority=3),
    ]

    local_only = _order_providers(
        providers=providers,
        route_tier="local_light",
        preferred_locality="local",
        allow_cloud_fallback=False,
    )
    assert [item.name for item in local_only] == ["local-light", "local-main"]

    with_cloud = _order_providers(
        providers=providers,
        route_tier="local_light",
        preferred_locality="local",
        allow_cloud_fallback=True,
    )
    assert [item.name for item in with_cloud] == ["local-light", "local-main", "cloud"]


def test_build_payload_uses_provider_tier_model_for_auto() -> None:
    provider = ProviderEndpoint(
        name="local-main",
        base_url="http://127.0.0.1:9000",
        timeout_seconds=5,
        max_retries=1,
        tier="local_main",
        locality="local",
        models_by_tier={"local_light": "phi-mini", "local_main": "phi-main"},
        model_default="phi-fallback",
    )
    payload = _build_payload_for_provider(
        path="/v1/chat/completions",
        payload={"model": "auto", "messages": [], "stream": False},
        provider=provider,
        route_tier="local_light",
    )
    assert payload is not None
    assert payload["model"] == "phi-mini"


def test_build_payload_keeps_explicit_model() -> None:
    provider = ProviderEndpoint(
        name="cloud",
        base_url="https://api.example.com",
        timeout_seconds=5,
        max_retries=1,
        tier="cloud_high",
        locality="cloud",
        models_by_tier={"cloud_high": "gpt-4.1"},
        model_default="gpt-4.1",
    )
    payload = _build_payload_for_provider(
        path="/v1/chat/completions",
        payload={"model": "my-fixed-model", "messages": [], "stream": False},
        provider=provider,
        route_tier="cloud_high",
    )
    assert payload is not None
    assert payload["model"] == "my-fixed-model"


def test_build_provider_headers_uses_env_token(monkeypatch) -> None:
    provider = ProviderEndpoint(
        name="secure-provider",
        base_url="https://api.example.com",
        timeout_seconds=5,
        max_retries=1,
        tier="cloud_high",
        locality="cloud",
        auth_mode="api",
        auth_header="X-API-Key",
        auth_scheme="",
        auth_env="MY_PROVIDER_TOKEN",
        static_headers={"X-Tenant": "marv"},
    )
    monkeypatch.setenv("MY_PROVIDER_TOKEN", "secret-123")
    headers = _build_provider_headers(provider)
    assert headers is not None
    assert headers["X-Tenant"] == "marv"
    assert headers["X-API-Key"] == "secret-123"


def test_provider_capabilities_and_model_catalog() -> None:
    client = CoreClient()
    providers = [
        ProviderEndpoint(
            name="local-main",
            base_url="http://127.0.0.1:9000",
            timeout_seconds=5,
            max_retries=1,
            tier="local_main",
            locality="local",
            auth_mode="api",
            priority=2,
            model_default="phi-main",
            models_by_tier={"local_light": "phi-mini", "local_main": "phi-main"},
        ),
        ProviderEndpoint(
            name="cloud-oauth",
            base_url="https://api.example.com",
            timeout_seconds=8,
            max_retries=1,
            tier="cloud_high",
            locality="cloud",
            auth_mode="oauth",
            priority=5,
            model_default="gpt-4.1",
            models_by_tier={"cloud_high": "gpt-4.1"},
        ),
    ]
    client._providers = providers  # type: ignore[attr-defined]

    capabilities = client.provider_capabilities()
    assert capabilities["count"] == 2
    summary = capabilities["summary"]
    assert summary["locality"]["local"] == 1
    assert summary["locality"]["cloud"] == 1
    assert summary["auth_mode"]["oauth"] == 1

    catalog = client.model_catalog()
    assert catalog["count"] >= 2
    model_keys = {(item["provider"], item["model"]) for item in catalog["models"]}
    assert ("local-main", "phi-main") in model_keys
    assert ("cloud-oauth", "gpt-4.1") in model_keys


def test_provider_auth_status_reports_loaded_tokens(monkeypatch) -> None:
    provider = ProviderEndpoint(
        name="cloud-oauth",
        base_url="https://api.example.com",
        timeout_seconds=8,
        max_retries=1,
        tier="cloud_high",
        locality="cloud",
        auth_mode="oauth",
        oauth_token_env="CLOUD_OAUTH_TOKEN",
    )
    monkeypatch.setenv("CLOUD_OAUTH_TOKEN", "oauth-secret")
    client = CoreClient()
    client._providers = [provider]  # type: ignore[attr-defined]

    payload = client.provider_auth_status()
    assert payload["count"] == 1
    assert payload["credential_loaded_count"] == 1
    item = payload["providers"][0]
    assert item["credential_loaded"] is True
    assert item["loaded_from_env"] == "CLOUD_OAUTH_TOKEN"
