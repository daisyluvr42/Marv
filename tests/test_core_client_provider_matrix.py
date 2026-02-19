from __future__ import annotations

from types import SimpleNamespace

from backend.core_client.openai_compat import ProviderEndpoint, _build_payload_for_provider, _load_providers, _order_providers


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
