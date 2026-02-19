from __future__ import annotations

from typing import Any

from backend.core_client.openai_compat import get_core_client


_EMBEDDING_CACHE: dict[tuple[str, str], list[float]] = {}


async def embed_text(text: str, model: str = "mock-embedding") -> list[float]:
    cache_key = (model, text)
    cached = _EMBEDDING_CACHE.get(cache_key)
    if cached is not None:
        return cached

    response = await get_core_client().embeddings(input_text=text, model=model)
    vector = _extract_single_embedding(response)
    _EMBEDDING_CACHE[cache_key] = vector
    return vector


def _extract_single_embedding(response: dict[str, Any]) -> list[float]:
    data = response.get("data", [])
    if not data:
        return []
    vector = data[0].get("embedding", [])
    return [float(v) for v in vector]


def embedding_cache_size() -> int:
    return len(_EMBEDDING_CACHE)
