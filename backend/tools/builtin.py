from __future__ import annotations

from backend.tools.registry import tool


@tool(
    name="mock_web_search",
    risk="read_only",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
    },
)
def mock_web_search(query: str) -> dict[str, object]:
    return {
        "query": query,
        "results": [
            {"title": f"Mock result for {query}", "url": "https://example.com/mock"},
        ],
    }


@tool(
    name="mock_external_write",
    risk="external_write",
    schema={
        "type": "object",
        "properties": {
            "target": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["target", "content"],
    },
)
def mock_external_write(target: str, content: str) -> dict[str, object]:
    return {
        "target": target,
        "status": "written",
        "bytes": len(content.encode("utf-8")),
    }
