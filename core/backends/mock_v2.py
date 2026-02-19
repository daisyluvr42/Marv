from __future__ import annotations

from typing import Any

from core.backends.base import BaseCoreBackend


class MockV2CoreBackend(BaseCoreBackend):
    async def chat_completions(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
    ) -> dict[str, Any]:
        latest_input = messages[-1].get("content", "") if messages else ""
        return {
            "id": "chatcmpl_mock_v2",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": f"echo-v2:{latest_input}",
                    },
                }
            ],
        }

    async def embeddings(self, model: str, inputs: list[str]) -> dict[str, Any]:
        data = []
        for idx, value in enumerate(inputs):
            seed = float((len(value) % 5) + 2)
            vector = [seed, seed / 3.0, seed / 9.0, seed / 27.0]
            data.append({"index": idx, "object": "embedding", "embedding": vector})
        return {
            "object": "list",
            "model": model,
            "data": data,
        }
