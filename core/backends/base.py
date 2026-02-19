from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseCoreBackend(ABC):
    @abstractmethod
    async def chat_completions(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def embeddings(self, model: str, inputs: list[str]) -> dict[str, Any]:
        raise NotImplementedError
