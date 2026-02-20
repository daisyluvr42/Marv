from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


PiRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class PiMessage:
    role: PiRole
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PiTurnContext:
    system_prompts: list[str] = field(default_factory=list)
    messages: list[PiMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

