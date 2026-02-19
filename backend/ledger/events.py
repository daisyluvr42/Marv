from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class BaseEvent:
    conversation_id: str
    ts: int
    task_id: Optional[str] = None
    actor_id: Optional[str] = None
    event_id: Optional[str] = None
    type: str = field(init=False)

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("type", None)
        return payload


@dataclass
class InputEvent(BaseEvent):
    message: str = ""

    def __post_init__(self) -> None:
        self.type = "InputEvent"


@dataclass
class PlanEvent(BaseEvent):
    plan: str = ""

    def __post_init__(self) -> None:
        self.type = "PlanEvent"


@dataclass
class RouteEvent(BaseEvent):
    route: str = ""

    def __post_init__(self) -> None:
        self.type = "RouteEvent"


@dataclass
class CompletionEvent(BaseEvent):
    response_text: str = ""

    def __post_init__(self) -> None:
        self.type = "CompletionEvent"


@dataclass
class PatchCommittedEvent(BaseEvent):
    proposal_id: str = ""
    revision: str = ""
    risk_level: str = ""
    patch: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type = "PatchCommittedEvent"


@dataclass
class PatchRolledBackEvent(BaseEvent):
    revision: str = ""
    rollback_revision: str = ""

    def __post_init__(self) -> None:
        self.type = "PatchRolledBackEvent"
