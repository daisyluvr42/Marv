from __future__ import annotations

import importlib
import json
import pkgutil
from dataclasses import dataclass
from typing import Any, Callable

from sqlmodel import select

from backend.agent.state import now_ts
from backend.storage.db import get_session
from backend.storage.models import ToolRegistry


@dataclass
class ToolSpec:
    name: str
    risk: str
    schema: dict[str, Any]
    version: str = "0.1.0"
    enabled: bool = True

    @property
    def requires_approval(self) -> bool:
        return self.risk == "external_write"


TOOL_REGISTRY: dict[str, ToolSpec] = {}
TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {}


def tool(name: str, risk: str, schema: dict[str, Any]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        spec = ToolSpec(name=name, risk=risk, schema=schema)
        TOOL_REGISTRY[name] = spec
        TOOL_FUNCTIONS[name] = func
        setattr(func, "__tool_spec__", spec)
        return func

    return decorator


def scan_tools(package_name: str = "backend.tools") -> None:
    package = importlib.import_module(package_name)
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module_info.name)


def list_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "version": spec.version,
            "risk": spec.risk,
            "requires_approval": spec.requires_approval,
            "schema": spec.schema,
            "enabled": spec.enabled,
        }
        for spec in sorted(TOOL_REGISTRY.values(), key=lambda item: item.name)
    ]


def get_tool_spec(name: str) -> ToolSpec | None:
    return TOOL_REGISTRY.get(name)


def get_tool_function(name: str) -> Callable[..., Any] | None:
    return TOOL_FUNCTIONS.get(name)


def register_runtime_tool(
    *,
    name: str,
    risk: str,
    schema: dict[str, Any],
    func: Callable[..., Any],
    version: str = "runtime",
    enabled: bool = True,
) -> None:
    TOOL_REGISTRY[name] = ToolSpec(name=name, risk=risk, schema=schema, version=version, enabled=enabled)
    TOOL_FUNCTIONS[name] = func


def sync_tools_registry() -> None:
    with get_session() as session:
        for tool_data in list_tools():
            existing = session.exec(select(ToolRegistry).where(ToolRegistry.name == tool_data["name"])).first()
            if existing is None:
                existing = ToolRegistry(
                    name=tool_data["name"],
                    version=tool_data["version"],
                    risk=tool_data["risk"],
                    requires_approval=tool_data["requires_approval"],
                    schema_payload=json.dumps(tool_data["schema"], ensure_ascii=True),
                    enabled=tool_data["enabled"],
                    updated_at=now_ts(),
                )
            else:
                existing.version = tool_data["version"]
                existing.risk = tool_data["risk"]
                existing.requires_approval = tool_data["requires_approval"]
                existing.schema_payload = json.dumps(tool_data["schema"], ensure_ascii=True)
                existing.enabled = tool_data["enabled"]
                existing.updated_at = now_ts()
            session.add(existing)
        session.commit()
