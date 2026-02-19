from __future__ import annotations

import json
import os
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any


VALID_SECURITY = {"deny", "allowlist", "full"}
VALID_ASK = {"off", "on-miss", "always"}
VALID_ASK_FALLBACK = {"deny", "allowlist", "full"}
DEFAULT_MAIN_AGENT = "main"


def _default_policy() -> dict[str, Any]:
    # Keep runtime backward compatible: full access unless user tightens policy.
    return {
        "security": "full",
        "ask": "off",
        "ask_fallback": "deny",
        "allowlist": [],
    }


def _default_config() -> dict[str, Any]:
    return {
        "version": 1,
        "defaults": _default_policy(),
        "agents": {},
    }


def _merge_policy(raw: dict[str, Any] | None) -> dict[str, Any]:
    policy = _default_policy()
    if not isinstance(raw, dict):
        return policy
    if raw.get("security") in VALID_SECURITY:
        policy["security"] = raw["security"]
    if raw.get("ask") in VALID_ASK:
        policy["ask"] = raw["ask"]
    if raw.get("ask_fallback") in VALID_ASK_FALLBACK:
        policy["ask_fallback"] = raw["ask_fallback"]
    allowlist = raw.get("allowlist", [])
    if isinstance(allowlist, list):
        policy["allowlist"] = [str(item).strip() for item in allowlist if str(item).strip()]
    return policy


def normalize_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    base = _default_config()
    if not isinstance(raw, dict):
        return base
    base["version"] = int(raw.get("version", 1))
    base["defaults"] = _merge_policy(raw.get("defaults"))
    agents = raw.get("agents", {})
    if isinstance(agents, dict):
        normalized_agents: dict[str, dict[str, Any]] = {}
        for key, value in agents.items():
            agent = str(key).strip()
            if not agent:
                continue
            normalized_agents[agent] = _merge_policy(value if isinstance(value, dict) else {})
        base["agents"] = normalized_agents
    return base


def get_exec_approvals_path() -> Path:
    env_path = os.getenv("EDGE_EXEC_APPROVALS_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    data_dir = Path(os.getenv("EDGE_DATA_DIR", "./data")).expanduser().resolve()
    return data_dir / "exec-approvals.json"


def load_exec_approvals(path: Path | None = None) -> dict[str, Any]:
    file_path = path or get_exec_approvals_path()
    if not file_path.exists():
        return _default_config()
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _default_config()
    return normalize_config(raw)


def save_exec_approvals(config: dict[str, Any], path: Path | None = None) -> Path:
    normalized = normalize_config(config)
    file_path = path or get_exec_approvals_path()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(normalized, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return file_path


def get_agent_policy(config: dict[str, Any], actor_id: str) -> dict[str, Any]:
    normalized = normalize_config(config)
    agents = normalized.get("agents", {})
    if isinstance(agents, dict):
        if actor_id in agents and isinstance(agents[actor_id], dict):
            return _merge_policy(agents[actor_id])
        if DEFAULT_MAIN_AGENT in agents and isinstance(agents[DEFAULT_MAIN_AGENT], dict):
            return _merge_policy(agents[DEFAULT_MAIN_AGENT])
    return _merge_policy(normalized.get("defaults"))


def evaluate_tool_permission(config: dict[str, Any], *, actor_id: str, tool_name: str) -> dict[str, str]:
    policy = get_agent_policy(config, actor_id)
    tool = tool_name.strip()
    allowlist = [item for item in policy["allowlist"] if item]
    allowed_by_allowlist = any(fnmatchcase(tool, pattern) for pattern in allowlist)

    security = policy["security"]
    ask = policy["ask"]

    if security == "deny":
        return {"decision": "deny", "reason": "security=deny"}

    if security == "full":
        if ask == "always":
            return {"decision": "ask", "reason": "ask=always"}
        return {"decision": "allow", "reason": "security=full"}

    # security=allowlist
    if allowed_by_allowlist:
        if ask == "always":
            return {"decision": "ask", "reason": "allowlist_hit + ask=always"}
        return {"decision": "allow", "reason": "allowlist_hit"}

    if ask in {"always", "on-miss"}:
        return {"decision": "ask", "reason": "allowlist_miss"}
    return {"decision": "deny", "reason": "allowlist_miss + ask=off"}

