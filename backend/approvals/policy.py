from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


VALID_APPROVAL_MODES = {"policy", "all", "risky"}
DEFAULT_RISKY_RISKS = {"external_write", "exec", "network", "write", "sandbox_escape"}


def get_approval_policy_path() -> Path:
    env_path = os.getenv("EDGE_APPROVAL_POLICY_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    data_dir = Path(os.getenv("EDGE_DATA_DIR", "./data")).expanduser().resolve()
    return data_dir / "approval-policy.json"


def normalize_approval_policy(raw: dict[str, Any] | None) -> dict[str, Any]:
    mode = "policy"
    risky_risks = sorted(DEFAULT_RISKY_RISKS)
    if isinstance(raw, dict):
        candidate_mode = str(raw.get("mode", "")).strip().lower()
        if candidate_mode in VALID_APPROVAL_MODES:
            mode = candidate_mode
        candidate_risks = raw.get("risky_risks")
        if isinstance(candidate_risks, list):
            normalized = [str(item).strip().lower() for item in candidate_risks if str(item).strip()]
            if normalized:
                risky_risks = sorted(set(normalized))
    return {
        "mode": mode,
        "risky_risks": risky_risks,
    }


def load_approval_policy(path: Path | None = None) -> dict[str, Any]:
    file_path = path or get_approval_policy_path()
    if not file_path.exists():
        return normalize_approval_policy(None)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return normalize_approval_policy(None)
    return normalize_approval_policy(payload if isinstance(payload, dict) else None)


def save_approval_policy(policy: dict[str, Any], path: Path | None = None) -> Path:
    normalized = normalize_approval_policy(policy)
    file_path = path or get_approval_policy_path()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(normalized, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return file_path


def decide_approval_mode(
    *,
    policy: dict[str, Any],
    tool_risk: str,
    policy_decision: str,
) -> tuple[bool, str]:
    mode = str(policy.get("mode", "policy")).strip().lower()
    normalized_risk = tool_risk.strip().lower()
    risky_risks = {str(item).strip().lower() for item in policy.get("risky_risks", [])}

    if mode == "all":
        return True, "approval_mode=all"
    if mode == "risky":
        if normalized_risk in risky_risks:
            return True, f"approval_mode=risky:{normalized_risk}"
        if policy_decision == "ask":
            return True, "approval_mode=risky + policy=ask"
        return False, "approval_mode=risky + low_risk"
    # mode=policy
    if policy_decision == "ask":
        return True, "approval_mode=policy + policy=ask"
    if normalized_risk == "external_write":
        return True, "approval_mode=policy + external_write"
    return False, "approval_mode=policy + direct_allow"
