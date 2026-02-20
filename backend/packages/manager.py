from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from backend.tools.ipc_bridge import load_ipc_tools_from_payload


def get_packages_root() -> Path:
    value = os.getenv("EDGE_PACKAGES_ROOT", "./packages")
    return Path(value).expanduser().resolve()


def list_installed_packages(root: Path | None = None) -> list[dict[str, Any]]:
    packages_root = (root or get_packages_root()).resolve()
    if not packages_root.exists():
        return []

    items: list[dict[str, Any]] = []
    for child in sorted(packages_root.iterdir()):
        if not child.is_dir():
            continue
        manifest = _load_package_manifest(child)
        if manifest is None:
            continue
        hooks = _normalize_hooks(manifest, package_dir=child)
        item = {
            "id": child.name,
            "name": str(manifest.get("name", child.name)).strip() or child.name,
            "version": str(manifest.get("version", "0.0.0")).strip() or "0.0.0",
            "description": str(manifest.get("description", "")).strip(),
            "enabled": bool(manifest.get("enabled", True)),
            "capabilities": _normalize_capabilities(manifest.get("capabilities")),
            "path": str(child),
            "hooks": hooks,
        }
        items.append(item)
    return items


def load_runtime_packages(root: Path | None = None) -> dict[str, Any]:
    packages_root = (root or get_packages_root()).resolve()
    items = list_installed_packages(packages_root)
    loaded_packages: list[dict[str, Any]] = []
    skipped_packages: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    ipc_tools_loaded: list[str] = []

    for item in items:
        package_name = str(item["name"])
        if not bool(item.get("enabled", True)):
            skipped_packages.append({"name": package_name, "reason": "disabled"})
            continue
        hooks = item.get("hooks")
        if not isinstance(hooks, dict):
            hooks = {}

        package_loaded_tools: list[str] = []
        ipc_path = hooks.get("ipc_tools")
        if isinstance(ipc_path, str) and ipc_path:
            candidate = Path(ipc_path).expanduser().resolve()
            if not candidate.exists():
                errors.append({"name": package_name, "error": f"ipc_tools file not found: {candidate}"})
            else:
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                    if isinstance(payload, list):
                        loaded = load_ipc_tools_from_payload(payload, source=f"package:{package_name}")
                        package_loaded_tools.extend(loaded)
                        ipc_tools_loaded.extend(loaded)
                    else:
                        errors.append({"name": package_name, "error": f"ipc_tools payload must be a JSON array: {candidate}"})
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append({"name": package_name, "error": f"failed loading ipc_tools from {candidate}: {exc}"})

        loaded_packages.append(
            {
                "name": package_name,
                "version": item["version"],
                "ipc_tools_loaded": package_loaded_tools,
            }
        )

    return {
        "root": str(packages_root),
        "count": len(items),
        "loaded_count": len(loaded_packages),
        "skipped_count": len(skipped_packages),
        "error_count": len(errors),
        "ipc_tools_loaded_count": len(ipc_tools_loaded),
        "loaded": loaded_packages,
        "skipped": skipped_packages,
        "errors": errors,
    }


def _load_package_manifest(package_dir: Path) -> dict[str, Any] | None:
    marv_manifest = package_dir / "MARV_PACKAGE.json"
    if marv_manifest.exists():
        try:
            payload = json.loads(marv_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    npm_manifest = package_dir / "package.json"
    if not npm_manifest.exists():
        return None
    try:
        payload = json.loads(npm_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    scoped = payload.get("marvPackage")
    if isinstance(scoped, dict):
        merged = dict(scoped)
        merged.setdefault("name", payload.get("name", package_dir.name))
        merged.setdefault("version", payload.get("version", "0.0.0"))
        merged.setdefault("description", payload.get("description", ""))
        return merged
    return None


def _normalize_capabilities(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _normalize_hooks(manifest: dict[str, Any], *, package_dir: Path) -> dict[str, str]:
    raw = manifest.get("hooks")
    if not isinstance(raw, dict):
        return {}
    hooks: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        cleaned_key = key.strip()
        cleaned_value = value.strip()
        if not cleaned_key or not cleaned_value:
            continue
        candidate = Path(cleaned_value)
        if not candidate.is_absolute():
            candidate = (package_dir / candidate).resolve()
        hooks[cleaned_key] = str(candidate)
    return hooks

