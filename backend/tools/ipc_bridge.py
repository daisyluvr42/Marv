from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from backend.sandbox.runtime import load_execution_config
from backend.tools.registry import register_runtime_tool


def get_ipc_tools_path() -> Path:
    value = os.getenv("EDGE_IPC_TOOLS_PATH")
    if value:
        return Path(value).expanduser().resolve()
    data_dir = Path(os.getenv("EDGE_DATA_DIR", "./data")).expanduser().resolve()
    return data_dir / "ipc-tools.json"


def load_ipc_tools() -> list[str]:
    path = get_ipc_tools_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []

    loaded: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        command = item.get("command")
        if not name or not isinstance(command, list) or not command:
            continue
        risk = str(item.get("risk", "read_only")).strip().lower() or "read_only"
        schema = item.get("schema")
        if not isinstance(schema, dict):
            schema = {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            }
        timeout_seconds = max(1.0, float(item.get("timeout_seconds", 12.0)))
        register_runtime_tool(
            name=name,
            risk=risk,
            schema=schema,
            func=_build_ipc_tool(command=[str(part) for part in command], timeout_seconds=timeout_seconds),
            version="ipc-v1",
            enabled=bool(item.get("enabled", True)),
        )
        loaded.append(name)
    return loaded


def _build_ipc_tool(*, command: list[str], timeout_seconds: float):
    def _tool(**kwargs: Any) -> dict[str, Any]:
        runtime_context = _extract_runtime_context(kwargs)
        request_payload = {"args": kwargs}
        config = load_execution_config()
        mode = _resolve_execution_mode(config=config, override=runtime_context["execution_mode"])
        shell_command = command
        if mode == "sandbox":
            shell_command = _build_docker_command(
                command=command,
                docker_image=str(config["docker_image"]),
                network_enabled=bool(config["network_enabled"]),
                session_workspace=runtime_context["session_workspace"],
            )
        elif mode == "auto":
            if _docker_available():
                shell_command = _build_docker_command(
                    command=command,
                    docker_image=str(config["docker_image"]),
                    network_enabled=bool(config["network_enabled"]),
                    session_workspace=runtime_context["session_workspace"],
                )

        result = subprocess.run(
            shell_command,
            input=json.dumps(request_payload, ensure_ascii=True),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ipc tool failed rc={result.returncode}, mode={mode}: {result.stderr.strip()}")
        output = result.stdout.strip()
        if not output:
            return {"status": "ok", "result": None, "mode": mode}
        try:
            decoded = json.loads(output)
        except json.JSONDecodeError:
            return {"status": "ok", "stdout": output, "mode": mode}
        if isinstance(decoded, dict):
            decoded.setdefault("_mode", mode)
            return decoded
        return {"status": "ok", "result": decoded, "mode": mode}

    return _tool


def _extract_runtime_context(kwargs: dict[str, Any]) -> dict[str, str | None]:
    session_workspace = kwargs.pop("__marv_session_workspace", None)
    execution_mode = kwargs.pop("__marv_execution_mode", None)
    session_workspace_text = str(session_workspace).strip() if session_workspace is not None else None
    execution_mode_text = str(execution_mode).strip().lower() if execution_mode is not None else None
    return {
        "session_workspace": session_workspace_text or None,
        "execution_mode": execution_mode_text or None,
    }


def _resolve_execution_mode(*, config: dict[str, Any], override: str | None) -> str:
    if override in {"auto", "local", "sandbox"}:
        return override
    mode = str(config.get("mode", "auto")).strip().lower()
    if mode in {"auto", "local", "sandbox"}:
        return mode
    return "auto"


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _build_docker_command(
    *,
    command: list[str],
    docker_image: str,
    network_enabled: bool,
    session_workspace: str | None,
) -> list[str]:
    if not _docker_available():
        raise RuntimeError("sandbox mode requires docker installed in PATH")

    docker_cmd: list[str] = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=64m",
    ]
    if not network_enabled:
        docker_cmd.extend(["--network", "none"])

    if session_workspace:
        workspace = Path(session_workspace).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        docker_cmd.extend(["-v", f"{workspace}:/workspace:rw", "-w", "/workspace"])
    else:
        docker_cmd.extend(["-w", "/tmp"])

    docker_cmd.append(docker_image)
    docker_cmd.extend(command)
    return docker_cmd
