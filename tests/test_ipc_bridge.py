from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from backend.tools.ipc_bridge import _build_ipc_tool, load_ipc_tools
from backend.tools.registry import get_tool_function


def test_load_ipc_tools_and_execute(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "ipc-tools.json"
    tool_name = "ipc_echo_test"
    config_path.write_text(
        json.dumps(
            [
                {
                    "name": tool_name,
                    "risk": "read_only",
                    "command": [
                        sys.executable,
                        "-c",
                        "import json,sys; p=json.load(sys.stdin); print(json.dumps({'status':'ok','echo':p.get('args',{})}))",
                    ],
                    "schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    "timeout_seconds": 5,
                }
            ],
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EDGE_IPC_TOOLS_PATH", str(config_path))

    loaded = load_ipc_tools()
    assert tool_name in loaded

    func = get_tool_function(tool_name)
    assert func is not None
    payload = func(query="hello")
    assert payload["status"] == "ok"
    assert payload["echo"]["query"] == "hello"


def test_ipc_tool_sandbox_mode_wraps_command_with_docker(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "backend.tools.ipc_bridge.load_execution_config",
        lambda: {"mode": "sandbox", "docker_image": "python:3.12-alpine", "network_enabled": False},
    )
    monkeypatch.setattr("backend.tools.ipc_bridge._docker_available", lambda: True)

    captured: dict[str, object] = {}

    def _fake_run(command, input, text, capture_output, timeout, check):  # type: ignore[no-untyped-def]
        captured["command"] = command
        captured["input"] = input
        return SimpleNamespace(returncode=0, stdout='{"status":"ok"}', stderr="")

    monkeypatch.setattr("backend.tools.ipc_bridge.subprocess.run", _fake_run)
    tool = _build_ipc_tool(command=[sys.executable, "-c", "print('ok')"], timeout_seconds=3)
    workspace = tmp_path / "session-1"
    payload = tool(query="hello", __marv_session_workspace=str(workspace))
    assert payload["status"] == "ok"

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:3] == ["docker", "run", "--rm"]
    assert "--network" in command
    assert "none" in command
    assert "python:3.12-alpine" in command

    raw_input = captured["input"]
    assert isinstance(raw_input, str)
    decoded = json.loads(raw_input)
    assert decoded["args"]["query"] == "hello"
    assert "__marv_session_workspace" not in decoded["args"]
