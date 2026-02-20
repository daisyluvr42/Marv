from __future__ import annotations

import json
import sys

from backend.packages.manager import list_installed_packages, load_runtime_packages
from backend.tools.registry import get_tool_function


def test_list_and_load_runtime_packages_with_ipc_hook(tmp_path, monkeypatch) -> None:
    packages_root = tmp_path / "packages"
    package_dir = packages_root / "demo-runtime"
    package_dir.mkdir(parents=True, exist_ok=True)
    ipc_config_path = package_dir / "tools" / "ipc-tools.json"
    ipc_config_path.parent.mkdir(parents=True, exist_ok=True)
    ipc_config_path.write_text(
        json.dumps(
            [
                {
                    "name": "pkg_echo_tool",
                    "risk": "read_only",
                    "command": [
                        sys.executable,
                        "-c",
                        "import json,sys; p=json.load(sys.stdin); print(json.dumps({'status':'ok','echo':p.get('args',{})}))",
                    ],
                    "schema": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
                }
            ],
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (package_dir / "MARV_PACKAGE.json").write_text(
        json.dumps(
            {
                "name": "demo-runtime",
                "version": "0.1.0",
                "description": "demo",
                "enabled": True,
                "capabilities": ["ipc_tools"],
                "hooks": {"ipc_tools": "tools/ipc-tools.json"},
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EDGE_PACKAGES_ROOT", str(packages_root))

    items = list_installed_packages()
    assert len(items) == 1
    assert items[0]["name"] == "demo-runtime"
    assert items[0]["enabled"] is True
    assert "ipc_tools" in items[0]["capabilities"]
    assert str(ipc_config_path.resolve()) == items[0]["hooks"]["ipc_tools"]

    report = load_runtime_packages()
    assert report["loaded_count"] == 1
    assert report["ipc_tools_loaded_count"] == 1
    assert report["error_count"] == 0

    func = get_tool_function("pkg_echo_tool")
    assert func is not None
    payload = func(x="ok")
    assert payload["status"] == "ok"
    assert payload["echo"]["x"] == "ok"


def test_list_packages_supports_package_json_contract(tmp_path, monkeypatch) -> None:
    packages_root = tmp_path / "packages"
    package_dir = packages_root / "demo-npm"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "package.json").write_text(
        json.dumps(
            {
                "name": "demo-npm",
                "version": "1.2.3",
                "description": "demo package",
                "marvPackage": {
                    "enabled": False,
                    "capabilities": ["ipc_tools", "skills"],
                    "hooks": {},
                },
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EDGE_PACKAGES_ROOT", str(packages_root))

    items = list_installed_packages()
    assert len(items) == 1
    assert items[0]["name"] == "demo-npm"
    assert items[0]["version"] == "1.2.3"
    assert items[0]["enabled"] is False
    assert "skills" in items[0]["capabilities"]

    report = load_runtime_packages()
    assert report["loaded_count"] == 0
    assert report["skipped_count"] == 1

