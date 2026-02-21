from __future__ import annotations

import json
import tarfile
from pathlib import Path

from backend.cli import (
    _build_doctor_report,
    _create_migration_archive,
    _doctor_exit_code,
    build_parser,
)


def test_create_migration_archive_contains_manifest_and_edge_db(tmp_path: Path) -> None:
    project_root = tmp_path / "Marv"
    data_dir = project_root / "data"
    docs_dir = project_root / "docs"
    output_dir = project_root / "dist" / "migrations"
    data_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    edge_db_path = data_dir / "edge.db"
    edge_db_path.write_bytes(b"sqlite-test")
    (project_root / ".env.example").write_text("EDGE_BASE_URL=http://127.0.0.1:8000\n", encoding="utf-8")
    (docs_dir / "DEPLOY_MACBOOK_PRO_M1.md").write_text("# deploy\n", encoding="utf-8")

    payload = _create_migration_archive(project_root=project_root, edge_db_path=edge_db_path, output_dir=output_dir)

    archive_path = Path(payload["archive_path"])
    assert archive_path.exists()
    bundle_name = payload["bundle_name"]

    with tarfile.open(archive_path, mode="r:gz") as tar:
        names = set(tar.getnames())
        assert f"{bundle_name}/data/edge.db" in names
        assert f"{bundle_name}/manifest.json" in names
        manifest_member = tar.extractfile(f"{bundle_name}/manifest.json")
        assert manifest_member is not None
        manifest = json.loads(manifest_member.read().decode("utf-8"))
        assert manifest["source_edge_db_path"] == str(edge_db_path)
        assert "data/edge.db" in manifest["files"]


def test_build_parser_includes_ops_doctor() -> None:
    parser = build_parser()
    args = parser.parse_args(["ops", "doctor", "--skip-network"])
    assert args.func.__name__ == "cmd_ops_doctor"
    assert args.skip_network is True


def test_build_parser_includes_im_security_commands() -> None:
    parser = build_parser()
    show_args = parser.parse_args(["im", "security-show"])
    assert show_args.func.__name__ == "cmd_im_security_show"
    set_args = parser.parse_args(
        ["im", "security-set", "--channel", "discord", "--dm-policy", "allowlist", "--add-allow-from", "u-1"]
    )
    assert set_args.func.__name__ == "cmd_im_security_set"
    assert set_args.channel == "discord"
    assert set_args.dm_policy == "allowlist"


def test_build_parser_includes_evolve_commands() -> None:
    parser = build_parser()
    run_args = parser.parse_args(["evolve", "run", "--config", "backend/evolution/evolution.example.json"])
    assert run_args.func.__name__ == "cmd_evolve_run"
    best_args = parser.parse_args(["evolve", "best", "--run", "evr_demo"])
    assert best_args.func.__name__ == "cmd_evolve_best"


def test_build_doctor_report_fix_creates_missing_runtime_config_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "Marv"
    project_root.mkdir(parents=True)
    (project_root / ".env").write_text("EDGE_BASE_URL=http://127.0.0.1:8000\n", encoding="utf-8")
    data_dir = project_root / "data"
    monkeypatch.setenv("EDGE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("EDGE_SKILLS_ROOT", str(project_root / "skill" / "modules"))
    monkeypatch.setenv("EDGE_PACKAGES_ROOT", str(project_root / "packages"))
    for key in (
        "EDGE_DB_PATH",
        "EDGE_EXEC_APPROVALS_PATH",
        "EDGE_APPROVAL_POLICY_PATH",
        "EDGE_EXECUTION_CONFIG_PATH",
        "EDGE_HEARTBEAT_CONFIG_PATH",
        "EDGE_IPC_TOOLS_PATH",
        "IM_INGRESS_TOKEN",
        "IM_INGRESS_TOKENS_JSON",
        "TELEGRAM_BOT_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)

    report = _build_doctor_report(
        project_root=project_root,
        edge_base_url="http://127.0.0.1:8000",
        actor_id="cli-user",
        actor_role="owner",
        timeout_seconds=1.0,
        skip_network=True,
        apply_fixes=True,
    )

    assert int(report["summary"]["fail"]) == 0
    assert Path(report["paths"]["exec_approvals_path"]).exists()
    assert Path(report["paths"]["approval_policy_path"]).exists()
    assert Path(report["paths"]["execution_config_path"]).exists()
    assert Path(report["paths"]["heartbeat_config_path"]).exists()


def test_build_doctor_report_flags_invalid_ingress_tokens_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "Marv"
    project_root.mkdir(parents=True)
    (project_root / ".env").write_text("IM_INGRESS_TOKENS_JSON={bad-json\n", encoding="utf-8")
    monkeypatch.setenv("EDGE_DATA_DIR", str(project_root / "data"))
    monkeypatch.delenv("IM_INGRESS_TOKEN", raising=False)
    monkeypatch.delenv("IM_INGRESS_TOKENS_JSON", raising=False)

    report = _build_doctor_report(
        project_root=project_root,
        edge_base_url="http://127.0.0.1:8000",
        actor_id="cli-user",
        actor_role="owner",
        timeout_seconds=1.0,
        skip_network=True,
        apply_fixes=True,
    )

    ingress = next(item for item in report["checks"] if item["id"] == "security.ingress_auth")
    assert ingress["status"] == "fail"


def test_doctor_exit_code_strict_mode() -> None:
    warning_summary = {"counts": {"ok": 0, "warn": 1, "fail": 0, "skip": 0}}
    failure_summary = {"counts": {"ok": 0, "warn": 0, "fail": 1, "skip": 0}}

    assert _doctor_exit_code(summary=warning_summary, strict=False) == 0
    assert _doctor_exit_code(summary=warning_summary, strict=True) == 2
    assert _doctor_exit_code(summary=failure_summary, strict=False) == 2
