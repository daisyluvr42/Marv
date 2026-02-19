from __future__ import annotations

import json
import tarfile
from pathlib import Path

from backend.cli import _create_migration_archive


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

