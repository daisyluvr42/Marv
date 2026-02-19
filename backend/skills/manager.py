from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".py",
    ".js",
    ".ts",
    ".rb",
}
BLOCKED_BINARY_EXTENSIONS = {
    ".exe",
    ".dll",
    ".dylib",
    ".so",
    ".bin",
    ".apk",
    ".ipa",
    ".pkg",
    ".dmg",
    ".class",
    ".jar",
}
SUSPICIOUS_PATTERNS = [
    r"rm\s+-rf\s+/\b",
    r"curl\s+[^|]*\|\s*(bash|sh|zsh)\b",
    r"wget\s+[^|]*\|\s*(bash|sh|zsh)\b",
    r"mkfs\.[a-z0-9]+",
    r"dd\s+if=/dev/zero",
    r":\(\)\s*\{\s*:\|\:&\s*;\s*\}:",
    r"sudo\s+rm\s+-rf",
]


@dataclass(frozen=True)
class ScanIssue:
    severity: str
    path: str
    reason: str


def get_skills_root() -> Path:
    env_path = Path(Path(os.getenv("EDGE_SKILLS_ROOT", "./skill/modules")).expanduser())
    return env_path.resolve()


def list_installed_skills(root: Path | None = None) -> list[dict[str, Any]]:
    skills_root = (root or get_skills_root()).resolve()
    if not skills_root.exists():
        return []
    items: list[dict[str, Any]] = []
    for skill_md in skills_root.rglob("SKILL.md"):
        skill_dir = skill_md.parent
        metadata = _read_skill_metadata(skill_md)
        items.append(
            {
                "id": str(skill_dir.relative_to(skills_root)),
                "path": str(skill_dir),
                "name": metadata.get("name") or skill_dir.name,
                "description": metadata.get("description") or "",
            }
        )
    return sorted(items, key=lambda item: str(item["id"]))


def import_skills_from_directory(
    *,
    source_dir: str,
    source_name: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    src_root = Path(source_dir).expanduser().resolve()
    if not src_root.exists():
        raise RuntimeError(f"source path not found: {src_root}")
    if not src_root.is_dir():
        raise RuntimeError(f"source path is not a directory: {src_root}")

    skills_root = (root or get_skills_root()).resolve()
    skills_root.mkdir(parents=True, exist_ok=True)
    source_key = _safe_name(source_name or src_root.name)
    destination_root = skills_root / source_key
    destination_root.mkdir(parents=True, exist_ok=True)

    imported: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    discovered = _discover_skill_dirs(src_root)
    for skill_dir in discovered:
        issues = _scan_skill_dir(skill_dir)
        relative = skill_dir.relative_to(src_root)
        if any(issue.severity == "critical" for issue in issues):
            blocked.append(
                {
                    "skill_path": str(relative),
                    "issues": [issue.__dict__ for issue in issues],
                }
            )
            continue

        dest = destination_root / relative
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(skill_dir, dest)
        imported.append(
            {
                "skill_path": str(relative),
                "destination": str(dest),
                "issues": [issue.__dict__ for issue in issues],
            }
        )

    manifest = {
        "source_root": str(src_root),
        "source_key": source_key,
        "imported_count": len(imported),
        "blocked_count": len(blocked),
        "imported": imported,
        "blocked": blocked,
    }
    manifest_path = destination_root / ".import-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def import_skills_from_git(
    *,
    git_url: str,
    subdir: str = "",
    source_name: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="marv_skills_") as temp_dir:
        checkout = Path(temp_dir) / "repo"
        result = subprocess.run(
            ["git", "clone", "--depth", "1", git_url, str(checkout)],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
        source_root = checkout / subdir if subdir else checkout
        return import_skills_from_directory(
            source_dir=str(source_root),
            source_name=source_name or _guess_source_name(git_url),
            root=root,
        )


def _discover_skill_dirs(root: Path) -> list[Path]:
    result: list[Path] = []
    for skill_md in root.rglob("SKILL.md"):
        if ".git" in skill_md.parts:
            continue
        result.append(skill_md.parent)
    return sorted(result)


def _scan_skill_dir(skill_dir: Path) -> list[ScanIssue]:
    issues: list[ScanIssue] = []
    for path in skill_dir.rglob("*"):
        if path.name == ".DS_Store":
            continue
        if path.is_symlink():
            resolved = path.resolve()
            try:
                resolved.relative_to(skill_dir.resolve())
            except ValueError:
                issues.append(
                    ScanIssue(
                        severity="critical",
                        path=str(path.relative_to(skill_dir)),
                        reason="symlink escapes skill directory",
                    )
                )
            continue
        if path.is_dir():
            continue

        ext = path.suffix.lower()
        rel = str(path.relative_to(skill_dir))
        if ext in BLOCKED_BINARY_EXTENSIONS:
            issues.append(ScanIssue(severity="critical", path=rel, reason=f"blocked binary extension: {ext}"))
            continue
        if ext not in TEXT_EXTENSIONS:
            # Unknown asset is allowed but tracked as warning.
            issues.append(ScanIssue(severity="warning", path=rel, reason=f"unclassified extension: {ext or '[none]'}"))
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(ScanIssue(severity="critical", path=rel, reason="text file is not valid UTF-8"))
            continue
        lowered = content.lower()
        for pattern in SUSPICIOUS_PATTERNS:
            if re.search(pattern, lowered, flags=re.IGNORECASE | re.MULTILINE):
                issues.append(ScanIssue(severity="critical", path=rel, reason=f"suspicious pattern: {pattern}"))
    return issues


def _read_skill_metadata(skill_md: Path) -> dict[str, str]:
    text = skill_md.read_text(encoding="utf-8")
    data: dict[str, str] = {}
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end > 0:
            frontmatter = text[4:end]
            for raw_line in frontmatter.splitlines():
                line = raw_line.strip()
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip().strip('"')
    if "name" not in data:
        heading = next((line.strip() for line in text.splitlines() if line.strip().startswith("#")), "")
        if heading:
            data["name"] = heading.lstrip("#").strip()
    return data


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.").lower()
    return cleaned or "skills"


def _guess_source_name(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1]
    if tail.endswith(".git"):
        tail = tail[:-4]
    return _safe_name(tail)
