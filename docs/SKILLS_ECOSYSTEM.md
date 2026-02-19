# Skill Ecosystem

运行时新增 `skill/modules` 生态，支持导入通用 `SKILL.md` 格式技能，并在导入前做安全扫描。

## 支持能力
- 列出已安装技能
- 从本地目录导入技能
- 从 Git 仓库导入技能
- 一键同步 Marv + LobsterAI 技能集合

## 安全扫描（前置）
导入时会阻断：
- 可疑危险命令模式（如 `curl|bash`, `wget|sh`, `rm -rf /` 等）
- 可执行二进制扩展（`.exe/.dll/.so/.dmg/...`）
- 越界 symlink

未识别扩展会标记 warning 并写入导入 manifest。

## CLI
```bash
uv run marv skills list
uv run marv skills import --source-path ./path/to/skills --source-name custom
uv run marv skills import --git-url https://github.com/example/repo.git --git-subdir skills --source-name upstream
uv run marv skills sync-upstream
```

## API
- `GET /v1/skills`
- `POST /v1/skills/import`
- `POST /v1/skills/sync-upstream`
