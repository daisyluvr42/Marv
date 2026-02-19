# Blackbox Agent Runtime

Blackbox Agent Runtime 是一个本地优先的 Agent 运行时项目，包含：
- Edge API（任务编排、事件溯源、工具执行、审批、记忆、配置补丁）
- Core API（OpenAI 风格 `/v1/chat/completions` 与 `/v1/embeddings`）
- Frontend Console（Chat/Tasks/Approvals/Memory/Config/Tools）

## Prerequisites
- Python 3.12.8（见 `.python-version`）
- uv
- Node 20.11.1（见 `.nvmrc`）

## Setup
```bash
uv sync
cd frontend
npm ci
```

## Run
```bash
# terminal 1
uv run uvicorn core.api:app --port 9000

# terminal 2
CORE_BASE_URL=http://127.0.0.1:9000 uv run uvicorn backend.agent.api:app --port 8000

# terminal 3
cd frontend
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

## Test
```bash
# backend tests
uv run pytest -q

# frontend build
cd frontend && npm run build

# end-to-end smoke
EDGE_BASE_URL=http://127.0.0.1:8000 bash scripts/smoke_test.sh
```

## CLI
项目提供统一 CLI：`marv`（通过 `uv run marv` 调用）。

```bash
# health
uv run marv health

# send message and follow events
uv run marv chat send --message "你好" --follow

# list tools and execute read-only tool
uv run marv tools list
uv run marv tools exec --tool mock_web_search --args '{"query":"llm"}'

# approvals
uv run marv approvals list --status pending
uv run marv approvals approve <approval_id>

# config patches
uv run marv config propose --text "更简洁" --scope-type channel --scope-id web:default
uv run marv config commit <proposal_id>
uv run marv config revisions --scope-type channel --scope-id web:default

# memory
uv run marv memory write --scope-id u1 --content "我偏好简洁回答" --requires-confirmation
uv run marv memory candidates --status pending
uv run marv memory approve <candidate_id>
uv run marv memory query --scope-id u1 --query "简洁回答"

# local ops (interactive confirmation required)
uv run marv ops stop-services
uv run marv ops package-migration --output-dir ./dist/migrations

# permissions (OpenClaw-like)
uv run marv permissions show
uv run marv permissions set-default --security allowlist --ask on-miss --ask-fallback deny
uv run marv permissions allowlist add --agent main --pattern mock_web_search
uv run marv permissions set-agent --agent telegram:123456 --security full --ask off
```

详细说明：`docs/PERMISSIONS.md`

## Telegram Adapter (MVP)
通过 Telegram Bot 长轮询把消息转发到 Edge API，再将回复发回聊天。

- 详细配置：`docs/TELEGRAM_SETUP.md`

```bash
cd /path/to/Marv
cp .env.example .env
# 编辑 .env，至少配置 TELEGRAM_BOT_TOKEN
```

单独启动 Telegram 适配器：
```bash
EDGE_BASE_URL=http://127.0.0.1:8000 TELEGRAM_BOT_TOKEN=xxx uv run marv-telegram
```

随栈一键启动（推荐）：
```bash
cd /path/to/Marv
bash scripts/start_stack.sh
# 若 .env 内存在 TELEGRAM_BOT_TOKEN，将自动拉起 telegram 进程
```

## MacBook Pro M1 Deployment
- 部署文档：`docs/DEPLOY_MACBOOK_PRO_M1.md`
- 自举脚本：`scripts/bootstrap_mbp_m1.sh`
- 后端栈启动/停止：`scripts/start_stack.sh` / `scripts/stop_stack.sh`

## CI
GitHub Actions: `.github/workflows/ci.yml`
- backend: `uv sync` + `uv run pytest -q`
- frontend: `npm ci` + `npm run build`

## Key Paths
- Edge API: `backend/agent/api.py`
- Core API: `core/api.py`
- Ledger: `backend/ledger/`
- Tools: `backend/tools/`
- Approvals: `backend/approvals/`
- Patch config: `backend/patch/`
- Memory: `backend/memory/`
- Permissions: `backend/permissions/exec_approvals.py`
- Telegram gateway: `backend/gateway/telegram.py`
- Console: `frontend/app/`
