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
- Console: `frontend/app/`
