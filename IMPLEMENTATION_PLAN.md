# IMPLEMENTATION_PLAN — 极细粒度分步开发计划（Feed to Coding Agent）

原则：
- 每一步都可独立执行、可验收（有明确输出）。
- 先跑通“消息 -> 任务 -> Core 推理 -> 回复”，再加工具/补丁/审批/记忆/审计。
- 始终遵循：原子车道、事件溯源、分级变更、RBAC。

---

## Phase 0 — 仓库与基础脚手架

### Step 0.1 初始化仓库结构
输出：
- 目录：
  - `backend/`
  - `frontend/`
  - `core/`（可与 backend 分开或同 repo）
  - `docs/`（可选）
- 根目录放置：PRD.md、APP_FLOW.md、TECH_STACK.md、FRONTEND_GUIDELINES.md、BACKEND_STRUCTURE.md、IMPLEMENTATION_PLAN.md

验收：
- repo 能跑基本命令（空项目不报错）

### Step 0.2 Python 环境与依赖锁定（uv）
输出：
- `pyproject.toml`（backend + core）
- `uv.lock`

验收：
- `uv sync` 成功
- `uv run python -c "import fastapi"` 成功

---

## Phase 1 — Edge MVP：消息 -> 任务 -> Core 推理 -> 回复

### Step 1.1 FastAPI 服务骨架（Edge）
输出：
- `backend/agent/api.py` FastAPI app
- `/health` 返回 ok
- 基础配置：data dir, env vars

验收：
- `uvicorn backend.agent.api:app --reload` 启动成功

### Step 1.2 SQLite 初始化与 migrations
输出：
- `backend/storage/db.py`（engine/session）
- `backend/storage/models.py`（conversations, tasks, ledger_events 最小表）
- Alembic 初始化（可选）
- 初始化脚本：启动时自动创建表（MVP）

验收：
- 启动后生成 `data/edge.db`，表存在

### Step 1.3 ledger（事件写入/读取）
输出：
- `backend/ledger/events.py`（事件 schema：InputEvent/PlanEvent/RouteEvent/CompletionEvent）
- `backend/ledger/store.py`（append_event, query_events）
- API：`GET /v1/audit/conversations/{conversation_id}/timeline`

验收：
- 能写入与读取事件；timeline 返回按 ts 排序

### Step 1.4 task queue（最小队列 + worker）
输出：
- `backend/agent/queue.py`：enqueue_task、worker loop
- `backend/agent/state.py`：task 状态更新
- API：`GET /v1/agent/tasks/{task_id}`

验收：
- 发起 task 后状态从 queued -> running -> completed（先用 mock completion）

### Step 1.5 Core Client（HTTP 调用）
输出：
- `backend/core_client/openai_compat.py`：
  - `chat_completions(messages, stream)`
  - 超时/重试
  - health check
- 配置：
  - `CORE_BASE_URL` 默认 `http://localhost:9000`

验收：
- Edge 能 ping Core `/health` 并调用 completions（先用 core mock）

### Step 1.6 Agent messages API
输出：
- `POST /v1/agent/messages`：
  - upsert conversation
  - 写 InputEvent
  - 创建 task 入队
  - 返回 task_id
- worker：
  - 写 PlanEvent（先简单：直接 call core）
  - 写 RouteEvent
  - 写 CompletionEvent（response text）
  - 更新 task 状态

验收：
- Console 或 curl 发送消息后能得到回复；ledger 可查全链路

---

## Phase 2 — Core MVP：OpenAI 风格本地网关（先 mock）

### Step 2.1 Core FastAPI 骨架
输出：
- `core/api.py`：`/health`
- `POST /v1/chat/completions`（先返回固定文本或 echo）
- `POST /v1/embeddings`（先返回 dummy vector）

验收：
- Edge 调用成功；联调完成

### Step 2.2 可插拔后端适配（预留）
输出：
- `core/backends/base.py`
- `core/backends/mock.py`
- 未来可接 Ollama/llama.cpp

验收：
- 配置切换 backend 生效

---

## Phase 3 — 工具系统 + RBAC + 审批

### Step 3.1 工具注册表（静态 + 动态扫描）
输出：
- `backend/tools/registry.py`：
  - Tool decorator（name, risk, schema）
  - list_tools()
- DB 表 tools_registry 初始化与同步

验收：
- `GET /v1/tools` 返回工具列表与 schema

### Step 3.2 tools:execute（read_only 先通）
输出：
- `POST /v1/tools:execute`
- `tool_calls` 表写入
- tool runner 执行内置工具（mock web_search）

验收：
- 对话中可调用 read_only 工具并返回结果

### Step 3.3 approvals（挂起与批准）
输出：
- approvals 表 + API：
  - `GET /v1/approvals?status=pending`
  - `POST /v1/approvals/{id}:approve`
  - `POST /v1/approvals/{id}:reject`
- tools:execute 若 risk=external_write 返回 pending_approval + approval_id

验收：
- 高危工具无法直接执行；批准后才能执行并回填 tool_result

### Step 3.4 Auth（owner/member）
输出：
- 简单 auth middleware：基于 header 或 JWT（MVP header）
- owner 才能 approve/commit

验收：
- member 调 approve 返回 403；owner 成功

---

## Phase 4 — Patch 编译器（自然语言改偏好）+ 回滚

### Step 4.1 Seed 与分层状态加载
输出：
- config_seed 表写入默认 seed（json）
- state resolver：seed + committed patches（按 scope）合成 effective config

验收：
- `GET /v1/config/revisions` 返回历史；effective config 可打印（内部）

### Step 4.2 patches:propose（规则+轻模型/LLM 可选）
输出：
- `POST /v1/config/patches:propose`
- MVP：基于规则/模板将常见偏好句转 patch
- risk_level 评估（L1/L2/L3）

验收：
- 输入“更简洁”→ 得到 L1 patch

### Step 4.3 patches:commit 与 rollback
输出：
- commit：写 config_revisions，写 ledger 事件
- rollback：写 rolled_back revision + 重建 effective config

验收：
- commit 后行为变化（至少在回复风格字段体现）
- rollback 后恢复

---

## Phase 5 — Memory（向量/候选区）+ Embeddings 接入

### Step 5.1 Embeddings client
输出：
- Edge 调 Core `/v1/embeddings`，缓存结果

验收：
- 输入文本返回 vector（mock 也可）

### Step 5.2 memory/write 与 candidates
输出：
- `POST /v1/memory/write`
- 低置信度/高影响 -> candidates
- approve/reject API

验收：
- candidates 列表可审批，审批后进入 memory_items

### Step 5.3 memory/query（top_k）
输出：
- `POST /v1/memory/query`
- sqlite-vec/LanceDB 检索 + 返回 top_k

验收：
- 能检索到刚写入的记忆内容

---

## Phase 6 — Audit render（按需渲染）

### Step 6.1 render 报告生成器
输出：
- `POST /v1/audit/render`
- 从 ledger 聚合 timeline：
  - plan, route, tool calls, approvals, patches, memory writes
- 输出结构化 JSON（前端可视化）

验收：
- 选 task_id 能生成完整因果链报告

---

## Phase 7 — 前端 Console

### Step 7.1 Next.js App Shell + 路由
输出：
- Sidebar：Chat/Tasks/Approvals/Memory/Config/Tools
- 基础 layout 与 theme

验收：
- 页面能打开并导航

### Step 7.2 Chat 页面联调
输出：
- 发送消息
- SSE 订阅 task events
- 显示流式输出与 tool 状态

验收：
- 对话可用

### Step 7.3 Approvals/Config/Memory 页面联调
输出：
- Approvals 列表与 approve/reject
- Config propose/commit/rollback
- Memory query + candidates 审批

验收：
- 关键功能可视化可操作

---

## Phase 8 — 可靠性（熔断/降级/重放）

### Step 8.1 Core health + 熔断
输出：
- health 检测
- 超时后切 cloud 或返回明确错误
- 事件链记录 fallback 原因

验收：
- 关掉 Core，系统不挂死，能给出可理解反馈

### Step 8.2 幂等与续跑
输出：
- tool_call_id 幂等
- pending_approval 恢复后继续执行

验收：
- 重启 Edge 后 pending approval 仍可批准并执行

---

## 每个 Phase 的最低验收脚本
- 提供 `scripts/smoke_test.sh`（curl 测试 Agent/Tools/Approvals/Config/Memory/Audit）