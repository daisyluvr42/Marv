# BACKEND_STRUCTURE — 后端结构、Schema、API、Auth、Storage

## 总体结构（Edge）
- `agentd`：主 API + 调度器 + 原子车道执行器 + Patch 编译器 + Policy 校验
- `ledger`：事件溯源（append-only）
- `memory`：向量+实体+候选区
- `tools`：注册表 + 执行入口 + 回调
- `approvals`：审批流
- `gateway`：接入适配器（MVP 先做 Web）

建议目录：
- `backend/agent/` FastAPI app
- `backend/core_client/` Core/Cloud 调用
- `backend/storage/` SQLite + migrations
- `backend/ledger/` event schemas + append/query
- `backend/policy/` risk model + validator
- `backend/patch/` compiler + revision store
- `backend/tools/` registry + runner + callbacks
- `backend/approvals/` approval model + API
- `backend/memory/` embeddings client + vec store + entities

---

## 数据库（SQLite）Schema（建议）

### 1) conversations
- id (PK, text) — conv_*
- channel (text) — web/telegram/slack/...
- channel_id (text, nullable) — 群/频道 id
- user_id (text, nullable)
- thread_id (text, nullable)
- created_at (int)
- updated_at (int)

Index：channel + channel_id + thread_id

### 2) tasks
- id (PK, text) — task_*
- conversation_id (FK)
- status (text) — queued/running/completed/failed/pending_approval
- created_at (int)
- updated_at (int)
- last_error (text, nullable)
- current_stage (text, nullable) — plan/tool/answer/...

Index：conversation_id, status

### 3) ledger_events (append-only)
- id (PK, integer autoincrement)
- event_id (text unique) — evt_*
- task_id (FK, nullable)
- conversation_id (FK)
- type (text) — InputEvent/PlanEvent/ToolCallEvent/...
- ts (int)
- actor_id (text, nullable)
- payload_json (text) — 事件内容（JSON）
- hash (text) — 可选：链式 hash
- prev_hash (text) — 可选

Index：conversation_id, task_id, ts, type

### 4) config_seed (single row)
- id (PK, text) = "seed"
- seed_json (text)
- created_at (int)

### 5) config_revisions
- revision (PK, text) — rev_yyyymmdd_xxxx
- scope_type (text) — global/user/channel/conversation
- scope_id (text)
- created_at (int)
- actor_id (text)
- patch_json (text) — json patch array
- explanation (text)
- risk_level (text) — L1/L2/L3
- status (text) — proposed/committed/rolled_back

Index：scope_type+scope_id, created_at

### 6) patch_proposals
- proposal_id (PK, text)
- scope_type, scope_id
- natural_language (text)
- patch_json (text)
- risk_level (text)
- explanation (text)
- needs_approval (int 0/1)
- created_at (int)
- actor_id (text)
- status (text) — open/committed/rejected

### 7) approvals
- approval_id (PK, text) — ap_*
- type (text) — tool_execute/config_change
- status (text) — pending/approved/rejected/expired
- summary (text)
- constraints_json (text) — one_time/expires/allowlist
- created_at (int)
- updated_at (int)
- actor_id (text) — requestor
- decided_by (text, nullable)

Index：status, created_at

### 8) tools_registry
- name (PK, text)
- version (text)
- risk (text) — read_only/internal_write/external_write
- requires_approval (int)
- schema_json (text)
- enabled (int)
- updated_at (int)

### 9) tool_calls
- tool_call_id (PK, text) — tc_*
- task_id (FK)
- tool (text)
- args_json (text)
- status (text) — pending/running/ok/error/pending_approval
- approval_id (FK nullable)
- created_at (int)
- updated_at (int)
- result_json (text nullable)
- error (text nullable)

Index：task_id, status

### 10) memory_items
- id (PK, text) — mem_*
- scope_type (text) — user/channel/global
- scope_id (text)
- kind (text) — preference/knowledge/entity
- content (text)
- embedding (blob or vec ref) — sqlite-vec 存储
- confidence (real)
- created_at (int)
- source_event_id (text nullable)

Index：scope_type+scope_id, created_at

### 11) memory_candidates
- id (PK, text) — cand_*
- scope_type, scope_id
- content (text)
- embedding (blob/vec ref)
- confidence (real)
- status (text) — pending/approved/rejected
- created_at (int)
- decided_at (int nullable)
- decided_by (text nullable)

### 12) secrets (加密存储简化版)
- key (PK, text) — CORE_TOKEN / CLOUD_API_KEY ...
- value_enc (blob)
- updated_at (int)

备注：MVP 可先用环境变量，后续补齐加密存储。

---

## API 规范（Edge）

### Agent
- `POST /v1/agent/messages`
- `GET /v1/agent/tasks/{task_id}`
- `GET /v1/agent/tasks/{task_id}/events` (SSE)

### Tools
- `GET /v1/tools`
- `POST /v1/tools:execute`
- `POST /v1/tools/callback`

### Config / Patches
- `POST /v1/config/patches:propose`
- `POST /v1/config/patches:commit`
- `POST /v1/config/revisions:rollback`
- `GET /v1/config/revisions?scope=...`

### Memory
- `POST /v1/memory/query`
- `POST /v1/memory/write`
- `POST /v1/memory/candidates/{id}:approve`
- `POST /v1/memory/candidates/{id}:reject`

### Approvals
- `GET /v1/approvals?status=pending`
- `POST /v1/approvals/{approval_id}:approve`
- `POST /v1/approvals/{approval_id}:reject`

### Audit
- `GET /v1/audit/conversations/{conversation_id}/timeline`
- `POST /v1/audit/render`

---

## Core API（Mac Studio）
- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/rerank`（推荐）
- `GET /health`

---

## 认证与权限（Auth）
MVP：先实现“单 Owner + Member”两级：
- Header：
  - `X-Actor-Id`
  - `X-Actor-Role` = owner|member
  - `X-Channel` / `X-Conversation-Id`
- 规则：
  - owner 才能：
    - commit L2/L3 patch
    - approve/reject approvals
    - 执行 external_write 工具（且仍需 approval）
  - member 只能：
    - 发送消息
    - 触发 read_only 工具
    - 查看自己会话的 task（可按 scope 限制）
- 双闸：
  1) actor/role/channel 是可信来源（Web 控制台先内置登录）
  2) policy validator 禁止降低安全阈值/扩大权限域

后续增强：
- JWT 登录（owner token）
- mTLS（Edge<->Core）
- request signing（webhook 回调）

---

## 存储规则（Storage）
- 数据目录：`./data/`（可配置）
  - `data/edge.db` SQLite
  - `data/ledger/` append-only 文件（可选）或 ledger_events 表
  - `data/media/` 附件与缓存（哈希命名）
- 备份：
  - 定期导出 SQLite + ledger snapshot
- PII/Secrets：
  - 禁止写入事件 payload 的明文字段（需脱敏）
  - secrets 仅存在 vault/环境变量，不进入 LLM 上下文