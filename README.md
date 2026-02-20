# Marv

Marv 是一个本地优先（local-first）的 Agent 运行时，提供完整的：
- Edge Runtime（任务编排、工具执行、审批、记忆、配置补丁、会话隔离）
- Core API（OpenAI 兼容 `/v1/chat/completions`、`/v1/embeddings`）
- Console（Web 前端）
- 多渠道 IM 接入、定时任务、技能生态、沙箱执行模式

## 1. 功能概览

- 多渠道 IM ingress：`telegram/discord/slack/dingtalk/feishu/webchat`
- 审批体系：`policy|all|risky` 三模式 + grant（session/actor 级临时放行）
- 沙箱执行：`auto|local|sandbox`（IPC 工具可走 Docker）
- 定时任务：Cron 调度，自动投递到现有任务链路
- 多会话隔离：session workspace + subagent（spawn/send/history）
- 记忆闭环：提取/检索/治理（update/delete/forget/decay）+ metrics
- 技能生态：支持通用 `SKILL.md` 导入，含恶意模式前置扫描
- MacOS + iOS 端形态：macOS Electron 壳 + iOS PWA

## 2. 模块介绍

### 2.1 后端模块

- `backend/agent/`：主 API、任务队列、会话流程、runtime 编排
- `backend/core_client/`：Core provider 调用与 fallback/retry
- `backend/gateway/`：Telegram 网关、统一 IM ingress、配对逻辑
- `backend/tools/`：工具注册/执行、IPC 动态工具桥接
- `backend/approvals/`：审批、grant、审批策略模式
- `backend/memory/`：记忆写入、检索、生命周期治理与观测
- `backend/sandbox/`：执行模式配置（auto/local/sandbox）
- `backend/scheduler/`：定时任务存储与 APScheduler 运行时
- `backend/permissions/`：Openclaw 风格执行权限策略
- `backend/patch/`：配置 patch proposal/revision/effective config
- `backend/ledger/`：事件模型与审计时间线
- `backend/storage/`：SQLModel 模型与 DB 初始化
- `backend/skills/`：技能导入、同步、安全扫描
- `backend/packages/`：插件包契约扫描、运行时 hook 加载

### 2.2 前端与客户端

- `frontend/app/`：Console 页面（chat/tasks/approvals/memory/config/tools）
- `frontend/public/sw.js`：PWA service worker（iOS 安装入口）
- `desktop/macos/`：macOS Electron 壳

### 2.3 运行数据

- `data/`：默认运行时数据目录（DB、配置、策略文件等）
- `skill/modules/`：导入后的技能仓

## 3. 安装

### 3.1 前置依赖

- Python `3.12.8`（见 `.python-version`）
- `uv`
- Node `20.11.1`（见 `.nvmrc`）

### 3.2 运行硬件要求（当前版本评估）

以下为当前代码栈（FastAPI + Next.js + SQLite + APScheduler + 可选 Electron + 可选 Docker 沙箱）在本地空载实测与容量估算结果：

- 空载进程内存（RSS）：
  - Core（`uvicorn core.api:app`）约 `15MB`
  - Edge（`uvicorn backend.agent.api:app`）约 `83MB`
  - Frontend Dev（`next dev`）约 `206MB`
- 当前环境磁盘占用（已安装依赖）：
  - `.venv` 约 `145MB`
  - `frontend/node_modules` 约 `371MB`
  - `frontend/.next` 约 `55MB`

| 运行场景 | CPU | 内存 | 磁盘可用空间 |
| --- | --- | --- | --- |
| 基础运行（Core + Edge + CLI） | 最低 2 核，推荐 4 核 | 最低 4GB，推荐 8GB | 最低 2GB，推荐 5GB |
| 本地开发（基础运行 + Next.js dev） | 最低 4 核，推荐 6 核 | 最低 8GB，推荐 16GB | 最低 5GB，推荐 10GB |
| 启用 Docker 沙箱（`execution=sandbox`） | 在对应场景基础上预留 2 核 | 在对应场景基础上额外预留 2-4GB | 额外 2-6GB（镜像与容器层） |
| MacOS Electron 壳（`desktop/macos`） | 与本地开发一致 | 在本地开发基础上额外预留 1GB | 额外 1GB |

iOS 端采用 PWA 形态：客户端仅需 Safari 支持“添加到主屏幕”，主要算力消耗仍在后端主机。

### 3.3 后端 + 前端安装

```bash
git clone https://github.com/daisyluvr42/Marv.git
cd Marv
uv sync
cd frontend
npm ci
```

### 3.4 可选：MacOS 桌面壳安装

```bash
cd Marv/desktop/macos
npm install
```

## 4. 启动方式

### 4.1 推荐：一键启动栈

```bash
cd Marv
bash scripts/start_stack.sh
```

### 4.2 手动启动（开发态）

```bash
# terminal 1
cd Marv
uv run uvicorn core.api:app --port 9000

# terminal 2
cd Marv
CORE_BASE_URL=http://127.0.0.1:9000 uv run uvicorn backend.agent.api:app --port 8000

# terminal 3
cd Marv/frontend
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

### 4.3 可选：Telegram Adapter

```bash
cd Marv
EDGE_BASE_URL=http://127.0.0.1:8000 TELEGRAM_BOT_TOKEN=<token> uv run marv-telegram
```

## 5. 关键环境变量

- `CORE_BASE_URL`：Edge 调 Core 的基础地址
- `EDGE_DATA_DIR` / `EDGE_DB_PATH`：运行数据目录与 DB 路径
- `CORE_PROVIDER_MATRIX_JSON`：provider fallback/retry 矩阵
- `TELEGRAM_BOT_TOKEN`：Telegram 网关 token
- `TELEGRAM_REQUIRE_PAIRING=true|false`：是否启用配对认证
- `EDGE_EXEC_APPROVALS_PATH`：执行权限策略文件路径
- `EDGE_APPROVAL_POLICY_PATH`：审批模式策略文件路径
- `EDGE_EXECUTION_CONFIG_PATH`：执行模式配置文件路径
- `EDGE_IPC_TOOLS_PATH`：IPC 工具配置路径
- `EDGE_SKILLS_ROOT`：技能安装目录（默认 `./skill/modules`）

## 6. CLI 使用总览

统一入口：

```bash
cd Marv
uv run marv --help
```

全局参数：
- `--edge-base-url`
- `--actor-id`
- `--actor-role owner|member`

### 6.1 基础链路：health/chat/task/audit

```bash
# health
uv run marv health

# send message
uv run marv chat send --message "你好" --channel web --follow
uv run marv chat send --message "hi" --conversation-id conv_xxx --channel telegram --channel-id 123 --user-id 456

# task status/events
uv run marv task get <task_id>
uv run marv task events <task_id>

# audit
uv run marv audit timeline <conversation_id>
uv run marv audit render <task_id>
```

### 6.2 工具与审批

```bash
# list tools
uv run marv tools list

# execute tool
uv run marv tools exec --tool mock_web_search --args '{"query":"llm"}'

# 绑定会话上下文
uv run marv tools exec --tool mock_external_write --args '{"target":"./tmp/x","content":"abc"}' --task-id <task_id> --session-id <conversation_id>

# 覆盖 IPC 执行模式（auto/local/sandbox）
uv run marv tools exec --tool ipc_echo --args '{"query":"hello"}' --execution-mode sandbox

# approvals
uv run marv approvals list --status pending
uv run marv approvals approve <approval_id>
uv run marv approvals reject <approval_id>

# 审批通过时创建临时 grant
uv run marv approvals approve <approval_id> --grant-scope session --grant-ttl-seconds 900
uv run marv approvals approve <approval_id> --grant-scope actor --grant-ttl-seconds 1800

# grants 管理
uv run marv approvals grants --status active
uv run marv approvals revoke-grant <grant_id>

# 审批模式
uv run marv approvals policy-show
uv run marv approvals policy-set --mode policy
uv run marv approvals policy-set --mode all
uv run marv approvals policy-set --mode risky --risky-risks external_write,exec,network
```

### 6.3 权限策略（Openclaw 风格）

```bash
uv run marv permissions show
uv run marv permissions set-default --security allowlist --ask on-miss --ask-fallback deny
uv run marv permissions allowlist add --agent main --pattern mock_web_search
uv run marv permissions allowlist sync-readonly --agent main
uv run marv permissions preset --name balanced
uv run marv permissions set-agent --agent telegram:123456 --security full --ask off
uv run marv permissions eval --agent telegram:123456 --tool mock_external_write
```

### 6.4 配置补丁（persona/runtime config）

```bash
uv run marv config propose --text "更简洁" --scope-type channel --scope-id web:default
uv run marv config commit <proposal_id>
uv run marv config rollback <revision>
uv run marv config revisions --scope-type channel --scope-id web:default
uv run marv config effective --channel telegram --channel-id 123456 --user-id 7890
```

### 6.5 记忆系统

```bash
# write/query
uv run marv memory write --scope-type user --scope-id u1 --kind preference --content "我偏好简洁回答" --confidence 0.9
uv run marv memory query --scope-type user --scope-id u1 --query "简洁" --top-k 5

# candidate 审核
uv run marv memory candidates --status pending
uv run marv memory approve <candidate_id>
uv run marv memory reject <candidate_id>

# lifecycle
uv run marv memory items --scope-type user --scope-id u1 --limit 20
uv run marv memory update <item_id> --content "我偏好先给结论再给细节" --confidence 0.95
uv run marv memory delete <item_id>
uv run marv memory forget --scope-type user --scope-id u1 --query "旧偏好" --threshold 0.8
uv run marv memory decay --half-life-days 60 --min-confidence 0.25
uv run marv memory metrics --window-hours 24
```

### 6.6 会话与 subagent

```bash
# sessions
uv run marv sessions list
uv run marv sessions get <conversation_id>
uv run marv sessions archive <conversation_id>

# spawn subagent
uv run marv sessions spawn <parent_conversation_id> --name analyst

# send to existing session
uv run marv sessions send <conversation_id> --message "做二次分析" --wait

# session history
uv run marv sessions history <conversation_id> --limit 100
```

### 6.7 定时任务

```bash
uv run marv schedule list
uv run marv schedule create --name "daily-report" --prompt "生成日报" --cron "0 9 * * *" --channel telegram --channel-id 123 --user-id 456
uv run marv schedule pause <schedule_id>
uv run marv schedule resume <schedule_id>
uv run marv schedule run <schedule_id>
uv run marv schedule delete <schedule_id>
```

### 6.8 多渠道 IM ingress

```bash
uv run marv im channels

# 结构化参数
uv run marv im ingest --channel discord --message "hi" --channel-id room-1 --user-id u-1 --wait

# 原始 payload
uv run marv im ingest --channel slack --payload-json '{"event":{"text":"hello","channel":"C1","user":"U1"}}' --wait
```

### 6.9 Telegram 配对

```bash
uv run marv telegram pair create-code --chat-id 123456 --user-id 7890 --ttl-seconds 900
uv run marv telegram pair codes --status open
uv run marv telegram pair list --chat-id 123456
uv run marv telegram pair revoke <pairing_id>
```

### 6.10 执行模式 / 沙箱

```bash
uv run marv execution show
uv run marv execution set --mode auto
uv run marv execution set --mode sandbox --docker-image python:3.12-alpine --no-network-enabled
```

### 6.11 系统诊断与运维

```bash
uv run marv system core-providers
uv run marv system core-capabilities
uv run marv system core-models
uv run marv system core-auth
uv run marv system ipc-reload

uv run marv heartbeat show
uv run marv heartbeat set --mode interval --interval-seconds 30
uv run marv heartbeat set --mode cron --cron "*/2 * * * *" --core-health-enabled --resume-approved-tools-enabled

# 本地运维
uv run marv ops probe --message "联调探针" --channel telegram --channel-id 123456 --user-id 7890
uv run marv ops stop-services
uv run marv ops package-migration --output-dir ./dist/migrations
```

### 6.12 插件包契约（pi-mono 风格）

```bash
uv run marv packages list
uv run marv packages reload
```

## 7. 技能生态（重点）

### 7.1 技能格式与目录

- 支持通用 `SKILL.md` 格式技能
- 安装目录：`./skill/modules/<source>/<skill>/SKILL.md`

### 7.2 技能命令

```bash
# 列出已安装技能
uv run marv skills list

# 从本地目录导入
uv run marv skills import --source-path ./path/to/skills --source-name custom

# 从 Git 导入
uv run marv skills import --git-url https://github.com/example/repo.git --git-subdir skills --source-name upstream

# 同步上游（Marv + LobsterAI）
uv run marv skills sync-upstream
```

### 7.3 安全导入机制

导入前会扫描并拦截高风险内容，包括：
- 可疑命令模式（例如 `curl|bash`、`wget|sh`、`rm -rf /`）
- 二进制可执行扩展（`.exe/.dll/.dylib/.so/.dmg/...`）
- 越界 symlink

导入结果会记录 manifest，包含 `imported_count/blocked_count` 与每个技能问题详情。

## 8. MacOS + iOS 形态（优先）

### 8.1 MacOS 桌面壳

```bash
cd Marv/desktop/macos
npm install
ELECTRON_START_URL=http://127.0.0.1:3000/chat npm run dev
```

### 8.2 iOS（PWA）

- 前端已内置 `manifest`、`apple-icon`、`service worker`
- iPhone Safari 打开 Console 地址后可“添加到主屏幕”以独立应用形态运行

详见：`docs/MACOS_IOS_CLIENTS.md`

## 9. 测试与验证

```bash
# backend tests
cd Marv
uv run pytest -q

# frontend build
cd Marv/frontend
npm run build

# e2e smoke
cd Marv
EDGE_BASE_URL=http://127.0.0.1:8000 bash scripts/smoke_test.sh
```

## 10. 文档索引

- 权限策略：`docs/PERMISSIONS.md`
- 心跳调度：`docs/HEARTBEAT.md`
- 实机联调调参：`docs/LIVE_TUNING.md`
- IPC 工具桥接：`docs/IPC_TOOLS.md`
- 多渠道 IM：`docs/IM_CHANNELS.md`
- 沙箱执行：`docs/SANDBOX.md`
- 定时任务：`docs/SCHEDULED_TASKS.md`
- Subagent：`docs/SUBAGENTS.md`
- PI-core 重塑路线：`docs/PI_CORE_RESHAPE.md`
- 插件包契约：`docs/PACKAGES.md`
- Skill 生态：`docs/SKILLS_ECOSYSTEM.md`
- MacOS + iOS 客户端：`docs/MACOS_IOS_CLIENTS.md`
- Telegram 配置：`docs/TELEGRAM_SETUP.md`

## 11. 关键路径

- Edge API：`backend/agent/api.py`
- Core API：`core/api.py`
- CLI：`backend/cli.py`
- Memory：`backend/memory/`
- Approvals：`backend/approvals/`
- Scheduler：`backend/scheduler/`
- Skills：`backend/skills/`
- Sandbox：`backend/sandbox/`
- Telegram Gateway：`backend/gateway/telegram.py`
- Console：`frontend/app/`
- MacOS Desktop Shell：`desktop/macos/`
