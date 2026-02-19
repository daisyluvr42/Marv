# TECH_STACK — 推荐技术栈（锁版本）

目标：Edge Runtime 在 macOS/Linux 可靠运行；Core 在 Mac Studio 提供本地推理 API；前端为轻量控制台；尽量减少兼容性地雷。

## 语言与运行时
- Python: **3.12.8**
- Node.js: **20.11.1**（仅用于前端构建与可选工具）
- Package manager (Python): **uv 0.5.24**（锁依赖 + venv）
- Web framework: **FastAPI 0.115.8**
- ASGI server: **uvicorn 0.34.0**（可选 gunicorn 23.0.0）

## 数据与存储（Edge）
- SQLite: **3.45+**（系统自带通常可用；建议通过 pysqlite3-binary 固化）
- ORM/DB: **SQLModel 0.0.22**（基于 SQLAlchemy 2）
- SQLAlchemy: **2.0.38**
- 向量检索：
  - 首选：**sqlite-vec 0.1.6**（作为 SQLite extension）
  - 备选：LanceDB **0.14.0**（若 sqlite-vec 安装不便）
- 迁移：Alembic **1.14.1**

## 事件溯源与队列
- 事件总线/队列：内置（SQLite 表 + async worker）
- 任务调度：APScheduler **3.10.4**（或纯 asyncio 心跳）
- 并发：anyio **4.8.0**（FastAPI 默认栈）

## 安全与认证
- JWT: PyJWT **2.10.1**
- 密码哈希：passlib[bcrypt] **1.7.4**
- mTLS / 证书（可选）：cryptography **44.0.1**
- 请求校验：pydantic **2.10.6**

## HTTP 客户端
- httpx **0.28.1**（Edge 调 Core/Cloud）
- tenacity **9.0.0**（重试/退避/熔断策略）

## Core 推理服务（Mac Studio）
- 建议实现“OpenAI 兼容网关”：
  - FastAPI 0.115.8 + uvicorn
- 模型推理后端（按你的偏好二选一）：
  - Ollama **0.5.11**（易部署，API 成熟）
  - llama.cpp server（版本随你本地编译，但在网关层统一成 OpenAI 风格）
- 备注：本项目只负责“网关与路由”，推理后端可插拔。

## 前端（控制台）
- Next.js **15.1.6**
- React **19.0.0**
- TypeScript **5.7.3**
- Tailwind CSS **3.4.17**
- shadcn/ui：按当前模板生成（锁定组件版本到仓库）
- 状态管理：zustand **4.5.6**
- 请求：fetch + SSE（EventSource）或 ws（可选）

## 工具系统与 Schema
- JSON Schema：pydantic 自动生成
- MCP 风格 registry：本项目自定义 Tool Manifest（JSON）+ Python decorator

## 测试
- pytest **8.3.4**
- pytest-asyncio **0.25.3**
- httpx[http2]（可选）
- 前端：vitest **2.1.8**（可选）

## 格式化与静态检查
- ruff **0.9.5**
- mypy **1.15.0**
- black **24.10.0**（可选，ruff 已可覆盖大部）

## 部署与运行
- Edge：`uv run python -m agentd`（或 `uvicorn agent.api:app`)
- Core：`uvicorn core.api:app --port 9000`
- 数据目录：`./data/`（可配置）