# PRD - Blackbox Agent Runtime (Edge) + Core Compute Center

## 项目概述
本项目实现一个“黑匣子”风格的 Agent 系统，采用三层架构：
- **Edge 节点（mac mini/笔记本）**：运行 Agent Runtime（会话隔离、任务队列、原子车道、事件溯源、工具执行与 RBAC、审批流、黑盒记忆、网关适配器）。
- **Core 节点（Mac Studio）**：提供本地算力中心（LLM / Embeddings / Rerank），对 Edge 暴露 OpenAI 风格 API。
- **Cloud（可选）**：当 Core 不可用或需要更强模型时兜底。

产品目标：用户不再编辑 AGENTS.md/SOUL.md 等文档，只保留一个 Seed（宪法层），后续全部通过自然语言交互驱动偏好与行为调整；系统内部通过“Patch 编译器 + 分级变更 + 双闸校验 + 事件溯源”保证可控、安全、可回滚。

## 用户角色
- **Owner（拥有者）**：系统管理员/唯一可信操作人。可提交 L2/L3 变更、审批高危工具。
- **Member（成员）**：群聊或协作成员。仅可触发普通对话与 L1 行为，不可改安全策略与权限。
- **Channel Adapter（渠道适配器）**：Telegram/Slack/Discord/Web 等接入层，提交标准化消息。
- **Tool Host（工具宿主）**：可执行外部工具的服务/进程（本机或局域网），通过统一工具执行接口被调用。
- **Core Compute（算力中心）**：提供 LLM/Embeddings/Rerank 的 HTTP 服务。

## 核心功能列表（Features）
### F1. 会话隔离与任务编排
- 多会话（channel/user/thread）隔离：每个会话独立上下文、记忆作用域与策略叠加。
- 任务队列：消息进入队列，生成 task_id，可查询状态。
- 原子车道：所有“写外部/写持久层/改配置/高危”动作强制串行；只读推理/检索可并行但提交前需合并校验。

### F2. 黑匣子“自我调整”与配置补丁
- 仅一个 Seed（宪法层）用于身份、边界、硬规则、安全基线、预算默认值。
- 自然语言 -> Patch Compiler：将“偏好/规则”转为结构化补丁（JSON Patch/DSL）。
- 变更分级：L1 自动；L2 需确认；L3 强制审批。
- 双闸校验：可信身份/通道 + Policy Validator（禁止降低安全阈值、禁止扩大权限域等）。
- 版本化与回滚：补丁链 revision 可回滚到任意稳定点。

### F3. 工具系统与 RBAC 沙盒
- 动态工具注册：扫描模块并生成 Schema（Function Calling 风格）。
- 工具风险分级：read_only / internal_write / external_write(high)。
- 高危工具默认挂起：生成 approval_id，通过批准后执行；支持一次性/限时/限域授权。

### F4. 事件溯源与按需审计渲染
- 内部不可变事件链：输入/计划/路由/工具调用/结果摘要/错误栈/状态变更/审批决策。
- 默认不输出人类日志；只在审查指令下渲染报告（timeline + diff + 因果链）。
- 支持导出审计报告（JSON）。

### F5. 记忆系统（向量 + 实体关系 + 候选区）
- 长期记忆：向量 + 实体表（user/channel/tool/endpoint 等）。
- 写入治理：重要性阈值、作用域控制；可能误记内容进入候选区需确认后转正。
- 记忆查询：top_k、作用域、时间/来源过滤。

### F6. Core Compute API（本地算力中心）
- OpenAI 风格：
  - `/v1/chat/completions`
  - `/v1/embeddings`
  - `/v1/rerank`（可选但推荐）
  - `/health`

### F7. 降级与熔断
- Core 不可用时：
  - 轻任务：Edge 本地轻模型/规则继续
  - 重任务：切 Cloud（可配置）
- 超时、重试、熔断策略可配置，事件链可重放/续跑。

## 非目标（Non-goals）
- 不实现复杂 UI 设计器/可视化工作流编辑器（先 API + 简易 Web 控制台）。
- 不实现多租户 SaaS（单用户/家庭/小团队为先）。
- 不实现“自动放开权限”的自我进化：安全阈值永远只能更严不能更松（除非 Owner 走管理员重置流程）。

## 成功标准（Definition of Done）
- ✅ 能从任一接入（先做 Web 控制台+Webhook）发送消息，得到 agent 回复（经 Core 推理）。
- ✅ 能执行至少 3 类工具：read_only（搜索/文件读取模拟）、internal_write（写 DB）、external_write（模拟：需审批）。
- ✅ 能完成自然语言偏好调整：`patches:propose` → `patches:commit`，并按 scope 生效（channel vs user）。
- ✅ L3 高危动作必须挂起，Owner 通过 `approvals:*` 批准后才能执行。
- ✅ 事件溯源完整：能渲染某 task 的 plan/tool_calls/policy_decisions/patches/memory_writes。
- ✅ 断开 Core 时，Edge 能检测并降级（至少给出明确错误与 fallback 路径），不会卡死。
- ✅ 所有关键数据（seed、revisions、ledger、memory）持久化且可备份，重启后不丢。