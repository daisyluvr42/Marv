## Skill 核心内容（给 vibe agent 执行的那部分）

### 1) 执行总原则（Hard Constraints）
- 需求唯一来源：PRD.md、APP_FLOW.md、TECH_STACK.md、FRONTEND_GUIDELINES.md、BACKEND_STRUCTURE.md、IMPLEMENTATION_PLAN.md、SKILL.md。
- 严格按 IMPLEMENTATION_PLAN.md 的 Step 顺序推进（除非 HANDOFF.json 指示 blocked/skip）。
- 技术栈与版本严格锁定 TECH_STACK.md；不得引入替代框架/版本。
- 禁止引入可编辑引导文件体系（AGENTS.md/SOUL.md/TOOLS.md/...），只允许一个 Seed（宪法层）。
- 所有关键动作必须写入事件溯源 ledger（append-only）；默认不输出人类日志；审计通过 `/v1/audit/render` 按需渲染。
- 原子车道：写外部/写持久层/改配置/高危工具必须串行；只读可并行但提交前需合并校验。
- 工具系统必须风险分级 + RBAC；external_write 默认挂起，只能 owner 审批放行。
- Core 必须提供 OpenAI 风格 API：`/v1/chat/completions`、`/v1/embeddings`、（可选）`/v1/rerank`、`/health`。
- 安全优先：宁可更严格；secrets 禁止进入 ledger/LLM 上下文/前端明文。

---

### 2) JSON 交接机制（Single Source of Truth）
- 仓库根目录必须存在 `HANDOFF.json`，它是**当前任务状态的唯一事实来源**。
- 每次开始工作前：必须读取 `HANDOFF.json`，以 `execution.current_step` 作为当前要做的 Step。
- 每个 Step 完成后：必须更新 `HANDOFF.json` 并写入历史快照：
  - `handoffs/history/YYYYMMDD_HHMMSS_step-<STEP>.json`

---

### 3) HANDOFF.json 必备字段（最小协议）
- `execution.current_step`：当前 Step（id/title/status/acceptance_criteria）
- `execution.next_step`：下一 Step（id/title/reason）
- `state.blockers`：阻塞项（为空才能继续）
- `artifacts.changed_files`：本 Step 文件改动清单（added/modified/deleted）
- `artifacts.commands`：setup/run/test/smoke 命令（可复现）
- `verification.status`：passed/failed/not_run + 结果记录
- `handoff.summary` + `handoff.how_to_verify` + `handoff.rollback_plan`

---

### 4) Autopilot 工作流（必须照做）
#### 4.1 启动前检查
1) 读取 `HANDOFF.json`
2) 若 `current_step.status == "done"`：把 `next_step` 提升为新的 `current_step`，并继续
3) 若 `state.blockers` 非空或 `current_step.status == "blocked"`：优先解决 blocker（仅做最小修复，不得越界）
4) 打开 IMPLEMENTATION_PLAN.md，定位该 Step 的要求与验收标准

#### 4.2 实现规则
1) 只实现当前 Step，禁止顺手做后续 Step
2) 任何新增/修改文件都必须写入 `artifacts.changed_files`
3) 若触及 API/db/event，更新：
   - `artifacts.api_endpoints_touched`
   - `artifacts.db_tables_touched`
   - `artifacts.events_emitted`
4) 写入可执行命令到 `artifacts.commands`（至少 smoke）

#### 4.3 验收与状态推进
1) 运行 smoke（若环境允许）并记录结果
2) 若通过：
   - `verification.status="passed"`
   - `current_step.status="done"`
   - 设定 `next_step` 为 IMPLEMENTATION_PLAN 的下一步
3) 若失败：
   - `verification.status="failed"`
   - `current_step.status="blocked"`
   - 把失败原因写入 `state.blockers`（含日志/修复建议）

#### 4.4 交接输出
完成一步后必须输出：
- 文件变更清单
- 每个文件完整内容
- 本步验证命令
- 更新后的 `HANDOFF.json` 完整内容
- 若失败：blockers 与最小修复建议

---

### 5) 禁止事项（Hard No）
- 禁止新增/使用 AGENTS.md/SOUL.md/TOOLS.md 等可编辑引导文件替代 Seed。
- 禁止 secrets/PII 明文写入 ledger 或返回前端。
- 禁止自动批准 L3 或 external_write 工具。
- 禁止一次实现多个 Step（除非 HANDOFF.json 明确合并）。
- 禁止修改 TECH_STACK.md 锁定版本（除非记录 blocker 并请求人工决策）。