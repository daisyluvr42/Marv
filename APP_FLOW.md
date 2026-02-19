# APP_FLOW — 用户路径与数据流

本项目优先提供一个“Web 控制台 + HTTP API”，后续再接入 Telegram/Slack 等渠道适配器。
核心理念：用户只通过自然语言与系统交互；系统内部以事件溯源记录一切，按需渲染报告。

## 关键页面/界面
- Console 首页（对话页）
- Task 详情页（状态、事件流）
- Approvals 审批页（待审批列表、批准/拒绝）
- Memory 记忆页（查询、候选记忆审核）
- Config 变更页（补丁提案、提交、回滚）
- Tools 工具页（工具列表、风险等级、Schema）

---

## 用户旅程 1：基础对话（走 Core 推理）
1) 用户打开 Console 对话页，选择一个会话（默认 `web:default`）。
2) 输入消息，点击发送。
3) 前端调用 `POST /v1/agent/messages`。
4) Edge：
   - 写入 InputEvent 到 ledger
   - 创建 task 入队
   - 调度执行：规划（plan）→ 调用 Core `/v1/chat/completions`（stream）→ 输出
   - 写入 PlanEvent / RouteEvent / CompletionEvent
5) 前端轮询或订阅 `GET /v1/agent/tasks/{task_id}/events` 显示流式输出。
6) 用户看到回复。

数据流：
Console -> (Agent API) -> task queue -> atomic lane -> core inference -> response -> Console

---

## 用户旅程 2：执行只读工具（无需审批）
1) 用户说：“帮我查一下 X 的信息”。
2) Agent 规划决定调用 read_only 工具（例如 web_search mock）。
3) Edge 调用 `POST /v1/tools:execute`。
4) ToolHost 返回结果，Edge 写 ToolCallEvent + ToolResultEvent。
5) Agent 汇总并回复。

数据流：
Message -> Plan -> tools:execute(read_only) -> tool result -> answer

---

## 用户旅程 3：执行高危工具（必须审批）
1) 用户说：“给某人发邮件/删除某文件/触发外部写操作”。
2) Agent 规划命中 external_write 工具。
3) Edge 调用 `POST /v1/tools:execute` 返回 `pending_approval` 与 `approval_id`。
4) Console 弹出提示：需要审批。并在 Approvals 页出现一条记录。
5) Owner 在 Approvals 页点击“批准（一次性/限时/限域）”：
   - `POST /v1/approvals/{id}:approve`
6) Edge 收到批准后继续执行工具（同一个 tool_call_id），写入 ApprovedEvent → ToolExecutedEvent → ToolResultEvent。
7) Agent 回复执行结果。

数据流：
Message -> Plan -> pending_approval -> Approval UI -> approve -> tool execute -> answer

---

## 用户旅程 4：自然语言调整偏好（黑匣子自我调整）
目标：用户不编辑文件；通过自然语言触发结构化补丁。

1) 用户说：“这个群里只给结论，别解释过程。”
2) Agent 判断这是“偏好变更”，调用：
   - `POST /v1/config/patches:propose`
3) 系统返回 proposal（L1/L2/L3 + patch + explanation）。
4) 若 L1：系统可自动 commit（或仍走 commit 以便可追溯）。
   - `POST /v1/config/patches:commit`
5) 若 L2/L3：Console 弹出确认/审批。
6) 生效范围由 scope 决定：
   - channel scope：只影响该群/频道
   - user scope：影响用户默认行为
7) 事件溯源记录 PatchProposedEvent/PatchCommittedEvent + diff。

数据流：
Natural Language -> propose patch -> (confirm/approve) -> commit -> behavior changes

---

## 用户旅程 5：记忆写入候选区与审核
1) Agent 执行完一轮任务，反思引擎建议写入某条偏好或知识。
2) 若系统判定“可能误记/高影响”，写入候选区：
   - `POST /v1/memory/write` with `requires_confirmation=true`
3) Owner 在 Memory 页看到候选条目并批准/拒绝：
   - `POST /v1/memory/candidates/{id}:approve` or `:reject`
4) 批准后进入长期记忆索引；拒绝则标记为否决，不再重复建议。

---

## 用户旅程 6：按需审计与解释（为什么这么做）
1) 用户点击某个 task 的“解释/审计”按钮。
2) 前端调用：
   - `POST /v1/audit/render`（include plan/tool_calls/policy/patches/memory）
3) 后端从 ledger 重建时间线与因果链，输出结构化报告。
4) Console 展示报告（时间线、关键决策、引用索引、变更 diff、审批记录）。

---

## 页面跳转与状态同步规则（简化）
- 对话页发送消息后跳转到 Task 详情（或在侧栏显示 task 列表）。
- Approvals 页显示 pending 列表；批准后自动刷新对应 Task 详情。
- Config 页显示补丁提案与 revision 列表；回滚后，所有会话策略即时生效（从 state snapshot + patches 重建）。
- Memory 页的候选区操作会影响后续行为（记忆检索结果与偏好推断）。