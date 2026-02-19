# FRONTEND_GUIDELINES — 控制台设计系统规范

目标：一个“可用、清晰、偏工程”的 Console，用于对话、任务、审批、记忆、配置与工具管理。

## 组件库与样式
- UI：**shadcn/ui** + **Tailwind CSS 3.4.17**
- 图标：lucide-react
- 布局：App Shell（左侧导航 + 主内容区）
- 响应式：移动端折叠侧边栏；表格改卡片列表。

## 配色（Hex）
- Background: `#0B0F14`
- Surface: `#121826`
- Surface-2: `#182235`
- Border: `#243047`
- Text Primary: `#E6EDF3`
- Text Secondary: `#9FB0C3`
- Muted: `#6B7A90`

- Primary: `#4F8CFF`
- Primary Hover: `#3A7BFF`
- Success: `#2DD4BF`
- Warning: `#FBBF24`
- Danger: `#F87171`

- Code/Log BG: `#0F172A`
- Highlight: `#A78BFA`

对比度要求：正文与背景对比 >= 4.5:1。

## 字体与排印
- Font family：`ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto`
- 基础字号：14px
- 标题：
  - H1: 24/32 (semibold)
  - H2: 18/28 (semibold)
  - H3: 16/24 (medium)
- 行高：正文 1.6，代码块 1.45
- 数字与 ID：使用等宽（`ui-monospace`）

## 间距（Spacing）
使用 4pt 网格：
- xs: 4
- sm: 8
- md: 12
- lg: 16
- xl: 24
- 2xl: 32

容器宽度：
- 主内容 max-width：1200px（超出时居中）
- 列表与表格：尽量全宽

## 页面结构（信息架构）
- Sidebar:
  - Chat
  - Tasks
  - Approvals
  - Memory
  - Config
  - Tools
  - Audit (可合并到 Tasks)

## 关键组件规范
### Chat
- 左侧：会话列表（未来可扩展多 channel）
- 右侧：消息流 + 输入框
- 支持：
  - 流式输出（SSE）
  - tool 调用提示块（pending_approval 高亮）
  - “解释/审计”按钮跳转到 Task

### Task Detail
- 顶部：状态（queued/running/completed/failed/pending_approval）
- Tabs：
  - Output（最终回复/流）
  - Timeline（事件时间线）
  - Tool Calls（参数+结果摘要）
  - Policy（权限与校验）
  - Patches（变更）
  - Memory（写入/候选）

### Approvals
- 列表：approval_id、type、summary、created_at、status
- 操作：approve / reject
- approve 弹窗：
  - one_time（默认 true）
  - expires_in_sec（默认 600）
  - constraints：工具名、to allowlist、路径 allowlist（可选）

### Memory
- Query：输入 query + scope（user/channel）
- Candidates：候选记忆审批列表

### Config
- Proposals：proposal_id、risk_level、explanation、patch diff
- Revisions：revision 列表 + rollback

### Tools
- 工具列表：risk、requires_approval、schema 折叠查看

## 响应式原则
- < 768px：侧边栏抽屉；表格变卡片；减少多列。
- 768–1024：两栏布局保留；长 JSON 默认折叠。
- >1024：完整表格、tab 布局。

## 可用性与反馈
- 所有写操作显示 toast（成功/失败），并提供 request_id。
- 长任务显示进度（事件数/阶段）。
- 错误必须可复制（error_id + stack 摘要）。