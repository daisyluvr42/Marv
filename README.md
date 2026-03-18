<p align="center">
  <img src="assets/marv-logo.png" width="200" alt="Marv" />
</p>

<pre align="center">
▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
██░▄░░▄░██░░▄▄░░██░▄▄▄▀░██░█░░█░██
██░█▀▀█░██░▄▀▀▄░██░▀▀▀▄░██░░██░░██
██░█░░█░██░█░░█░██░█░░█░██░░▀▀░░██
▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
</pre>

# Marv

A multi-channel AI agent gateway forked from [Openclaw](https://github.com/nicepkg/openclaw), with redesigned agent architecture, sub-agent management, and a tiered memory system.

---

## What's Different from Openclaw

### Agent Architecture

- Embedded Pi runner with streaming output, token compaction, and adaptive thinking levels
- External CLI tool delegation — dispatch coding tasks to Claude Code, Codex, Gemini CLI, or Aider
- Tool policy system with pre-call validation, permission tiers, and sandbox isolation

### Sub-Agent Management

- Persistent sub-agent registry that survives gateway restarts with auto-retry
- Parent-child session hierarchy with cross-agent task context propagation
- Automatic complexity classification (simple / moderate / complex) for thinking level selection
- Multi-channel result routing — sub-agent results are delivered back to the originating channel

### Memory System

Four-tier memory model with clarity decay and automatic promotion:

| Tier | Purpose                  | Half-life |
| ---- | ------------------------ | --------- |
| P0   | Core identity & persona  | 365 days  |
| P1   | Verified important facts | 45 days   |
| P2   | General knowledge        | 10 days   |
| P3   | Ephemeral context        | 3 days    |

- Multi-scope weighting: session > agent > user > channel
- Hybrid retrieval: vector RRF + BM25 + lexical + graph expansion + clustering
- Automatic promotion (P3→P2→P1), confidence decay, retrieval reinforcement, semantic evolution
- P3 episodic compaction: clusters similar fragments into distilled P2 semantic knowledge nodes

## Quick Start

```bash
# Global install
npm install -g agentmarv@latest
marv onboard --install-daemon

# Or from source
git clone https://github.com/daisyluvr42/Marv.git
cd Marv && pnpm install && pnpm build
pnpm marv onboard
```

## Repository Layout

```
src/agents/          Agent runtime, sub-agents, tools, external CLI delegation
src/memory/          Memory storage, retrieval, decay, compaction pipeline
src/core/            Gateway core, sessions, configuration
src/channels/        Built-in channels (WhatsApp, Telegram, Discord, Slack, Signal)
src/commands/        CLI command implementations
src/plugins/         Plugin runtime & SDK
extensions/          Channel and capability extension plugins
```

## Development

```bash
pnpm build          # Build
pnpm tsgo           # Type check
pnpm check          # lint + format + ts
pnpm test           # Test
pnpm dev            # Dev mode
```

## Docs

- [Getting Started](/start/getting-started)
- [Deployment](/operations)
- [Gateway](/gateway)
- [Channels](/channels)
- [Models & Providers](/providers/models)
- [CLI Reference](/cli)
- [Plugins](/plugins/community)
- [Proxy Config](/gateway/proxy)
- [Troubleshooting](/help)

## License

[MIT](LICENSE)

---

# Marv（中文）

基于 [Openclaw](https://github.com/nicepkg/openclaw) 衍生的多渠道 AI Agent 网关，重新设计了 agent 运行架构、子 agent 管理和多层记忆系统。

## 与 Openclaw 的主要差异

### Agent 架构

- 内嵌式 Pi runner，支持流式输出、token 压缩和 thinking level 自适应
- 外部 CLI 工具委托：可将编码任务分发给 Claude Code、Codex、Gemini CLI、Aider
- 工具策略系统：调用前置校验、权限分级、沙箱隔离

### 子 Agent 管理

- 持久化子 agent 注册表，网关重启后自动恢复
- 父子会话层级追踪，任务上下文跨 agent 传播
- 复杂度自动分类（simple / moderate / complex），按需选择 thinking level
- 多渠道结果路由：子 agent 完成后自动回送到发起渠道

### 记忆系统

四层记忆模型，按重要性分级存储和衰减：

| 层级 | 用途             | 半衰期 |
| ---- | ---------------- | ------ |
| P0   | 核心身份与人格   | 365 天 |
| P1   | 已验证的重要事实 | 45 天  |
| P2   | 一般知识         | 10 天  |
| P3   | 临时上下文       | 3 天   |

- 多维度作用域：session > agent > user > channel，按权重融合检索
- 混合检索：向量 RRF + BM25 + 词法 + 图谱扩展 + 聚类
- 自动晋升（P3→P2→P1）、置信度衰减、检索增强、语义演化
- P3 episodic 压缩：将相似片段聚类为 P2 语义知识节点

## 快速开始

```bash
# 全局安装
npm install -g agentmarv@latest
marv onboard --install-daemon

# 或从源码
git clone https://github.com/daisyluvr42/Marv.git
cd Marv && pnpm install && pnpm build
pnpm marv onboard
```

## 仓库结构

```
src/agents/          Agent 运行时、子 agent、工具、外部 CLI 委托
src/memory/          记忆存储、检索、衰减、压缩管线
src/core/            Gateway 核心、会话、配置
src/channels/        内建渠道（WhatsApp、Telegram、Discord、Slack、Signal）
src/commands/        CLI 命令实现
src/plugins/         插件运行时与 SDK
extensions/          渠道与能力扩展插件
```

## 开发

```bash
pnpm build          # 构建
pnpm tsgo           # 类型检查
pnpm check          # lint + format + ts
pnpm test           # 测试
pnpm dev            # 开发运行
```

## 文档索引

- [快速上手](/start/getting-started)
- [部署指南](/operations)
- [Gateway 配置](/gateway)
- [渠道配置](/channels)
- [模型与提供方](/providers/models)
- [CLI 参考](/cli)
- [插件](/plugins/community)
- [代理配置](/gateway/proxy)
- [排障与帮助](/help)

## 许可证

[MIT](LICENSE)
