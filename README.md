# Marv

Marv 是一个可自托管的多渠道 AI Gateway。
你只需要维护一个网关进程，就可以在 WhatsApp、Telegram、Discord、Slack、Signal、WebChat 等入口和同一个智能体对话，并通过 CLI / Web UI / Node 设备统一管理。

## 核心特性

- 多渠道统一接入：一个 Gateway 管理多个聊天平台
- 本地优先与可控部署：配置、会话与凭据保留在你自己的环境
- 智能体运行与路由：支持多智能体、会话隔离、消息路由
- 可扩展插件体系：通过 `extensions/*` 按需安装渠道与能力
- 完整运维工具链：CLI、状态探测、日志、健康检查、诊断

## 架构概览

```text
Chat Channels / WebChat / Nodes
              |
              v
      Marv Gateway (WebSocket)
              |
   -----------------------------
   |            |             |
 Marv CLI   Control UI    Agent Runtime
              |
          Plugins/Extensions
```

默认本地网关地址：`ws://127.0.0.1:18789`

## 仓库模块地图

### 核心源码（`src/`）

| 路径                                    | 作用                                                          |
| --------------------------------------- | ------------------------------------------------------------- |
| `src/cli`                               | CLI 入口与参数编排                                            |
| `src/commands`                          | 各命令实现（`gateway` / `channels` / `memory` / `models` 等） |
| `src/core`                              | Gateway 核心流程、会话与配置核心逻辑                          |
| `src/channels`                          | 内建渠道实现与通道路由                                        |
| `src/providers`                         | 模型提供方接入层                                              |
| `src/memory`                            | 记忆索引、检索与存储                                          |
| `src/routing`                           | 多渠道/多会话路由策略                                         |
| `src/media` + `src/media-understanding` | 媒体处理与理解流水线                                          |
| `src/infra`                             | 守护进程、网络、更新、基础设施封装                            |
| `src/plugins` + `src/plugin-sdk`        | 插件运行时与插件 SDK                                          |
| `src/node-host` + `src/nodes`           | 节点设备接入与命令能力                                        |
| `src/web` + `src/browser` + `src/tui`   | Web 与终端交互面                                              |

### 扩展模块（`extensions/`）

`extensions/*` 为插件工作区，包含渠道扩展（如 `msteams`、`matrix`、`zalo`、`nextcloud-talk`、`twitch` 等）以及能力扩展（如 `voice-call`、`llm-task`、`diagnostics-otel`）。

## 快速开始

### 1) 环境要求

- Node.js `>=22.12.0`
- 推荐包管理器：`pnpm@10`

### 2) 安装依赖

```bash
pnpm install
```

### 3) 初始化并启动

```bash
pnpm marv onboard --install-daemon
pnpm marv gateway status
pnpm marv dashboard
```

### 4) 前台运行（调试）

```bash
pnpm marv gateway run --port 18789 --verbose
```

## 常见使用方式（当前 CLI）

### 渠道管理

```bash
pnpm marv channels list
pnpm marv channels login --channel whatsapp
pnpm marv channels status --probe
```

### 消息与智能体

```bash
pnpm marv message send --target +15555550123 --message "Hello from Marv"
pnpm marv agent --to +15555550123 --message "总结今天的变更"
```

### 模型与记忆

```bash
pnpm marv models status
pnpm marv models list
pnpm marv memory status
pnpm marv memory search --query "deployment notes"
```

### 运行诊断

```bash
pnpm marv status --all
pnpm marv health
pnpm marv doctor
pnpm marv logs --follow
```

## 开发工作流

### 常用脚本

```bash
pnpm build          # 构建
pnpm tsgo           # TypeScript 检查
pnpm check          # format + ts + lint
pnpm test           # 测试
pnpm test:coverage  # 覆盖率
pnpm dev            # 开发运行
```

### 代码组织约定

- 业务逻辑优先放在 `src/commands`、`src/core`、`src/channels`、`src/memory`
- 渠道/能力插件优先放在对应 `extensions/<name>`，避免污染根包依赖
- 测试与源码同目录，命名 `*.test.ts`

## 文档导航

- 快速上手：[/start/getting-started](/start/getting-started)
- CLI 参考：[/cli](/cli)
- Gateway：[/gateway](/gateway)
- 渠道配置：[/channels](/channels)
- 模型提供方：[/providers/models](/providers/models)
- 插件：[/plugins/community](/plugins/community)
- 排障与帮助：[/help](/help)

## 许可证

[MIT](LICENSE)
