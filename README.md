# 🤖 Marv — Personal Multi-Channel AI Assistant

<p align="center">
  <strong>EXFOLIATE! EXFOLIATE!</strong>
</p>

**Marv** is a self-hosted AI gateway that lets one assistant live across the chat surfaces you already use.
Run it locally (or on your own server), connect channels, and message your agent from anywhere.

## Why Marv

- One gateway for many chat apps and device nodes.
- Local-first deployment and data ownership.
- Strong CLI + automation surface (gateway, channels, routing, memory, tools).
- Plugin/extension architecture for channels, providers, and custom commands.
- Companion apps for macOS/iOS/Android.

## How It Works

```text
Chat apps (WhatsApp/Telegram/Discord/Slack/...) + WebChat + device nodes
                              |
                              v
                 Marv Gateway (ws://127.0.0.1:18789)
                              |
          -------------------------------------------------
          |               |               |               |
        Agent           CLI         Control UI       Apps/Nodes
```

The Gateway is the control plane: sessions, routing, channels, tool access, and health.

## User Installation, Deployment, and Operations Guide

This section is a complete end-user flow:

1. Install Marv
2. Deploy and start the Gateway
3. Configure auth/safety and connect channels
4. Run daily operations (status/logs/restart/update/backup)

### 0) Requirements

- Node.js **22.12.0+**
- macOS, Linux, or Windows (WSL2 recommended for Windows)
- Docker (optional, only for Docker deployment)

Check runtime:

```bash
node -v
npm -v
```

### 1) Install Marv

#### Option A (recommended): Installer script

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash

# Windows (PowerShell)
iwr -useb https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.ps1 | iex
```

#### Option B: npm / pnpm global install

```bash
npm install -g "git+https://github.com/daisyluvr42/Marv.git#main"
# or
pnpm add -g "github:daisyluvr42/Marv#main"
```

Then run onboarding:

```bash
marv onboard --install-daemon
```

#### Option C: From source (development / contributors)

```bash
git clone https://github.com/daisyluvr42/Marv.git
cd Marv
pnpm install
pnpm ui:build
pnpm build
pnpm marv onboard --install-daemon
```

#### Option D: Docker deployment

From repo root:

```bash
./docker-setup.sh
```

Manual Docker Compose path:

```bash
docker build -t marv:local -f Dockerfile .
docker compose run --rm marv-cli onboard
docker compose up -d marv-gateway
```

#### Option E: 24/7 VPS deployment

If you want always-on remote hosting (for example Hetzner), use the production VPS guide.

---

### 2) Deploy and start the Gateway

#### Local foreground run

```bash
marv gateway --port 18789
# debug logs to terminal
marv gateway --port 18789 --verbose
```

#### Service/supervised run (recommended for daily use)

```bash
marv gateway install
marv gateway restart
marv gateway status
```

Onboarding (`marv onboard --install-daemon`) already installs service mode for most users.

#### Verify health

```bash
marv gateway status
marv status
marv channels status --probe
marv health
marv logs --follow
```

Healthy baseline: gateway is running and probe/health checks are OK.

#### Open Control UI

```bash
marv dashboard
```

Default local URL: `http://127.0.0.1:18789/`

#### Remote access to a remote gateway host

Preferred: VPN/Tailscale.
Fallback: SSH tunnel.

```bash
ssh -N -L 18789:127.0.0.1:18789 user@host
```

After tunneling, connect locally to `ws://127.0.0.1:18789`.

---

### 3) Initial configuration and security setup

#### Run wizard setup

```bash
marv onboard
# or for config-only wizard
marv configure
```

#### Config CLI basics

```bash
marv config get agents.defaults.workspace
marv config set agents.defaults.heartbeat.every "2h"
marv config unset tools.web.search.apiKey
```

Main config path: `~/.marv/marv.json`

#### Recommended DM safety policy

Use pairing mode for DMs (default on major channels):

```json5
{
  channels: {
    telegram: {
      enabled: true,
      dmPolicy: "pairing",
    },
  },
}
```

Approve pairing requests:

```bash
marv pairing list telegram
marv pairing approve telegram <CODE>
```

---

### 4) Connect channels (quick recipes)

Marv supports built-in and extension-backed channels.

#### WhatsApp

```bash
marv channels login --channel whatsapp
marv gateway
```

Optional account-scoped login:

```bash
marv channels login --channel whatsapp --account work
```

#### Telegram

1. Create token with `@BotFather`.
2. Set token and start gateway:

```bash
marv config set channels.telegram.botToken '"123:abc"' --json
marv config set channels.telegram.enabled true --json
marv gateway
```

3. Approve first DM pairing code:

```bash
marv pairing list telegram
marv pairing approve telegram <CODE>
```

#### Discord

1. Create bot app in Discord Developer Portal and get token.
2. Configure and start:

```bash
marv config set channels.discord.token '"YOUR_BOT_TOKEN"' --json
marv config set channels.discord.enabled true --json
marv gateway restart
```

3. Approve pairing:

```bash
marv pairing list discord
marv pairing approve discord <CODE>
```

#### Extension channels (Matrix/Teams/Zalo/etc.)

Install plugin first, then configure that channel.

```bash
marv plugins list
marv plugins install <path-or-spec>
```

---

### 5) Daily operations (runbook)

#### Core ops commands

```bash
marv status
marv health
marv doctor
marv gateway status --deep
marv logs --follow
marv channels status --probe
```

#### Start/stop/restart

```bash
marv gateway stop
marv gateway restart
marv gateway status
```

#### Send messages / run agent

```bash
marv message send --target +15555550123 --message "Hello from Marv"
marv agent --message "Summarize today's issues" --thinking high
```

#### Update safely

```bash
# installer path
curl -fsSL https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash

# or package manager path
npm i -g "git+https://github.com/daisyluvr42/Marv.git#main"
# or
pnpm add -g "github:daisyluvr42/Marv#main"

marv doctor
marv gateway restart
marv health
```

#### Backup paths (important)

Back up these directories/files before major upgrades or migrations:

- `~/.marv/marv.json`
- `~/.marv/credentials/`
- `~/.marv/workspace/`

---

### 6) Troubleshooting checklist

```bash
marv doctor
marv gateway status --deep
marv channels status --probe
marv logs --follow
```

Common quick fixes:

- `marv` not found: ensure global npm/pnpm bin is in `PATH`
- port conflict: start with `marv gateway --force`
- auth error on remote access: confirm gateway token/password in client settings
- post-update issues: run `marv doctor`, then restart gateway

## External AI Agent Memory Integration

You can connect external AI agents to Marv memory without building a custom DB adapter.

### Option A (recommended): MCP over HTTP (`/mcp`)

Marv exposes an MCP-compatible endpoint from the Gateway:

- `POST /mcp`
- Example URL: `http://127.0.0.1:18789/mcp`
- Auth: `Authorization: Bearer <gateway-token-or-password>`

Exposed memory tools:

- `memory_search`
- `memory_get`
- `memory_write`

Minimal `memory_search` call:

```bash
curl -sS http://127.0.0.1:18789/mcp \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "search-1",
    "method": "tools/call",
    "params": {
      "name": "memory_search",
      "sessionKey": "agent:main:main",
      "arguments": {
        "query": "deployment checklist",
        "maxResults": 6
      }
    }
  }'
```

### Option B: Direct tool invoke API (`/tools/invoke`)

If your external orchestrator is not MCP-native, call tools directly:

- `POST /tools/invoke`
- Example URL: `http://127.0.0.1:18789/tools/invoke`
- Auth: `Authorization: Bearer <gateway-token-or-password>`

Minimal `memory_search` call:

```bash
curl -sS http://127.0.0.1:18789/tools/invoke \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "tool": "memory_search",
    "sessionKey": "agent:main:main",
    "args": {
      "query": "deployment checklist",
      "maxResults": 6
    }
  }'
```

### Notes for external agents

- This is fully automatic at retrieval time (no per-call human confirmation required).
- `memory_search` results may include:
  - `references` (explicit citation chain from `[ref:mem_xxx]`)
  - `salienceScore`, `salienceDecay`, `salienceReinforcement`
  - `referenceBoost` (multi-hop chain-weight boost)
- Use `memory_get` to fetch exact content by `soul-memory/<itemId>` path.
- Use `memory_write` to persist durable memory entries (`kind/scope/source/confidence`).

## Supported Channels

Marv supports built-in and extension-backed channels.

Core channels:

- WhatsApp
- Telegram
- Discord
- IRC
- Google Chat
- Slack
- Signal
- iMessage (legacy)

Extension/plugin channels (available in this repo and docs):

- BlueBubbles (recommended iMessage path)
- Feishu
- Mattermost
- Microsoft Teams
- Matrix
- LINE
- Nextcloud Talk
- Nostr
- Tlon
- Twitch
- Zalo
- Zalo Personal
- WebChat (gateway web surface)

## Common CLI Commands

```bash
# Setup
marv setup
marv onboard
marv configure
marv doctor

# Gateway
marv gateway run
marv gateway status
marv gateway health

# Messaging and agent
marv message send --target <id> --message "..."
marv agent --message "Ship checklist" --thinking high

# Channels / plugins
marv channels list
marv channels status --probe
marv plugins list
marv plugins install <path-or-spec>

# Operations
marv status
marv health
marv logs
marv update
```

## Development Commands

```bash
pnpm build            # Build TypeScript + bundled assets
pnpm tsgo             # TypeScript checks
pnpm check            # format check + type-aware lint
pnpm test             # Test suite
pnpm test:coverage    # Coverage run
pnpm ui:dev           # Control UI dev server
pnpm gateway:watch    # Gateway watch/dev loop
```

## Repository Layout

```text
src/             CLI, gateway, channels, routing, media pipeline
extensions/      Built-in extension packages (channels/providers/features)
apps/            macOS, iOS, Android clients
ui/              Web Control UI
docs/            Mintlify documentation
scripts/         Build, release, QA, and automation scripts
```

## Plugins and Extension Development

Marv exposes a plugin SDK via `marv/plugin-sdk` and supports extensions from:

- bundled `extensions/*`
- `~/.marv/extensions`
- `<workspace>/.marv/extensions`

## Security

Marv connects to real messaging surfaces. Treat inbound messages as untrusted input.

- Run `marv security audit` for local checks.
- Keep gateway bind mode loopback unless you intentionally configure remote access.
- Prefer pairing/allowlists for DM safety.

## Release Channels

- **stable**: tagged releases (`vYYYY.M.D`), npm `latest`
- **beta**: prereleases (`vYYYY.M.D-beta.N`), npm `beta`
- **dev**: moving `main`

Switch/update with:

```bash
marv update --channel stable|beta|dev
```

## Contributing

- Contribution guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security policy: [`SECURITY.md`](SECURITY.md)

## License

MIT. See [`LICENSE`](LICENSE).

---

## README 中文版

# 🤖 Marv — 个人多渠道 AI 助手

**Marv** 是一个自托管 AI 网关，让同一个助手同时运行在你常用的聊天平台中。  
你可以本地部署（或部署到自有服务器），连接渠道后随时从任意端给 Agent 发消息。

## 为什么选择 Marv

- 一个网关连接多个聊天应用与设备节点
- 本地优先，数据由你掌控
- CLI 与自动化能力完善（网关、渠道、路由、记忆、工具）
- 插件/扩展架构，方便扩展渠道、模型与命令
- 提供 macOS/iOS/Android 配套应用

## 工作原理

```text
聊天应用（WhatsApp/Telegram/Discord/Slack/...）+ WebChat + 设备节点
                              |
                              v
                 Marv Gateway (ws://127.0.0.1:18789)
                              |
          -------------------------------------------------
          |               |               |               |
        Agent           CLI         Control UI       Apps/Nodes
```

Gateway 是控制平面：负责会话、路由、渠道、工具访问与健康状态。

## 用户安装、部署与运维指南

建议按下面流程进行：

1. 安装 Marv
2. 部署并启动 Gateway
3. 完成鉴权/安全配置并连接渠道
4. 日常运维（状态、日志、重启、更新、备份）

### 0）环境要求

- Node.js **22.12.0+**
- macOS / Linux / Windows（Windows 推荐 WSL2）
- Docker（可选，仅 Docker 部署时需要）

检查运行时：

```bash
node -v
npm -v
```

### 1）安装 Marv

#### 方案 A（推荐）：安装脚本

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash

# Windows (PowerShell)
iwr -useb https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.ps1 | iex
```

#### 方案 B：npm / pnpm 全局安装

```bash
npm install -g "git+https://github.com/daisyluvr42/Marv.git#main"
# 或
pnpm add -g "github:daisyluvr42/Marv#main"
```

安装后执行初始化：

```bash
marv onboard --install-daemon
```

#### 方案 C：源码安装（开发者）

```bash
git clone https://github.com/daisyluvr42/Marv.git
cd Marv
pnpm install
pnpm ui:build
pnpm build
pnpm marv onboard --install-daemon
```

#### 方案 D：Docker 部署

在仓库根目录执行：

```bash
./docker-setup.sh
```

或手动 Docker Compose：

```bash
docker build -t marv:local -f Dockerfile .
docker compose run --rm marv-cli onboard
docker compose up -d marv-gateway
```

#### 方案 E：24/7 VPS 常驻部署

如果你希望远程长期在线运行（例如 Hetzner），请使用生产 VPS 部署文档。

### 2）部署并启动 Gateway

#### 本地前台运行

```bash
marv gateway --port 18789
# 输出调试日志
marv gateway --port 18789 --verbose
```

#### 服务化运行（推荐）

```bash
marv gateway install
marv gateway restart
marv gateway status
```

多数场景下，`marv onboard --install-daemon` 已自动完成服务安装。

#### 健康检查

```bash
marv gateway status
marv status
marv channels status --probe
marv health
marv logs --follow
```

正常基线：Gateway 运行中，`probe/health` 均通过。

#### 打开控制台 UI

```bash
marv dashboard
```

默认地址：`http://127.0.0.1:18789/`

#### 访问远程 Gateway

优先使用 VPN/Tailscale；备选方案为 SSH 隧道：

```bash
ssh -N -L 18789:127.0.0.1:18789 user@host
```

隧道建立后，本地连接 `ws://127.0.0.1:18789`。

### 3）初始配置与安全设置

#### 启动向导

```bash
marv onboard
# 或仅配置
marv configure
```

#### 配置命令基础

```bash
marv config get agents.defaults.workspace
marv config set agents.defaults.heartbeat.every "2h"
marv config unset tools.web.search.apiKey
```

主配置文件：`~/.marv/marv.json`

#### 推荐私聊安全策略

建议开启 `pairing`（主流渠道默认如此）：

```json5
{
  channels: {
    telegram: {
      enabled: true,
      dmPolicy: "pairing",
    },
  },
}
```

审批配对请求：

```bash
marv pairing list telegram
marv pairing approve telegram <CODE>
```

### 4）连接渠道（快速示例）

#### WhatsApp

```bash
marv channels login --channel whatsapp
marv gateway
```

可选账号维度登录：

```bash
marv channels login --channel whatsapp --account work
```

#### Telegram

1. 用 `@BotFather` 创建 Token
2. 写入配置并启动：

```bash
marv config set channels.telegram.botToken '"123:abc"' --json
marv config set channels.telegram.enabled true --json
marv gateway
```

3. 审批首次私聊配对：

```bash
marv pairing list telegram
marv pairing approve telegram <CODE>
```

#### Discord

1. 在 Discord Developer Portal 创建 Bot 并获取 Token
2. 配置并启动：

```bash
marv config set channels.discord.token '"YOUR_BOT_TOKEN"' --json
marv config set channels.discord.enabled true --json
marv gateway restart
```

3. 审批配对：

```bash
marv pairing list discord
marv pairing approve discord <CODE>
```

#### 扩展渠道（Matrix/Teams/Zalo 等）

先安装插件，再做对应渠道配置：

```bash
marv plugins list
marv plugins install <path-or-spec>
```

### 5）日常运维（Runbook）

#### 核心命令

```bash
marv status
marv health
marv doctor
marv gateway status --deep
marv logs --follow
marv channels status --probe
```

#### 启停与重启

```bash
marv gateway stop
marv gateway restart
marv gateway status
```

#### 发送消息与执行 Agent

```bash
marv message send --target +15555550123 --message "Hello from Marv"
marv agent --message "Summarize today's issues" --thinking high
```

#### 安全更新

```bash
# 安装脚本更新
curl -fsSL https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash

# 或包管理器更新
npm i -g "git+https://github.com/daisyluvr42/Marv.git#main"
# 或
pnpm add -g "github:daisyluvr42/Marv#main"

marv doctor
marv gateway restart
marv health
```

#### 关键备份路径

- `~/.marv/marv.json`
- `~/.marv/credentials/`
- `~/.marv/workspace/`

### 6）故障排查清单

```bash
marv doctor
marv gateway status --deep
marv channels status --probe
marv logs --follow
```

常见问题：

- `marv` 命令找不到：确认 npm/pnpm 全局 bin 在 `PATH` 中
- 端口冲突：尝试 `marv gateway --force`
- 远程访问鉴权失败：检查 token/password 配置
- 更新后异常：先 `marv doctor`，再重启 Gateway

## 外部 AI Agent 对接 Memory

无需直接连接数据库，可通过网关 API 对接记忆系统。

### 方式 A（推荐）：MCP over HTTP（`/mcp`）

- 地址：`POST /mcp`
- 示例：`http://127.0.0.1:18789/mcp`
- 认证：`Authorization: Bearer <gateway-token-or-password>`

可调用工具：

- `memory_search`
- `memory_get`
- `memory_write`

最小 `memory_search` 请求：

```bash
curl -sS http://127.0.0.1:18789/mcp \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "search-1",
    "method": "tools/call",
    "params": {
      "name": "memory_search",
      "sessionKey": "agent:main:main",
      "arguments": {
        "query": "deployment checklist",
        "maxResults": 6
      }
    }
  }'
```

### 方式 B：直接工具调用（`/tools/invoke`）

- 地址：`POST /tools/invoke`
- 示例：`http://127.0.0.1:18789/tools/invoke`
- 认证：`Authorization: Bearer <gateway-token-or-password>`

最小 `memory_search` 请求：

```bash
curl -sS http://127.0.0.1:18789/tools/invoke \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "tool": "memory_search",
    "sessionKey": "agent:main:main",
    "args": {
      "query": "deployment checklist",
      "maxResults": 6
    }
  }'
```

### 对接说明

- 检索流程默认自动执行，不需要每次人工确认
- `memory_search` 结果可能包含：
  - `references`（显式引用链，如 `[ref:mem_xxx]`）
  - `salienceScore`、`salienceDecay`、`salienceReinforcement`
  - `referenceBoost`（多跳引用链加权提升）
- 通过 `memory_get` 用 `soul-memory/<itemId>` 精确读取内容
- 通过 `memory_write` 写入结构化长期记忆（`kind/scope/source/confidence`）

## 支持的渠道

核心渠道：

- WhatsApp
- Telegram
- Discord
- IRC
- Google Chat
- Slack
- Signal
- iMessage（legacy）

扩展/插件渠道（仓库和文档中可见）：

- BlueBubbles（推荐 iMessage 路径）
- Feishu
- Mattermost
- Microsoft Teams
- Matrix
- LINE
- Nextcloud Talk
- Nostr
- Tlon
- Twitch
- Zalo
- Zalo Personal
- WebChat（网关 Web 入口）

## 常用 CLI 命令

```bash
# 初始化
marv setup
marv onboard
marv configure
marv doctor

# Gateway
marv gateway run
marv gateway status
marv gateway health

# 消息与 Agent
marv message send --target <id> --message "..."
marv agent --message "Ship checklist" --thinking high

# 渠道与插件
marv channels list
marv channels status --probe
marv plugins list
marv plugins install <path-or-spec>

# 运维
marv status
marv health
marv logs
marv update
```

## 开发命令

```bash
pnpm build            # 构建 TypeScript 与静态资源
pnpm tsgo             # TypeScript 检查
pnpm check            # 格式检查 + 类型感知 lint
pnpm test             # 测试
pnpm test:coverage    # 覆盖率
pnpm ui:dev           # 控制台 UI 开发模式
pnpm gateway:watch    # Gateway watch/dev
```

## 仓库结构

```text
src/             CLI、gateway、channels、routing、media
extensions/      内置扩展包（渠道/模型/功能）
apps/            macOS、iOS、Android 客户端
ui/              Web 控制台
docs/            Mintlify 文档
scripts/         构建、发布、QA 与自动化脚本
```

## 插件与扩展开发

Marv 通过 `marv/plugin-sdk` 提供插件 SDK，扩展来源包括：

- 仓库内 `extensions/*`
- `~/.marv/extensions`
- `<workspace>/.marv/extensions`

## 安全

Marv 会接入真实聊天渠道，请把所有入站消息都视为不可信输入。

- 本地执行 `marv security audit`
- 非必要不要开放网关绑定地址（优先 loopback）
- 私聊建议使用 pairing/allowlist

## 发布通道

- **stable**：带标签正式版（`vYYYY.M.D`），npm `latest`
- **beta**：预发布版（`vYYYY.M.D-beta.N`），npm `beta`
- **dev**：`main` 持续更新

切换方式：

```bash
marv update --channel stable|beta|dev
```

## 参与贡献

- 贡献指南：[`CONTRIBUTING.md`](CONTRIBUTING.md)
- 安全策略：[`SECURITY.md`](SECURITY.md)

## 许可证

MIT，详见 [`LICENSE`](LICENSE)。
