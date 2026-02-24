# 🦞 Marv — Personal Multi-Channel AI Assistant

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/marv/marv/main/docs/assets/marv-logo-text-dark.png">
    <img src="https://raw.githubusercontent.com/marv/marv/main/docs/assets/marv-logo-text.png" alt="Marv" width="500">
  </picture>
</p>

<p align="center">
  <strong>EXFOLIATE! EXFOLIATE!</strong>
</p>

<p align="center">
  <a href="https://github.com/marv/marv/actions/workflows/ci.yml?branch=main"><img src="https://img.shields.io/github/actions/workflow/status/marv/marv/ci.yml?branch=main&style=for-the-badge" alt="CI status"></a>
  <a href="https://github.com/marv/marv/releases"><img src="https://img.shields.io/github/v/release/marv/marv?include_prereleases&style=for-the-badge" alt="GitHub release"></a>
  <a href="https://discord.gg/clawd"><img src="https://img.shields.io/discord/1456350064065904867?label=Discord&logo=discord&logoColor=white&color=5865F2&style=for-the-badge" alt="Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
</p>

**Marv** is a self-hosted AI gateway that lets one assistant live across the chat surfaces you already use.
Run it locally (or on your own server), connect channels, and message your agent from anywhere.

- Website: [marv.ai](https://marv.ai)
- Docs: [docs.marv.ai](https://docs.marv.ai)
- Getting started: [https://docs.marv.ai/start/getting-started](https://docs.marv.ai/start/getting-started)
- CLI reference: [https://docs.marv.ai/cli](https://docs.marv.ai/cli)
- Channel docs: [https://docs.marv.ai/channels](https://docs.marv.ai/channels)
- Security: [https://docs.marv.ai/gateway/security](https://docs.marv.ai/gateway/security)

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

Full list and setup guides:
[https://docs.marv.ai/channels](https://docs.marv.ai/channels)

## Quick Start

Runtime requirement: **Node.js 22.12.0+**.

### 1) Install

```bash
# macOS/Linux
curl -fsSL https://marv.ai/install.sh | bash

# Windows (PowerShell)
iwr -useb https://marv.ai/install.ps1 | iex
```

Or install from npm directly:

```bash
npm install -g marv@latest
# or: pnpm add -g marv@latest
```

### 2) Run onboarding

```bash
marv onboard --install-daemon
```

### 3) Verify gateway and open UI

```bash
marv gateway status
marv dashboard
```

Local Control UI default: `http://127.0.0.1:18789/`

### 4) Optional: connect a channel and send a test message

```bash
marv channels login --channel whatsapp
marv message send --target +15555550123 --message "Hello from Marv"
```

## Common CLI Commands

```bash
# Core setup
marv setup
marv onboard
marv doctor

# Gateway
marv gateway run
marv gateway status
marv gateway health

# Messaging and agent
marv message send --target <id> --message "..."
marv agent --message "Summarize today's issues" --thinking high

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

Complete command docs:
[https://docs.marv.ai/cli](https://docs.marv.ai/cli)

## Install From Source (Development)

```bash
git clone https://github.com/marv/marv.git
cd marv
pnpm install
pnpm ui:build
pnpm build

# Run CLI from source
pnpm marv --help
pnpm marv onboard --install-daemon
```

Useful dev loop:

```bash
pnpm gateway:watch
```

## Development Commands

```bash
pnpm build            # Build TypeScript + bundled assets
pnpm tsgo             # TypeScript checks
pnpm check            # format check + type-aware lint
pnpm test             # Test suite
pnpm test:coverage    # Coverage run
pnpm ui:dev           # Control UI dev server
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

Start here:

- Plugin CLI: [https://docs.marv.ai/cli/plugins](https://docs.marv.ai/cli/plugins)
- Plugin manifest/schema: [https://docs.marv.ai/plugins/manifest](https://docs.marv.ai/plugins/manifest)
- Plugin tooling overview: [https://docs.marv.ai/tools/plugin](https://docs.marv.ai/tools/plugin)

## Security

Marv connects to real messaging surfaces. Treat inbound messages as untrusted input.

- Run `marv security audit` for local checks.
- Keep gateway bind mode loopback unless you intentionally configure remote access.
- Prefer pairing/allowlists for DM safety.

Security docs:
[https://docs.marv.ai/gateway/security](https://docs.marv.ai/gateway/security)

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
- GitHub issues: [https://github.com/marv/marv/issues](https://github.com/marv/marv/issues)
- Discord: [https://discord.gg/clawd](https://discord.gg/clawd)

## License

MIT. See [`LICENSE`](LICENSE).
