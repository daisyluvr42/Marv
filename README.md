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
docker compose run --rm openclaw-cli onboard
docker compose up -d openclaw-gateway
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
