---
title: Operations Manual
description: From zero to running Marv - install, configure, daily use, maintain, troubleshoot.
---

This guide walks you through every stage of running Marv: from first install to daily operations and long-term maintenance. Follow it top to bottom for a fresh setup, or jump to the section you need.

## 1. Install

### macOS (recommended path)

The installer detects macOS, installs the CLI, then automatically downloads and launches the **Marv Mac app** for GUI-based setup.

```bash
curl -fsSL https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash
```

What happens:

1. Ensures Node.js 22+ and Git are available (installs via Homebrew if missing)
2. Runs `npm install -g agentmarv@latest`
3. Downloads `Marv.app` from the matching GitHub Release
4. Installs to `/Applications/Marv.app` and opens it
5. The app walks you through onboarding with a GUI wizard

If you prefer CLI-only setup, skip the Mac app:

```bash
curl -fsSL https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash -s -- --no-mac-app
```

### Linux / WSL

```bash
curl -fsSL https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash
```

This installs the CLI and runs `marv onboard --install-daemon` in the terminal.

### Manual install (npm)

```bash
npm install -g agentmarv@latest
marv onboard --install-daemon
```

### From source (development)

```bash
git clone https://github.com/daisyluvr42/Marv.git
cd Marv
pnpm install
pnpm ui:build && pnpm build
pnpm marv onboard --install-daemon
```

### Docker

```bash
docker compose up -d
docker compose logs -f
```

### Other methods

See the [Install overview](/install) for Docker, Podman, Nix, Ansible, Fly.io, Hetzner, GCP, and other hosting options.

---

## 2. First-time setup

### Onboarding wizard

The onboarding wizard (GUI or CLI) configures these essentials:

| Step           | What it does                        | CLI equivalent                                                             |
| -------------- | ----------------------------------- | -------------------------------------------------------------------------- |
| Gateway mode   | Local or remote                     | `marv config set gateway.mode local`                                       |
| Model provider | API key for Anthropic, OpenAI, etc. | `marv config set providers.anthropic.apiKey sk-...`                        |
| Default model  | Which model to use                  | `marv config set agents.defaults.model anthropic/claude-sonnet-4-20250514` |
| First channel  | Connect WhatsApp, Telegram, etc.    | `marv channels login --channel telegram`                                   |
| Daemon install | Register as system service          | `marv onboard --install-daemon`                                            |

### Verify everything is running

```bash
marv status --all        # overview of gateway, channels, models
marv health              # deep health probe
marv channels status --probe   # probe each channel
```

Expected healthy output: gateway shows `running`, channels show `connected`, models show `ready`.

### iOS companion (optional)

If you have a spare iPhone connected via USB and want to use it as a monitoring dashboard:

```bash
bash scripts/ios-deploy.sh
```

This builds and installs the iOS companion app to the first connected iPhone. The agent can also do this itself when you tell it to.

---

## 3. Configuration

### Where config lives

```
~/.marv/marv.json        # main config (JSON5)
~/.marv/credentials/     # channel credentials
~/.marv/sessions/        # session data
```

### Editing config

```bash
# Interactive wizard (recommended for first time)
marv setup

# One-liner changes
marv config set agents.defaults.model anthropic/claude-sonnet-4-20250514
marv config set channels.telegram.proxy "http://127.0.0.1:7890"

# View current config
marv config get

# Full reset
marv config reset
```

### Hot reload

Config changes are picked up automatically (default mode: `hybrid`). No restart needed for most changes. See [Configuration](/gateway/configuration) for reload modes.

### Common config tasks

**Add a channel:**

```bash
marv channels login --channel whatsapp
marv channels login --channel telegram
marv channels login --channel discord
```

**Set up proxy per channel:**

```bash
marv config set channels.telegram.proxy "http://127.0.0.1:7890"
marv config set channels.discord.proxy "socks5://127.0.0.1:1080"
```

**Configure model providers:**

```bash
marv config set providers.anthropic.apiKey "sk-ant-..."
marv config set providers.openai.apiKey "sk-..."
marv config set providers.openrouter.apiKey "sk-or-..."
```

**Set identity:**

```bash
marv config set agents.defaults.identity.name "Marv"
marv config set agents.defaults.identity.personality "helpful and concise"
```

For complete reference, see [Configuration reference](/gateway/configuration-reference) and [Configuration examples](/gateway/configuration-examples).

---

## 4. Daily use

### Starting and stopping

**Mac app users:** The app manages the gateway as a launchd service. Use the menu bar icon to start/stop.

**CLI users:**

```bash
# Check if running
marv gateway status

# Start as daemon (background service)
marv onboard --install-daemon

# Start in foreground (debug)
marv gateway run --bind loopback --port 4242

# Restart
marv gateway run --force
```

### Interacting

**TUI (terminal UI):**

```bash
marv tui
```

**Web dashboard:**

```bash
marv dashboard    # opens Control UI in browser
```

**Send a message from CLI:**

```bash
marv message send --target +15555550123 --message "Hello from Marv"
```

**Run an agent task:**

```bash
marv agent --message "summarize today's changes"
marv agent --to +15555550123 --message "check server status"
```

### Channel management

```bash
marv channels list                    # list all channels
marv channels status --probe          # probe connectivity
marv channels login --channel telegram   # (re)authenticate
marv channels logs --channel whatsapp    # channel-specific logs
```

### Models

```bash
marv models status    # current model + provider health
marv models list      # all available models
```

### Memory

```bash
marv memory status                          # memory store health
marv memory search --query "deployment"     # search memories
```

### Sessions

```bash
marv sessions list       # active sessions
marv sessions prune      # clean up old sessions
```

### Automation (cron jobs)

```bash
marv cron list           # list scheduled tasks
marv cron status         # cron service health
marv cron add --schedule "0 9 * * *" --task "morning briefing"
```

See [Cron jobs](/automation/cron-jobs) for details.

---

## 5. Monitoring and logs

### Quick health check

```bash
marv status --all      # one-screen summary
marv health --json     # machine-readable deep probe
```

### Logs

```bash
# Follow live logs (recommended)
marv logs --follow

# JSON output for piping
marv logs --follow --json

# Filter by level
marv logs --follow --level error

# Channel-specific
marv channels logs --channel whatsapp
```

**Log file location:** `/tmp/marv/marv-YYYY-MM-DD.log` (JSON lines, date in local timezone).

**macOS unified log:**

```bash
./scripts/clawlog.sh --follow
```

### Web monitoring

Open the Control UI dashboard for real-time monitoring:

```bash
marv dashboard
```

The dashboard shows gateway status, active sessions, channel health, and live logs.

---

## 6. Updating

### Global install

```bash
npm install -g agentmarv@latest
marv --version
```

On macOS, the Mac app updates itself via Sparkle (automatic update prompts).

### From source

```bash
git pull --rebase origin main
pnpm install
pnpm ui:build && pnpm build
```

### Post-update verification

```bash
marv doctor            # run automated checks and repairs
marv status --all      # verify everything is healthy
marv channels status --probe
```

See [Updating](/install/updating) for detailed guidance.

---

## 7. Troubleshooting

### 60-second triage

Run these commands in order. Stop when you find the problem:

```bash
marv --version              # CLI is installed?
marv gateway status         # gateway running?
marv status --all           # channels connected? models ready?
marv health                 # deep probe
marv logs --follow          # look for errors
marv doctor                 # automated repair
```

### Common issues

| Symptom                       | Fix                                                        |
| ----------------------------- | ---------------------------------------------------------- |
| Gateway wont start            | `marv gateway run --force` (kills stale process)           |
| Channel disconnected          | `marv channels login --channel <name>`                     |
| No replies to messages        | Check `marv status --all` for model/channel errors         |
| Port conflict (4242)          | `marv gateway run --port 4243` or kill conflicting process |
| Config validation error       | `marv doctor --repair`                                     |
| Missing dependencies (source) | `pnpm install && pnpm ui:build && pnpm build`              |

### Doctor (automated repair)

```bash
marv doctor                # interactive mode
marv doctor --yes          # auto-fix all
marv doctor --repair       # non-interactive repair
marv doctor --deep         # deep diagnostics
```

Doctor runs 19+ checks including config validation, credential health, service migrations, port conflicts, and more. See [Doctor](/gateway/doctor).

### Deep troubleshooting

See [Troubleshooting guide](/help/troubleshooting) for symptom-based triage trees and [Gateway troubleshooting](/gateway/troubleshooting) for detailed runbooks.

---

## 8. Maintenance

### Backup

Back up these directories regularly:

```
~/.marv/marv.json          # configuration
~/.marv/credentials/       # channel auth tokens
~/.marv/sessions/          # conversation history
~/.marv/memory/            # memory index
```

### Session cleanup

```bash
marv sessions prune        # remove expired sessions
```

### Auth token refresh

Some channels (WhatsApp, OAuth-based providers) need periodic re-authentication:

```bash
marv channels status --probe     # check which channels need attention
marv channels login --channel whatsapp   # re-authenticate
```

Set up automated auth monitoring: see [Auth monitoring](/automation/auth-monitoring).

### Resource monitoring

- **Disk:** `~/.marv/` grows with sessions and memory. Monitor and prune periodically.
- **Memory:** Gateway typically uses 200-500 MB RAM.
- **CPU:** Spikes during agent runs; idle is near zero.

### Service management

**macOS (launchd):**

```bash
# Check service status
launchctl print gui/$UID | grep marv

# The Mac app manages the service; use its menu bar controls
# Or restart via script:
./scripts/restart-mac.sh
```

**Linux (systemd):**

```bash
systemctl --user status marv-gateway
systemctl --user restart marv-gateway
journalctl --user -u marv-gateway -f
```

---

## 9. Platform-specific notes

### macOS

- Mac app provides menu bar controls, Canvas, Camera, Screen Recording, voice input
- TCC permissions: grant Notifications, Accessibility, Screen Recording, Microphone when prompted
- See [macOS platform guide](/platforms/macos)

### Linux

- Headless server: use `--bind loopback` or `--bind all` as needed
- systemd linger required for user services: `loginctl enable-linger $USER`
- See [Linux platform guide](/platforms/linux)

### iOS companion

- USB-connected iPhone becomes a monitoring dashboard
- Deploy: `bash scripts/ios-deploy.sh` or ask the agent
- Shows gateway status, sessions, cron jobs, usage
- See [iOS platform guide](/platforms/ios)

### Android companion

- See [Android platform guide](/platforms/android)

---

## 10. Quick reference

### Most-used commands

| Command                                | Purpose                          |
| -------------------------------------- | -------------------------------- |
| `marv status --all`                    | Full system overview             |
| `marv health`                          | Deep health probe                |
| `marv logs --follow`                   | Live log tail                    |
| `marv doctor`                          | Automated diagnostics and repair |
| `marv tui`                             | Terminal UI                      |
| `marv dashboard`                       | Web dashboard                    |
| `marv channels status --probe`         | Channel connectivity check       |
| `marv channels login --channel <name>` | (Re)authenticate a channel       |
| `marv models status`                   | Model provider health            |
| `marv config set <key> <value>`        | Change config                    |
| `marv gateway run --force`             | Restart gateway                  |
| `marv agent --message "..."`           | Run agent task                   |

### Key file paths

| Path                   | Contents            |
| ---------------------- | ------------------- |
| `~/.marv/marv.json`    | Main configuration  |
| `~/.marv/credentials/` | Channel auth tokens |
| `~/.marv/sessions/`    | Session data        |
| `~/.marv/memory/`      | Memory index        |
| `~/.marv/agents/`      | Agent session logs  |
| `/tmp/marv/`           | Runtime logs        |

### Useful links

- [Getting started](/start/getting-started)
- [Configuration reference](/gateway/configuration-reference)
- [Configuration examples](/gateway/configuration-examples)
- [CLI command index](/cli)
- [Gateway runbook](/gateway)
- [Troubleshooting](/help/troubleshooting)
- [Model providers](/providers)
- [Channel setup](/channels)
