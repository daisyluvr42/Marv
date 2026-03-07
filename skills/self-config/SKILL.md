---
name: self-config
description: "Marv self-configuration operations manual. Use when the user asks about configuring Marv itself: proxy settings, channel setup (Telegram/Discord/Slack/Signal/WhatsApp), model provider configuration, gateway settings, TTS, networking, deployment, messaging, scheduling, plugins, skills, security, nodes, memory, or troubleshooting Marv config/CLI issues. Also use when the agent needs to resolve its own configuration problems (e.g. connectivity failures, proxy routing, channel not responding) or needs to run any marv CLI command."
---

# Marv Self-Configuration

Operational reference for Marv CLI commands, gateway, channels, models, proxies, and infrastructure.

## Quick reference

All configuration lives in `~/.marv/marv.json` (JSON5). Edit via:

```bash
marv config set <key> <value>      # set a value
marv config get <key>              # read a value
marv config unset <key>            # remove a value
marv configure                     # interactive wizard
```

Diagnostics: when something is broken, run `marv doctor` first, then `marv channels status --probe`.

Before any config write through the gateway tool or RPC (`config.get`, `config.patch`, `config.apply`, `config.patches.*`), read [references/config-writes.md](references/config-writes.md). This is required when the agent is modifying Marv's own config.

## Reference guides

Read the relevant file based on the topic:

### Configuration

- **Config CLI** (get/set/unset/validate): [references/cli-config.md](references/cli-config.md)
- **Gateway config writes** (`config.get`, `config.patch`, `config.apply`, `baseHash`, redaction): [references/config-writes.md](references/config-writes.md)
- **Proxy / network routing**: [references/proxy.md](references/proxy.md)
- **Channel config** (allowlists, DM policy, multi-account): [references/channels.md](references/channels.md)
- **Gateway config** (bind, auth, Tailscale): [references/gateway.md](references/gateway.md)

### CLI commands by area

- **Gateway & system** (run, start/stop, tui, acp, dns, browser): [references/cli-gateway.md](references/cli-gateway.md)
- **Channels** (list, status, login, pairing, directory): [references/cli-channels.md](references/cli-channels.md)
- **Agent & sessions** (agent, agents, sessions, task): [references/cli-agent.md](references/cli-agent.md)
- **Messaging** (send, broadcast, read, edit, delete, poll, react, pin, search): [references/cli-message.md](references/cli-message.md)
- **Models & auth** (list, set, aliases, fallbacks, auth): [references/cli-models.md](references/cli-models.md)
- **Nodes & devices** (nodes, node host, devices, qr): [references/cli-nodes.md](references/cli-nodes.md)
- **Plugins, skills & hooks**: [references/cli-plugins.md](references/cli-plugins.md)
- **Memory** (search, reindex, list, status): [references/cli-memory.md](references/cli-memory.md)
- **Ops & diagnostics** (status, health, doctor, logs, security, sandbox, approvals, cron, dashboard): [references/cli-ops.md](references/cli-ops.md)
- **Setup & maintenance** (setup, onboard, update, reset, uninstall, completion): [references/cli-setup.md](references/cli-setup.md)

## General rules

- Always verify config changes with `marv config get <key>` after setting
- Use `marv channels status --probe` to verify channel connectivity
- Use `marv health` and `marv doctor` to check overall system health
- Prefer `marv config set` over manual JSON editing to avoid syntax errors
- Do not guess config keys; check the configuration reference docs
- For gateway config writes, do not rely on repo docs being present in the runtime workspace; use the bundled `references/config-writes.md` manual
