---
summary: "CLI reference for `marv config` (get/set/unset config values)"
read_when:
  - You want to read or edit config non-interactively
title: "config"
---

# `marv config`

Config helpers: get/set/unset values by path. Run without a subcommand to open
the configure wizard (same as `marv configure`).

## Examples

```bash
marv config get browser.executablePath
marv config set browser.executablePath "/usr/bin/google-chrome"
marv config set agents.defaults.heartbeat.every "2h"
marv config set agents.defaults.tools.exec.node "node-id-or-name"
marv config unset tools.web.search.apiKey
```

## Paths

Paths use dot or bracket notation:

```bash
marv config get agents.defaults.workspace
marv config get agents.defaults.identity.name
```

Main-only durable agent config lives under `agents.defaults`:

```bash
marv config get agents.defaults
marv config set agents.defaults.tools.exec.node "node-id-or-name"
```

`agents.list` and `bindings` were removed and are rejected.

## Values

Values are parsed as JSON5 when possible; otherwise they are treated as strings.
Use `--json` to require JSON5 parsing.

```bash
marv config set agents.defaults.heartbeat.every "0m"
marv config set gateway.port 19001 --json
marv config set channels.whatsapp.groups '["*"]' --json
```

Restart the gateway after edits.
