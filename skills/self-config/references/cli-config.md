# Config CLI Reference

## marv config get

```bash
marv config get <dot.path>
```

Read a value from `~/.marv/marv.json`.

```bash
marv config get channels.telegram.proxy
marv config get agents.defaults.workspace
marv config get gateway.auth.mode
```

## marv config set

```bash
marv config set <dot.path> <value>
```

Set a value. For arrays/objects pass JSON:

```bash
marv config set channels.telegram.proxy "http://127.0.0.1:7890"
marv config set gateway.port 18789
marv config set agents.defaults.heartbeat.every "2h"
marv config set channels.whatsapp.allowFrom '["*"]'
```

## marv config unset

```bash
marv config unset <dot.path>
```

Remove a value:

```bash
marv config unset channels.telegram.proxy
marv config unset tools.web.search.apiKey
```

## marv config validate

Check config syntax and schema without starting the gateway:

```bash
marv config validate
```

## marv configure

Interactive setup wizard (credentials, channels, gateway, agent defaults):

```bash
marv configure
```

## Config file

- Path: `~/.marv/marv.json` (JSON5, supports comments and trailing commas)
- Credentials: `~/.marv/credentials/`
- Sessions: `~/.marv/sessions/`
- Full reference: docs `/gateway/configuration-reference`
