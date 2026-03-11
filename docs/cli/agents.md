---
summary: "CLI reference for `marv agents` in the main-only architecture"
read_when:
  - You want to inspect the durable agent or update its identity
title: "agents"
---

# `marv agents`

The durable top-level agent is now always `main`.

`marv agents` is kept for inspecting that durable agent and for updating its
identity. Legacy top-level create/delete flows were removed.

Related:

- Agent workspace: [Agent workspace](/concepts/agent-workspace)
- Subagents: [Subagents](/tools/subagents)

## Examples

```bash
marv agents list
marv agents set-identity --workspace ~/.marv/workspace --from-identity
marv agents set-identity --agent main --avatar avatars/marv.png
```

## Removed commands

- `marv agents add`
- `marv agents delete`

Use `agents.defaults` for durable agent config and enhanced subagents for
delegated work.

## Identity files

The main workspace can include an `IDENTITY.md` at the workspace root:

- example path: `~/.marv/workspace/IDENTITY.md`
- `set-identity --from-identity` reads from the workspace root unless you pass
  `--identity-file`

Avatar paths resolve relative to the workspace root.

## Set identity

`set-identity` writes fields into `agents.defaults.identity`:

- `name`
- `theme`
- `emoji`
- `avatar` (workspace-relative path, http(s) URL, or data URI)

Load from `IDENTITY.md`:

```bash
marv agents set-identity --workspace ~/.marv/workspace --from-identity
```

Override fields explicitly:

```bash
marv agents set-identity --agent main --name "Marv" --emoji "🤖" --avatar avatars/marv.png
```

Config sample:

```json5
{
  agents: {
    defaults: {
      identity: {
        name: "Marv",
        theme: "space lobster",
        emoji: "🤖",
        avatar: "avatars/marv.png",
      },
    },
  },
}
```
