---
summary: "Inspect the agent's own runtime, settings, models, scheduled tasks, tools, and context"
read_when:
  - You want to inspect the agent's current state
  - The user asks what the agent is using, knows, or is currently configured to do
title: "Self inspecting"
---

# Self inspecting

`self_inspecting` is the read-first tool for querying the agent's own current
state.

Use it when the user asks about:

- current runtime or session status
- current settings or overrides
- available or active models
- scheduled tasks
- available tools and limits
- context or context pollution

## Query types

The tool normalizes natural-language requests into these query modes:

- `summary`
- `runtime`
- `settings`
- `models`
- `tasks`
- `context`
- `tools`
- `all`

If the request mixes several categories, the tool generally resolves to `all`.

## What it reads

Depending on the query, `self_inspecting` can inspect:

- current session status
- session overrides from the session store
- active and default model selection
- runtime model registry and candidate models
- cron scheduler status and jobs
- available tool names
- context pollution signals for the current session

## Context cleanup

`self_inspecting` also supports a bounded cleanup action:

```json
{
  "query": "context",
  "cleanupContextPollution": true
}
```

That cleanup path is only allowed for direct user instructions. Indirect or
forwarded requests are denied.

## When to use this instead of `self_settings`

Use `self_inspecting` first when the user wants information about the agent's
current behavior or state.

Use [Self settings](/tools/self-settings) only when the user directly wants the
agent to change its own current session behavior.

## Related docs

- [Self settings](/tools/self-settings)
- [Tools](/tools)
