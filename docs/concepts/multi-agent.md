---
summary: "Top-level multi-agent routing was removed; use main plus enhanced subagents."
title: Multi-Agent Routing
read_when: "You are looking for the old top-level multi-agent routing model."
status: active
---

# Multi-Agent Routing

Top-level multi-agent routing has been removed.

Marv now keeps a single durable configured agent, `main`, and uses enhanced
subagents for delegation and collaboration.

## What changed

- `agents.list` is no longer supported.
- `bindings` is no longer supported.
- Inbound channel routing no longer picks between multiple durable agents.
- The durable agent is always `main`, configured under `agents.defaults`.

## What to use instead

- Configure the durable agent under `agents.defaults`.
- Use subagent tools such as `sessions_spawn`, `subagents`, and `task_dispatch`
  when `main` needs delegated work.
- If you need hard isolation between personas, workspaces, or credentials, run
  separate Marv profiles or separate Marv installs instead of one gateway with
  multiple top-level agents.

## Session model

Session keys still include the agent prefix for storage compatibility, but the
durable top-level agent id is always `main`:

- direct chat: `agent:main:<mainKey>`
- group chat: `agent:main:<channel>:group:<id>`
- channel/thread chat: `agent:main:<channel>:channel:<id>[:thread:<threadId>]`

## Related docs

- Channel routing: [Channel Routing](/channels/channel-routing)
- Subagents: [Subagents](/tools/subagents)
- Agent workspace: [Agent workspace](/concepts/agent-workspace)
