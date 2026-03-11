---
summary: "Top-level broadcast groups were removed with the single-agent cutover."
read_when:
  - You are looking for the old multi-agent broadcast feature
title: "Broadcast Groups"
---

# Broadcast Groups

Top-level broadcast groups were removed with the single-agent cutover.

Inbound messages now resolve to the durable `main` agent only. If one
conversation needs parallel or role-based collaboration, let `main` delegate
with enhanced subagents instead of configuring multiple top-level durable
agents for the same peer.

## What to use instead

- configure the durable agent under `agents.defaults`
- keep normal channel/group gating under `channels.*` and `messages.*`
- use subagent tools such as `task_dispatch` when one message needs parallel
  analysis or execution

## Related docs

- Channel routing: [Channel Routing](/channels/channel-routing)
- Subagents: [Subagents](/tools/subagents)
