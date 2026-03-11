---
summary: "Routing rules per channel and how session keys are chosen in main-only mode"
read_when:
  - Changing channel routing or inbox behavior
title: "Channel Routing"
---

# Channels & routing

Marv routes replies back to the same channel/account/conversation a message came
from. The host decides routing deterministically.

After the single-agent cutover, the durable top-level agent is always `main`.
Channel routing now decides the session key and outbound target, not which
durable agent to run.

## Key terms

- **Channel**: `whatsapp`, `telegram`, `discord`, `slack`, `signal`, `imessage`, `webchat`
- **AccountId**: a per-channel account instance when the channel supports it
- **AgentId**: the durable top-level agent id, always `main`
- **SessionKey**: the bucket key used to store context and control concurrency

## Session key shapes

Direct messages collapse to the main session:

- `agent:main:<mainKey>` (default: `agent:main:main`)

Groups and channels stay isolated per conversation:

- groups: `agent:main:<channel>:group:<id>`
- channels/rooms: `agent:main:<channel>:channel:<id>`
- threads append `:thread:<threadId>` when supported

Examples:

- `agent:main:telegram:group:-1001234567890:topic:42`
- `agent:main:discord:channel:123456:thread:987654`

## Routing rules

Inbound routing now resolves to `main` in all cases.

The following inputs still matter because they shape session selection and
reply targeting:

- channel id
- account id
- peer kind/id
- thread/topic context
- provider-specific delivery metadata

There is no supported top-level `bindings` or `agents.list` routing layer.

## Delegation

If one conversation needs collaborative work, let `main` delegate internally
with enhanced subagents instead of routing the inbound message to another
durable top-level agent.

## Related docs

- Multi-agent routing removal: [Multi-Agent Routing](/concepts/multi-agent)
- Broadcast groups: [Broadcast Groups](/channels/broadcast-groups)
- Subagents: [Subagents](/tools/subagents)
