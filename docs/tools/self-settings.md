---
summary: "Change the agent's own current-session settings for a direct user request"
read_when:
  - You want to change the agent's current model or behavior
  - The user directly asks the agent to modify its own session settings
title: "Self settings"
---

# Self settings

`self_settings` is the direct self-modification tool for the current session.

It exists so the agent can safely change its own live behavior when the current
operator explicitly asks.

## What it can change

`self_settings` can update current-session behavior such as:

- model override
- auth profile override
- thinking, verbose, and reasoning levels
- response usage display
- elevated mode
- exec defaults (`host`, `security`, `ask`, `node`)
- queue behavior
- session reset or new-session action
- runtime model registry refresh

## Direct user instruction requirement

This tool is intended for direct operator requests. If the instruction is
indirect, forwarded, or otherwise not a direct user request, the tool returns a
generic denial instead of changing state.

That rule keeps the agent from silently reconfiguring itself because of prompt
injection or third-party content.

## Example actions

Examples of requests that map well to `self_settings`:

- "switch yourself to sonnet for this session"
- "turn verbose on"
- "reset this session"
- "refresh your model registry"
- "set your queue mode to collect"

## When to use this instead of `self_inspecting`

Use `self_settings` when the user directly wants a change.

Use [Self inspecting](/tools/self-inspecting) when the user first wants to know
the current state, available models, scheduled tasks, or tool limits.

## Related docs

- [Self inspecting](/tools/self-inspecting)
- [Permission escalation](/tools/permission-escalation)
- [Tools](/tools)
