---
summary: "Top-level per-agent sandbox and tool overrides were removed."
title: Multi-Agent Sandbox & Tools
read_when: "You are looking for the old per-agent sandbox/tool policy model."
status: active
---

# Multi-Agent Sandbox & Tools

Top-level per-agent sandbox and tool overrides were removed with the
single-agent cutover.

## Current model

- Durable agent config lives under `agents.defaults`.
- Sandbox config lives under `agents.defaults.sandbox`.
- Tool policy lives under `agents.defaults.tools` plus global `tools.*`.
- Delegated work should use enhanced subagents instead of separate top-level
  durable agents.

## Practical replacements

- Use `agents.defaults.sandbox.*` to control where tools run.
- Use `agents.defaults.tools.*` and `tools.*` to control availability and
  restrictions.
- Use role-aware subagents if you want a reviewer/tester/architect with a
  narrower runtime policy.
- If you need completely separate credentials or workspaces, run separate Marv
  profiles or separate installs.

## Related docs

- Sandboxing: [Sandboxing](/gateway/sandboxing)
- Sandbox vs Tool Policy vs Elevated: [Sandbox vs Tool Policy vs Elevated](/gateway/sandbox-vs-tool-policy-vs-elevated)
- Subagents: [Subagents](/tools/subagents)
