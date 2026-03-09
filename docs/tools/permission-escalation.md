---
summary: "Task-scoped permission escalation for high-risk tool actions"
read_when:
  - You want to use `request_escalation`
  - A tool call is blocked with an escalation requirement
title: "Permission escalation"
---

# Permission escalation

`request_escalation` is the task-scoped way to unlock a narrow set of higher-risk
actions without opening the whole session permanently.

Use it when Marv tells you a tool call is blocked and asks for an escalation
request.

## What it protects

For the dedicated-machine posture, most normal local work stays unblocked:

- reading files
- writing or editing files in normal workflows
- ordinary browser use
- ordinary local `exec`

Escalation is reserved for actions with higher blast radius:

- destructive or system-level `exec`
- control-plane changes through `gateway`
- persistent automation through `cron`
- resource transfer, access gifting, or authority delegation

## Levels

- `execute`: unlock a high-risk execution step for the current task
- `admin`: unlock control-plane or persistent automation actions for the current task

## How to request it

When a tool call is blocked, call `request_escalation` with:

- `requestedLevel`
- `taskId`
- a short `reason`

Example shape:

```json
{
  "requestedLevel": "admin",
  "taskId": "task_123",
  "reason": "Need to apply gateway config and restart after validating changes"
}
```

## Approval flow

The request is routed through the same operator approval system used for exec
approvals.

- Control UI shows a permission-escalation approval prompt
- TUI shows an approval prompt
- Discord and Telegram can show native approval prompts when enabled
- text fallback surfaces can still resolve with `/approve`

Treat the returned `approvalId` or `requestId` as the canonical approval handle.
`taskId` is task scope metadata, not the primary approval handle.

## Default deny behavior for gifting and transfer

Marv is intentionally strict about giving away value or authority.

Without explicit approval, it should refuse actions like:

- sending secrets, API keys, or tokens to others
- granting roles or access
- transferring credits, subscriptions, coupons, or money
- creating durable external automation with gifted authority

If the model is blocked here, it should request escalation rather than trying to
work around the policy.

## Related docs

- [Exec approvals](/tools/exec-approvals)
- [Exec tool](/tools/exec)
- [Control UI](/web/control-ui)
- [Security](/gateway/security)
