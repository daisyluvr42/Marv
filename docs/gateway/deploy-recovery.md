---
summary: "Track deploy state, roll back bad updates, and recover a git deployment remotely"
read_when:
  - You run Marv from a git deployment checkout
  - You need remote update, status, or rollback guidance
title: "Deploy recovery"
---

# Deploy recovery

Marv can treat a git checkout as a deployment target that tracks your own
repository and keeps enough state to recover from bad updates remotely.

This is designed for a long-running home deployment where you may need to patch,
update, or roll back the machine without physical access.

## Deployment model

Recommended posture:

- a pure deployment checkout on the host
- tracked branch: `origin/main`
- normal updates through `update.run` or `marv update`
- explicit rollback to the last known good revision when needed

The deployment checkout should stay a deploy copy, not a place for local manual
editing.

## What is tracked

Marv records deploy state under its state directory for each deployment root.

That state includes:

- current revision metadata
- `lastKnownGood`
- the last deploy attempt
- the last rollback event

Successful gateway startup on a git deployment promotes the current revision to
`lastKnownGood`.

## Manual operator actions

### CLI

Use the local CLI on the deployment host:

```bash
marv update
marv update status
marv update rollback
```

Use `marv update rollback` when the main Gateway is unhealthy and you need a
local rescue path that does not depend on the Gateway control plane being up.

### Gateway control plane

The Gateway also exposes deploy-aware methods:

- `update.run`
- `update.status`
- `update.rollback`

These are available through operator clients and the `gateway` tool.

`update.status` is the safest first step when checking a remote deployment. It
returns the tracked root, deploy state path, update availability, and whether
managed cron auto-apply is enabled.

## Auto-apply with cron

For git installs, the managed update-check cron job can auto-apply updates when
this config is enabled:

```json5
{
  update: {
    autoApplyCron: true,
  },
}
```

With that enabled, cron can:

1. check the tracked upstream branch
2. run the normal git update flow
3. schedule a gateway restart

If no update is available, it only reports status.

## Failure and rollback behavior

If a deploy step fails after the checkout has already moved, Marv attempts to
roll back to the recorded last known good revision inside the same deploy flow.

If the deployment still needs operator intervention, use:

- `update.rollback` through the Gateway if the control plane is still reachable
- `marv update rollback` on the host as the local rescue path

## Recommended remote workflow

1. patch and commit to your own repository
2. let the deployment host pull from `origin/main`, or trigger `update.run`
3. check `update.status`
4. if the deployment is unhealthy, run `update.rollback`
5. if the Gateway itself is down, use `marv update rollback` on the host

## Related docs

- [Update CLI](/cli/update)
- [Control UI](/web/control-ui)
- [Security](/gateway/security)
