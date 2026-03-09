---
summary: "CLI reference for `marv update` (safe-ish source update + gateway auto-restart)"
read_when:
  - You want to update a source checkout safely
  - You need to understand `--update` shorthand behavior
title: "update"
---

# `marv update`

Safely update Marv and switch between stable/beta/dev channels.

If you installed via **npm/pnpm** (global install, no git metadata), updates happen via the package manager flow in [Updating](/install/updating).

## Usage

```bash
marv update
marv update status
marv update rollback
marv update wizard
marv update --channel beta
marv update --channel dev
marv update --tag beta
marv update --no-restart
marv update --json
marv --update
```

## Options

- `--no-restart`: skip restarting the Gateway service after a successful update.
- `--channel <stable|beta|dev>`: set the update channel (git + npm; persisted in config).
- `--tag <dist-tag|version>`: override the npm dist-tag or version for this update only.
- `--json`: print machine-readable `UpdateRunResult` JSON.
- `--timeout <seconds>`: per-step timeout (default is 1200s).

Note: downgrades require confirmation because older versions can break configuration.

## `update status`

Show the active update channel + git tag/branch/SHA (for source checkouts), plus update availability.

```bash
marv update status
marv update status --json
marv update status --timeout 10
```

Options:

- `--json`: print machine-readable status JSON.
- `--timeout <seconds>`: timeout for checks (default is 3s).

## `update rollback`

Restore the last known good git deployment on the local machine. This is the
local rescue path to use when the Gateway is unhealthy or a deploy needs to be
reverted immediately.

```bash
marv update rollback
marv update rollback --no-restart
marv update rollback --json
```

Options:

- `--no-restart`: restore code without restarting the Gateway service.
- `--json`: print machine-readable rollback JSON.
- `--timeout <seconds>`: per-step timeout (default is 1200s).

Rollback only works for git deployments that have recorded deploy state and a
`lastKnownGood` revision.

## `update wizard`

Interactive flow to pick an update channel and confirm whether to restart the Gateway
after updating (default is to restart). If you select `dev` without a git checkout, it
offers to create one.

## What it does

When you switch channels explicitly (`--channel ...`), Marv also keeps the
install method aligned:

- `dev` → ensures a git checkout (default: `~/marv`, override with `MARV_GIT_DIR`),
  updates it, and installs the global CLI from that checkout.
- `stable`/`beta` → installs from npm using the matching dist-tag.

## Git checkout flow

Channels:

- `stable`: checkout the latest non-beta tag, then build + doctor.
- `beta`: checkout the latest `-beta` tag, then build + doctor.
- `dev`: checkout `main`, then fetch + rebase.

High-level:

1. Requires a clean worktree (no uncommitted changes).
2. Switches to the selected channel (tag or branch).
3. Fetches upstream (dev only).
4. Dev only: preflight lint + TypeScript build in a temp worktree; if the tip fails, walks back up to 10 commits to find the newest clean build.
5. Rebases onto the selected commit (dev only).
6. Installs deps (pnpm preferred; npm fallback).
7. Builds + builds the Control UI.
8. Runs `marv doctor` as the final “safe update” check.
9. Syncs plugins to the active channel (dev uses bundled extensions; stable/beta uses npm) and updates npm-installed plugins.

## Deploy recovery for git installs

Git deployments also keep deploy state so operators and agents can recover from
bad updates without physical access to the machine.

- Successful startup promotes the current revision to `lastKnownGood`.
- `marv update status` shows the tracked install/update state.
- `marv update rollback` restores `lastKnownGood` locally.
- Gateway operators can also use `update.status` and `update.rollback` through
  the Gateway control plane.
- `update.autoApplyCron: true` lets the managed update-check cron job auto-apply
  updates for git deployments.

For the Gateway-side flow and the tracked deploy state model, see
[Deploy recovery](/gateway/deploy-recovery).

## `--update` shorthand

`marv --update` rewrites to `marv update` (useful for shells and launcher scripts).

## See also

- `marv doctor` (offers to run update first on git checkouts)
- [Development channels](/install/development-channels)
- [Updating](/install/updating)
- [Deploy recovery](/gateway/deploy-recovery)
- [CLI reference](/cli)
