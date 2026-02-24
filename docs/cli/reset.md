---
summary: "CLI reference for `marv reset` (reset local state/config)"
read_when:
  - You want to wipe local state while keeping the CLI installed
  - You want a dry-run of what would be removed
title: "reset"
---

# `marv reset`

Reset local config/state (keeps the CLI installed).

```bash
marv reset
marv reset --dry-run
marv reset --scope config+creds+sessions --yes --non-interactive
```
