---
summary: "CLI reference for `marv logs` (tail gateway logs via RPC)"
read_when:
  - You need to tail Gateway logs remotely (without SSH)
  - You want JSON log lines for tooling
title: "logs"
---

# `marv logs`

Tail Gateway file logs over RPC (works in remote mode).

Related:

- Logging overview: [Logging](/logging)

## Examples

```bash
marv logs
marv logs --follow
marv logs --json
marv logs --limit 500
marv logs --local-time
marv logs --follow --local-time
```

Use `--local-time` to render timestamps in your local timezone.
