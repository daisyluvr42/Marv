---
summary: "CLI reference for `marv tui` (terminal UI connected to the Gateway)"
read_when:
  - You want a terminal UI for the Gateway (remote-friendly)
  - You want to pass url/token/session from scripts
title: "tui"
---

# `marv tui`

Open the terminal UI connected to the Gateway.

Related:

- TUI guide: [TUI](/web/tui)

## Examples

```bash
marv tui
marv tui --url ws://127.0.0.1:4242 --token <token>
marv tui --session main --deliver
```
