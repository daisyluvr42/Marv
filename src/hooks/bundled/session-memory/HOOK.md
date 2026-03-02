---
name: session-memory
description: "Save session context to memory when /new command is issued"
homepage: /automation/hooks#session-memory
metadata:
  {
    "marv":
      {
        "emoji": "💾",
        "events": ["command:new"],
        "requires": { "config": ["workspace.dir"] },
        "install": [{ "id": "bundled", "kind": "bundled", "label": "Bundled with Marv" }],
      },
  }
---

# Session Memory Hook

Automatically saves session context to your workspace memory when you issue the `/new` command.

## What It Does

When you run `/new` to start a fresh session:

1. **Finds the previous session** - Uses the pre-reset session entry to locate the correct transcript
2. **Extracts conversation** - Reads the last N user/assistant messages from the session (default: 15, configurable)
3. **Generates descriptive slug** - Uses LLM to summarize the session topic
4. **Saves to memory** - Writes a structured `session_summary` item into soul memory
5. **Sends confirmation** - Logs a `soul-memory/<id>` reference

## Output Format

Structured entries are written with session metadata and summary text:

```text
session_date=2026-01-16
session_time_utc=14:30:00
session_topic_slug=api-design
session_key=agent:main:main
session_id=abc123def456
source=telegram

conversation_summary:
user: Discuss API architecture options.
assistant: We should use a typed contract-first approach.
```

## Slug Examples

The LLM still generates descriptive topic slugs:

- `vendor-pitch` - Discussion about vendor evaluation
- `api-design` - API architecture planning
- `bug-fix` - Debugging session
- `1430` - Fallback timestamp if slug generation fails

## Requirements

- **Config**: `workspace.dir` must be set (automatically configured during onboarding)

The hook uses your configured LLM provider to generate slugs, so it works with any provider (Anthropic, OpenAI, etc.).

## Configuration

The hook supports optional configuration:

| Option     | Type   | Default | Description                                                        |
| ---------- | ------ | ------- | ------------------------------------------------------------------ |
| `messages` | number | 15      | Number of user/assistant messages to include in the memory summary |

Example configuration:

```json
{
  "hooks": {
    "internal": {
      "entries": {
        "session-memory": {
          "enabled": true,
          "messages": 25
        }
      }
    }
  }
}
```

The hook automatically:

- Uses your workspace directory (`~/.marv/workspace` by default)
- Uses your configured LLM for slug generation
- Falls back to timestamp slugs if LLM is unavailable

## Disabling

To disable this hook:

```bash
marv hooks disable session-memory
```

Or remove it from your config:

```json
{
  "hooks": {
    "internal": {
      "entries": {
        "session-memory": { "enabled": false }
      }
    }
  }
}
```
