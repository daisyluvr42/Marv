# Agent & Session CLI Reference

## marv agent

Run one agent turn via the Gateway.

```bash
marv agent -m <message> [options]
```

| Flag                   | Description                                 |
| ---------------------- | ------------------------------------------- |
| `-m, --message <text>` | Message body (required)                     |
| `-t, --to <number>`    | Recipient (E.164) for session routing       |
| `--session-id <id>`    | Explicit session id                         |
| `--agent <id>`         | Agent id (overrides routing)                |
| `--thinking <level>`   | off, minimal, low, medium, high             |
| `--verbose <on\|off>`  | Persist verbose for the session             |
| `--channel <channel>`  | Delivery channel                            |
| `--reply-to <target>`  | Delivery target override                    |
| `--reply-channel <ch>` | Delivery channel override                   |
| `--reply-account <id>` | Delivery account override                   |
| `--local`              | Run embedded agent locally (needs API keys) |
| `--deliver`            | Send reply to the channel                   |
| `--json`               | JSON output                                 |
| `--timeout <seconds>`  | Override timeout (default 600)              |

Examples:

```bash
marv agent --to +15555550123 --message "status update"
marv agent --agent ops --message "Summarize logs"
marv agent --session-id 1234 --message "Summarize inbox" --thinking medium
marv agent --agent ops --message "Report" --deliver --reply-channel slack --reply-to "#reports"
```

## marv agents

Manage agent configurations.

```bash
marv agents list                # list configured agents
marv agents add                 # add a new agent
marv agents delete              # delete an agent
marv agents set-identity        # set agent identity
```

## marv sessions

List stored conversation sessions:

```bash
marv sessions
```

## marv task

Manage task context windows and archives.

```bash
marv task list                  # list task contexts for an agent
marv task show                  # show task details, decisions, entries
marv task archive               # archive a completed task
```
