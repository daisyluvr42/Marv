---
summary: "Logging overview: file logs, console output, CLI tailing, and the Control UI"
read_when:
  - You need a beginner-friendly overview of logging
  - You want to configure log levels or formats
  - You are troubleshooting and need to find logs quickly
title: "Logging"
---

# Logging

Marv logs in two places:

- **File logs** (JSON lines) written by the Gateway.
- **Console output** shown in terminals and the Control UI.

This page explains where logs live, how to read them, and how to configure log
levels and formats.

## Where logs live

By default, the Gateway writes a rolling log file under:

`/tmp/marv/marv-YYYY-MM-DD.log`

The date uses the gateway host's local timezone.

You can override this in `~/.marv/marv.json`:

```json
{
  "logging": {
    "file": "/path/to/marv.log"
  }
}
```

## How to read logs

### CLI: live tail (recommended)

Use the CLI to tail the gateway log file via RPC:

```bash
marv logs --follow
```

Output modes:

- **TTY sessions**: pretty, colorized, structured log lines.
- **Non-TTY sessions**: plain text.
- `--json`: line-delimited JSON (one log event per line).
- `--plain`: force plain text in TTY sessions.
- `--no-color`: disable ANSI colors.

In JSON mode, the CLI emits `type`-tagged objects:

- `meta`: stream metadata (file, cursor, size)
- `log`: parsed log entry
- `notice`: truncation / rotation hints
- `raw`: unparsed log line

If the Gateway is unreachable, the CLI prints a short hint to run:

```bash
marv doctor
```

### Control UI (web)

The Control UI’s **Logs** tab tails the same file using `logs.tail`.
See [/web/control-ui](/web/control-ui) for how to open it.

### Channel-only logs

To filter channel activity (WhatsApp/Telegram/etc), use:

```bash
marv channels logs --channel whatsapp
```

## Log formats

### File logs (JSONL)

Each line in the log file is a JSON object. The CLI and Control UI parse these
entries to render structured output (time, level, subsystem, message).

### Console output

Console logs are **TTY-aware** and formatted for readability:

- Subsystem prefixes (e.g. `gateway/channels/whatsapp`)
- Level coloring (info/warn/error)
- Optional compact or JSON mode

Console formatting is controlled by `logging.consoleStyle`.

## Configuring logging

All logging configuration lives under `logging` in `~/.marv/marv.json`.

```json
{
  "logging": {
    "level": "info",
    "file": "/tmp/marv/marv-YYYY-MM-DD.log",
    "consoleLevel": "info",
    "consoleStyle": "pretty",
    "redactSensitive": "tools",
    "redactPatterns": ["sk-.*"]
  }
}
```

### Log levels

- `logging.level`: **file logs** (JSONL) level.
- `logging.consoleLevel`: **console** verbosity level.

`--verbose` only affects console output; it does not change file log levels.

### Console styles

`logging.consoleStyle`:

- `pretty`: human-friendly, colored, with timestamps.
- `compact`: tighter output (best for long sessions).
- `json`: JSON per line (for log processors).

### Redaction

Tool summaries can redact sensitive tokens before they hit the console:

- `logging.redactSensitive`: `off` | `tools` (default: `tools`)
- `logging.redactPatterns`: list of regex strings to override the default set

Redaction affects **console output only** and does not alter file logs.

## Diagnostics + OpenTelemetry

Diagnostics are structured, machine-readable events for model runs **and**
message-flow telemetry (webhooks, queueing, session state). They do **not**
replace logs; they exist to feed metrics, traces, and other exporters.

Diagnostics events are emitted in-process, but exporters only attach when
diagnostics + the exporter plugin are enabled.

### OpenTelemetry vs OTLP

- **OpenTelemetry (OTel)**: the data model + SDKs for traces, metrics, and logs.
- **OTLP**: the wire protocol used to export OTel data to a collector/backend.
- Marv exports via **OTLP/HTTP (protobuf)** today.

### Signals exported

- **Metrics**: counters + histograms (token usage, message flow, queueing).
- **Traces**: spans for model usage + webhook/message processing.
- **Logs**: exported over OTLP when `diagnostics.otel.logs` is enabled. Log
  volume can be high; keep `logging.level` and exporter filters in mind.

### Diagnostic event catalog

Model usage:

- `model.usage`: tokens, cost, duration, context, provider/model/channel, session ids.

Message flow:

- `webhook.received`: webhook ingress per channel.
- `webhook.processed`: webhook handled + duration.
- `webhook.error`: webhook handler errors.
- `message.queued`: message enqueued for processing.
- `message.processed`: outcome + duration + optional error.

Queue + session:

- `queue.lane.enqueue`: command queue lane enqueue + depth.
- `queue.lane.dequeue`: command queue lane dequeue + wait time.
- `session.state`: session state transition + reason.
- `session.stuck`: session stuck warning + age.
- `run.attempt`: run retry/attempt metadata.
- `diagnostic.heartbeat`: aggregate counters (webhooks/queue/session).

### Enable diagnostics (no exporter)

Use this if you want diagnostics events available to plugins or custom sinks:

```json
{
  "diagnostics": {
    "enabled": true
  }
}
```

### Diagnostics flags (targeted logs)

Use flags to turn on extra, targeted debug logs without raising `logging.level`.
Flags are case-insensitive and support wildcards (e.g. `telegram.*` or `*`).

```json
{
  "diagnostics": {
    "flags": ["telegram.http"]
  }
}
```

Env override (one-off):

```
MARV_DIAGNOSTICS=telegram.http,telegram.payload
```

Notes:

- Flag logs go to the standard log file (same as `logging.file`).
- Output is still redacted according to `logging.redactSensitive`.
- Full guide: [/diagnostics/flags](/diagnostics/flags).

### Export to OpenTelemetry

Diagnostics can be exported via the `diagnostics-otel` plugin (OTLP/HTTP). This
works with any OpenTelemetry collector/backend that accepts OTLP/HTTP.

```json
{
  "plugins": {
    "allow": ["diagnostics-otel"],
    "entries": {
      "diagnostics-otel": {
        "enabled": true
      }
    }
  },
  "diagnostics": {
    "enabled": true,
    "otel": {
      "enabled": true,
      "endpoint": "http://otel-collector:4318",
      "protocol": "http/protobuf",
      "serviceName": "marv-gateway",
      "traces": true,
      "metrics": true,
      "logs": true,
      "sampleRate": 0.2,
      "flushIntervalMs": 60000
    }
  }
}
```

Notes:

- You can also enable the plugin with `marv plugins enable diagnostics-otel`.
- `protocol` currently supports `http/protobuf` only. `grpc` is ignored.
- Metrics include token usage, cost, context size, run duration, and message-flow
  counters/histograms (webhooks, queueing, session state, queue depth/wait).
- Traces/metrics can be toggled with `traces` / `metrics` (default: on). Traces
  include model usage spans plus webhook/message processing spans when enabled.
- Set `headers` when your collector requires auth.
- Environment variables supported: `OTEL_EXPORTER_OTLP_ENDPOINT`,
  `OTEL_SERVICE_NAME`, `OTEL_EXPORTER_OTLP_PROTOCOL`.

### Exported metrics (names + types)

Model usage:

- `marv.tokens` (counter, attrs: `marv.token`, `marv.channel`,
  `marv.provider`, `marv.model`)
- `marv.cost.usd` (counter, attrs: `marv.channel`, `marv.provider`,
  `marv.model`)
- `marv.run.duration_ms` (histogram, attrs: `marv.channel`,
  `marv.provider`, `marv.model`)
- `marv.context.tokens` (histogram, attrs: `marv.context`,
  `marv.channel`, `marv.provider`, `marv.model`)

Message flow:

- `marv.webhook.received` (counter, attrs: `marv.channel`,
  `marv.webhook`)
- `marv.webhook.error` (counter, attrs: `marv.channel`,
  `marv.webhook`)
- `marv.webhook.duration_ms` (histogram, attrs: `marv.channel`,
  `marv.webhook`)
- `marv.message.queued` (counter, attrs: `marv.channel`,
  `marv.source`)
- `marv.message.processed` (counter, attrs: `marv.channel`,
  `marv.outcome`)
- `marv.message.duration_ms` (histogram, attrs: `marv.channel`,
  `marv.outcome`)

Queues + sessions:

- `marv.queue.lane.enqueue` (counter, attrs: `marv.lane`)
- `marv.queue.lane.dequeue` (counter, attrs: `marv.lane`)
- `marv.queue.depth` (histogram, attrs: `marv.lane` or
  `marv.channel=heartbeat`)
- `marv.queue.wait_ms` (histogram, attrs: `marv.lane`)
- `marv.session.state` (counter, attrs: `marv.state`, `marv.reason`)
- `marv.session.stuck` (counter, attrs: `marv.state`)
- `marv.session.stuck_age_ms` (histogram, attrs: `marv.state`)
- `marv.run.attempt` (counter, attrs: `marv.attempt`)

### Exported spans (names + key attributes)

- `marv.model.usage`
  - `marv.channel`, `marv.provider`, `marv.model`
  - `marv.sessionKey`, `marv.sessionId`
  - `marv.tokens.*` (input/output/cache_read/cache_write/total)
- `marv.webhook.processed`
  - `marv.channel`, `marv.webhook`, `marv.chatId`
- `marv.webhook.error`
  - `marv.channel`, `marv.webhook`, `marv.chatId`,
    `marv.error`
- `marv.message.processed`
  - `marv.channel`, `marv.outcome`, `marv.chatId`,
    `marv.messageId`, `marv.sessionKey`, `marv.sessionId`,
    `marv.reason`
- `marv.session.stuck`
  - `marv.state`, `marv.ageMs`, `marv.queueDepth`,
    `marv.sessionKey`, `marv.sessionId`

### Sampling + flushing

- Trace sampling: `diagnostics.otel.sampleRate` (0.0–1.0, root spans only).
- Metric export interval: `diagnostics.otel.flushIntervalMs` (min 1000ms).

### Protocol notes

- OTLP/HTTP endpoints can be set via `diagnostics.otel.endpoint` or
  `OTEL_EXPORTER_OTLP_ENDPOINT`.
- If the endpoint already contains `/v1/traces` or `/v1/metrics`, it is used as-is.
- If the endpoint already contains `/v1/logs`, it is used as-is for logs.
- `diagnostics.otel.logs` enables OTLP log export for the main logger output.

### Log export behavior

- OTLP logs use the same structured records written to `logging.file`.
- Respect `logging.level` (file log level). Console redaction does **not** apply
  to OTLP logs.
- High-volume installs should prefer OTLP collector sampling/filtering.

## Troubleshooting tips

- **Gateway not reachable?** Run `marv doctor` first.
- **Logs empty?** Check that the Gateway is running and writing to the file path
  in `logging.file`.
- **Need more detail?** Set `logging.level` to `debug` or `trace` and retry.
