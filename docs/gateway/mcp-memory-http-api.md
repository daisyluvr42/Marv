---
summary: "Expose a Marv-mem MCP-compatible /mcp HTTP endpoint from the Gateway"
read_when:
  - Connecting external MCP clients to Marv memory
  - Calling memory_search/memory_get/memory_write over JSON-RPC
title: "Marv-mem MCP API"
---

# Marv-mem MCP (HTTP)

Marv’s Gateway exposes an MCP-compatible memory endpoint over HTTP JSON-RPC.

- `POST /mcp`
- Same port as the Gateway (WS + HTTP multiplex): `http://<gateway-host>:<port>/mcp`

Default max payload size is 2 MB.

## Authentication

Uses the same Gateway bearer authentication as other HTTP endpoints.

- `Authorization: Bearer <token>`

Notes:

- When `gateway.auth.mode="token"`, use `gateway.auth.token` (or `MARV_GATEWAY_TOKEN`).
- When `gateway.auth.mode="password"`, use `gateway.auth.password` (or `MARV_GATEWAY_PASSWORD`) as bearer value.

## Supported MCP methods

The endpoint accepts JSON-RPC `2.0` requests and currently supports:

- `initialize`
- `ping`
- `tools/list`
- `tools/call`
- `notifications/initialized` (notification)

## Exposed tools

`tools/list` returns the Marv memory tools:

- `memory_search`
- `memory_get`
- `memory_write`

Each `tools/call` can optionally pass `sessionKey` in `params` or in `arguments`.

If omitted, the Gateway resolves the configured main session key.

## JSON-RPC examples

### Initialize

```bash
curl -sS http://127.0.0.1:18789/mcp \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {}
  }'
```

### List tools

```bash
curl -sS http://127.0.0.1:18789/mcp \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "tools",
    "method": "tools/list",
    "params": {}
  }'
```

### Call memory_search

```bash
curl -sS http://127.0.0.1:18789/mcp \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "search-1",
    "method": "tools/call",
    "params": {
      "name": "memory_search",
      "sessionKey": "agent:main:main",
      "arguments": {
        "query": "deployment checklist",
        "maxResults": 6
      }
    }
  }'
```

### Call memory_write

```bash
curl -sS http://127.0.0.1:18789/mcp \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "write-1",
    "method": "tools/call",
    "params": {
      "name": "memory_write",
      "arguments": {
        "content": "release retrospective: rollback playbook updated",
        "kind": "session_summary",
        "scopeType": "agent",
        "scopeId": "main",
        "source": "manual_log"
      }
    }
  }'
```

## MCP client config examples

Use `Marv-mem` as the MCP server name.

### Claude Code / Claude Desktop

```json
{
  "mcpServers": {
    "Marv-mem": {
      "type": "streamableHttp",
      "url": "http://127.0.0.1:18789/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      }
    }
  }
}
```

### Cursor (MCP)

```json
{
  "mcpServers": {
    "Marv-mem": {
      "type": "streamableHttp",
      "url": "http://127.0.0.1:18789/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      }
    }
  }
}
```

## Response semantics

- Standard JSON-RPC `2.0` envelope.
- Request/response calls return `200`.
- Notification-only calls may return `202` with an empty body.
- Tool input validation errors return JSON-RPC error `-32602`.
- Unknown methods/tools return `-32601`.
