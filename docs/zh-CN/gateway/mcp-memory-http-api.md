---
summary: 从 Gateway 网关暴露兼容 MCP 的 Marv-mem /mcp HTTP 端点
read_when:
  - 将外部 MCP 客户端接入 Marv 记忆系统时
  - 通过 JSON-RPC 调用 memory_search/memory_get/memory_write 时
title: Marv-mem MCP API
---

# Marv-mem MCP（HTTP）

Marv 的 Gateway 网关提供一个兼容 MCP 的记忆端点（基于 HTTP JSON-RPC）。

- `POST /mcp`
- 与 Gateway 网关相同的端口（WS + HTTP 多路复用）：`http://<gateway-host>:<port>/mcp`

默认最大负载大小为 2 MB。

## 认证

与其他 Gateway 网关 HTTP 端点一致，使用 Bearer 认证：

- `Authorization: Bearer <token>`

说明：

- 当 `gateway.auth.mode="token"` 时，使用 `gateway.auth.token`（或 `OPENCLAW_GATEWAY_TOKEN`）。
- 当 `gateway.auth.mode="password"` 时，也通过 Bearer 头传 `gateway.auth.password`（或 `OPENCLAW_GATEWAY_PASSWORD`）。

## 支持的 MCP 方法

端点接收 JSON-RPC `2.0` 请求，目前支持：

- `initialize`
- `ping`
- `tools/list`
- `tools/call`
- `notifications/initialized`（通知）

## 暴露的工具

`tools/list` 返回以下 Marv 记忆工具：

- `memory_search`
- `memory_get`
- `memory_write`

`tools/call` 可在 `params` 或 `arguments` 中传 `sessionKey`。

若未提供，Gateway 网关将自动使用配置的主会话键。

## JSON-RPC 示例

### 初始化

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

### 列出工具

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

### 调用 memory_search

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

### 调用 memory_write

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

## MCP 客户端配置示例

MCP 服务名使用 `Marv-mem`。

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

### Cursor（MCP）

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

## 响应语义

- 使用标准 JSON-RPC `2.0` 包装格式。
- 请求-响应调用返回 `200`。
- 纯通知可能返回 `202` 且响应体为空。
- 工具参数校验错误返回 JSON-RPC 错误 `-32602`。
- 未知方法/工具返回 `-32601`。
