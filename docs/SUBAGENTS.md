# Subagent Sessions

运行时支持从主会话派生 `subagent` 会话，每个会话有独立 workspace，并可单独收发消息。

## API
- `POST /v1/agent/sessions/{conversation_id}:spawn`
- `POST /v1/agent/sessions/{conversation_id}:send`
- `GET /v1/agent/sessions/{conversation_id}/history`

## CLI
```bash
uv run marv sessions spawn <parent_conversation_id> --name analysis
uv run marv sessions send <child_conversation_id> --message "请做深入分析" --wait
uv run marv sessions history <child_conversation_id> --limit 100
```

子会话 ID 形如：
`subagent:<parent_conversation_id>:<name>:<suffix>`
