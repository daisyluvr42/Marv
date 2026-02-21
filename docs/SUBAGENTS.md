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

## 自动分身编排（实验）
当 `auto_subagents.enabled=true` 且任务复杂度达到阈值时，主会话会自动执行三段式编排：
- `blueprint`：先产出执行蓝图与验收标准
- `executor`：按蓝图执行
- `supervisor`：监督校验并产出最终答复

默认会在流程结束后自动归档子会话（`auto_archive_children=true`），用于“收回分身”。
