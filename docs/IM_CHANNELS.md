# Multi-Channel IM Ingress

运行时提供统一 IM 入站接口，用于 Telegram/Discord/Slack/钉钉/飞书/WebChat 的消息接入。

## 支持渠道
- `telegram`
- `discord`
- `slack`
- `dingtalk`
- `feishu`
- `webchat`

查询：
```bash
uv run marv im channels
```

## 统一入站 API
`POST /v1/gateway/im/{channel}/inbound`

最简 payload：
```json
{
  "text": "hello",
  "channel_id": "room-1",
  "user_id": "u-1"
}
```

可选参数：
- `thread_id`
- `conversation_id`
- `actor_id`
- `metadata`

查询参数：
- `wait=true|false`：是否等待任务结束并返回 `completion_text`
- `wait_timeout_seconds`：等待超时

## Slack URL Verification
当 `channel=slack` 且 payload 为 `{"type":"url_verification","challenge":"..."}` 时，接口会原样回传 challenge。

## 安全控制
可选配置入口 token：
- `IM_INGRESS_TOKEN`：单 token（全渠道）
- `IM_INGRESS_TOKENS_JSON`：按渠道 token，例如：
```json
{
  "slack": "xxx",
  "discord": "yyy",
  "*": "fallback-token"
}
```

请求头支持：
- `X-Marv-Token: <token>`
- `Authorization: Bearer <token>`

## CLI 联调
```bash
uv run marv im ingest --channel discord --message "hi" --channel-id room-1 --user-id u-1 --wait
```

或直接提交原始 JSON：
```bash
uv run marv im ingest --channel slack --payload-json '{"event":{"text":"hi","channel":"C1","user":"U1"}}'
```
