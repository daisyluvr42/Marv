# Live Integration & Tuning

本指南用于在真实运行环境中完成 Telegram + Edge 的联调，并对 heartbeat/persona 参数做可回放调优。

## 1. 前置条件
- 已完成 `uv sync`
- 已启动服务：`bash scripts/start_stack.sh`
- `.env` 已配置 `TELEGRAM_BOT_TOKEN`

## 2. 实机联调前置检查
```bash
bash scripts/telegram_live_check.sh | jq
```

检查重点：
- `edge_health.status == ok`
- `telegram_process.running == true`
- `bot.username` 正确
- `updates.latest` 能看到最近消息（说明 Bot API 连通）

## 3. 查看当前会话生效配置（persona runtime）
按 Telegram chat/user 维度查看最终配置：
```bash
uv run marv config effective \
  --channel telegram \
  --channel-id <chat_id> \
  --user-id <telegram_user_id>
```

按具体会话查看（会额外叠加 conversation 层 patch）：
```bash
uv run marv config effective \
  --conversation-id telegram:<chat_id>:0 \
  --channel telegram \
  --channel-id <chat_id> \
  --user-id <telegram_user_id>
```

## 4. 单次联调探针（建议先跑）
```bash
uv run marv ops probe \
  --message "联调探针" \
  --conversation-id telegram:<chat_id>:0 \
  --channel telegram \
  --channel-id <chat_id> \
  --user-id <telegram_user_id>
```

输出包含：
- `wall_time_ms`：端到端耗时
- `event_latency_ms`：InputEvent -> CompletionEvent 耗时
- `plan/route/completion_text`：联调核心观测字段
- `effective_config`：本次任务最终生效配置

## 5. 批量参数调优
### 5.1 可选：先调 heartbeat
```bash
uv run marv heartbeat set --mode interval --interval-seconds 45
```

### 5.2 批量 probe（自动写入 JSONL）
```bash
TUNE_CHANNEL_ID=<chat_id> \
TUNE_USER_ID=<telegram_user_id> \
TUNE_RUNS=5 \
TUNE_HEARTBEAT_INTERVAL_SECONDS=45 \
TUNE_STYLE_TEXT="更简洁" \
bash scripts/live_tuning.sh
```

结果文件：`logs/live_tuning_*.jsonl`

常用变量：
- `TUNE_CHANNEL`（默认 `telegram`）
- `TUNE_CHANNEL_ID`（必填）
- `TUNE_USER_ID`
- `TUNE_RUNS`
- `TUNE_TIMEOUT_SECONDS`
- `TUNE_POLL_INTERVAL_SECONDS`
- `TUNE_STYLE_TEXT`（为空时不自动提交 patch）

## 6. 推荐调优顺序
1. 固定权限策略（例如 `permissions preset --name balanced`）
2. 固定 heartbeat（先 `interval` 再试 `cron`）
3. 用 `ops probe` 打基线（至少 3 次）
4. 修改 persona patch 后再次批量 probe
5. 对比 JSONL 中 `wall_time_ms/event_latency_ms` 与回复风格变化
