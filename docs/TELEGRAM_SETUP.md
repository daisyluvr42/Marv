# Telegram Setup (MVP)

## 1. 创建 Bot
1. 在 Telegram 打开 `@BotFather`
2. 发送 `/newbot`，按提示完成创建
3. 记录返回的 `TELEGRAM_BOT_TOKEN`

## 2. 配置环境变量
在项目根目录创建 `.env`：

```bash
cp .env.example .env
```

至少配置：

```bash
TELEGRAM_BOT_TOKEN=<your_bot_token>
```

可选配置：
- `TELEGRAM_OWNER_IDS`：逗号分隔 Telegram user id，命中的用户将以 owner 角色调用 Edge
- `TELEGRAM_ALLOWED_CHAT_IDS`：聊天白名单（逗号分隔），为空表示不限制
- `TELEGRAM_REQUIRE_PAIRING=true|false`：是否要求首次配对码认证（默认 false）

## 3. 启动
```bash
bash scripts/start_stack.sh
```

进程与日志：
- PID: `.run/telegram.pid`
- 日志: `logs/telegram.log`

联调前可执行：
```bash
bash scripts/telegram_live_check.sh | jq
```

## 5. 配对码认证（可选）
当 `TELEGRAM_REQUIRE_PAIRING=true` 时，用户首次需先发送 `/pair <code>`。

生成配对码（owner）：
```bash
uv run marv telegram pair create-code --chat-id <chat_id> --user-id <telegram_user_id> --ttl-seconds 900
```

查看配对状态：
```bash
uv run marv telegram pair codes --status open
uv run marv telegram pair list --chat-id <chat_id>
```

## 4. 获取自己的 Telegram user id（可选）
给 bot 发一条消息后执行：

```bash
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getUpdates" | jq
```

在返回里查看 `message.from.id`，填入 `TELEGRAM_OWNER_IDS`。
