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

## 3. 启动
```bash
bash scripts/start_stack.sh
```

进程与日志：
- PID: `.run/telegram.pid`
- 日志: `logs/telegram.log`

## 4. 获取自己的 Telegram user id（可选）
给 bot 发一条消息后执行：

```bash
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getUpdates" | jq
```

在返回里查看 `message.from.id`，填入 `TELEGRAM_OWNER_IDS`。
