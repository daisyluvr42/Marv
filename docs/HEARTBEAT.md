# Heartbeat Scheduler (APScheduler / Cron)

系统已改为 Marv 风格的周期心跳任务，由 APScheduler 驱动。

## 初始化机制
- 首次启动 Edge 时，若不存在心跳配置文件，会从环境变量生成默认配置：
  - 文件：`data/heartbeat-config.json`
  - 可覆盖路径：`EDGE_HEARTBEAT_CONFIG_PATH`
- 默认任务：
  - Core health 周期探活
  - 已批准但待执行工具调用的周期恢复

## 环境变量（初始化默认值）
- `HEARTBEAT_ENABLED`
- `HEARTBEAT_MODE=interval|cron`
- `HEARTBEAT_INTERVAL_SECONDS`
- `HEARTBEAT_CRON`（crontab 格式）
- `HEARTBEAT_CORE_HEALTH_ENABLED`
- `HEARTBEAT_RESUME_APPROVED_TOOLS_ENABLED`
- `HEARTBEAT_EMIT_EVENTS`
- `HEARTBEAT_MEMORY_DECAY_ENABLED`
- `HEARTBEAT_MEMORY_DECAY_HALF_LIFE_DAYS`
- `HEARTBEAT_MEMORY_DECAY_MIN_CONFIDENCE`

## 运行后修改（热更新）
```bash
uv run marv heartbeat show
uv run marv heartbeat set --mode interval --interval-seconds 30
uv run marv heartbeat set --mode cron --cron "*/2 * * * *"
uv run marv heartbeat set --no-emit-events
uv run marv heartbeat set --memory-decay-enabled --memory-decay-half-life-days 60 --memory-decay-min-confidence 0.25
```

`heartbeat set` 会写入配置文件并立即重载调度器。

## API
- `GET /v1/system/heartbeat`
- `POST /v1/system/heartbeat/config`（owner）
