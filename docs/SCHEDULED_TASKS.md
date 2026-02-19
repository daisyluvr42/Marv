# Scheduled Tasks

运行时提供 Cron 定时任务，触发时会自动创建任务并进入现有队列执行。

## API
- `GET /v1/scheduled/tasks`
- `POST /v1/scheduled/tasks`
- `POST /v1/scheduled/tasks/{schedule_id}:pause`
- `POST /v1/scheduled/tasks/{schedule_id}:resume`
- `POST /v1/scheduled/tasks/{schedule_id}:run`
- `POST /v1/scheduled/tasks/{schedule_id}:delete`

## CLI
```bash
uv run marv schedule list
uv run marv schedule create --name "daily-report" --prompt "生成日报" --cron "0 9 * * *" --channel telegram --channel-id 123 --user-id 456
uv run marv schedule pause <schedule_id>
uv run marv schedule resume <schedule_id>
uv run marv schedule run <schedule_id>
uv run marv schedule delete <schedule_id>
```

## 触发行为
- 调度器会在目标会话写入一条 `InputEvent`（actor=`scheduler:<schedule_id>`）
- 再由现有任务处理链路完成推理与工具执行
