# Execution Mode & Sandbox

运行时支持三种执行模式（主要作用于 IPC 工具）：
- `auto`：优先 Docker 沙箱（若可用），否则本地执行
- `local`：本地执行
- `sandbox`：强制 Docker 沙箱执行（无 Docker 时失败）

## 配置文件
- 默认：`data/execution-config.json`
- 可覆盖：`EDGE_EXECUTION_CONFIG_PATH`

字段：
- `mode`
- `docker_image`
- `network_enabled`

## CLI
```bash
uv run marv execution show
uv run marv execution set --mode sandbox --docker-image python:3.12-alpine --no-network-enabled
```

## API
- `GET /v1/system/execution-mode`
- `POST /v1/system/execution-mode`（owner）

## 单次工具调用覆盖
```bash
uv run marv tools exec --tool ipc_echo --args '{"query":"hello"}' --execution-mode sandbox
```

当工具绑定会话时，运行时会自动把会话工作区挂载到沙箱 `/workspace`。
