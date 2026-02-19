# IPC Tools Bridge

运行时支持从配置文件动态加载 IPC 工具，便于接入本地脚本/微服务。

默认配置路径：
- `data/ipc-tools.json`
- 可通过 `EDGE_IPC_TOOLS_PATH` 覆盖

## 配置格式
```json
[
  {
    "name": "ipc_echo",
    "risk": "read_only",
    "command": ["python3", "-c", "import json,sys; p=json.load(sys.stdin); print(json.dumps({'status':'ok','echo':p.get('args',{})}))"],
    "schema": {
      "type": "object",
      "properties": {
        "query": { "type": "string" }
      },
      "required": ["query"]
    },
    "timeout_seconds": 8
  }
]
```

进程输入（stdin）：
```json
{"args": {"query": "hello"}}
```

进程输出（stdout）建议返回 JSON；非 JSON 文本也会被封装返回。

## 运行
```bash
uv run marv system ipc-reload
uv run marv tools list
uv run marv tools exec --tool ipc_echo --args '{"query":"hello"}'
```
