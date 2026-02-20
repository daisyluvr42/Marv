# Packages Contract

项目支持 `pi-mono` 风格的“包契约层”，用于把扩展能力以目录包方式接入运行时。

## 1. 目录

- 默认根目录：`./packages`
- 可通过环境变量覆盖：`EDGE_PACKAGES_ROOT`
- 每个包是根目录下一个子目录。

## 2. Manifest 规范

支持两种格式：

1. `MARV_PACKAGE.json`（推荐）
2. `package.json` + `marvPackage` 字段

最小字段示例：

```json
{
  "name": "demo-package",
  "version": "0.1.0",
  "description": "demo",
  "enabled": true,
  "capabilities": ["ipc_tools"],
  "hooks": {
    "ipc_tools": "tools/ipc-tools.json"
  }
}
```

说明：

- `enabled=false` 的包会被扫描但不会加载运行时 hook。
- `hooks.ipc_tools` 为相对路径时，会按包目录解析为绝对路径。

## 3. 运行时加载

启动时会自动扫描并加载包 hook：

- IPC 工具 hook 会注册到工具系统（与 `data/ipc-tools.json` 一致的格式）。

## 4. API

- `GET /v1/packages`：列出包契约信息
- `POST /v1/packages:reload`：重新加载包 hook（owner）

## 5. CLI

```bash
uv run marv packages list
uv run marv packages reload
```

