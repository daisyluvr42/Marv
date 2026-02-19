# Permissions (Marv-style)

项目支持 Marv 风格的执行权限策略，主要控制工具执行是否直接放行、拒绝、或进入审批。

策略文件：
- 默认路径：`data/exec-approvals.json`
- 可通过 `EDGE_EXEC_APPROVALS_PATH` 指定

核心字段：
- `security`: `deny | allowlist | full`
- `ask`: `off | on-miss | always`
- `ask_fallback`: `deny | allowlist | full`（当前保留字段）
- `allowlist`: glob 模式列表（例如 `mock_*`）

策略层级：
1. 先看 actor 专属策略（`agents.<actor_id>`）
2. 再看 `agents.main`
3. 最后回退 `defaults`

示例：
```bash
uv run marv permissions set-default --security allowlist --ask on-miss --ask-fallback deny
uv run marv permissions allowlist add --agent main --pattern mock_web_search
uv run marv permissions set-agent --agent telegram:123456 --security full --ask off
uv run marv permissions eval --agent telegram:123456 --tool mock_external_write
```

决策结果：
- `allow`: 直接执行
- `ask`: 生成 `pending_approval`，需 owner 审批后执行
- `deny`: 直接返回 403

审批模式（新增）：
- 策略文件：`data/approval-policy.json`（可通过 `EDGE_APPROVAL_POLICY_PATH` 覆盖）
- `mode=policy|all|risky`
  - `policy`：沿用原策略（兼容）
  - `all`：所有工具都先审批
  - `risky`：命中高风险集合时审批

CLI：
```bash
uv run marv approvals policy-show
uv run marv approvals policy-set --mode all
uv run marv approvals policy-set --mode risky --risky-risks external_write,exec,network
```

会话级临时放行（新增）：
```bash
# 审批通过时创建 session grant（默认 one_time）
uv run marv approvals approve <approval_id> --grant-scope session --grant-ttl-seconds 900

# 查看/撤销 grant
uv run marv approvals grants --status active
uv run marv approvals revoke-grant <grant_id>
```

Marv 交互增强：
- `permissions preset --name strict|balanced|full`
- `permissions allowlist sync-readonly --agent main`（把所有 read_only 工具加入 allowlist）
- `tools exec --prompt-approval`：当返回 `pending_approval` 时，CLI 立即提示 approve/reject/skip
