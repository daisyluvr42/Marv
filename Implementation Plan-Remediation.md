# 重构遗留问题整改方案

> 基于 2026-03-02 审计报告，针对 Phase 2/3 未完成项制定分步整改计划。
> 原则：**每步独立可提交**，每步完成后 `tsc --noEmit` 验证编译通过。

---

## 整改一：删除残留空目录（🟢 零风险）

### 操作

- 删除 `src/telegram/`（空目录，实际 telegram 实现已在 `src/channels/telegram/`）
- 删除 `src/web/`（空目录，实际 web 实现已在 `src/channels/web/`）
- 删除 `marv-legacy.mjs`（根目录，无任何文件引用）

### 验证

```bash
tsc --noEmit
pnpm build
```

---

## 整改二：迁移 `src/whatsapp/` → `src/channels/whatsapp/`（🟡 低风险）

### 现状

`src/whatsapp/` 包含 4 个文件：
| 文件 | 用途 |
|------|------|
| `normalize.ts` | WhatsApp JID/号码标准化 |
| `normalize.test.ts` | 对应测试 |
| `resolve-outbound-target.ts` | 出站目标解析 |
| `resolve-outbound-target.test.ts` | 对应测试 |

### 引用点（共 4 处需更新 import 路径）

| 引用文件                                   | 当前路径                                                              | 新路径                                                                                  |
| ------------------------------------------ | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `src/channels/dock.ts`                     | `../whatsapp/normalize.js`                                            | `./whatsapp/normalize.js`                                                               |
| `src/plugin-sdk/index.ts`                  | `../whatsapp/normalize.js` + `../whatsapp/resolve-outbound-target.js` | `../channels/whatsapp/normalize.js` + `../channels/whatsapp/resolve-outbound-target.js` |
| `src/channels/plugins/directory-config.ts` | `../../whatsapp/normalize.js`                                         | `../whatsapp/normalize.js`                                                              |
| `src/infra/outbound/outbound-session.ts`   | `../../whatsapp/normalize.js`                                         | `../../channels/whatsapp/normalize.js`                                                  |

### 操作步骤

1. 将 4 个文件从 `src/whatsapp/` 移入 `src/channels/whatsapp/`（该目录下已有其他 whatsapp channel 文件）
2. 更新上述 4 处 import 路径
3. 删除空的 `src/whatsapp/` 目录

### 验证

```bash
tsc --noEmit
pnpm vitest run --project unit -- whatsapp
```

---

## 整改三：迁移 `src/sessions/` → `src/core/session/`（🟠 中风险）

### 现状

`src/sessions/` 包含 8 个文件：
| 文件 | 用途 |
|------|------|
| `session-key-utils.ts` | Session key 解析（agent/cron/subagent/acp/thread） |
| `send-policy.ts` + `.test.ts` | 发送策略 |
| `model-overrides.ts` | 模型覆盖 |
| `level-overrides.ts` | 日志级别覆盖 |
| `input-provenance.ts` | 输入来源追踪 |
| `session-label.ts` | Session 标签 |
| `transcript-events.ts` | 转录事件 |

### 引用规模

**约 23 处 import** 分布在以下模块：

- `src/routing/` (2 处)
- `src/cron/` (2 处)
- `src/commands/` (3 处)
- `src/agents/` (5 处)
- `src/auto-reply/` (6 处)
- `src/core/gateway/` (4 处)
- `src/memory/` (1 处)

### 操作步骤

1. 在 `src/core/` 下创建 `session/` 目录
2. 将 8 个文件移入 `src/core/session/`
3. 批量更新约 23 处 import 路径（使用 sed 或逐文件修改）
4. 删除空的 `src/sessions/` 目录

### 风险控制

- 这是此次整改中影响面最大的一步，涉及 7 个模块
- 建议在单独的 Git commit 中完成，方便回滚

### 验证

```bash
tsc --noEmit
pnpm build
pnpm vitest run --project unit -- session
```

---

## 整改四：创建 `src/channels/_interface.ts`（🟡 低风险，新增文件）

### 动机

当前 channel 的统一接口类型分散在 `src/channels/plugins/types.ts` 中的 `ChannelMeta`。计划创建 `_interface.ts` 作为统一的 ChannelAdapter 抽象层，让每个 channel 实现只依赖此接口。

### 现状分析

- `src/channels/plugins/types.ts` 已定义 `ChannelMeta`、`ChannelId` 等类型
- `src/channels/registry.ts` 已有 channel 注册 + 元数据机制
- 各 channel 实现通过 plugin 系统注册

### 操作步骤

1. 审查 `src/channels/plugins/types.ts` 中现有类型定义
2. 从中提取/复用 `ChannelAdapter` 接口定义到 `src/channels/_interface.ts`
3. 导出统一的消息发送/接收/配置接口
4. 逐步让各 channel 实现依赖此接口（可分步推进，不必一次完成）

### 验证

```bash
tsc --noEmit
```

> [!NOTE]
> 这一步偏向架构优化，可作为后续持续改进任务，不阻塞其他整改。

---

## 整改五：IRC Channel 实现目录（🔵 功能新增，非整改）

### 现状

- `src/channels/registry.ts` 中已注册 `irc` 作为 `CHAT_CHANNEL_ORDER` 的成员
- 已有完整的 IRC 元数据（label、docsPath、blurb 等）
- **但 `src/channels/irc/` 实现目录不存在**

### 建议

IRC channel 的实现属于**功能开发**而非结构整改。当前的 registry 注册不影响编译和运行（只要不激活 IRC channel）。

**优先级：低。** 建议归入后续功能迭代，不纳入本次整改。

---

## 整改六：清理 `.pi/` 目录（⚪ 暂缓）

### 现状

- `.pi/` 含 `extensions/`、`git/`、`prompts/` 三个子目录
- `CLEANUP_LOG.md` 明确记录：保留 `.pi/*`，因仍被本地 coding-agent 工具链使用

### 建议

维持现状，待工具链迁移后再清理。

---

## 执行优先级

| 次序 | 整改项                         | 风险  | 预估工作量 |
| ---- | ------------------------------ | ----- | ---------- |
| 1    | 删除空目录 + `marv-legacy.mjs` | 🟢 零 | 1 分钟     |
| 2    | 迁移 `src/whatsapp/`           | 🟡 低 | 10 分钟    |
| 3    | 迁移 `src/sessions/`           | 🟠 中 | 30 分钟    |
| 4    | 创建 `_interface.ts`           | 🟡 低 | 按需推进   |
| 5    | IRC 实现                       | 🔵 —  | 功能迭代   |
| 6    | `.pi/` 清理                    | ⚪ —  | 暂缓       |

> [!IMPORTANT]
> 整改 1-3 建议在同一个工作 session 中完成，每步独立 commit。
> 整改 4-6 可作为后续 backlog。
