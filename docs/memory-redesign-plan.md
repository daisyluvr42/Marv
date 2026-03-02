# Memory System Redesign & MCP Integration Layer — 开发计划

> 创建时间：2026-03-02  
> 状态：待审批

## 背景

当前 Memory 系统存在两个核心问题：

1. **`soul-memory-store.ts` 过于臃肿**（2383 行），salience score 计算（含 vector/lexical/bm25/graph/cluster 五路融合）、temporal decay、reinforcement、reference boost 等核心机制全部内嵌，缺乏独立的类型定义和可测试性。
2. **MCP 对接层**（`/mcp` 和 `tools/call` 端点）位于 `core/gateway/mcp-memory-http.ts`，与 Memory 领域分离，缺乏清晰的请求/响应类型定义。

---

## Part 1: Memory System 模块化改造

### 1.1 新建 `src/memory/salience/` 模块

| 文件                     | 职责                                                                                                                                                                                    |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `salience-types.ts`      | `SalienceScoreBreakdown`, `SalienceWeights`, `FusionWeights` 类型定义                                                                                                                   |
| `salience-compute.ts`    | `computeFusionSemanticMatch()`, `computeWeightedScore()`, `tierPriorityFactor()`, `resolveScopePenalty()`, `computeCurrentClarity()`, `clarityDecayFactor()`, `shouldPruneMemoryItem()` |
| `reinforcement.ts`       | `reinforceRetrievedItems()`, `recordScopeHits()`, `computeReinforcementFactor()`                                                                                                        |
| `reference-expansion.ts` | `applyReferenceExpansion()`, `loadReferencesBySourceIds()`, `upsertItemReferences()` 及相关常量                                                                                         |
| `index.ts`               | Barrel export                                                                                                                                                                           |

### 1.2 增强 Temporal Decay 类型

修改 `src/memory/search/temporal-decay.ts`：

- 新增 `TemporalDecayResult` 类型

### 1.3 瘦身 `soul-memory-store.ts`

- 删除已提取的函数/常量
- 改为从 `../salience/` import
- 保留 DB 操作、schema、query orchestration
- **所有公共 export 保持兼容**

### 1.4 新增测试

| 测试文件                      | 覆盖范围                            |
| ----------------------------- | ----------------------------------- |
| `salience-compute.test.ts`    | 五路融合、权重极端值、clarity decay |
| `reinforcement.test.ts`       | reinforcement factor log 曲线       |
| `reference-expansion.test.ts` | 多跳扩展边界条件                    |

---

## Part 2: MCP 对接层

### 2.1 新建 `src/memory/api/` 模块

| 文件                  | 职责                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| `mcp-types.ts`        | `McpJsonRpcRequest/Response/Error`, `McpToolCallParams`, `McpMemorySearchArgs/GetArgs/WriteArgs` |
| `mcp-handler.ts`      | `handleRpcRequest()`, `callMemoryTool()`, `buildToolList()`, `resolveToolContext()`              |
| `mcp-handler.test.ts` | 纯 JSON-RPC 层集成测试                                                                           |
| `index.ts`            | Barrel export                                                                                    |

### 2.2 瘦身 `core/gateway/mcp-memory-http.ts`

- 保留为 HTTP 薄垫片
- 委托 JSON-RPC 逻辑给 `memory/api/mcp-handler.ts`
- 仅负责 HTTP body 解析、auth、status code

---

## 目录结构预览

```
src/memory/
├── api/                          # [NEW]
│   ├── index.ts
│   ├── mcp-handler.ts
│   ├── mcp-handler.test.ts
│   └── mcp-types.ts
├── salience/                     # [NEW]
│   ├── index.ts
│   ├── salience-types.ts
│   ├── salience-compute.ts
│   ├── salience-compute.test.ts
│   ├── reinforcement.ts
│   ├── reinforcement.test.ts
│   ├── reference-expansion.ts
│   └── reference-expansion.test.ts
├── search/
│   └── temporal-decay.ts         # [MODIFY]
├── storage/
│   └── soul-memory-store.ts      # [MODIFY] 瘦身
├── embeddings/                   # (不变)
├── manager.ts                    # (不变)
├── types.ts                      # (不变)
└── index.ts                      # (不变)
```

---

## 验证计划

```bash
# 1. TypeScript 编译检查
npx tsc --noEmit

# 2. 全量单元测试（无回归）
npx vitest run --project unit

# 3. 新增 salience 测试
npx vitest run src/memory/salience/

# 4. 新增 MCP handler 测试
npx vitest run src/memory/api/

# 5. 原 MCP HTTP 测试兼容性
npx vitest run src/core/gateway/mcp-memory-http.test.ts

# 6. 原 soul-memory-store 测试
npx vitest run src/memory/storage/soul-memory-store.test.ts
```

---

## 风险评估

| 风险项                                      | 等级 | 缓解措施                                 |
| ------------------------------------------- | ---- | ---------------------------------------- |
| `soul-memory-store.ts` 拆分引入 import 循环 | 中   | 单向依赖：salience → types，不回指 store |
| MCP 端点路径变化                            | 低   | 保留 `core/gateway/` 薄垫片 re-export    |
| 提取函数遗漏参数变更                        | 低   | 函数签名 1:1 Copy，re-export 确保兼容    |
