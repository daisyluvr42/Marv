# Marv Migration Record: Semantic Config Patching + Event Sourcing Ledger

Date: 2026-02-23
Workspace Root: /Users/daisyluvr/Documents/Marv
Target Codebase: /Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main

## 1. Background

This migration continues the Marv memory upgrade track and ports two capabilities from the original Marv backend design into the OpenClaw-based architecture:

1. Semantic Config Patching
2. Event Sourcing Ledger

The goal is to keep OpenClaw as the runtime core while reintroducing structured, auditable config evolution and event timeline replay/query.

## 2. Scope of This Change

### 2.1 Semantic Config Patching

Implemented a proposal/commit/rollback lifecycle:

- propose: natural language -> structured patch proposal
- commit: apply proposal patch to active gateway config
- rollback: revert a committed semantic revision using stored snapshot
- list: inspect semantic revision history

Includes persistent storage for proposals and revisions in SQLite under OpenClaw state dir.

### 2.2 Event Sourcing Ledger

Implemented a durable event ledger store in SQLite with:

- append API for domain events
- query API by conversation/task/type/time range
- ordered replay (`ts ASC, id ASC`)

Also wired key flows to append events:

- config.apply
- config.patch
- semantic proposal / commit / rollback
- memory_write

## 3. New/Updated RPC Methods

Added gateway methods:

- `config.patches.propose`
- `config.patches.commit`
- `config.revisions.rollback`
- `config.revisions.list`
- `ledger.events.query`

## 4. Implementation Files

### 4.1 New Files

- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/config/semantic-patches.ts`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/config/semantic-patches.test.ts`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/ledger/event-store.ts`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/ledger/event-store.test.ts`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/gateway/server-methods/ledger.ts`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/gateway/protocol/schema/ledger.ts`

### 4.2 Updated Files (Core Wiring)

- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/gateway/server-methods/config.ts`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/gateway/server-methods.ts`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/gateway/server-methods-list.ts`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/gateway/protocol/index.ts`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/gateway/protocol/schema/config.ts`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/gateway/protocol/schema.ts`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/gateway/protocol/schema/types.ts`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/gateway/protocol/schema/protocol-schemas.ts`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/agents/tools/memory-tool.ts`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/agents/tools/gateway-tool.ts`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/agents/openclaw-gateway-tool.e2e.test.ts`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/src/gateway/server.config-patch.e2e.test.ts`

### 4.3 Updated Docs

- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/docs/gateway/configuration.md`
- `/Users/daisyluvr/Documents/Marv/external_reference_code/openclaw-main/docs/zh-CN/gateway/configuration.md`

## 5. Runtime Storage Layout

State-backed SQLite files introduced by this migration:

- semantic patches DB:
  - `<OPENCLAW_STATE_DIR>/config/semantic-patches.sqlite`
- ledger DB:
  - `<OPENCLAW_STATE_DIR>/ledger/events.sqlite`

If `OPENCLAW_STATE_DIR` is unset, OpenClaw default state resolution applies.

## 6. Compatibility Notes

- Existing `config.patch` and `config.apply` behavior remains available.
- New semantic patch APIs are additive and do not remove old config RPC methods.
- Ledger writes are best-effort (fail-open for primary operations).
- `memory_write` now emits `MemoryWrittenEvent` into ledger for timeline/audit.

## 7. Validation & Test Results

Executed and passed:

- `src/config/semantic-patches.test.ts`
- `src/ledger/event-store.test.ts`
- `src/gateway/server.config-patch.e2e.test.ts`
- `src/agents/openclaw-gateway-tool.e2e.test.ts`
- `src/agents/tools/memory-tool.e2e.test.ts`

All targeted tests for this migration passed on 2026-02-23.

## 8. Example Calls

### 8.1 Semantic proposal + commit

```bash
openclaw gateway call config.patches.propose --params '{
  "naturalLanguage": "请更简洁一点",
  "scopeType": "global",
  "scopeId": "gateway"
}'

openclaw gateway call config.patches.commit --params '{
  "proposalId": "pp_xxx"
}'
```

### 8.2 Revision rollback

```bash
openclaw gateway call config.revisions.rollback --params '{
  "revision": "rev_xxx"
}'
```

### 8.3 Ledger query

```bash
openclaw gateway call ledger.events.query --params '{
  "conversationId": "config:global:gateway",
  "limit": 100
}'
```

## 9. Follow-up Suggestions

1. Add stricter role/approval gating for L2/L3 semantic commits (currently rule-based risk tags exist, but policy can be tightened further).
2. Extend ledger event taxonomy to include more runtime stages (planning/routing/execution milestones).
3. Add a small admin UI panel for semantic proposals/revisions/ledger query to reduce CLI-only operations.
