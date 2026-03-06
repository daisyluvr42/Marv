# LobsterAI Integration Design

Date: 2026-03-06

## Summary

Integrate the highest-value ideas from `external_source/LobsterAI-main` into Marv without importing LobsterAI's Electron runtime or creating parallel subsystems.

This work will be implemented in four layers:

- Add conservative memory-write heuristics before long-term memory persistence.
- Strengthen OpenResponses compatibility for tool-call and reasoning edge cases.
- Improve cron observability and failure handling using existing Marv cron infrastructure.
- Enhance Feishu diagnostics and add a new DingTalk channel plugin as an extension.

## Goals

- Reuse Marv's existing memory, gateway, cron, and channel plugin architecture.
- Borrow LobsterAI's strongest ideas where they improve correctness or operator experience.
- Keep changes incremental, testable, and independently reviewable.
- Avoid introducing a second scheduler, memory system, or channel framework.

## Non Goals

- Copying LobsterAI modules wholesale into Marv.
- Replacing Marv's existing runner, sandbox runtime, or skill system.
- Rebuilding `/v1/responses` around a separate compatibility proxy.
- Shipping a fully feature-complete DingTalk implementation in the first pass.

## Findings From LobsterAI

The LobsterAI areas with the best integration value are:

- `coworkMemoryExtractor.ts`: durable-fact and preference extraction heuristics
- `coworkOpenAICompatProxy.ts`: tool-call and Responses compatibility edge cases
- `scheduler.ts`: repeated-failure handling and task-to-session linkage ideas
- `imGatewayManager.ts` and `dingtalkGateway.ts`: channel diagnostics and DingTalk connection patterns

The following areas are not worth transplanting directly:

- LobsterAI skill manager
- LobsterAI sandbox/runtime distribution system
- LobsterAI Electron-specific IM manager structure

## Recommended Approach

Implement each improvement inside the existing Marv subsystem it naturally belongs to.

### Why this approach

- It minimizes architectural conflict.
- It keeps provider and channel behavior consistent with the rest of Marv.
- It preserves current operational tooling and tests.
- It lets each change land with a narrow blast radius.

## 1. Memory Heuristics

### Goal

Reduce accidental long-term memory writes while preserving explicit "remember this" behavior.

### Placement

Add a pure heuristics layer near the existing memory write paths and apply it before persistence. The first write path to cover is the agent-facing memory tool. Automatic session-memory archival can optionally reuse the same classifier for filtering decisions where appropriate.

### Behavior

The heuristics layer should classify candidate writes into outcomes such as:

- explicit_memory
- durable_preference
- durable_identity_fact
- project_convention
- reject_transient
- reject_question
- reject_small_talk

### Rules

- Strongly allow explicit user directives to remember something.
- Allow stable user preferences and durable project conventions.
- Reject obvious one-off context, ephemeral task chatter, greetings, and most question-shaped text.
- Keep the logic conservative and deterministic. If uncertain, prefer skipping automatic persistence.

### Integration

- Keep `writeSoulMemory` and the existing storage format unchanged.
- Return a short reason or classification in tool results when a write is skipped by heuristics.
- Preserve manual/explicit writes where the user clearly asked to store something.

## 2. OpenResponses Compatibility

### Goal

Harden Marv's `/v1/responses` implementation for edge cases already handled in LobsterAI's compatibility layer.

### Placement

Extend the current gateway path in `src/core/gateway/openresponses-http.ts` with small helpers as needed. Do not introduce a second proxy service.

### Focus Areas

- Streaming function-call argument assembly and boundary handling
- Stable `call_id` correlation for follow-up `function_call_output`
- Better preservation and graceful degradation of reasoning or extra-content metadata across providers

### Strategy

- Start from regression tests.
- Port ideas as targeted behavior fixes, not broad rewrites.
- Keep endpoint shapes and existing contracts stable.

## 3. Cron Observability And Failure Handling

### Goal

Make cron health easier to inspect and repeated failures easier to understand without replacing the current scheduler.

### Placement

Build on `src/cron/service/*`, `src/cron/run-log.ts`, gateway cron handlers, and the agent cron tool.

### Behavior

Improve visibility for:

- consecutive execution failures
- schedule computation failures
- auto-disabled jobs
- latest session/sessionKey associated with runs
- last delivery and last known run health in list/status results

### Strategy

- Prefer optional additive fields so existing clients stay compatible.
- Reuse the current run log rather than adding a second tracking store.
- Surface the job-to-session linkage more clearly in gateway and agent tool responses.

## 4. Feishu Diagnostics And DingTalk Plugin MVP

### Feishu

Improve probe and channel-status diagnostics so operators can distinguish between:

- missing credentials
- token/auth failures
- bot metadata lookup failures
- transport-mode issues such as webhook versus websocket setup problems

This should remain inside the existing Feishu extension.

### DingTalk

Add a new `extensions/dingtalk` channel plugin using the same extension/plugin SDK style as `extensions/feishu`.

First version scope:

- account/config parsing
- probe/status support
- inbound text message normalization
- outbound text sending
- reply/session mapping for current conversation replies
- channel registration and basic docs metadata

Deferred from first version:

- media uploads
- rich cards
- advanced interactive workflows
- deep enterprise directory sync

### Strategy

- Use LobsterAI's DingTalk code as protocol and failure-mode reference only.
- Avoid LobsterAI's Electron gateway/event-manager structure.
- Keep DingTalk isolated as an extension so Marv core channel architecture remains unchanged.

## Data Flow

### Memory

candidate write -> heuristics classifier -> accepted write path or skip result -> existing memory storage

### OpenResponses

HTTP request -> existing request normalization -> improved tool/reasoning compatibility helpers -> existing agent execution path

### Cron

job execution -> existing state/run log updates -> enriched gateway/tool responses -> operator-visible health details

### DingTalk

DingTalk event/webhook or stream -> message normalization -> Marv channel plugin runtime -> existing agent routing/session model -> outbound text reply/send helpers

## Error Handling

### Memory

- Skipped writes should fail safely and explain only the minimal reason needed.
- Explicit remember requests should not be silently dropped.

### OpenResponses

- Invalid or incomplete tool-call follow-up data should produce clear client-facing validation errors.
- Missing provider-specific reasoning metadata should degrade gracefully rather than breaking responses.

### Cron

- Repeated execution and schedule failures should remain visible in job state.
- Auto-disable state should be surfaced clearly in list/status outputs.

### Channels

- Probe results should identify the failing phase without exposing secrets.
- DingTalk transport failures should map to clean status summaries and reconnect-ready states where possible.

## Testing Plan

### Memory

- Unit tests for heuristics classification
- Integration tests for memory tool writes that should be accepted or skipped

### OpenResponses

- Regression tests for tool-call streaming boundaries
- `function_call_output` correlation tests
- Reasoning or extra-content compatibility tests

### Cron

- Tests for additive health/status fields
- Repeated-failure and auto-disable visibility tests
- Session linkage visibility in run history

### Feishu

- Probe/status tests for key failure modes

### DingTalk

- Config normalization tests
- Probe tests
- Inbound normalization tests
- Outbound send target tests
- Plugin registration/status tests

## Rollout Order

1. Memory heuristics
2. OpenResponses compatibility
3. Cron observability improvements
4. Feishu diagnostics and DingTalk plugin MVP

## Notes

- The four workstreams are intentionally layered so the first three can land without being blocked by DingTalk.
- DingTalk should start as a minimal, operational text channel and grow only after the transport and session model are solid.
