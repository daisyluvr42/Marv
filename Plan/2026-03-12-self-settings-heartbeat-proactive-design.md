# Self Settings, Heartbeat, and Proactive Behavior Design

Date: 2026-03-12
Status: Draft
Relates to: `Plan/2026-03-10-proactive-agent-behavior.md`

## Summary

Expand `self_settings` from a narrow session-only tool into a unified self-
configuration entry point with an explicit system-level allowlist.

The design also formalizes heartbeat as a lightweight proactive engine:

- `HEARTBEAT.md` becomes the durable, user-steerable checklist for routine
  awareness tasks
- heartbeat system settings remain user-authorized and persist through config
- heartbeat runs are allowed to take bounded low-risk task actions
- exact-time or heavyweight work remains the job of cron or explicit user turns

This design is intentionally conservative compared with the most aggressive
"always working" OpenClaw community patterns. It borrows their proactive
benefits without turning heartbeat into an unbounded autonomous loop.

## Problem

The current `self_settings` tool was introduced in a multi-agent world and keeps
its scope intentionally small. That no longer matches the desired product
behavior.

The current gaps are:

- users cannot set heartbeat cadence, delivery, or checklist behavior via
  natural language through `self_settings`
- the agent can update some session behavior, but there is no single
  allowlisted path for persistent system-level settings
- heartbeat has no first-class product definition as a bounded proactive engine
- there is no formal policy for what heartbeat may do on its own versus what
  requires a direct user instruction

## Research Inputs

### Local Marv/OpenClaw code and docs

- Marv heartbeat currently runs as a full agent turn with main-session context
  and default `HEARTBEAT.md` prompt semantics
- interval heartbeats are skipped when `HEARTBEAT.md` is missing or effectively
  empty, which means the checklist file is already the practical control plane
  for routine heartbeat work
- `heartbeat.model` exists specifically to let heartbeat runs use a distinct
  model from normal chat
- `HEARTBEAT.md` is already treated as a normal workspace file and can be
  updated by the agent

### Upstream OpenClaw positioning

The vendored upstream docs consistently position heartbeat as:

- periodic awareness
- batched routine checks
- context-aware prioritization
- conversational continuity
- suppression-friendly notification logic via `HEARTBEAT_OK`

OpenClaw's official guidance strongly separates heartbeat from cron:

- heartbeat for recurring awareness and triage
- cron for exact timing, isolated runs, or heavyweight one-off jobs

### Public user/community patterns

Public usage patterns fall into two broad families:

1. Conservative users
   - use `HEARTBEAT.md` as a short checklist
   - ask heartbeat to scan inbox, calendar, blockers, alerts
   - rely on digesting, drafts, and approval boundaries

2. Aggressive users
   - use heartbeat as a quasi-continuous proactive loop
   - ask the agent to review backlog, choose next work, write logs, dispatch
     subagents, and keep projects moving

The aggressive pattern is attractive because it aligns with the product goal of
more proactive behavior. The risk is that it can turn heartbeat into a vague,
always-on optimizer with weak stop conditions, high token use, and fuzzy safety
boundaries.

## Product Positioning

Heartbeat should be positioned as:

**A lightweight, checklist-driven proactive loop that periodically observes the
current situation, performs bounded low-risk follow-through, and only escalates
to the user when something is worth attention.**

Heartbeat should not be positioned as:

- a precise scheduler
- a replacement for cron
- an always-on autonomous builder
- an open-ended self-directed planning loop

## User Mental Model

Users generally organize this type of automation in three layers:

1. `HEARTBEAT.md`
   - what to check every cycle
   - what counts as urgent
   - what can stay internal

2. Heartbeat config
   - how often to run
   - where to deliver
   - what model to use
   - what hours are valid

3. Cron
   - exactly-timed tasks
   - one-shot reminders
   - heavyweight isolated work

This design should keep those layers clear in both implementation and prompt
guidance.

## Goals

- Let users change heartbeat behavior through direct natural language requests.
- Let the agent maintain `HEARTBEAT.md` as part of normal proactive operation.
- Preserve a hard boundary: system-level settings require direct user
  instruction.
- Allow heartbeat to perform bounded, low-risk task actions without explicit
  user confirmation each time.
- Keep heartbeat cheap enough to run on smaller or local models when the
  checklist is narrow.

## Non Goals

- Turning `self_settings` into a free-form config editor
- Allowing heartbeat to change arbitrary global config on its own
- Making heartbeat the primary engine for exact-time automation
- Defaulting to high-autonomy external side effects

## Design

### 1. Expand `self_settings` into a unified allowlisted config entry point

`self_settings` should continue to support session-level adjustments, but it
should no longer be described as session-only.

It should gain an explicit second tier:

- session/task-level settings
- system-level allowlisted settings

System-level writes remain narrow and schema-backed. The tool must not become a
generic natural-language proxy for arbitrary config changes.

### 2. Authority model

The authority model should be explicit and easy to reason about:

- Session-level changes:
  - may be applied during the current conversation
  - may be adjusted autonomously when the task demands it

- Task-level low-risk behavior:
  - may be performed autonomously by heartbeat or the main agent
  - must stay inside pre-approved low-risk categories

- System-level changes:
  - only allowed when the request is a direct user instruction
  - denied for indirect, quoted, forwarded, or inferred requests
  - restricted to an allowlist

This preserves user control over persistent system behavior while still letting
the agent be operationally proactive.

### 3. Heartbeat autonomy model

Heartbeat behavior should be split into three layers:

#### Observe

Heartbeat may always:

- read `HEARTBEAT.md`
- inspect current session/main-session state
- check routine signals listed in the checklist
- decide whether there is anything worth surfacing

#### Propose

Heartbeat may autonomously:

- update `HEARTBEAT.md` when the checklist is stale or obviously redundant
- write notes, work logs, or memory entries
- prepare drafts, summaries, or follow-up suggestions
- reorganize low-risk internal task lists

#### Act

Heartbeat may autonomously perform bounded low-risk actions inside the current
task domain, such as:

- gather logs or status output
- refresh local project context
- summarize finished background work
- create internal notes or TODO scaffolding
- prepare but not send external communications

Heartbeat must not autonomously:

- change system-level config
- create or modify scheduling policy outside its allowlisted checklist files
- send high-risk external actions by default
- start open-ended heavy work loops without clear task scope

### 4. First-wave system allowlist

The first-wave system-level `self_settings` allowlist should include heartbeat
settings only.

Recommended fields:

- `heartbeatEvery`
- `heartbeatPrompt`
- `heartbeatModel`
- `heartbeatTarget`
- `heartbeatTo`
- `heartbeatAccountId`
- `heartbeatIncludeReasoning`
- `heartbeatSuppressToolErrorWarnings`
- `heartbeatAckMaxChars`
- `heartbeatActiveHoursStart`
- `heartbeatActiveHoursEnd`
- `heartbeatActiveHoursTimezone`
- `heartbeatFileAction`
- `heartbeatFileContent`

The first twelve map to `agents.defaults.heartbeat`.

The file action fields control `HEARTBEAT.md`.

This keeps the initial design small while proving the larger architecture.
Once stable, the same allowlist pattern can be extended to other system-level
domains.

### 5. `HEARTBEAT.md` as the behavioral control plane

`HEARTBEAT.md` should be treated as the main durable behavioral surface for
heartbeat.

It should remain:

- short
- stable
- direct
- checklist-shaped

It should contain:

- routine checks
- urgency criteria
- notification rules
- low-risk proactive permissions

Recommended structure:

```md
# HEARTBEAT.md

## Mission

- Maintain lightweight awareness of the user's current priorities.

## Routine Checks

- Check inbox for urgent items.
- Check calendar for events in the next 2 hours.
- Check project blockers or failing tasks.

## Escalate When

- There is a same-day deadline risk.
- A customer-facing issue is blocked.
- A background task failed and needs a decision.

## Low-Risk Actions You May Take

- Collect logs or status.
- Update notes, memory, or internal TODOs.
- Rewrite this file to stay short and current.
- Draft messages, but do not send risky external replies without approval.

## Stay Quiet When

- Nothing is urgent.
- The result is only routine noise.
```

### 6. Heartbeat execution profile

Heartbeat should default to a low-intensity operational profile:

- narrow checklist
- low or medium reasoning
- cheaper model when possible
- `target: "none"` during early rollout or on setups where silent observation is
  preferred

This matches both official guidance and the practical reality that heartbeat
burns fewer tokens and behaves more predictably when it is narrow.

## Recommended Natural Language Mapping

The agent should route these requests through `self_settings`:

- "Set heartbeat to every 30 minutes"
- "Only run heartbeats between 9am and 10pm"
- "Use a cheaper local model for heartbeats"
- "Send heartbeat alerts to Telegram"
- "Rewrite the heartbeat checklist to focus on inbox and blockers"
- "Add a rule saying heartbeat may collect logs automatically"

Tool-side mapping examples:

- `heartbeatEvery: "30m"`
- `heartbeatActiveHoursStart: "09:00"`
- `heartbeatActiveHoursEnd: "22:00"`
- `heartbeatModel: "ollama/qwen2.5:3b"`
- `heartbeatTarget: "telegram"`
- `heartbeatFileAction: "replace"`
- `heartbeatFileContent: "<new HEARTBEAT.md body>"`

## Write Paths

### System config writes

System-level `self_settings` writes should go through the normal config write
flow:

- `config.get`
- `config.patch` with `baseHash`

This avoids direct config-file edits and keeps behavior aligned with the rest of
the platform.

### `HEARTBEAT.md` writes

`HEARTBEAT.md` updates should use the existing workspace file APIs:

- `agents.files.get`
- `agents.files.set`

Supported file actions should initially be:

- `replace`
- `append`
- `clear`

This is simpler and safer than trying to build structured patch semantics into
the first version.

## Prompt and Skill Changes

The prompt contract around `self_settings` should be updated to say:

- use `self_settings` when the user asks to change your own behavior or
  allowlisted system settings
- system-level settings require a direct user instruction
- heartbeat settings and `HEARTBEAT.md` maintenance are valid `self_settings`
  operations

The skill guidance should add trigger examples such as:

- "把 heartbeat 改成每 30 分钟"
- "只在白天跑 heartbeat"
- "把 heartbeat 改用本地小模型"
- "把 HEARTBEAT.md 改成只看 inbox 和 blockers"

## Testing Plan

Add focused tests for:

### `self_settings`

- accepts heartbeat allowlist fields
- rejects system-level writes when `directUserInstruction` is false
- preserves existing session-level behavior
- rejects invalid heartbeat durations, targets, and active-hours input

### `HEARTBEAT.md` file operations

- replace
- append
- clear
- denied when system-write authority is absent

### Prompt/skill routing

- prompt teaches that heartbeat settings now go through `self_settings`
- skill examples include heartbeat configuration and checklist maintenance

### Regression

- ordinary session-only requests still work
- heartbeat model override continues to affect heartbeat only

## Rollout Strategy

### Phase 1

- expand `self_settings` schema and tool logic
- add heartbeat system allowlist
- add `HEARTBEAT.md` file actions
- update prompt + skill guidance

### Phase 2

- ship a starter `HEARTBEAT.md` pattern oriented around observation and drafts
- encourage `target: "none"` for first-run validation
- observe whether low-risk actions remain bounded

### Phase 3

- if stable, extend the same allowlist pattern to additional system domains
- consider a dedicated "light context" heartbeat mode if the local branch wants
  to follow newer upstream guidance

## Success Criteria

- Users can set heartbeat cadence and related behavior using natural language.
- Users can ask the agent to rewrite or maintain `HEARTBEAT.md`.
- Heartbeat can perform bounded low-risk operational work without needing a new
  user instruction every cycle.
- System-level persistence still requires a direct user request.
- The resulting heartbeat setup remains checklist-driven and small-model-
  friendly rather than turning into an open-ended autonomous loop.
