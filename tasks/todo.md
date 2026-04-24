# Todo

## MarvMem Extraction (2026-04-18)

- [x] Confirm the reusable boundary between Marv's long-term memory, runtime recall, and Marv-specific working memory.
- [x] Write the standalone `marvmem` design doc under `Plan/`.
- [x] Create a new `marvmem/` project directory with its own package metadata and TypeScript config.
- [x] Implement a standalone memory core with structured records, scopes, file-backed persistence, and hybrid search.
- [x] Implement runtime recall/capture helpers that produce prompt-ready context for other agents.
- [x] Implement MCP handlers and thin adapters for Marv, Hermes-style agents, and OpenClaw-style agents.
- [x] Add focused tests and a README with integration examples.

## Review

- Added a new standalone project at `marvmem/` instead of folding the extraction into Marv internals, so the memory system now has its own package metadata, build/test scripts, and README.
- Kept the extracted boundary focused on reusable long-term memory and runtime recall: `src/core` handles durable records, scopes, file-backed persistence, and hybrid lexical/hash-embedding search; `src/runtime` handles prompt injection and turn capture; `src/mcp` exposes tool calls; `src/adapters` provides thin agent-facing wrappers.
- Chose a file-backed JSON store for the first pass so external agents can embed MarvMem without native SQLite/vector dependencies, while keeping the public API stable enough to add a different backend later.
- Added focused tests for core search/recall, runtime capture/recall, and MCP write/search flow.
- Verified with:
  - `pnpm exec tsc -p marvmem/tsconfig.json --noEmit`
  - `pnpm exec tsc -p marvmem/tsconfig.json`
  - `node --import tsx --test marvmem/test/*.test.ts`

## Webchat Metadata Sanitization (2026-04-13)

- [x] Centralize user-message display sanitization so envelope stripping, message-id hint stripping, and inbound metadata stripping share one code path.
- [x] Apply the shared sanitizer to gateway chat history, webchat extraction, and transcript preview/title surfaces that render user text.
- [x] Add focused regression tests for server sanitization, webchat rendering, and transcript previews; then run targeted vitest verification.

## Review

- Added `sanitizeUserChatTextForDisplay()` in the shared chat text layer so inbound metadata stripping now lives in the same display-path sanitizer as envelope and `message_id` hint cleanup.
- Wired the shared sanitizer into gateway chat history sanitation, webchat message extraction, and transcript-derived title/preview helpers so user-visible surfaces stop diverging.
- Added focused regressions for gateway sanitization, webchat extraction, and transcript previews, and also registered `ui/src/ui/chat/message-extract.test.ts` with the `unit` and `unit-fast` Vitest projects so the UI regression test actually runs.
- Verified with:
  - `pnpm vitest run src/core/gateway/chat-sanitize.test.ts ui/src/ui/chat/message-extract.test.ts src/core/gateway/session-utils.fs.test.ts`
  - `pnpm tsgo`
  - `pnpm exec oxfmt --check src/shared/chat-envelope.ts src/core/gateway/chat-sanitize.ts ui/src/ui/chat/message-extract.ts src/core/gateway/session-utils.fs.ts src/core/gateway/chat-sanitize.test.ts ui/src/ui/chat/message-extract.test.ts src/core/gateway/session-utils.fs.test.ts vitest.config.ts tasks/todo.md Plan/2026-04-13-webchat-metadata-sanitization.md`

## Local Model Provider Review Fixes (2026-04-12)

- [x] Rework provider-config hot reload so provider changes rebuild runtime model state instead of only clearing availability cache.
- [x] Route `models.providers.*.timeoutMs` through embedded OpenAI-compatible execution so provider timeouts apply to real inference requests.
- [x] Restore session model resolution so manual selection wins over stale persisted runtime model for display/selection surfaces.
- [x] Add focused regression tests for reload behavior, provider timeout resolution, and session model precedence.

## Review

- Hot reload for `models.providers.*` now invalidates the gateway model catalog, clears stale model availability entries, regenerates `models.json`, forces a runtime model registry refresh, and updates the long-lived refresh loop to use the new config instead of the startup snapshot.
- Embedded model execution now resolves provider-level `timeoutMs` from `models.providers.*` and applies it at the actual `runEmbeddedAttempt()` boundary, so configured local/custom provider timeouts affect real inference requests instead of only discovery/warmup.
- Session display/model resolution now treats manual selection state, including legacy override fields, as the source of truth ahead of stale persisted runtime model fields; the status surface was aligned to the same precedence.
- Verified with:
  - `pnpm vitest run src/core/gateway/config-reload.test.ts src/core/gateway/server-reload-handlers.test.ts src/core/gateway/session-utils.test.ts src/auto-reply/support/status.test.ts src/agents/pi-embedded-runner/provider-timeout.test.ts src/agents/model/runtime-model-registry.test.ts`
  - `pnpm tsgo`

## Model Management Code Audit (2026-04-06)

- [x] Inventory all model-management entrypoints across config, registration, routing, switching, and display.
- [x] Trace the runtime flow from configured defaults through registry/pool resolution and fallback behavior.
- [x] Summarize the current architecture, overlapping responsibilities, and likely sources of routing confusion.

## Review

- The model-management system is split across several layers with overlapping names: static config/schema (`core/config`), models.json generation and provider discovery (`src/agents/model/models-config*`), merged catalog/registry (`model-catalog`, `runtime-model-registry`), runtime candidate filtering (`model-pool` + availability state), execution-time failover (`model-fallback`), session/manual overrides (`core/session/model-selection-state`, `model-overrides`, `session-status-tool`, `self-settings-session`), and multiple display surfaces (`status`, `session_status`, `self_inspecting`).
- The biggest source of confusion is that several different concepts all answer “what model are we using?” with different data: configured default model, model allowlist/selections, runtime pool head candidate, session manual override, last-run persisted runtime model, subagent auto-route recommendation, and the actual fallback winner after a failed attempt.
- Auth/onboarding writes model state in more than one place: provider config under `models.providers`, default model under `agents.defaults.model`, picker allowlist under `models.selections`, aliases/params under `models.metadata`, and then forces a runtime registry refresh so the pool can change immediately.
- Manual switching is split between global config mutation (`marv models set`, fallback list commands, picker helpers) and session-scoped mutation (`session_status model=...`, `self_settings_session model=...`). Session switching writes both the new manual-selection state and legacy `providerOverride`/`modelOverride` fields for compatibility, then clears stale runtime fields.
- Verification for this task was code-audit only. No product code changes were made beyond this local task note, and no tests were needed/run.

## Model Pool Refactor Design (2026-04-06)

- [x] Re-anchor the desired architecture around a single product-level provider list and models pool.
- [x] Compare implementation approaches for unifying picker, autorouting, session switching, and footer display around the models pool.
- [x] Present the proposed design for approval before implementation.

## Review

- Chose the “single runtime models pool” direction: provider list and provider model list stay as product/config sources, while all runtime selection surfaces converge on one pool view.
- Locked the user’s key behavioral rules into the design:
  - provider auth/config adds that provider’s full model list into the pool by default
  - pool inventory is ordered high-to-low by capability for inspection and fallback traversal
  - main user sessions still default to a lightweight/fast model unless the user manually changes it
  - manual session selection outranks autorouting, and footer display must show the session’s current model immediately after each switch
- Next step is implementation planning only; no product code has been changed yet for this refactor.

## Model Pool Refactor Implementation Plan (2026-04-06)

- [in progress] Write a concrete implementation plan under `Plan/` covering data model, runtime resolution order, display semantics, and verification.
- [pending] Hand off the implementation plan summary for approval before coding.

## Review

- In progress.

## Prompt Cache Optimization Plan Review (2026-04-06)

- [x] Read `Plan/prompt-cache-optimization.md` and verify its key assumptions against the current system-prompt and Anthropic payload wiring.
- [x] Append concrete review notes to the end of the plan document.
- [x] Run a quick format/verification pass on the touched markdown files and capture the result below.

## Review

- Reviewed the plan against the current `system-prompt.ts`, `extra-params.ts`, and the Anthropic provider payload builder in `@mariozechner/pi-ai`.
- Appended targeted notes covering the main cross-provider leakage risk for the boundary marker, the Anthropic OAuth vs API-key payload shape difference, and the validation gap around wrapper composition.
- No product code was changed for this task; only local planning notes were updated.
- Verified with `pnpm exec oxfmt --check Plan/prompt-cache-optimization.md tasks/todo.md` after formatting the initial markdown wrap issue in the plan note.

## Cron Mutation Escalation Prompt Fix (2026-04-05)

- [x] Confirm whether cron add/update/remove are still hard-gated or whether the issue is model guidance.
- [x] Patch the agent guidance so normal cron mutations default to direct execution plus notification, not `request_escalation`.
- [x] Add focused tests and record the verified result below.

## Review

- Confirmed the hard gate already treats `cron` mutations as notify-and-audit operations, not escalation-gated ones, in [src/agents/tools/policy/escalation-policy.ts](/Users/daisyluvr/Documents/Marv/src/agents/tools/policy/escalation-policy.ts).
- The likely regression was model guidance, so I added an explicit rule to the system prompt telling the agent that normal `cron` add/update/remove work should be executed directly and audited via notifications instead of `request_escalation`.
- I also added the same rule to the `cron` tool description so the tool contract itself reinforces the expected behavior.
- Verified with:
  - `pnpm vitest run --config vitest.e2e.config.ts src/agents/tools/pi-tools.before-tool-call.e2e.test.ts -t 'allows cron mutations without escalation'`
  - `pnpm vitest run --config vitest.e2e.config.ts src/agents/prompt/system-prompt.e2e.test.ts -t 'tells the agent not to escalate ordinary cron mutations'`
  - `pnpm vitest run --config vitest.e2e.config.ts src/agents/tools/cron-tool.e2e.test.ts -t 'documents that normal cron mutations do not need escalation'`
  - `pnpm exec oxfmt --check src/agents/prompt/system-prompt.ts src/agents/prompt/system-prompt.e2e.test.ts src/agents/tools/cron-tool.ts src/agents/tools/cron-tool.e2e.test.ts tasks/todo.md`
- Broader note:
  - `pnpm vitest run --config vitest.e2e.config.ts src/agents/prompt/system-prompt.e2e.test.ts` still reports multiple pre-existing failures elsewhere in that file; the new cron-specific prompt assertion passed in isolation and the broader failures were not introduced by this change.

## Toolset Planner Runtime Wiring And Read-First Workbench (2026-04-05)

- [x] Wire the Phase 0 toolset planner contract into coding-tool assembly, session snapshots, and inspection surfaces without bypassing existing hard tool policy.
- [x] Expose a unified workbench snapshot through gateway RPC and add the Control UI overview/workbench surfaces with route-aware polling.
- [x] Run focused unit, e2e, typecheck, and format verification for the new planner/workbench paths and capture the reviewed results below.

## Review

- `createMarvCodingTools()` now computes a session-level toolset plan after the existing hard policy pipeline, logs the selected mode/intent, and only shrinks the final tool surface in `enforce` mode.
- The current toolset plan is now stored on the session entry and exposed through `self_inspecting`, so operators can see planner mode, intent, reasons, and suppressed tools alongside the effective tool list.
- `ensureSkillSnapshot()` now records planner metadata next to the workspace skill snapshot, while the planner-filtered tool list is reused as the skill eligibility surface so tool-incompatible skills stop inflating prompts when tool shrinking is active.
- Added a new `src/workbench/` aggregation module plus `workbench.status` gateway RPC so task-context rows, proactive rows, and deliverable counts are normalized once on the server.
- Control UI now has a Workbench workspace section, overview summary card, dedicated read-first workbench view, and 30s polling that only stays active while the workbench route is open.
- Verified with:
  - `pnpm tsgo`
  - `pnpm vitest run src/agents/tools/policy/toolset-plan.test.ts src/workbench/types.test.ts src/workbench/snapshot.test.ts src/core/gateway/server-methods/dashboard.test.ts ui/src/ui/controllers/workbench.test.ts ui/src/ui/navigation.test.ts ui/src/ui/storage.test.ts ui/src/ui/trusted-device.test.ts ui/src/ui/app-settings.test.ts ui/src/ui/app-gateway.node.test.ts ui/src/ui/navigation.browser.test.ts src/agents/tools/self/marv-tools.self-inspecting.test.ts src/auto-reply/execution/get-reply-run.test.ts src/agents/pi-embedded-runner/run.overflow-compaction.test.ts`
  - `pnpm vitest run --config vitest.e2e.config.ts src/agents/skills.buildworkspaceskillsnapshot.e2e.test.ts src/agents/tools/policy/pi-tools.policy.e2e.test.ts`
  - `pnpm exec oxfmt --check $(git status --porcelain --untracked-files=all | awk '{print $2}' | rg '\\.(ts|md)$')`

## Phase 0 Contract Lock-In For Tool Shrinking And Task Workbench (2026-04-05)

- [x] Re-read the revised plans and extract the explicit Phase 0 contract requirements before coding.
- [x] Add a pure contract module for toolset planning intent/order/suppression rules and override detection.
- [x] Add a pure contract module for workbench row/snapshot/status/deep-link types and refresh defaults.
- [x] Run focused tests for the new contract modules and capture results below.

## Review

- Added `src/agents/tools/policy/toolset-plan.ts` as the Phase 0 contract source for tool-selection modes, intent priority, suppression groups, suppression expansion, and explicit-tool override detection.
- Added `src/workbench/types.ts` as the Phase 0 contract source for unified workbench statuses, row/snapshot shapes, deep-link inventory, polling defaults, and summary truncation rules.
- Added focused tests in `src/agents/tools/policy/toolset-plan.test.ts` and `src/workbench/types.test.ts` so the revised Phase 0 rules are locked before any runtime integration work begins.
- Updated `tasks/lessons.md` after the user correction so future implementation work re-reads revised plans before coding.
- Verified with:
  - `pnpm vitest run src/agents/tools/policy/toolset-plan.test.ts src/workbench/types.test.ts`
  - `pnpm exec oxfmt --check src/agents/tools/policy/toolset-plan.ts src/agents/tools/policy/toolset-plan.test.ts src/workbench/types.ts src/workbench/types.test.ts tasks/lessons.md tasks/todo.md`

## Goose-Inspired Tool Shrinking and Task Workbench Plans (2026-04-05)

- [x] Review the current tool-policy, plugin-tool, task-context, proactive, and dashboard surfaces to anchor both plans in existing Marv architecture.
- [x] Write a local implementation plan for session-level tool/plugin shrinking in `Plan/2026-04-05-session-tool-shrinking-implementation-plan.md`.
- [x] Write a local implementation plan for a first-class task workbench in `Plan/2026-04-05-task-workbench-implementation-plan.md`.
- [x] Record the planning review below.

## Review

- Wrote two implementation plans under `Plan/` so they stay as local operator notes instead of repo docs.
- The tool-shrinking plan is intentionally subtractive-first and reuses the current `createMarvTools()` -> `resolvePluginTools()` -> `applyToolPolicyPipeline()` path instead of introducing a parallel tool system.
- The task-workbench plan is intentionally read-first and unifies `task-context` plus `proactive` data through a gateway snapshot instead of forcing a storage merge up front.
- No code or behavior changes were made for this task, so there was no test run beyond source inspection.

## iOS Companion Runtime Follow-up Plan (2026-04-02)

- [x] Align `apps/ios/Sources/DashboardModels.swift` with the current gateway response shapes and add focused contract coverage for the drifted payloads.
- [x] Rework the iOS root/dashboard presentation so the app fills the usable iPhone screen height and the dashboard uses glanceable cards instead of plain sections.
- [x] Reconnect the operator/node sessions automatically when the app returns to the foreground and only refresh dashboard data after the connection is healthy again.
- [x] Relax first-party iOS shared-auth pairing so valid companion connections auto-approve without the manual pairing prompt.
- [x] Re-run focused iOS and gateway verification, then capture the reviewed results below.

## Review

- `apps/ios/Sources/DashboardModels.swift` now matches the current gateway contract again: session rows decode `updatedAt`, cost usage decodes `daily` plus nested `totals`, and the iOS views read those current shapes directly instead of older fields.
- `apps/ios/Sources/DashboardTab.swift`, `apps/ios/Sources/RootView.swift`, `apps/ios/Sources/MarvCompanionApp.swift`, and `apps/ios/Sources/Info.plist` now present the companion as a full-height portrait iPhone app and replace the old dashboard list with glanceable cards that fit shorter screens better.
- `apps/ios/Sources/CompanionAppModel.swift` now keeps explicit operator/node connection state, checks session health on foreground return, reconnects when needed, and only refreshes the dashboard after the operator session is healthy again.
- `src/core/gateway/server/ws-connection/message-handler.ts` now silently auto-approves first-party iOS device pairing when shared token/password auth already succeeded, and the iOS app now identifies itself with the canonical `marv-ios` client id plus `ui`/`node` modes.
- Focused contract coverage now guards the drifted shapes in `src/core/gateway/server-methods/usage.test.ts`, `src/core/gateway/server.sessions.gateway-server-sessions-a.e2e.test.ts`, and the new iOS pairing path in `src/core/gateway/server.auth.e2e.test.ts`.
- Verified with:
  - `xcodebuild -project apps/ios/MarvCompanion.xcodeproj -scheme MarvCompanion -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build`
  - `pnpm vitest run src/core/gateway/server-methods/usage.test.ts`
  - `pnpm vitest run --config vitest.e2e.config.ts src/core/gateway/server.auth.e2e.test.ts src/core/gateway/server.ios-client-id.e2e.test.ts`
  - `pnpm vitest run --config vitest.e2e.config.ts src/core/gateway/server.sessions.gateway-server-sessions-a.e2e.test.ts -t 'lists and patches session store via sessions.* RPC'`
- Broader note:
  - `pnpm vitest run --config vitest.e2e.config.ts src/core/gateway/server.sessions.gateway-server-sessions-a.e2e.test.ts src/core/gateway/server.auth.e2e.test.ts src/core/gateway/server.ios-client-id.e2e.test.ts` still reports multiple pre-existing failures elsewhere in `server.sessions.gateway-server-sessions-a.e2e.test.ts`; those failures were outside this iOS follow-up scope, so I verified the specific updated `sessions.list` contract path with a targeted run instead of widening the task.

## iOS Local Deploy Follow-up Plan (2026-04-02)

- [x] Patch `scripts/ios-deploy.sh` to prefer the hardware UDID, add earlier deploy preflights, and keep repo-root deploy reliable.
- [x] Patch `scripts/ios-team-id.sh` to fall back to usable Xcode-managed provisioning profiles when plist/keychain discovery is incomplete.
- [x] Patch the iOS Swift sources so shared settings rows compile across files, settings storage stays Swift 6-friendly, and the UI fits shorter iPhone 11 Pro-height layouts.
- [x] Re-run targeted verification for shell syntax, package resolution, and a generic iOS device build.

## Review

- `scripts/ios-deploy.sh` now prefers the classic hardware UDID exposed by CoreDevice JSON, validates iPhoneOS platform/SDK availability before device selection, checks per-device readiness before build, and keeps `--dry-run` working end to end instead of failing after the mocked build step.
- `scripts/ios-team-id.sh` now falls back to local provisioning profiles under both Xcode UserData and MobileDevice profile directories, and the missing-directory path no longer trips `set -e`.
- `apps/shared/OpenClawKit/Package.swift` now checks for the optional test target relative to the manifest location instead of relying on the caller’s current working directory.
- The iOS app now compiles cleanly again: `KeyValueRow` is module-visible for `SettingsTab`, `CompanionSettingsStore` no longer keeps `UserDefaults.standard` in shared static stored state, and the list-based tabs use inline titles while the transcript editor relaxes its minimum height in compact vertical space for shorter iPhone layouts.
- Verified with:
  - `bash -n scripts/ios-deploy.sh scripts/ios-team-id.sh scripts/ios-configure-signing.sh`
  - `cd apps/shared/OpenClawKit && swift package dump-package`
  - `xcodebuild -project apps/ios/MarvCompanion.xcodeproj -scheme MarvCompanion -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build`
  - mocked `scripts/ios-team-id.sh` run returning `PROFIL1234` from a provisioning profile with keychain fallback disabled
  - mocked `scripts/ios-deploy.sh --yes --dry-run --force` path proving hardware UDID selection and explicit `-project .../MarvCompanion.xcodeproj` usage
  - checked local simulator inventory for `iPhone 11 Pro`; no matching simulator runtime was installed, so exact simulator rendering was not available for verification

## iOS Local Deploy Fixes Plan (2026-04-01)

- [x] Inspect the current iOS deploy, signing, and package setup and confirm the reported blockers in code.
- [x] Patch `scripts/ios-deploy.sh` so repo-root deploy targets the generated Xcode project and performs an early platform-support preflight.
- [x] Patch `apps/shared/OpenClawKit/Package.swift` so missing packaged test sources do not block local package resolution.
- [x] Patch `scripts/ios-team-id.sh` so Apple Development identities in the keychain are accepted by default.
- [x] Run targeted verification and add a short review note with the verified results.

## Review

- `scripts/ios-deploy.sh` now builds against `apps/ios/MarvCompanion.xcodeproj` explicitly, so repo-root deploy no longer depends on the current working directory containing an Xcode project.
- The deploy flow now compares the connected device OS version with the locally installed iOS SDK/platform support and fails early with an Xcode Components install hint when the device requires a newer iOS platform.
- `apps/shared/OpenClawKit/Package.swift` now omits `OpenClawKitTests` when the packaged snapshot does not include `Tests/OpenClawKitTests`, allowing project/package resolution to proceed.
- `scripts/ios-team-id.sh` now allows keychain-based Apple Development identity discovery by default, which unblocks local signing setups where Xcode account plist metadata is incomplete.
- Verified with:
  - `bash -n scripts/ios-deploy.sh scripts/ios-team-id.sh scripts/ios-configure-signing.sh`
  - `cd apps/shared/OpenClawKit && swift package dump-package`
  - `IOS_DEVELOPMENT_TEAM=ABCDEFGHIJ bash scripts/ios-open.sh --generate-only`
  - `xcodebuild -project apps/ios/MarvCompanion.xcodeproj -list`
  - mocked `scripts/ios-team-id.sh` run returning a keychain-only Apple Development team id
  - mocked `scripts/ios-deploy.sh --yes --dry-run` run showing the explicit `-project` build command
  - mocked `scripts/ios-deploy.sh --yes --dry-run` run failing early for a device on iOS 26.4 when local support only reached iOS 26.2

- [x] Map the current memory system from code, covering entrypoints, storage layers, read/write paths, and compaction/distillation.
- [x] Collect concrete file and line references for each major component and flow.
- [x] Call out any mismatches between docs/comments and current code behavior.
- [x] Add a short review note once the audit handoff is complete.

## Audit Notes

- Scope: code-only audit of the current memory system implementation for handoff.

## Review

- The current memory implementation is split across multiple persistence layers rather than a single subsystem: config-backed soul/identity/user sections, structured soul-memory SQLite (Memory Palace), builtin legacy memory index SQLite, optional QMD index state, task-context SQLite plus archive files, and experience markdown files.
- Agent recall paths are not uniform: `memory_search` queries active soul memory, archive, and legacy fallback; CLI search omits archive; gateway `memory.search` hits active soul memory only.
- The strongest implementation mismatches are:
  - builtin manager advertises FTS-only search but skips provider-less indexing, so fresh FTS-only indexes are not built
  - `memory_write` is described as a direct durable write, but it only queues distillation
  - weekly calibration requests recent episodic fragments, but the loader filters on a non-existent `record_kind = 'episodic'`
  - public soul-memory normalization erases the `"knowledge"` memory type back to `"episodic"` on reads

---

- [x] Fix the CLI startup crash by removing the problematic import cycle from the built entrypoint path.
- [x] Align installer docs with the actual supported `install.sh` and `install-cli.sh` interfaces.
- [x] Fix first-run and agent CLI docs/examples so they use valid commands.
- [x] Run source and packaged smoke checks after the fixes.
- [x] Audit the current memory system implementation across Soul/Experience/Context, markdown memory search, and auto-recall.
- [x] Trace the end-to-end read/write/maintenance flow from message ingest through retrieval, compaction, and distillation.
- [x] Identify severe risks in scope isolation, privacy boundaries, data consistency, and silent failure paths.
- [x] Summarize the verified architecture and highest-priority findings with file references.

## Review

- Root cause fixed by splitting OpenAI default model ids into a leaf constants module, removing the `model-picker -> onboard-auth.config-core -> auth-choice -> model-picker` cycle that caused the built CLI to throw before command execution.
- Installer and first-run docs now match the real CLI/script interfaces, including valid `marv agent` usage and supported installer flags/env vars.
- Commander negated-option parsing was corrected at the CLI boundary so `--no-open`, `--no-workspace-suggestions`, `--no-prefix-cwd`, and global `--no-color` behave as designed.
- Shell-completion checks no longer depend on opaque runtime imports. `doctor`, `update`, and the lazy `completion` subcommand now use bundle-safe module edges, so packaged builds do not fail with `ERR_MODULE_NOT_FOUND`.
- Verified with:
  - `pnpm vitest run src/commands/dashboard.e2e.test.ts src/cli/program/register.maintenance.test.ts src/cli/acp-cli.option-collisions.test.ts`
  - `pnpm vitest run src/cli/update-cli.test.ts src/cli/program/register.subclis.e2e.test.ts src/cli/program/register.maintenance.test.ts src/cli/acp-cli.option-collisions.test.ts`
  - `MARV_HOME=$(mktemp -d) pnpm marv --version`
  - `MARV_HOME=$(mktemp -d) pnpm marv dashboard --no-open`
  - `MARV_HOME=$(mktemp -d) pnpm marv doctor --non-interactive`
  - `MARV_HOME=$(mktemp -d) pnpm marv completion --help`
  - extracted tarball `node package/marv.mjs --version`
  - extracted tarball `node package/marv.mjs dashboard --no-open`
  - extracted tarball `node package/marv.mjs doctor --non-interactive`
  - extracted tarball `node package/marv.mjs completion --help`

## Memory Audit Plan

- [x] Inventory memory-related entrypoints: tool surfaces, gateway routes, CLI commands, and plugin hooks.
- [x] Trace scope derivation and filtering for writes, retrieval, and transcript/session lookup.
- [x] Inspect indexing paths for session transcripts and external files, including default directories and watchers.
- [x] Validate access-control boundaries for cross-session and cross-user access.
- [x] Record only concrete severe findings with file/line citations and supporting evidence.

## Memory Audit Review

- Confirmed a raw gateway/session isolation gap: `sessions.list`/`sessions.preview`/`sessions.resolve` and `chat.history` are only gated by generic gateway read scope, while the agent-facing session tools separately implement `tools.sessions.visibility` and agent-to-agent restrictions that the raw gateway methods do not enforce.
- Confirmed a raw memory isolation gap: `memory.list` and `memory.search` accept caller-controlled `agentId`, `sessionKey`, and arbitrary `scopes`, then query soul-memory directly without intersecting those inputs with any caller/session visibility boundary.
- Confirmed an MCP/QMD bypass path: `/mcp` is bearer-authenticated and dispatches straight into memory tools with caller-supplied `sessionKey`; `memory_get` then calls `manager.readFile()` without passing session context, and the QMD backend will read `qmd/<collection>/...` files directly, including exported session transcripts when `memory.qmd.sessions.enabled` is on.

## Review: Memory Audit

- Verified current memory architecture across `src/memory`, `src/agents`, `src/auto-reply`, `src/knowledge`, and gateway/MCP memory surfaces.
- Confirmed the system now spans multiple layers: config-backed soul/identity/user injection, structured Soul SQLite memory (Memory Palace), legacy/QMD markdown indexers, task-context stores, and `MARV_EXPERIENCE.md` / `MARV_CONTEXT.md`.
- Highest-risk findings:
  - `memory_write` is advertised and used as a structured factual memory writer, but actually routes into experience distillation, so scope/kind/factual persistence semantics do not match the implementation.
  - Active Soul recall is scored globally with scope penalties rather than hard scope filtering, which can surface unrelated-session memory under strong matches.
  - Runtime ingest stores assistant replies as facts and reinforcement raises retrieved items toward full confidence, creating a self-reinforcing false-memory path.
  - Knowledge vault indexing keys document chunks only by relative path, so same-path files across vaults can clobber each other.
  - Several consistency paths are brittle: FTS-only fallback indexing is broken, runtime ingest can silently lose writes under maintenance lock contention, and deep-consolidation fragment loading filters on the wrong field.

## Defect Fix Plan (2026-03-31)

- [x] Read `Plan/2026-03-31-code-defect-review.md` and confirm the highest-risk problem clusters in code.
- [x] Fix SSRF-guard inconsistencies for embeddings, reranker, and Ollama discovery so local/LAN services use the same guarded private-network path as local LLM inference.
- [x] Fix onboarding/provider configuration gaps that currently skip memory search setup or accept unreachable local/self-hosted model endpoints.
- [x] Fix deterministic config/command defects called out in the review where the current code is clearly wrong.
- [x] Add or update focused tests for the repaired paths.
- [x] Run targeted verification and summarize confirmed fixes plus any rejected findings.

## Review

- Confirmed and repaired the SSRF/private-network inconsistency across memory embedding requests, reranker calls, batch embedding HTTP paths, Ollama/vLLM discovery, and local/self-hosted endpoint verification. These paths now use the same guarded private-network fetch flow as local LLM inference instead of raw `fetch()`.
- Quickstart onboarding no longer skips memory search when a model has actually been configured. It now runs the memory-search step on that path, validates the configured embedding provider before continuing, and surfaces explicit fallback notes instead of silently degrading.
- Added a first-class Ollama onboarding/auth choice that configures the native Ollama provider directly and verifies the selected model against `/api/tags` before writing config. vLLM setup now verifies `/v1/models` and the selected model before persisting credentials/config.
- Fixed the deterministic command/config defects from the review that were real in current code: `channels add` no longer shadows `nextConfig`, and duplicated environment-variable branches in path resolution were collapsed to the intended single checks/fallbacks.
- Extended reranker config to accept explicit request headers so non-`Bearer` local/self-hosted reranker auth can be configured cleanly without patch behavior.
- Telegram token-file resolution now reports empty token files instead of silently falling through, and Telegram/Google Chat webhook URLs are validated at config-schema time as real `http(s)` URLs.
- `install/install.sh` now fails fast when `HOME` is missing instead of generating broken `/.npm-global` and shell-profile paths.
- Confirmed one reported issue as a false positive in current code: the SSRF hostname blocking for `.local` / `.internal` is already bypassed when `allowPrivateNetwork: true`, so no code change was needed there.
- Focused verification passed:
  - `pnpm vitest run src/memory/embeddings/embeddings-remote-fetch.test.ts src/memory/search/reranker.test.ts src/commands/vllm-setup.test.ts src/commands/ollama-setup.test.ts src/commands/auth-choice-options.e2e.test.ts src/core/config/paths.test.ts src/wizard/onboarding.test.ts src/memory/storage/local-llm-client.test.ts src/memory/embeddings/embeddings.test.ts src/commands/channels.add.test.ts`
  - `pnpm vitest run src/core/config/config-misc.test.ts src/memory/search/reranker.test.ts src/wizard/onboarding.test.ts src/commands/vllm-setup.test.ts src/commands/ollama-setup.test.ts`
  - `pnpm vitest run src/channels/telegram/token.test.ts src/core/config/telegram-webhook-secret.test.ts src/core/config/googlechat-webhook-url.test.ts src/core/config/config-misc.test.ts`
  - `pnpm tsgo`
  - `pnpm exec oxfmt --check src/core/config/zod-schema.agent-runtime.ts src/core/config/io.ts src/wizard/onboarding.ts src/commands/ollama-setup.ts src/commands/vllm-setup.ts src/memory/search/reranker.ts src/infra/net/private-network-fetch.ts`
  - `pnpm exec oxlint --type-aware src/core/config/zod-schema.agent-runtime.ts src/core/config/io.ts src/wizard/onboarding.ts src/commands/ollama-setup.ts src/commands/vllm-setup.ts src/memory/search/reranker.ts src/infra/net/private-network-fetch.ts`

## Skill System Optimization Plan (2026-03-31)

- [x] Implement `skill_view` and lazy-mode skill retrieval plumbing.
- [x] Add tool-aware conditional skill activation filtering.
- [x] Add canonical skill package metadata and normalization helpers for managed skills.
- [x] Implement normalized skill install/materialization flow without compatibility wrappers.
- [x] Extend skill usage records to capture outcomes and validated success.
- [x] Implement canonical skill crystallization so one validated successful adaptation rewrites the installed skill.
- [x] Make workspace skill sync preserve user-modified targets instead of destructive overwrite.
- [x] Add focused tests for lazy loading, activation filtering, normalization, crystallization, and sync behavior.
- [x] Run targeted verification and record results here.

## Review

- Added first-class skill lifecycle tools: `skill_view` for lazy-mode retrieval and `skill_crystallize` for verified canonical rewrites of skills.
- Skill metadata now supports activation gating and package-level sidecar metadata (`source`, `originHash`, `trust`, `adaptedAt`, `adaptedFrom`) outside the skill body.
- Skill eligibility now considers the current run's available tools, so browser-only or fallback skills stop polluting prompts when their tool prerequisites do not match.
- Skill usage records now track `lastUsedAt`, `successCount`, `failureCount`, `lastOutcome`, and `lastValidatedAt`, which supports usage-driven crystallization after one validated success.
- Workspace skill sync is no longer destructive: it writes a sync manifest, preserves locally modified target skills, and only replaces entries that still match the last synced hash.
- Focused verification passed:
  - `pnpm vitest run --config vitest.e2e.config.ts src/agents/skills.buildworkspaceskillsnapshot.e2e.test.ts src/agents/skills.build-workspace-skills-prompt.syncs-merged-skills-into-target-workspace.e2e.test.ts src/agents/tools/skill-tools.e2e.test.ts`
  - `pnpm vitest run --config vitest.e2e.config.ts src/agents/tools/policy/tool-policy.e2e.test.ts`
  - `pnpm tsgo`
- Broader spot checks surfaced pre-existing failures outside this skill work:
  - `pnpm vitest run --config vitest.e2e.config.ts src/agents/prompt/system-prompt.e2e.test.ts`
  - `pnpm vitest run --config vitest.e2e.config.ts src/agents/tools/pi-tools-agent-config.e2e.test.ts`

## Local Model Context Fix (2026-03-31)

- [x] Replace the custom local-provider `maxTokens: 4096` default with a usable value during onboarding/config generation.
- [x] Ensure local provider discovery keeps a usable default context window when `/v1/models` omits context metadata.
- [x] Add focused regression tests and run targeted verification for the missing-metadata local-model path.

## Review

- Custom OpenAI-compatible local providers now write `maxTokens: 16384` instead of `4096`, which keeps the fallback path above the agent's 16k minimum when upstream metadata is missing.
- Runtime local-model discovery now assigns a default `contextWindow: 128000` for local providers when `/v1/models` omits `context_window` and `max_model_len`, so discovered local models stay usable instead of appearing undersized.
- Added a regression test for runtime local-model discovery and a focused assertion for custom-provider onboarding defaults.
- Verified with:
  - `pnpm vitest run src/agents/model/runtime-model-registry.test.ts`
  - `pnpm vitest run --config vitest.e2e.config.ts src/agents/context-window-guard.e2e.test.ts`
  - `pnpm tsgo`
  - `pnpm exec oxfmt --check src/commands/onboard-custom.ts src/commands/onboard-custom.e2e.test.ts src/agents/model/runtime-model-registry.ts src/agents/model/runtime-model-registry.test.ts`
  - `pnpm exec tsx -e 'import { applyCustomApiConfig } from "./src/commands/onboard-custom.ts"; ...'` smoke check confirming custom-provider output now writes `contextWindow: 128000` and `maxTokens: 16384`

## Cron Mutation Policy Plan (2026-03-31)

- [x] Remove escalation blocking for `cron add/update/remove`.
- [x] Enrich cron mutation events so notifications can show readable job details.
- [x] Add non-blocking Web UI notifications for cron mutations.
- [x] Add Telegram cron mutation notifications for configured operator recipients.
- [x] Update focused tests and docs, then record verification results.

## Review

- `cron add/update/remove` no longer require permission escalation. The policy now treats cron mutations as notify-and-audit operations instead of approval-gated ones.
- Cron mutation events now carry readable metadata (`jobName`, `agentId`, `sessionKey`, `sessionTarget`, `deliveryMode`, `nextRunAtMs`) so operator surfaces can notify without reloading or re-querying first.
- Control UI now shows non-blocking toast notifications for cron adds, updates, and removals from gateway `cron` events.
- Telegram now forwards the same cron mutation events to configured operator chats, reusing `channels.telegram.execApprovals.approvers` as the recipient list for now.
- Docs were updated to remove the old claim that cron mutations need escalation and to note the notification-first behavior.
- Verified with:
  - `pnpm vitest run src/agents/tools/policy/escalation-policy.test.ts src/agents/tools/pi-tools.before-tool-call.e2e.test.ts src/channels/telegram/monitor/cron-mutations.test.ts ui/src/ui/controllers/cron-mutation-notice.test.ts ui/src/ui/app-gateway.node.test.ts`
  - `pnpm tsgo`
  - `pnpm exec oxfmt --check src/agents/tools/policy/escalation-policy.ts src/agents/tools/policy/escalation-policy.test.ts src/agents/tools/pi-tools.before-tool-call.e2e.test.ts src/cron/service/state.ts src/cron/service/ops.ts src/channels/telegram/bot.ts src/channels/telegram/monitor.ts src/channels/telegram/monitor/cron-mutations.ts src/channels/telegram/monitor/cron-mutations.test.ts ui/src/ui/app-gateway.ts ui/src/ui/app.ts ui/src/ui/app-view-state.ts ui/src/ui/app-render.ts ui/src/ui/controllers/cron-mutation-notice.ts ui/src/ui/controllers/cron-mutation-notice.test.ts ui/src/ui/views/cron-mutation-notice.ts ui/src/ui/app-gateway.node.test.ts`
  - `pnpm exec oxlint --type-aware src/agents/tools/policy/escalation-policy.ts src/agents/tools/policy/escalation-policy.test.ts src/agents/tools/pi-tools.before-tool-call.e2e.test.ts src/cron/service/state.ts src/cron/service/ops.ts src/channels/telegram/bot.ts src/channels/telegram/monitor.ts src/channels/telegram/monitor/cron-mutations.ts src/channels/telegram/monitor/cron-mutations.test.ts ui/src/ui/app-gateway.ts ui/src/ui/app.ts ui/src/ui/app-view-state.ts ui/src/ui/app-render.ts ui/src/ui/controllers/cron-mutation-notice.ts ui/src/ui/controllers/cron-mutation-notice.test.ts ui/src/ui/views/cron-mutation-notice.ts ui/src/ui/app-gateway.node.test.ts`

## Gateway Restart Validation Plan (2026-04-01)

- [x] Trace the current `marv gateway restart` flow and identify the exact success path that can report success while an old listener still serves the port.
- [x] Add a post-restart verification step that proves the old gateway listener is gone and the serving process matches the configured gateway command.
- [x] Improve startup logs so the serving gateway reports version/build identity and PID clearly at boot.
- [x] Add focused tests for restart verification and startup logging.
- [x] Run targeted verification and record the results.

## Review

- `marv gateway restart` no longer reports success immediately after `launchctl/systemd/schtasks` returns. The CLI now captures the pre-restart gateway PID/port state, polls the configured gateway port after restart, and fails if the old PID still survives or if the port is held by a process that does not match the configured gateway command.
- When port inspection cannot identify the serving process but the service runtime PID clearly changed and the port is busy, restart succeeds with a warning instead of a silent false positive.
- Gateway startup logs now include a `build:` line with version, executable path, and entrypoint path before the listen address line, making it much easier to confirm which build is actually serving.
- Verified with:
  - `pnpm vitest run src/cli/daemon-cli/lifecycle-core.test.ts src/cli/daemon-cli/lifecycle.test.ts src/core/gateway/server-startup-log.test.ts`
  - `pnpm exec oxfmt --check src/cli/daemon-cli/lifecycle-core.ts src/cli/daemon-cli/lifecycle.ts src/core/gateway/server-startup-log.ts src/cli/daemon-cli/lifecycle-core.test.ts src/cli/daemon-cli/lifecycle.test.ts src/core/gateway/server-startup-log.test.ts`
  - `pnpm exec oxlint --type-aware src/cli/daemon-cli/lifecycle-core.ts src/cli/daemon-cli/lifecycle.ts src/core/gateway/server-startup-log.ts src/cli/daemon-cli/lifecycle-core.test.ts src/cli/daemon-cli/lifecycle.test.ts src/core/gateway/server-startup-log.test.ts`
  - `pnpm tsgo`

## Tavily Config Schema Plan (2026-04-01)

- [x] Confirm Tavily web search runtime support and identify the config-layer gaps.
- [x] Add Tavily support to the config schema and config types.
- [x] Update config help and web tool docs so Tavily is listed as a supported provider.
- [x] Add focused Tavily validation tests and record verification results.

## Review

- Added `tavily` to the strict config schema for `tools.web.search.provider`, and added a strict `tools.web.search.tavily` object with `apiKey`, `searchDepth`, and `includeAnswer`.
- Updated the config TypeScript types and config help/labels so the config surface now matches the existing Tavily runtime support.
- Updated the web tools docs to list Tavily as a supported provider, include a Tavily config example, and note the Tavily API key path in the requirements section.
- Added focused validation coverage for Tavily config acceptance plus invalid `searchDepth` rejection.
- Verified with:
  - `pnpm vitest run src/core/config/config-misc.test.ts`
  - `pnpm tsgo`
  - `pnpm exec oxfmt --check src/core/config/zod-schema.agent-runtime.ts src/core/config/types.tools.ts src/core/config/schema.help.ts src/core/config/schema.labels.ts src/core/config/config-misc.test.ts docs/tools/web.md`
  - `pnpm exec oxlint --type-aware src/core/config/zod-schema.agent-runtime.ts src/core/config/types.tools.ts src/core/config/schema.help.ts src/core/config/schema.labels.ts src/core/config/config-misc.test.ts`
  - `HOME="$(mktemp -d)" pnpm marv config set tools.web.search.provider tavily`
  - `HOME="$(mktemp -d)" pnpm marv config set tools.web.search.tavily.searchDepth advanced`

## Local Context Window Guard Removal Plan (2026-04-01)

- [x] Trace the embedded-runner context-window precheck that blocks small/unknown local model metadata.
- [x] Remove the runtime hard block so context window stays advisory instead of gating execution.
- [x] Add focused regression coverage for local/baseUrl providers with undersized or missing context metadata.
- [x] Run targeted verification and record the results.

## Review

- Embedded runner no longer fail-fast rejects models just because their reported `contextWindow` is below 16k. The value is still used for advisory warnings and budgeting, but not as a preflight availability gate.
- Added a regression test proving a local `baseUrl` provider with `contextWindow: 4096` and `maxTokens: 4096` still completes a run instead of entering failover.
- Verified with:
  - `pnpm vitest run --config vitest.e2e.config.ts src/agents/context-window-guard.e2e.test.ts`
  - `pnpm vitest run --config vitest.e2e.config.ts src/agents/runner/pi-embedded-runner.e2e.test.ts -t "does not pre-block local baseUrl providers when context metadata is tiny"`
  - `pnpm exec oxfmt --check src/agents/context-window-guard.ts src/agents/pi-embedded-runner/run.ts src/agents/context-window-guard.e2e.test.ts src/agents/runner/pi-embedded-runner.e2e.test.ts`
  - `pnpm tsgo`
- Broader suite note:
  - `pnpm vitest run --config vitest.e2e.config.ts src/agents/context-window-guard.e2e.test.ts src/agents/runner/pi-embedded-runner.e2e.test.ts` still hits a pre-existing failure in `persists prompt transport errors as transcript entries` (`ENOENT` on the session file after an unhandled `transport failed` rejection).

## Models List Base URL Visibility Fix (2026-04-01)

- [x] Keep configured keyless `baseUrl` custom provider models visible when runtime discovery omits them.
- [x] Preserve those models across fresh auth-sync reloads and fallback availability heuristics.
- [x] Add focused regression coverage for the config-backed listing path.

## Review

- `models list` now merges configured keyless `baseUrl` provider models into the registry-backed output instead of hiding them whenever runtime discovery omits `/v1/models` metadata.
- The availability/auth sync path now treats those configured local providers as usable without requiring an API key, so a fresh reload keeps them visible instead of marking them missing.
- Added focused regression coverage in `models list` and auth-sync tests for the config-backed fallback path.
