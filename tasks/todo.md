# Todo

- [x] Map the current memory system from code, covering entrypoints, storage layers, read/write paths, and compaction/distillation.
- [x] Collect concrete file and line references for each major component and flow.
- [x] Call out any mismatches between docs/comments and current code behavior.
- [x] Add a short review note once the audit handoff is complete.

## Audit Notes

- Scope: code-only audit of the current memory system implementation for handoff.

## Review

- The current memory implementation is split across multiple persistence layers rather than a single subsystem: config-backed P0 sections, structured soul-memory SQLite, builtin legacy memory index SQLite, optional QMD index state, task-context SQLite plus archive files, and experience markdown files.
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
- Confirmed the system now spans multiple layers: config-backed P0/Soul injection, structured Soul SQLite memory, legacy/QMD markdown indexers, task-context stores, and `MARV_EXPERIENCE.md` / `MARV_CONTEXT.md`.
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
