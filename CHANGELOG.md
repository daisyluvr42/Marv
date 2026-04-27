# Changelog

Docs: https://github.com/daisyluvr42/Marv/tree/main/docs

## Unreleased

## 2026.4.27

### Changes

- Models: honor configured primary and fallback routes consistently across agent runs, followups, status output, and session execution.
- Local Models: when a configured local model endpoint is temporarily unavailable, fall back to the next configured model without deleting the local selection.

### Fixes

- Cron: use a job's cron timezone when building scheduled-run time context and include an explicit local date for daily reports.
- Models: treat connection and unavailable-model failures as fallbackable runtime errors, while keeping public custom `baseUrl` providers distinct from local/private endpoints.

## 2026.4.24

### Fixes

- Memory: prevent pre-compaction memory flushes from livelocking when persisting session summaries.
- Models: keep runtime model availability and selected session model status fresh across provider/config changes.
- Chat: hide inbound metadata markers from user-visible transcript displays.
- Sub-agents: avoid selecting models already marked unavailable for delegated runs.

## 2026.4.3

### Changes

- Local Models: custom providers with a `baseUrl` are now classified as local in the runtime candidate pool, prioritized over cloud models in fallback ordering.
- CLI: add `marv models pool list` and `marv models pool clear [model]` to inspect and manage runtime model availability state, useful for recovering local models marked unavailable after slow cold starts.
- Local Models: alias re-assignment within the same provider now transfers the alias automatically instead of rejecting the change.
- Update: `marv update` now detects when run from a Marv git checkout and uses the git update path automatically; stored `update.channel=dev` is honored without requiring `--channel dev` each time.

### Fixes

- Models: fix session model drift where an explicitly selected model matching the configured default had its override silently cleared, causing later auto-runs to fall back to the first runnable cloud candidate (e.g. Gemini) instead of staying on the pinned local model.
- Update: fix stored `update.channel=dev` being ignored on subsequent `marv update` calls, requiring `--channel dev` to be passed again each time.
- Update: fix default git checkout target resolving to `~/.marv` (state directory) instead of a dedicated `~/.marv/source` path.
- Update: when running from a Marv git checkout, `marv update` now detects and uses the cwd checkout directly.

## 2026.3.31

### Changes

- Sub-agents: add goal-driven orchestration loop that evaluates delegated sub-agent output against the parent's success criteria, delivers structured feedback, and iterates until accepted or budget exhausted.
- Migration: `marv migrate export --scopes memory` now includes Soul.md identity files and Experience/Context files alongside vector databases, enabling complete one-command memory portability.
- Onboarding/Local Models: add first-class Ollama setup, verify vLLM/Ollama endpoints before saving config, and include memory-search setup in quickstart when a model is configured.
- Memory: make `memory_write` persist structured Soul entries directly, hard-filter unrelated scoped recall, and isolate knowledge-vault document scopes per vault.

### Fixes

- Build/CLI: fix packaged and built CLI startup initialization cycles so release artifacts boot cleanly again instead of failing before command execution.
- CLI: restore negated option handling for `--no-open`, `--no-workspace-suggestions`, `--no-prefix-cwd`, and `--no-color`.
- Build/CLI: keep `doctor` and `completion` working in packaged builds instead of importing missing bundle-time CLI modules.
- Install/Docs: align installer and first-use guidance with the actual supported installer flags and `marv agent` command behavior.
- Local Models: route local/LAN embedding, reranker, batch embedding, model discovery, and onboarding verification through the guarded private-network fetch path.
- Memory: stop runtime-event memories from being reinforced into high-confidence facts, fix episodic fragment loading, and add SQLite busy timeouts across Soul maintenance jobs.

## 2026.3.16

### Changes

- Memory: add P3 episodic compaction pipeline that clusters similar footage into distilled P2 semantic knowledge nodes, replacing blind confidence decay with structured knowledge extraction.
- Memory: add semantic evolution so compacted knowledge updates when new explicit evidence arrives, retiring old versions with full lineage tracking via supersedes edges.
- Memory: add temporal filtering to search queries, excluding retired semantic nodes by default and supporting point-in-time retrieval via the new `temporalMs` parameter.
- Memory: add conflict-aware retrieval that annotates search results with unresolved conflict IDs, enabling upstream consumers to surface knowledge contradictions.
- Memory: add per-item archival with session tag grouping, preserving original footage content and blood lineage metadata instead of merging into lossy episode summaries.

### Fixes

- Memory: prevent orphan P3 episodic items from accumulating indefinitely via a configurable safety valve (orphanAgeDays, default 60 days).
- Memory: exclude P3 episodic items from dedupe, consolidation, promotion, and confidence decay pipelines when compaction is enabled, preventing cross-pipeline interference.

## 2026.3.15

### Changes

- Web UI: reshape the web control surface into an operations-first console with a cleaner sidebar, a real overview page, and clearer sections for operations, channels, agents, workspace, chat, and settings.
- Models/Auth: replace the legacy `agents.defaults.models` flow with runtime selections plus model metadata, so newly configured providers can contribute discovered models without awkward allowlist plumbing.
- Telegram/Onboarding: validate bot tokens during onboarding and require a retry when Telegram rejects the token, while clarifying follow-up DM access setup.
- Install/Docs: clarify source-vs-global deployment guidance, remove stale installer links, and refresh release/install documentation around the current project URLs.

### Fixes

- Build/CLI: fix release-build startup initialization cycles so the compiled `dist` CLI boots cleanly again instead of failing before command execution.
- Models/TUI: keep explicit primary-model choices stable in model resolution and TUI displays instead of drifting to the first default candidate.
- Release notes: replace inherited upstream-style 2026.3.14 notes with this project's actual shipped changes.

## 2026.3.14

### Changes

- Agents/Memory: add soul-memory and deep-consolidation pipelines, archive session episodes, and expand local-first recall and knowledge tooling.
- Agents: add lightweight goal-loop steering, tool-synthesis fallback, self-inspection controls/reporting, and an external CLI fallback tool.
- Prompting: add response-language auto-detection, persist reply-language context in memory, and reduce prompt duplication for better cache friendliness.
- Models/CLI: unify provider auth/config setup under `marv models auth`, add runtime model-registry pool management, and centralize command policy definitions.
- Browser/Tools: add pinned tabs, page-text extraction, and a managed CLI synthesis workflow.
- Web UI: add dashboard status cards and workspace tabs.
- iOS: add companion app source and setup scripts, plus shared OpenClawKit package sources for the mobile app workspace.
- Onboarding/Gateway: add staged onboarding guidance, heartbeat proactive planning scaffolding, and CLI tool event streaming coverage.
- Media/Build: formalize storage targets and add plugin SDK aliases for `agentmarv` extensions.

### Fixes

- Memory: fix session-memory and digest-buffer regressions uncovered by the expanded memory pipeline work.
- Prompting: sync Chinese reply prompts with runtime model status and keep the new language-aware prompt flow consistent.
- Telegram: fix exec approval message formatting in agent conversations.
- Security/Web UI: harden approval handling, web UI trust flow, and deploy-recovery paths around signed approvals.
- Release/Build: fix the typecheck blockers that were preventing the 2026.3.14 release build from passing.
- Plugins: add `before_agent_start` model/provider overrides before resolution. (#18568) Thanks @natefikru.
- Mattermost: add emoji reaction actions plus reaction event notifications, including an explicit boolean `remove` flag to avoid accidental removals. (#18608) Thanks @echo931.
- Memory/Search: add FTS fallback plus query expansion for memory search. (#18304) Thanks @irchelper.
- Agents/Models: support per-model `thinkingDefault` overrides in model config. (#18152) Thanks @wu-tian807.
- Agents: enable `llms.txt` discovery in default behavior. (#18158) Thanks @yolo-maxi.
- Extensions/Auth: add OpenAI Codex CLI auth provider integration. (#18009) Thanks @jiteshdhamaniya.
- Feishu: add Bitable create-app/create-field tools for automation workflows. (#17963) Thanks @gaowanqi08141999.
- Docker: add optional `MARV_INSTALL_BROWSER` build arg to preinstall Chromium + Xvfb in the Docker image, avoiding runtime Playwright installs. (#18449)

### Fixes

- Tests/Telegram: add regression coverage for command-menu sync that asserts all `setMyCommands` entries are Telegram-safe and hyphen-normalized across native/custom/plugin command sources. (#19703) Thanks @obviyus.
- Agents/Image: collapse resize diagnostics to one line per image and include visible pixel/byte size details in the log message for faster triage.
- Agents/Subagents: preemptively guard accumulated tool-result context before model calls by truncating oversized outputs and compacting oldest tool-result messages to avoid context-window overflow crashes. Thanks @tyler6204.
- Agents/Subagents/CLI: fail `sessions_spawn` when subagent model patching is rejected, allow subagent model patch defaults from `subagents.model`, and keep `sessions list`/`status` model reporting aligned to runtime model resolution. (#18660) Thanks @robbyczgw-cla.
- Agents/Subagents: add explicit subagent guidance to recover from `[compacted: tool output removed to free context]` / `[truncated: output exceeded context limit]` markers by re-reading with smaller chunks instead of full-file `cat`. Thanks @tyler6204.
- Agents/Tools: make `read` auto-page across chunks (when no explicit `limit` is provided) and scale its per-call output budget from model `contextWindow`, so larger contexts can read more before context guards kick in. Thanks @tyler6204.
- Agents/Tools: strip duplicated `read` truncation payloads from tool-result `details` and make pre-call context guarding account for heavy tool-result metadata, so repeated `read` calls no longer bypass compaction and overflow model context windows. Thanks @tyler6204.
- Reply threading: keep reply context sticky across streamed/split chunks and preserve `replyToId` on all chunk sends across shared and channel-specific delivery paths (including iMessage, BlueBubbles, Telegram, Discord, and Matrix), so follow-up bubbles stay attached to the same referenced message. Thanks @tyler6204.
- Gateway/Agent: defer transient lifecycle `error` snapshots with a short grace window so `agent.wait` does not resolve early during retry/failover. Thanks @tyler6204.
- Gateway/Presence: centralize presence snapshot broadcasts and unify runtime version precedence (`MARV_VERSION` > `MARV_SERVICE_VERSION` > `npm_package_version`) so self-presence and websocket `hello-ok` report consistent versions.
- Hooks/Automation: bridge outbound/inbound message lifecycle into internal hook events (`message:received`, `message:sent`) with session-key correlation guards, while keeping per-payload success/error reporting accurate for chunked and best-effort deliveries. (PR #9387)
- Media understanding: honor `agents.defaults.imageModel` during auto-discovery so implicit image analysis uses configured primary/fallback image models. (PR #7607)
- iOS/Onboarding: stop auth Step 3 retry-loop churn by pausing reconnect attempts on unauthorized/missing-token gateway errors and keeping auth/pairing issue state sticky during manual retry. (#19153) Thanks @mbelinky.
- Voice-call: auto-end calls when media streams disconnect to prevent stuck active calls. (#18435) Thanks @JayMishra-source.
- Voice call/Gateway: prevent overlapping closed-loop turn races with per-call turn locking, route transcript dedupe via source-aware fingerprints with strict cache eviction bounds, and harden `voicecall latency` stats for large logs without spread-operator stack overflow. (#19140) Thanks @mbelinky.
- iOS/Chat: route ChatSheet RPCs through the operator session instead of the node session to avoid node-role authorization failures for `chat.history`, `chat.send`, and `sessions.list`. (#19320) Thanks @mbelinky.
- macOS/Update: correct the Sparkle appcast version for 2026.2.15 so updates are offered again. (#18201)
- Gateway/Auth: clear stale device-auth tokens after device token mismatch errors so re-paired clients can re-auth. (#18201)
- Telegram: enable DM voice-note transcription with CLI fallback handling. (#18564) Thanks @thhuang.
- Telegram/Polls: restore Telegram poll action wiring in channel handlers. (#18122) Thanks @akyourowngames.
- WebChat: strip reply/audio directive tags from rendered chat output. (#18093) Thanks @aldoeliacim.
- Discord: honor configured HTTP proxy for app-id and allowlist REST resolution. (#17958) Thanks @k2009.
- BlueBubbles: add fallback path to recover outbound `message_id` from `fromMe` webhooks when platform message IDs are missing. Thanks @tyler6204.
- BlueBubbles: match outbound message-id fallback recovery by chat identifier as well as account context. Thanks @tyler6204.
- BlueBubbles: include sender identifier in untrusted conversation metadata for conversation info payloads. Thanks @tyler6204.
- Security/Exec: fix the OC-09 credential-theft path via environment-variable injection. (#18048) Thanks @aether-ai-agent.
- Security/Config: confine `$include` resolution to the top-level config directory, harden traversal/symlink checks with cross-platform-safe path containment, and add doctor hints for invalid escaped include paths. (#18652) Thanks @aether-ai-agent.
- Providers: improve error messaging for unconfigured local `ollama`/`vllm` providers. (#18183) Thanks @arosstale.
- TTS: surface all provider errors instead of only the last error in aggregated failures. (#17964) Thanks @ikari-pl.
- CLI/Doctor/Configure: skip gateway auth checks for loopback-only setups. (#18407) Thanks @sggolakiya.
- CLI/Doctor: reconcile gateway service-token drift after re-pair flows. (#18525) Thanks @norunners.
- Process/Windows: disable detached spawn in exec runs to prevent empty command output. (#18067) Thanks @arosstale.
- Process: gracefully terminate process trees with SIGTERM before SIGKILL. (#18626) Thanks @sauerdaniel.
- Sessions/Windows: use atomic session-store writes to prevent context loss on Windows. (#18347) Thanks @twcwinston.
- Agents/Image: validate base64 image payloads before provider submission. (#18263) Thanks @sriram369.
- Models CLI: validate catalog entries in `marv models set`. (#18129) Thanks @carrotRakko.
- Usage: isolate last-turn totals in token usage reporting to avoid mixed-turn totals. (#18052) Thanks @arosstale.
- Cron: resolve `accountId` from agent bindings in isolated sessions. (#17996) Thanks @simonemacario.
- Gateway/HTTP: preserve unbracketed IPv6 `Host` headers when normalizing requests. (#18061) Thanks @Clawborn.
- Sandbox: fix workspace-directory orphaning during SHA-1 -> SHA-256 slug migration. (#18523) Thanks @yinghaosang.
- Ollama/Qwen: handle Qwen 3 reasoning field format in Ollama responses. (#18631) Thanks @mr-sk.
- OpenAI/Transcripts: always drop orphaned reasoning blocks from transcript repair. (#18632) Thanks @TySabs.
- Fix types in all tests. Typecheck the whole repository.
- Gateway/Channels: wire `gateway.channelHealthCheckMinutes` into strict config validation, treat implicit account status as managed for health checks, and harden channel auto-restart flow (preserve restart-attempt caps across crash loops, propagate enabled/configured runtime flags, and stop pending restart backoff after manual stop). Thanks @Monad_lab.
- Gateway/WebChat: hard-cap `chat.history` oversized payloads by truncating high-cost fields and replacing over-budget entries with placeholders, so history fetches stay within configured byte limits and avoid chat UI freezes. (#18505)
- UI/Usage: replace lingering undefined `var(--text-muted)` usage with `var(--muted)` in usage date-range and chart styles to keep muted text visible across themes. (#17975) Thanks @jogelin.
- UI/Usage: preserve selected-range totals when timeline data is downsampled by bucket-aggregating timeseries points (instead of dropping intermediate points), so filtered tokens/cost stay accurate. (#17959) Thanks @jogelin.
- UI/Sessions: refresh the sessions table only after successful deletes and preserve delete errors on cancel/failure paths, so deleted sessions disappear automatically without masking delete failures. (#18507)
- Scripts/UI/Windows: fix `pnpm ui:*` spawn `EINVAL` failures by restoring shell-backed launch for `.cmd`/`.bat` runners, narrowing shell usage to launcher types that require it, and rejecting unsafe forwarded shell metacharacters in UI script args. (#18594)
- Hooks/Session-memory: recover `/new` conversation summaries when session pointers are reset-path or missing `sessionFile`, and consistently prefer the newest `.jsonl.reset.*` transcript candidate for fallback extraction. (#18088)
- Auto-reply/Sessions: prevent stale thread ID leakage into non-thread sessions so replies stay in the main DM after topic interactions. (#18528) Thanks @j2h4u.
- Slack: restrict forwarded-attachment ingestion to explicit shared-message attachments and skip non-Slack forwarded `image_url` fetches, preventing non-forward attachment unfurls from polluting inbound agent context while preserving forwarded message handling.
- Feishu: detect bot mentions in post messages with embedded docs when `message.mentions` is empty. (#18074) Thanks @popomore.
- Agents/Sessions: align session lock watchdog hold windows with run and compaction timeout budgets (plus grace), preventing valid long-running turns from being force-unlocked mid-run while still recovering hung lock owners. (#18060)
- Cron: preserve default model fallbacks for cron agent runs when only `model.primary` is overridden, so failover still follows configured fallbacks unless explicitly cleared with `fallbacks: []`. (#18210) Thanks @mahsumaktas.
- Cron: route text-only announce output through the main session announce flow via runSubagentAnnounceFlow so cron text-only output remains visible to the initiating session. Thanks @tyler6204.
- Cron: treat `timeoutSeconds: 0` as no-timeout (not clamped to 1), ensuring long-running cron runs are not prematurely terminated. Thanks @tyler6204.
- Cron announce injection now targets the session determined by delivery config (`to` + channel) instead of defaulting to the current session. Thanks @tyler6204.
- Cron/Heartbeat: canonicalize session-scoped reminder `sessionKey` routing and preserve explicit flat `sessionKey` cron tool inputs, preventing enqueue/wake namespace drift for session-targeted reminders. (#18637) Thanks @vignesh07.
- Cron/Webhooks: reuse existing session IDs for webhook/cron runs when the session key is stable and still fresh, preserving conversation history. (#18031) Thanks @Operative-001.
- Cron: prevent spin loops when cron jobs complete within the scheduled second by advancing the next run and enforcing a minimum refire gap. (#18073) Thanks @widingmarcus-cyber.
- MarvKit/iOS ChatUI: accept canonical session-key completion events for local pending runs and preserve message IDs across history refreshes, preventing stuck "thinking" state and message flicker after gateway replies. (#18165) Thanks @mbelinky.
- iOS/Onboarding: add QR-first onboarding wizard with setup-code deep link support, pairing/auth issue guidance, and device-pair QR generation improvements for Telegram/Web/TUI fallback flows. (#18162) Thanks @mbelinky and @Marvae.
- iOS/Gateway: stabilize connect/discovery state handling, add onboarding reset recovery in Settings, and fix iOS gateway-controller coverage for command-surface and last-connection persistence behavior. (#18164) Thanks @mbelinky.
- iOS/Talk: harden mobile talk config handling by ignoring redacted/env-placeholder API keys, support secure local keychain override, improve accessibility motion/contrast behavior in status UI, and tighten ATS to local-network allowance. (#18163) Thanks @mbelinky.
- iOS/Location: restore the significant location monitor implementation (service hooks + protocol surface + ATS key alignment) after merge drift so iOS builds compile again. (#18260) Thanks @ngutman.
- iOS/Signing: auto-select local Apple Development team during iOS project generation/build, prefer the canonical Marv team when available, and support local per-machine signing overrides without committing team IDs. (#18421) Thanks @ngutman.
- Discord/Telegram: make per-account message action gates effective for both action listing and execution, and preserve top-level gate restrictions when account overrides only specify a subset of `actions` keys (account key -> base key -> default fallback). (#18494)
- Telegram: keep DM-topic replies and draft previews in the originating private-chat topic by preserving positive `message_thread_id` values for DM threads. (#18586) Thanks @sebslight.
- Telegram: preserve private-chat topic `message_thread_id` on outbound sends (message/sticker/poll), keep thread-not-found retry fallback, and avoid masking `chat not found` routing errors. (#18993) Thanks @obviyus.
- Discord: prevent duplicate media delivery when the model uses the `message send` tool with media, by skipping media extraction from messaging tool results since the tool already sent the message directly. (#18270)
- Discord: route `audioAsVoice` auto-replies through the voice message API so opt-in audio renders as voice messages. (#18041) Thanks @zerone0x.
- Discord: skip auto-thread creation in forum/media/voice/stage channels and keep group session last-route metadata fresh to avoid invalid thread API errors and lost follow-up sends. (#18098) Thanks @Clawborn.
- Discord/Commands: normalize `commands.allowFrom` entries with `user:`/`discord:`/`pk:` prefixes and `<@id>` mentions so command authorization matches Discord allowlist behavior. (#18042)
- Telegram: keep draft-stream preview replies attached to the user message for `replyToMode: "all"` in groups and DMs, preserving threaded reply context from preview through finalization. (#17880) Thanks @yinghaosang.
- Telegram: prevent streaming final replies from being overwritten by later final/error payloads, and suppress fallback tool-error warnings when a recovered assistant answer already exists after tool calls. (#17883) Thanks @Marvae and @obviyus.
- Telegram: debounce the first draft-stream preview update (30-char threshold) and finalize short responses by editing the stop-time preview message, improving first push notifications and avoiding duplicate final sends. (#18148) Thanks @Marvae.
- Telegram: disable block streaming when `channels.telegram.streamMode` is `off`, preventing newline/content-block replies from splitting into multiple messages. (#17679) Thanks @saivarunk.
- Telegram: keep `streamMode: "partial"` draft previews in a single message across assistant-message/reasoning boundaries, preventing duplicate preview bubbles during partial-mode tool-call turns. (#18956) Thanks @obviyus.
- Telegram: normalize native command names for Telegram menu registration (`-` -> `_`) to avoid `BOT_COMMAND_INVALID` command-menu wipeouts, and log failed command syncs instead of silently swallowing them. (#19257) Thanks @akramcodez.
- Telegram: route non-abort slash commands on the normal chat/topic sequential lane while keeping true abort requests (`/stop`, `stop`) on the control lane, preventing command/reply race conditions from control-lane bypass. (#17899) Thanks @obviyus.
- Telegram: ignore `<media:...>` placeholder lines when extracting `MEDIA:` tool-result paths, preventing false local-file reads and dropped replies. (#18510) Thanks @yinghaosang.
- Telegram: skip retries when inbound media `getFile` fails with Telegram's 20MB limit and continue processing message text, avoiding dropped messages for oversized attachments. (#18531) Thanks @brandonwise.
- Telegram: clear stored polling offsets when bot tokens change or accounts are deleted, preventing stale offsets after token rotations. (#18233)
- Telegram: enable `autoSelectFamily` by default on Node.js 22+ so IPv4 fallback works on broken IPv6 networks. (#18272) Thanks @nacho9900.
- Auto-reply/TTS: keep tool-result media delivery enabled in group chats and native command sessions (while still suppressing tool summary text) so `NO_REPLY` follow-ups do not drop successful TTS audio. (#17991) Thanks @zerone0x.
- Agents/Tools: deliver tool-result media even when verbose tool output is off so media attachments are not dropped. (#16679)
- Discord: optimize reaction notification handling to skip unnecessary message fetches in `off`/`all`/`allowlist` modes, streamline reaction routing, and improve reaction emoji formatting. (#18248) Thanks @thewilloftheshadow and @victorGPT.
- CLI/Pairing: make `marv qr --remote` prefer `gateway.remote.url` over tailscale/public URL resolution and register the `marv clawbot qr` legacy alias path. (#18091)
- CLI/QR: restore fail-fast validation for `marv qr --remote` when neither `gateway.remote.url` nor tailscale `serve`/`funnel` is configured, preventing unusable remote pairing QR flows. (#18166) Thanks @mbelinky.
- CLI: fix parent/subcommand option collisions across gateway, daemon, update, ACP, and browser command flows, while preserving legacy `browser set headers --json <payload>` compatibility.
- CLI/Doctor: ensure `marv doctor --fix --non-interactive --yes` exits promptly after completion so one-shot automation no longer hangs. (#18502)
- CLI/Doctor: auto-repair `dmPolicy="open"` configs missing wildcard allowlists and write channel-correct repair paths (including `channels.googlechat.dm.allowFrom`) so `marv doctor --fix` no longer leaves Google Chat configs invalid after attempted repair. (#18544)
- CLI/Doctor: detect gateway service token drift when the gateway token is only provided via environment variables, keeping service repairs aligned after token rotation.
- Gateway/Update: prevent restart crash loops after failed self-updates by restarting only on successful updates, stopping early on failed install/build steps, and running `marv doctor --fix` during updates to sanitize config. (#18131) Thanks @RamiNoodle733.
- Gateway/Update: preserve update.run restart delivery context so post-update status replies route back to the initiating channel/thread. (#18267) Thanks @yinghaosang.
- CLI/Update: run a standalone restart helper after updates, honoring service-name overrides and reporting restart initiation separately from confirmed restarts. (#18050)
- CLI/Daemon: warn when a gateway restart sees a stale service token so users can reinstall with `marv gateway install --force`, and skip drift warnings for non-gateway service restarts. (#18018)
- CLI/Daemon: prefer the active version-manager Node when installing daemons and include macOS version-manager bin directories in the service PATH so launchd services resolve user-managed runtimes.
- CLI/Status: fix `marv status --all` token summaries for bot-token-only channels so Mattermost/Zalo no longer show a bot+app warning. (#18527) Thanks @echo931.
- CLI/Configure: make the `/model picker` allowlist prompt searchable with tokenized matching in `marv configure` so users can filter huge model lists by typing terms like `gpt-5.2 openai/`. (#19010) Thanks @bjesuiter.
- CLI/Message: preserve `--components` JSON payloads in `marv message send` so Discord component payloads are no longer dropped. (#18222) Thanks @saurabhchopade.
- Voice Call: add an optional stale call reaper (`staleCallReaperSeconds`) to end stuck calls when enabled. (#18437)
- Auto-reply/Subagents: propagate group context (`groupId`, `groupChannel`, `space`) when spawning via `/subagents spawn`, matching tool-triggered subagent spawn behavior.
- Subagents: route nested announce results back to the parent session after the parent run ends, falling back only when the parent session is deleted. (#18043) Thanks @tyler6204.
- Subagents: cap announce retry loops with max attempts and expiry to prevent infinite retry spam after deferred announces. (#18444)
- Agents/Tools/exec: add a preflight guard that detects likely shell env var injection (e.g. `$DM_JSON`, `$TMPDIR`) in Python/Node scripts before execution, preventing recurring cron failures and wasted tokens when models emit mixed shell+language source. (#12836)
- Agents/Tools/exec: treat normal non-zero exit codes as completed and append the exit code to tool output to avoid false tool-failure warnings. (#18425)
- Agents/Tools: make loop detection progress-aware and phased by hard-blocking known `process(action=poll|log)` no-progress loops, warning on generic identical-call repeats, warning + no-progress-blocking ping-pong alternation loops (10/20), coalescing repeated warning spam into threshold buckets (including canonical ping-pong pairs), adding a global circuit breaker at 30 no-progress repeats, and emitting structured diagnostic `tool.loop` warning/error events for loop actions. (#16808) Thanks @akramcodez and @beca-oc.
- Agents/Hooks: preserve the `before_tool_call` wrapped-marker across abort-signal tool wrapping so the hook runs once per tool call in normal agent sessions. (#16852) Thanks @sreuter.
- Agents/Tests: add `before_message_write` persistence regression coverage for block/mutate behavior (including synthetic tool-result flushes) and thrown-hook fallback persistence. (#18197) Thanks @shakkernerd
- Agents/Tools: scope the `message` tool schema to the active channel so Telegram uses `buttons` and Discord uses `components`. (#18215) Thanks @obviyus.
- Agents/Image tool: replace Anthropic-incompatible union schema with explicit `image` (single) and `images` (multi) parameters, keeping tool schemas `anyOf`/`oneOf`/`allOf`-free while preserving multi-image analysis support. (#18551, #18566) Thanks @aldoeliacim.
- Agents/Models: probe the primary model when its auth-profile cooldown is near expiry (with per-provider throttling), so runs recover from temporary rate limits without staying on fallback models until restart. (#17478) Thanks @PlayerGhost.
- Agents/Failover: classify provider abort stop-reason errors (`Unhandled stop reason: abort`, `stop reason: abort`, `reason: abort`) as timeout-class failures so configured model fallback chains trigger instead of surfacing raw abort failures. (#18618) Thanks @sauerdaniel.
- Models/CLI: sync auth-profiles credentials into agent `auth.json` before registry availability checks so `marv models list --all` reports auth correctly for API-key/token providers, normalize provider-id aliases when bridging credentials, and skip expired token mirrors. (#18610, #18615)
- Agents/Context: raise default total bootstrap prompt cap from `24000` to `150000` chars (keeping `bootstrapMaxChars` at `20000`), include total-cap visibility in `/context`, and mark truncation from injected-vs-raw sizes so total-cap clipping is reflected accurately.
- Memory/QMD: scope managed collection names per agent and precreate glob-backed collection directories before registration, preventing cross-agent collection clobbering and startup ENOENT failures in fresh workspaces. (#17194) Thanks @jonathanadams96.
- Cron: preserve per-job schedule-error isolation in post-run maintenance recompute so malformed sibling jobs no longer abort persistence of successful runs. (#17852) Thanks @pierreeurope.
- Gateway/Config: prevent `config.patch` object-array merges from falling back to full-array replacement when some patch entries lack `id`, so partial `agents.list` updates no longer drop unrelated agents. (#17989) Thanks @stakeswky.
- Gateway/Auth: trim whitespace around trusted proxy entries before matching so configured proxies with stray spaces still authorize. (#18084) Thanks @Clawborn.
- Config/Discord: require string IDs in Discord allowlists, keep onboarding inputs string-only, and add doctor repair for numeric entries. (#18220) Thanks @thewilloftheshadow.
- Security/Sessions: create new session transcript JSONL files with user-only (`0o600`) permissions and extend `marv security audit --fix` to remediate existing transcript file permissions.
- Sessions/Maintenance: archive transcripts when pruning stale sessions, clean expired media in subdirectories, and purge `.deleted` transcript archives after the prune window to prevent disk leaks. (#18538)
- Infra/Fetch: ensure foreign abort-signal listener cleanup never masks original fetch successes/failures, while still preventing detached-finally unhandled rejection noise in `wrapFetchWithAbortSignal`. Thanks @Jackten.
- Heartbeat: allow suppressing tool error warning payloads during heartbeat runs via a new heartbeat config flag. (#18497) Thanks @thewilloftheshadow.
- Heartbeat: include sender metadata (From/To/Provider) in heartbeat prompts so model context matches the delivery target. (#18532) Thanks @dinakars777.
- Heartbeat/Telegram: strip configured `responsePrefix` before heartbeat ack detection (with boundary-safe matching) so prefixed `HEARTBEAT_OK` replies are correctly suppressed instead of leaking into DMs. (#18602)

- Skills/Security: restrict `download` installer `targetDir` to the per-skill tools directory to prevent arbitrary file writes. Thanks @Adam55A-code.
- Skills/Linux: harden go installer fallback on apt-based systems by handling root/no-sudo environments safely, doing best-effort apt index refresh, and returning actionable errors instead of failing with spawn errors. (#17687) Thanks @mcrolly.
- Web Fetch/Security: cap downloaded response body size before HTML parsing to prevent memory exhaustion from oversized or deeply nested pages. Thanks @xuemian168.
- Config/Gateway: make sensitive-key whitelist suffix matching case-insensitive while preserving `passwordFile` path exemptions, preventing accidental redaction of non-secret config values like `maxTokens` and IRC password-file paths. (#16042) Thanks @akramcodez.
- Dev tooling: harden git `pre-commit` hook against option injection from malicious filenames (for example `--force`), preventing accidental staging of ignored files. Thanks @mrthankyou.
- Gateway/Agent: reject malformed `agent:`-prefixed session keys (for example, `agent:main`) in `agent` and `agent.identity.get` instead of silently resolving them to the default agent, preventing accidental cross-session routing. (#15707) Thanks @rodrigouroz.
- Gateway/Chat: harden `chat.send` inbound message handling by rejecting null bytes, stripping unsafe control characters, and normalizing Unicode to NFC before dispatch. (#8593) Thanks @fr33d3m0n.
- Gateway/Send: return an actionable error when `send` targets internal-only `webchat`, guiding callers to use `chat.send` or a deliverable channel. (#15703) Thanks @rodrigouroz.
- Gateway/Commands: keep webchat command authorization on the internal `webchat` context instead of inferring another provider from channel allowlists, fixing dropped `/new`/`/status` commands in Control UI when channel allowlists are configured. (#7189) Thanks @karlisbergmanis-lv.
- Control UI: prevent stored XSS via assistant name/avatar by removing inline script injection, serving bootstrap config as JSON, and enforcing `script-src 'self'`. Thanks @Adam55A-code.
- Agents/Security: sanitize workspace paths before embedding into LLM prompts (strip Unicode control/format chars) to prevent instruction injection via malicious directory names. Thanks @aether-ai-agent.
- Agents/Sandbox: clarify system prompt path guidance so sandbox `bash/exec` uses container paths (for example `/workspace`) while file tools keep host-bridge mapping, avoiding first-attempt path misses from host-only absolute paths in sandbox command execution. (#17693) Thanks @app/juniordevbot.
- Agents/Context: apply configured model `contextWindow` overrides after provider discovery so `lookupContextTokens()` honors operator config values (including discovery-failure paths). (#17404) Thanks @michaelbship and @vignesh07.
- Agents/Context: derive `lookupContextTokens()` from auth-available model metadata and keep the smallest discovered context window for duplicate model ids, preventing cross-provider cache collisions from overestimating session context limits. (#17586) Thanks @githabideri and @vignesh07.
- Agents/OpenAI: force `store=true` for direct OpenAI Responses/Codex runs to preserve multi-turn server-side conversation state, while leaving proxy/non-OpenAI endpoints unchanged. (#16803) Thanks @mark9232 and @vignesh07.
- Memory/FTS: make `buildFtsQuery` Unicode-aware so non-ASCII queries (including CJK) produce keyword tokens instead of falling back to vector-only search. (#17672) Thanks @KinGP5471.
- Auto-reply/Compaction: resolve `memory/YYYY-MM-DD.md` placeholders with timezone-aware runtime dates and append a `Current time:` line to memory-flush turns, preventing wrong-year memory filenames without making the system prompt time-variant. (#17603, #17633) Thanks @nicholaspapadam-wq and @vignesh07.
- Auth/Cooldowns: auto-expire stale auth profile cooldowns when `cooldownUntil` or `disabledUntil` timestamps have passed, and reset `errorCount` so the next transient failure does not immediately escalate to a disproportionately long cooldown. Handles `cooldownUntil` and `disabledUntil` independently. (#3604) Thanks @nabbilkhan.
- Agents: return an explicit timeout error reply when an embedded run times out before producing any payloads, preventing silent dropped turns during slow cache-refresh transitions. (#16659) Thanks @liaosvcaf and @vignesh07.
- Group chats: always inject group chat context (name, participants, reply guidance) into the system prompt on every turn, not just the first. Prevents the model from losing awareness of which group it's in and incorrectly using the message tool to send to the same group. (#14447) Thanks @tyler6204.
- Browser/Agents: when browser control service is unavailable, return explicit non-retry guidance (instead of "try again") so models do not loop on repeated browser tool calls until timeout. (#17673) Thanks @austenstone.
- Subagents: use child-run-based deterministic announce idempotency keys across direct and queued delivery paths (with legacy queued-item fallback) to prevent duplicate announce retries without collapsing distinct same-millisecond announces. (#17150) Thanks @widingmarcus-cyber.
- Subagents/Models: preserve `agents.defaults.model.fallbacks` when subagent sessions carry a model override, so subagent runs fail over to configured fallback models instead of retrying only the overridden primary model.
- Agents/Tools: scope the `message` tool schema to the active channel so Telegram uses `buttons` and Discord uses `components`. (#18215) Thanks @obviyus.
- Telegram: omit `message_thread_id` for DM sends/draft previews and keep forum-topic handling (`id=1` general omitted, non-general kept), preventing DM failures with `400 Bad Request: message thread not found`. (#10942) Thanks @garnetlyx.
- Telegram: replace inbound `<media:audio>` placeholder with successful preflight voice transcript in message body context, preventing placeholder-only prompt bodies for mention-gated voice messages. (#16789) Thanks @Limitless2023.
- Telegram: retry inbound media `getFile` calls (3 attempts with backoff) and gracefully fall back to placeholder-only processing when retries fail, preventing dropped voice/media messages on transient Telegram network errors. (#16154) Thanks @yinghaosang.
- Telegram: finalize streaming preview replies in place instead of sending a second final message, preventing duplicate Telegram assistant outputs at stream completion. (#17218) Thanks @obviyus.
- Discord: preserve channel session continuity when runtime payloads omit `message.channelId` by falling back to event/raw `channel_id` values for routing/session keys, so same-channel messages keep history across turns/restarts. Also align diagnostics so active Discord runs no longer appear as `sessionKey=unknown`. (#17622) Thanks @shakkernerd.
- Discord: dedupe native skill commands by skill name in multi-agent setups to prevent duplicated slash commands with `_2` suffixes. (#17365) Thanks @seewhyme.
- Discord: ensure role allowlist matching uses raw role IDs for message routing authorization. Thanks @xinhuagu.
- Discord: skip text-based exec approval forwarding in favor of Discord's component-based approval UI. Thanks @thewilloftheshadow.
- Web UI/Agents: hide `BOOTSTRAP.md` in the Agents Files list after onboarding is completed, avoiding confusing missing-file warnings for completed workspaces. (#17491) Thanks @gumadeiras.
- Memory/QMD: scope managed collection names per agent and precreate glob-backed collection directories before registration, preventing cross-agent collection clobbering and startup ENOENT failures in fresh workspaces. (#17194) Thanks @jonathanadams96.
- Gateway/Memory: initialize QMD startup sync for every configured agent (not just the default agent), so `memory.qmd.update.onBoot` is effective across multi-agent setups. (#17663) Thanks @HenryLoenwind.
- Auto-reply/WhatsApp/TUI/Web: when a final assistant message is `NO_REPLY` and a messaging tool send succeeded, mirror the delivered messaging-tool text into session-visible assistant output so TUI/Web no longer show `NO_REPLY` placeholders. (#7010) Thanks @Morrowind-Xie.
- Cron: infer `payload.kind="agentTurn"` for model-only `cron.update` payload patches, so partial agent-turn updates do not fail validation when `kind` is omitted. (#15664) Thanks @rodrigouroz.
- TUI: make searchable-select filtering and highlight rendering ANSI-aware so queries ignore hidden escape codes and no longer corrupt ANSI styling sequences during match highlighting. (#4519) Thanks @bee4come.
- TUI/Windows: coalesce rapid single-line submit bursts in Git Bash into one multiline message as a fallback when bracketed paste is unavailable, preventing pasted multiline text from being split into multiple sends. (#4986) Thanks @adamkane.
- TUI: suppress false `(no output)` placeholders for non-local empty final events during concurrent runs, preventing external-channel replies from showing empty assistant bubbles while a local run is still streaming. (#5782) Thanks @LagWizard and @vignesh07.
- TUI: preserve copy-sensitive long tokens (URLs/paths/file-like identifiers) during wrapping and overflow sanitization so wrapped output no longer inserts spaces that corrupt copy/paste values. (#17515, #17466, #17505) Thanks @abe238, @trevorpan, and @JasonCry.
- CLI/Build: make legacy daemon CLI compatibility shim generation tolerant of minimal tsdown daemon export sets, while preserving restart/register compatibility aliases and surfacing explicit errors for unavailable legacy daemon commands. Thanks @vignesh07.
