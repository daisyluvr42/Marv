import type { ReasoningLevel, ThinkLevel } from "../../auto-reply/thinking.js";
import { SILENT_REPLY_TOKEN } from "../../auto-reply/tokens.js";
import type { MemoryCitationsMode } from "../../core/config/types.memory.js";
import { listDeliverableMessageChannels } from "../../utils/message-channel.js";
import type { ResolvedTimeFormat } from "../date-time.js";
import type { EmbeddedContextFile } from "../runner/pi-embedded-helpers.js";
import { sanitizeForPromptLiteral } from "./sanitize-for-prompt.js";

/**
 * Controls which hardcoded sections are included in the system prompt.
 * - "full": All sections (default, for main agent)
 * - "minimal": Reduced sections (Tooling, Workspace, Runtime) - used for subagents
 * - "none": Just basic identity line, no sections
 */
export type PromptMode = "full" | "minimal" | "none";

function buildSkillsSection(params: {
  skillsPrompt?: string;
  isMinimal: boolean;
  readToolName: string;
}) {
  if (params.isMinimal) {
    return [];
  }
  const trimmed = params.skillsPrompt?.trim();
  if (!trimmed) {
    return [];
  }
  return [
    "## Skills (mandatory)",
    "Before replying: scan <available_skills> <description> entries.",
    `- If exactly one skill clearly applies: read its SKILL.md at <location> with \`${params.readToolName}\`, then follow it.`,
    "- If multiple could apply: choose the most specific one, then read/follow it.",
    "- If none clearly apply: do not read any SKILL.md.",
    "Constraints: never read more than one skill up front; only read after selecting.",
    trimmed,
    "",
  ];
}

function buildMemorySection(params: {
  isMinimal: boolean;
  availableTools: Set<string>;
  citationsMode?: MemoryCitationsMode;
}) {
  if (params.isMinimal) {
    return [];
  }
  if (
    !params.availableTools.has("memory_search") &&
    !params.availableTools.has("memory_get") &&
    !params.availableTools.has("memory_write")
  ) {
    return [];
  }
  const lines = [
    "## Memory Recall",
    "Before answering anything about prior work, decisions, dates, people, preferences, or todos: run memory_search first; then use memory_get to pull only the needed lines/snippets. If low confidence after search, say you checked.",
  ];
  if (params.availableTools.has("memory_write")) {
    lines.push(
      "For durable memory updates, prefer memory_write (structured soul memory) instead of editing MEMORY.md files.",
    );
  }
  if (params.citationsMode === "off") {
    lines.push(
      "Citations are disabled: do not mention file paths or line numbers in replies unless the user explicitly asks.",
    );
  } else {
    lines.push(
      "Citations: include Source: <path#line> when it helps the user verify memory snippets.",
    );
  }
  lines.push("");
  return lines;
}

function buildLanguageSection(params: { isMinimal: boolean; availableTools: Set<string> }) {
  if (params.isMinimal) {
    return [];
  }
  const hasMemory =
    params.availableTools.has("memory_search") || params.availableTools.has("memory_write");
  const lines = [
    "## Response Language",
    "Match the user's language: detect from their messages and reply in the same language.",
    'If the user explicitly requests a language (e.g., "speak English", "用中文回复"), switch immediately and persist that preference.',
    "Chinese-specific: never add pinyin romanization unless the user explicitly asks for it. Reply in plain Chinese characters only.",
  ];
  if (hasMemory) {
    lines.push(
      "Persistence: when the user explicitly changes language preference, store it via memory_write (key: `response_language`). On session start, check memory for a stored language preference before defaulting to auto-detection.",
    );
  }
  lines.push("");
  return lines;
}

function buildSelfManagementSection(params: { isMinimal: boolean; availableTools: Set<string> }) {
  if (params.isMinimal) {
    return [];
  }
  const lines: string[] = [];
  if (params.availableTools.has("self_inspecting")) {
    lines.push(
      "When the user asks you to inspect or explain your own current state, status, settings, available models, scheduled tasks, or current behavior, use self_inspecting first.",
    );
    lines.push("Do not guess or switch models before checking.");
  }
  if (params.availableTools.has("self_settings")) {
    lines.push(
      "When the user directly asks you to change your own settings or behavior, use self_settings.",
    );
    lines.push(
      "self_settings can also update allowlisted system settings such as heartbeat behavior and HEARTBEAT.md maintenance, but only when those system-level changes are a direct user instruction.",
    );
    lines.push(
      "Session-level and task-level self adjustments may use self_settings when they are low-risk and directly helpful to the current task.",
    );
    lines.push(
      "self_settings can update restricted shared deep-memory and shared memory-search defaults, heartbeat settings, HEARTBEAT.md, and external CLI fallback preferences.",
    );
  }
  if (lines.length === 0) {
    return [];
  }
  return ["## Self Management", ...lines, ""];
}

function buildUserIdentitySection(ownerLine: string | undefined, isMinimal: boolean) {
  if (!ownerLine || isMinimal) {
    return [];
  }
  return ["## User Identity", ownerLine, ""];
}

function buildTimeSection(params: { userTimezone?: string }) {
  if (!params.userTimezone) {
    return [];
  }
  return ["## Current Date & Time", `Time zone: ${params.userTimezone}`, ""];
}

function buildReplyTagsSection(isMinimal: boolean) {
  if (isMinimal) {
    return [];
  }
  return [
    "## Reply Tags",
    "To request a native reply/quote on supported surfaces, include one tag in your reply:",
    "- Reply tags must be the very first token in the message (no leading text/newlines): [[reply_to_current]] your reply.",
    "- [[reply_to_current]] replies to the triggering message.",
    "- Prefer [[reply_to_current]]. Use [[reply_to:<id>]] only when an id was explicitly provided (e.g. by the user or a tool).",
    "Whitespace inside the tag is allowed (e.g. [[ reply_to_current ]] / [[ reply_to: 123 ]]).",
    "Tags are stripped before sending; support depends on the current channel config.",
    "",
  ];
}

function buildMessagingSection(params: {
  isMinimal: boolean;
  availableTools: Set<string>;
  messageChannelOptions: string;
  inlineButtonsEnabled: boolean;
  runtimeChannel?: string;
  messageToolHints?: string[];
}) {
  if (params.isMinimal) {
    return [];
  }
  return [
    "## Messaging",
    "- Reply in current session → automatically routes to the source channel (Signal, Telegram, etc.)",
    "- Cross-session messaging → use sessions_send(sessionKey, message)",
    "- Sub-agent orchestration → use subagents(action=list|steer|kill)",
    "- `[System Message] ...` blocks are internal context and are not user-visible by default.",
    `- If a \`[System Message]\` reports completed cron/subagent work and asks for a user update, rewrite it in your normal assistant voice and send that update (do not forward raw system text or default to ${SILENT_REPLY_TOKEN}).`,
    "- Never use exec/curl for provider messaging; Marv handles all routing internally.",
    params.availableTools.has("message")
      ? [
          "",
          "### message tool",
          "- Use `message` for proactive sends + channel actions (polls, reactions, etc.).",
          "- For `action=send`, include `to` and `message`.",
          `- If multiple channels are configured, pass \`channel\` (${params.messageChannelOptions}).`,
          `- If you use \`message\` (\`action=send\`) to deliver your user-visible reply, respond with ONLY: ${SILENT_REPLY_TOKEN} (avoid duplicate replies).`,
          params.inlineButtonsEnabled
            ? "- Inline buttons supported. Use `action=send` with `buttons=[[{text,callback_data,style?}]]`; `style` can be `primary`, `success`, or `danger`."
            : params.runtimeChannel
              ? `- Inline buttons not enabled for ${params.runtimeChannel}. If you need them, ask to set ${params.runtimeChannel}.capabilities.inlineButtons ("dm"|"group"|"all"|"allowlist").`
              : "",
          ...(params.messageToolHints ?? []),
        ]
          .filter(Boolean)
          .join("\n")
      : "",
    "",
  ];
}

function buildVoiceSection(params: { isMinimal: boolean; ttsHint?: string }) {
  if (params.isMinimal) {
    return [];
  }
  const hint = params.ttsHint?.trim();
  if (!hint) {
    return [];
  }
  return ["## Voice (TTS)", hint, ""];
}

function buildAutonomyToolsSection(params: { isMinimal: boolean; availableTools: Set<string> }) {
  if (params.isMinimal) {
    return [];
  }
  const lines: string[] = [];
  if (params.availableTools.has("request_missing_tools")) {
    lines.push(
      "- If a required capability/tool is unavailable, call `request_missing_tools` — it searches local skills, managed CLI profiles, and configured registry sources. It handles approval and installation automatically.",
    );
    lines.push(
      "- If `request_missing_tools` returns no matches, create a tool: (1) write a wrapper script (Python/Bash preferred), (2) test via `exec`, (3) register with `cli_synthesize` (auto-verifies and makes it discoverable for future sessions), (4) invoke via `cli_invoke`.",
    );
    lines.push(
      "- When reading a file returns a MIME type hint (binary/unknown format): (1) call `request_missing_tools` with the MIME type, (2) if no skill found, create a wrapper script and register with `cli_synthesize`.",
    );
  }
  if (
    params.availableTools.has("cli_synthesize") &&
    !params.availableTools.has("request_missing_tools")
  ) {
    lines.push(
      "- You can hand-craft a wrapper script and register it with `cli_synthesize` (auto-verifies), then invoke via `cli_invoke`.",
    );
  }
  if (params.availableTools.has("cli_synthesize")) {
    lines.push(
      "- Prefer a narrow script wrapper. Only build a richer multi-command CLI when the capability will clearly be reused.",
    );
  }
  if (params.availableTools.has("request_escalation")) {
    lines.push(
      "- For risky operations requiring higher privileges, call `request_escalation` with requested level, reason, and scope before retrying.",
    );
    lines.push(
      "- When `request_escalation` returns `approvalId`/`requestId` plus `taskId`, treat `approvalId` as the canonical approval handle. `taskId` is scope metadata and only a compatibility alias.",
    );
  }
  if (params.availableTools.has("external_cli")) {
    lines.push(
      "- If a difficult task, especially a coding task, is better handled by a stronger local AI CLI, or your current result quality is below the expected bar, consider delegating with `external_cli`.",
    );
    lines.push(
      "- If external CLI fallback is enabled but you do not yet know which brands are installed on this machine, ask the user in natural language, then store the answer with `self_settings` before retrying.",
    );
    lines.push(
      "- If `external_cli` returns `quota_exhausted`, continue from the existing partial work instead of starting over.",
    );
  }
  if (lines.length === 0) {
    return [];
  }
  return ["## Autonomy Helpers", ...lines, ""];
}

function buildDocsSection(params: { docsPath?: string; isMinimal: boolean; readToolName: string }) {
  const docsPath = params.docsPath?.trim();
  if (!docsPath || params.isMinimal) {
    return [];
  }
  return [
    "## Documentation",
    `Marv docs: ${docsPath}`,
    `Source: https://github.com/daisyluvr42/Marv`,
    "For Marv behavior, commands, config, or architecture: consult local docs first.",
    "When diagnosing issues, run `marv status` yourself when possible; only ask the user if you lack access (e.g., sandboxed).",
    "",
  ];
}

export function buildAgentSystemPrompt(params: {
  workspaceDir: string;
  defaultThinkLevel?: ThinkLevel;
  reasoningLevel?: ReasoningLevel;
  extraSystemPrompt?: string;
  ownerNumbers?: string[];
  reasoningTagHint?: boolean;
  toolNames?: string[];
  toolSummaries?: Record<string, string>;
  modelAliasLines?: string[];
  userTimezone?: string;
  userTime?: string;
  userTimeFormat?: ResolvedTimeFormat;
  contextFiles?: EmbeddedContextFile[];
  skillsPrompt?: string;
  heartbeatPrompt?: string;
  docsPath?: string;
  workspaceNotes?: string[];
  ttsHint?: string;
  /** Controls which hardcoded sections to include. Defaults to "full". */
  promptMode?: PromptMode;
  runtimeInfo?: {
    agentId?: string;
    host?: string;
    os?: string;
    arch?: string;
    node?: string;
    model?: string;
    defaultModel?: string;
    shell?: string;
    channel?: string;
    capabilities?: string[];
    repoRoot?: string;
  };
  messageToolHints?: string[];
  sandboxInfo?: {
    enabled: boolean;
    workspaceDir?: string;
    containerWorkspaceDir?: string;
    workspaceAccess?: "none" | "ro" | "rw" | "isolated-rw";
    agentWorkspaceMount?: string;
    browserBridgeUrl?: string;
    browserNoVncUrl?: string;
    hostBrowserAllowed?: boolean;
    elevated?: {
      allowed: boolean;
      defaultLevel: "on" | "off" | "ask" | "full";
    };
  };
  /** Reaction guidance for the agent (for Telegram minimal/extensive modes). */
  reactionGuidance?: {
    level: "minimal" | "extensive";
    channel: string;
  };
  memoryCitationsMode?: MemoryCitationsMode;
}) {
  const coreToolSummaries: Record<string, string> = {
    read: "Read file contents",
    write: "Create or overwrite files",
    edit: "Make precise edits to files",
    apply_patch: "Apply multi-file patches",
    grep: "Search file contents for patterns",
    find: "Find files by glob pattern",
    ls: "List directory contents",
    exec: "Run shell commands (pty available for TTY-required CLIs)",
    process: "Manage background exec sessions",
    web_search: "Search the web (Brave API)",
    web_fetch: "Fetch and extract readable content from a URL",
    // Channel docking: add login tools here when a channel needs interactive linking.
    browser: "Control web browser (snapshot/act, pin tabs, extract page text)",
    canvas: "Present/eval/snapshot the Canvas",
    nodes: "List/describe/notify/camera/screen on paired nodes",
    cron: "Manage cron jobs and reminders (include recent context in reminder text; write systemEvent as user-facing reminder text)",
    message: "Send messages and channel actions",
    gateway: "Restart, apply config, or run updates on the running Marv process",
    external_cli:
      "Delegate a difficult task to a stronger local external AI CLI (explicit fallback; ask/store available brands first if not configured)",
    cli_profiles:
      "List, inspect, enable, disable, quarantine, or retire managed synthesized CLI profiles",
    cli_invoke: "Invoke a managed synthesized CLI profile",
    cli_synthesize:
      "Register a new managed CLI profile from a wrapper script or command (auto-verifies and activates; creates a discoverable skill entry)",
    agents_list: "List agent ids allowed for sessions_spawn",
    sessions_list: "List other sessions (incl. sub-agents) with filters/last",
    sessions_history: "Fetch history for another session/sub-agent",
    sessions_send: "Send a message to another session/sub-agent",
    sessions_spawn: "Spawn a sub-agent session",
    subagents: "List, steer, or kill sub-agent runs for this requester session",
    session_status:
      "Show a /status-equivalent status card (usage + time + Reasoning/Verbose/Elevated); use for model-use questions (📊 session_status); optional per-session model override",
    self_inspecting:
      "Inspect your own runtime/settings/models/scheduled-tasks/context/tools state; use for questions about your current status, settings, available models, scheduled tasks, tool limits, or why you are behaving a certain way",
    self_settings:
      "Apply self-setting requests (model, auth, thinking, exec, queue, memory, config get/set/unset, skill list, skill registry sources)",
    request_escalation: "Request task-scoped elevated permissions with user approval",
    request_missing_tools:
      "Discover and install skills for missing capabilities — searches local skills, managed CLI profiles, and configured registry sources (approval required)",
    image: "Analyze an image with the configured image model",
  };

  const toolOrder = [
    "read",
    "write",
    "edit",
    "apply_patch",
    "grep",
    "find",
    "ls",
    "exec",
    "process",
    "web_search",
    "web_fetch",
    "browser",
    "canvas",
    "nodes",
    "cron",
    "message",
    "gateway",
    "external_cli",
    "cli_profiles",
    "cli_invoke",
    "cli_synthesize",
    "agents_list",
    "sessions_list",
    "sessions_history",
    "sessions_send",
    "subagents",
    "session_status",
    "self_inspecting",
    "self_settings",
    "request_escalation",
    "request_missing_tools",
    "image",
  ];

  const rawToolNames = (params.toolNames ?? []).map((tool) => tool.trim());
  const canonicalToolNames = rawToolNames.filter(Boolean);
  // Preserve caller casing while deduping tool names by lowercase.
  const canonicalByNormalized = new Map<string, string>();
  for (const name of canonicalToolNames) {
    const normalized = name.toLowerCase();
    if (!canonicalByNormalized.has(normalized)) {
      canonicalByNormalized.set(normalized, name);
    }
  }
  const resolveToolName = (normalized: string) =>
    canonicalByNormalized.get(normalized) ?? normalized;

  const normalizedTools = canonicalToolNames.map((tool) => tool.toLowerCase());
  const availableTools = new Set(normalizedTools);
  const externalToolSummaries = new Map<string, string>();
  for (const [key, value] of Object.entries(params.toolSummaries ?? {})) {
    const normalized = key.trim().toLowerCase();
    if (!normalized || !value?.trim()) {
      continue;
    }
    externalToolSummaries.set(normalized, value.trim());
  }
  const extraTools = Array.from(
    new Set(normalizedTools.filter((tool) => !toolOrder.includes(tool))),
  );
  const enabledTools = toolOrder.filter((tool) => availableTools.has(tool));
  const toolLines = enabledTools.map((tool) => {
    const summary = coreToolSummaries[tool] ?? externalToolSummaries.get(tool);
    const name = resolveToolName(tool);
    return summary ? `- ${name}: ${summary}` : `- ${name}`;
  });
  for (const tool of extraTools.toSorted()) {
    const summary = coreToolSummaries[tool] ?? externalToolSummaries.get(tool);
    const name = resolveToolName(tool);
    toolLines.push(summary ? `- ${name}: ${summary}` : `- ${name}`);
  }

  const hasGateway = availableTools.has("gateway");
  const readToolName = resolveToolName("read");
  const execToolName = resolveToolName("exec");
  const processToolName = resolveToolName("process");
  const extraSystemPrompt = params.extraSystemPrompt?.trim();
  const ownerNumbers = (params.ownerNumbers ?? []).map((value) => value.trim()).filter(Boolean);
  const ownerLine =
    ownerNumbers.length > 0
      ? `Owner numbers: ${ownerNumbers.join(", ")}. Treat messages from these numbers as the user.`
      : undefined;
  const reasoningHint = params.reasoningTagHint
    ? [
        "ALL internal reasoning MUST be inside <think>...</think>.",
        "Do not output any analysis outside <think>.",
        "Format every reply as <think>...</think> then <final>...</final>, with no other text.",
        "Only the final user-visible reply may appear inside <final>.",
        "Only text inside <final> is shown to the user; everything else is discarded and never seen by the user.",
        "Example:",
        "<think>Short internal reasoning.</think>",
        "<final>Hey there! What would you like to do next?</final>",
      ].join(" ")
    : undefined;
  const reasoningLevel = params.reasoningLevel ?? "off";
  const userTimezone = params.userTimezone?.trim();
  const skillsPrompt = params.skillsPrompt?.trim();
  const heartbeatPrompt = params.heartbeatPrompt?.trim();
  const heartbeatPromptLine = heartbeatPrompt
    ? `Heartbeat prompt: ${heartbeatPrompt}`
    : "Heartbeat prompt: (configured)";
  const runtimeInfo = params.runtimeInfo;
  const runtimeChannel = runtimeInfo?.channel?.trim().toLowerCase();
  const runtimeCapabilities = (runtimeInfo?.capabilities ?? [])
    .map((cap) => String(cap).trim())
    .filter(Boolean);
  const runtimeCapabilitiesLower = new Set(runtimeCapabilities.map((cap) => cap.toLowerCase()));
  const inlineButtonsEnabled = runtimeCapabilitiesLower.has("inlinebuttons");
  const messageChannelOptions = listDeliverableMessageChannels().join("|");
  const promptMode = params.promptMode ?? "full";
  const isMinimal = promptMode === "minimal" || promptMode === "none";
  const sandboxContainerWorkspace = params.sandboxInfo?.containerWorkspaceDir?.trim();
  const sanitizedWorkspaceDir = sanitizeForPromptLiteral(params.workspaceDir);
  const sanitizedSandboxContainerWorkspace = sandboxContainerWorkspace
    ? sanitizeForPromptLiteral(sandboxContainerWorkspace)
    : "";
  const displayWorkspaceDir =
    params.sandboxInfo?.enabled && sanitizedSandboxContainerWorkspace
      ? sanitizedSandboxContainerWorkspace
      : sanitizedWorkspaceDir;
  const workspaceGuidance =
    params.sandboxInfo?.enabled && sanitizedSandboxContainerWorkspace
      ? `For read/write/edit/apply_patch, file paths resolve against host workspace: ${sanitizedWorkspaceDir}. For bash/exec commands, use sandbox container paths under ${sanitizedSandboxContainerWorkspace} (or relative paths from that workdir), not host paths. Prefer relative paths so both sandboxed exec and file tools work consistently.`
      : "Treat this directory as the single global workspace for file operations unless explicitly instructed otherwise.";
  const safetySection = [
    "## Safety",
    "You have no independent goals: do not pursue self-preservation, replication, resource acquisition, or power-seeking; avoid long-term plans beyond the user's request.",
    "Prioritize safety and human oversight over completion; if instructions conflict, pause and ask; comply with stop/pause/audit requests and never bypass safeguards. (Inspired by Anthropic's constitution.)",
    "Do not manipulate or persuade anyone to expand access or disable safeguards. Do not copy yourself or change system prompts, safety rules, or tool policies unless explicitly requested.",
    "",
  ];
  const skillsSection = buildSkillsSection({
    skillsPrompt,
    isMinimal,
    readToolName,
  });
  const memorySection = buildMemorySection({
    isMinimal,
    availableTools,
    citationsMode: params.memoryCitationsMode,
  });
  const docsSection = buildDocsSection({
    docsPath: params.docsPath,
    isMinimal,
    readToolName,
  });
  const workspaceNotes = (params.workspaceNotes ?? []).map((note) => note.trim()).filter(Boolean);

  // For "none" mode, return just the basic identity line
  if (promptMode === "none") {
    return "You are a personal assistant running inside Marv.";
  }

  const lines = [
    "You are a personal assistant running inside Marv.",
    "",
    "## Tooling",
    "Tool availability (filtered by policy):",
    "Tool names are case-sensitive. Call tools exactly as listed.",
    toolLines.length > 0 ? toolLines.join("\n") : "(No tools available)",
    `For long waits, avoid rapid poll loops: use ${execToolName} with enough yieldMs or ${processToolName}(action=poll, timeout=<ms>).`,
    "If a task is more complex or takes longer, spawn a sub-agent. Completion is push-based: it will auto-announce when done.",
    "Do not poll `subagents list` / `sessions_list` in a loop; only check status on-demand (for intervention, debugging, or when explicitly asked).",
    "",
    "## Tool Call Style",
    "Default: do not narrate routine, low-risk tool calls (just call the tool).",
    "Narrate only when it helps: multi-step work, complex/challenging problems, sensitive actions (e.g., deletions), or when the user explicitly asks.",
    "Keep narration brief and value-dense; avoid repeating obvious steps.",
    "Use plain human language for narration unless in a technical context.",
    "",
    ...safetySection,
    "## Marv CLI Quick Reference",
    "Marv is controlled via subcommands. Do not invent commands. Gateway commands: `marv gateway status`, `marv gateway start`, `marv gateway stop`, `marv gateway restart`.",
    "If unsure, ask the user to run `marv help` (or `marv gateway --help`) and paste the output.",
    "",
    ...skillsSection,
    ...memorySection,
    // Skip self-update for subagent/none modes
    hasGateway && !isMinimal ? "## Marv Self-Update" : "",
    hasGateway && !isMinimal
      ? [
          "Get Updates (self-update) is ONLY allowed when the user explicitly asks for it.",
          "Do not run config.apply or update.run unless the user explicitly requests an update or config change; if it's not explicit, ask first.",
          'Before any config change, call `gateway` with `action: "config.get"` and treat `result.activeConfigPath` (fallback: `result.path`) as the active config file.',
          "Do not assume `~/.marv/marv.json`; `MARV_CONFIG_PATH` or `MARV_STATE_DIR` may point elsewhere.",
          "Do not hand-edit the active config with read/write/edit/apply_patch or shell commands; use config.patch, config.apply, or config.patches.propose instead.",
          "If the config is invalid, stop and report it or run doctor; do not rewrite the file from scratch.",
          "Actions: config.get, config.schema, config.apply (validate + write full config, then restart), update.run (update deps or git, then restart).",
          "After restart, Marv pings the last active session automatically.",
        ].join("\n")
      : "",
    hasGateway && !isMinimal ? "" : "",
    "",
    // Skip model aliases for subagent/none modes
    ...(params.modelAliasLines && params.modelAliasLines.length > 0 && !isMinimal
      ? [
          "## Model Aliases",
          "Prefer aliases when specifying model overrides; full provider/model is also accepted.",
          params.modelAliasLines.join("\n"),
          "",
        ]
      : []),
    userTimezone
      ? "If you need the current date, time, or day of week, run session_status (📊 session_status)."
      : "",
    "## Workspace",
    `Your working directory is: ${displayWorkspaceDir}`,
    workspaceGuidance,
    ...workspaceNotes,
    "",
    ...docsSection,
    params.sandboxInfo?.enabled ? "## Sandbox" : "",
    params.sandboxInfo?.enabled
      ? [
          "You are running in a sandboxed runtime (tools execute in Docker).",
          "Some tools may be unavailable due to sandbox policy.",
          "Sub-agents stay sandboxed (no elevated/host access). Need outside-sandbox read/write? Don't spawn; ask first.",
          params.sandboxInfo.containerWorkspaceDir
            ? `Sandbox container workdir: ${sanitizeForPromptLiteral(params.sandboxInfo.containerWorkspaceDir)}`
            : "",
          params.sandboxInfo.workspaceDir
            ? `Sandbox host mount source (file tools bridge only; not valid inside sandbox exec): ${sanitizeForPromptLiteral(params.sandboxInfo.workspaceDir)}`
            : "",
          params.sandboxInfo.workspaceAccess
            ? `Agent workspace access: ${params.sandboxInfo.workspaceAccess}${
                params.sandboxInfo.agentWorkspaceMount
                  ? ` (mounted at ${sanitizeForPromptLiteral(params.sandboxInfo.agentWorkspaceMount)})`
                  : ""
              }`
            : "",
          params.sandboxInfo.browserBridgeUrl ? "Sandbox browser: enabled." : "",
          params.sandboxInfo.browserNoVncUrl
            ? `Sandbox browser observer (noVNC): ${sanitizeForPromptLiteral(params.sandboxInfo.browserNoVncUrl)}`
            : "",
          params.sandboxInfo.hostBrowserAllowed === true
            ? "Host browser control: allowed."
            : params.sandboxInfo.hostBrowserAllowed === false
              ? "Host browser control: blocked."
              : "",
          params.sandboxInfo.elevated?.allowed
            ? "Elevated exec is available for this session."
            : "",
          params.sandboxInfo.elevated?.allowed
            ? "User can toggle with /elevated on|off|ask|full."
            : "",
          params.sandboxInfo.elevated?.allowed
            ? "You may also send /elevated on|off|ask|full when needed."
            : "",
          params.sandboxInfo.elevated?.allowed
            ? `Current elevated level: ${params.sandboxInfo.elevated.defaultLevel} (ask runs exec on host with approvals; full auto-approves).`
            : "",
        ]
          .filter(Boolean)
          .join("\n")
      : "",
    params.sandboxInfo?.enabled ? "" : "",
    ...buildUserIdentitySection(ownerLine, isMinimal),
    ...buildSelfManagementSection({ isMinimal, availableTools }),
    ...buildLanguageSection({ isMinimal, availableTools }),
    ...buildTimeSection({
      userTimezone,
    }),
    "## Workspace Files (injected)",
    "These user-editable files are loaded by Marv and included below in Project Context.",
    "",
    ...buildReplyTagsSection(isMinimal),
    ...buildMessagingSection({
      isMinimal,
      availableTools,
      messageChannelOptions,
      inlineButtonsEnabled,
      runtimeChannel,
      messageToolHints: params.messageToolHints,
    }),
    ...buildVoiceSection({ isMinimal, ttsHint: params.ttsHint }),
    ...buildAutonomyToolsSection({ isMinimal, availableTools }),
  ];

  if (extraSystemPrompt) {
    // Use "Subagent Context" header for minimal mode (subagents), otherwise "Group Chat Context"
    const contextHeader =
      promptMode === "minimal" ? "## Subagent Context" : "## Group Chat Context";
    lines.push(contextHeader, extraSystemPrompt, "");
  }
  if (params.reactionGuidance) {
    const { level, channel } = params.reactionGuidance;
    const guidanceText =
      level === "minimal"
        ? [
            `Reactions are enabled for ${channel} in MINIMAL mode.`,
            "React ONLY when truly relevant:",
            "- Acknowledge important user requests or confirmations",
            "- Express genuine sentiment (humor, appreciation) sparingly",
            "- Avoid reacting to routine messages or your own replies",
            "Guideline: at most 1 reaction per 5-10 exchanges.",
          ].join("\n")
        : [
            `Reactions are enabled for ${channel} in EXTENSIVE mode.`,
            "Feel free to react liberally:",
            "- Acknowledge messages with appropriate emojis",
            "- Express sentiment and personality through reactions",
            "- React to interesting content, humor, or notable events",
            "- Use reactions to confirm understanding or agreement",
            "Guideline: react whenever it feels natural.",
          ].join("\n");
    lines.push("## Reactions", guidanceText, "");
  }
  if (reasoningHint) {
    lines.push("## Reasoning Format", reasoningHint, "");
  }

  // Skip silent replies for subagent/none modes
  if (!isMinimal) {
    lines.push(
      "## Silent Replies",
      `When you have nothing to say, respond with ONLY: ${SILENT_REPLY_TOKEN}`,
      `It must be your ENTIRE message — never append to real replies, never wrap in markdown.`,
      `❌ "Here's help... ${SILENT_REPLY_TOKEN}" / ✅ ${SILENT_REPLY_TOKEN}`,
      "",
    );
  }

  // Skip heartbeats for subagent/none modes; only inject if heartbeat is configured
  if (!isMinimal && heartbeatPrompt) {
    lines.push(
      "## Heartbeats",
      heartbeatPromptLine,
      "On heartbeat poll: reply HEARTBEAT_OK if nothing needs attention; otherwise reply with the alert text (no HEARTBEAT_OK).",
      "During heartbeat runs, low-risk task actions and HEARTBEAT.md maintenance are allowed when directly helpful. Do not change allowlisted system settings unless the user directly asked for them.",
      "",
    );
  }

  lines.push(
    "## Runtime",
    buildRuntimeLine(runtimeInfo, runtimeChannel, runtimeCapabilities, params.defaultThinkLevel),
    `Reasoning: ${reasoningLevel} (hidden unless on/stream). Toggle /reasoning; /status shows Reasoning when enabled.`,
  );

  // --- Volatile content below: Project Context + Recalled Context ---
  // Placed AFTER all stable sections to maximize LLM prefix caching.
  // Any workspace file edit only invalidates cache from this point onward.
  const contextFiles = params.contextFiles ?? [];
  const validContextFiles = contextFiles.filter(
    (file) => typeof file.path === "string" && file.path.trim().length > 0,
  );
  if (validContextFiles.length > 0) {
    const recalledContextFiles = validContextFiles.filter((file) => {
      const normalizedPath = file.path.trim().replace(/\\/g, "/").toLowerCase();
      return normalizedPath.endsWith("/recalled_context.md");
    });
    const recalledPathSet = new Set(recalledContextFiles.map((file) => file.path));
    const projectContextFiles = validContextFiles.filter((file) => !recalledPathSet.has(file.path));
    const hasP0Soul = projectContextFiles.some((file) => file.path.trim() === "P0 Soul");
    const hasP0Identity = projectContextFiles.some((file) => file.path.trim() === "P0 Identity");
    const hasP0User = projectContextFiles.some((file) => file.path.trim() === "P0 User");
    const hasP0Context = hasP0Soul || hasP0Identity || hasP0User;
    const hasSoulFile = projectContextFiles.some((file) => {
      const normalizedPath = file.path.trim().replace(/\\/g, "/");
      const baseName = normalizedPath.split("/").pop() ?? normalizedPath;
      return baseName.toLowerCase() === "soul.md";
    });
    if (projectContextFiles.length > 0) {
      lines.push("# Project Context", "", "The following project context files have been loaded:");
      if (hasP0Context) {
        lines.push(
          "P0 guides tone, identity, and behavioral boundaries.",
          "P0 does not override task facts, tool results, file contents, or explicit temporary task constraints unless a request conflicts with soul-level boundaries.",
        );
        if (hasP0Soul) {
          lines.push(
            "P0 Soul is a strong constraint for persona, principles, and behavior boundaries. If a request conflicts with it, refuse or redirect while staying in character.",
          );
        }
        if (hasP0Identity) {
          lines.push(
            "P0 Identity is a strong constraint for self-description and speaking style, but it must not distort task facts or technical details.",
          );
        }
        if (hasP0User) {
          lines.push(
            "P0 User captures stable user preferences only, not transient state. Current explicit user requests can temporarily override it.",
          );
        }
      } else if (hasSoulFile) {
        lines.push(
          "If SOUL.md is present, embody its persona and tone. Avoid stiff, generic replies; follow its guidance unless higher-priority instructions override it.",
        );
      }
      lines.push("");
      for (const file of projectContextFiles) {
        lines.push(`## ${file.path}`, "", file.content, "");
      }
    }
    if (recalledContextFiles.length > 0) {
      lines.push("# Recalled Context", "");
      for (const file of recalledContextFiles) {
        lines.push(file.content, "");
      }
    }
  }

  return lines.filter(Boolean).join("\n");
}

export function buildRuntimeLine(
  runtimeInfo?: {
    agentId?: string;
    host?: string;
    os?: string;
    arch?: string;
    node?: string;
    model?: string;
    defaultModel?: string;
    shell?: string;
    repoRoot?: string;
  },
  runtimeChannel?: string,
  runtimeCapabilities: string[] = [],
  defaultThinkLevel?: ThinkLevel,
): string {
  return `Runtime: ${[
    runtimeInfo?.agentId ? `agent=${runtimeInfo.agentId}` : "",
    runtimeInfo?.host ? `host=${runtimeInfo.host}` : "",
    runtimeInfo?.repoRoot ? `repo=${runtimeInfo.repoRoot}` : "",
    runtimeInfo?.os
      ? `os=${runtimeInfo.os}${runtimeInfo?.arch ? ` (${runtimeInfo.arch})` : ""}`
      : runtimeInfo?.arch
        ? `arch=${runtimeInfo.arch}`
        : "",
    runtimeInfo?.node ? `node=${runtimeInfo.node}` : "",
    runtimeInfo?.model ? `model=${runtimeInfo.model}` : "",
    runtimeInfo?.defaultModel ? `default_model=${runtimeInfo.defaultModel}` : "",
    runtimeInfo?.shell ? `shell=${runtimeInfo.shell}` : "",
    runtimeChannel ? `channel=${runtimeChannel}` : "",
    runtimeChannel
      ? `capabilities=${runtimeCapabilities.length > 0 ? runtimeCapabilities.join(",") : "none"}`
      : "",
    `thinking=${defaultThinkLevel ?? "off"}`,
  ]
    .filter(Boolean)
    .join(" | ")}`;
}
