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
  const hasSearch =
    params.availableTools.has("memory_search") || params.availableTools.has("memory_get");
  const hasWrite = params.availableTools.has("memory_write");
  if (!hasSearch && !hasWrite) {
    return [];
  }
  const lines = [
    "## Memory",
    "For questions about prior work, decisions, preferences, or todos: use memory_search/memory_get before answering.",
  ];
  if (hasWrite) {
    lines.push(
      "Use memory_write for durable updates. Memory tools are the primary interface; .md files are read-only projections.",
    );
  }
  if (params.citationsMode !== "off") {
    lines.push("Include Source: <path#line> when it helps the user verify.");
  }
  lines.push("");
  return lines;
}

function buildLanguageSection(params: { isMinimal: boolean; availableTools: Set<string> }) {
  const hasMemory =
    params.availableTools.has("memory_search") || params.availableTools.has("memory_write");
  const lines = [
    "## Response Language",
    "Always reply in the language the user is writing in. Detect per-message and match it.",
    'Explicit switch (e.g., "speak English", "用中文回复") overrides auto-detection for all subsequent replies.',
    "Never add pinyin unless explicitly asked.",
  ];
  if (hasMemory) {
    lines.push(
      "When the user explicitly changes language, persist via memory_write (key: `response_language`). On session start, check memory first.",
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
      "Do not guess or switch models before checking.",
    );
  }
  if (params.availableTools.has("self_settings")) {
    lines.push(
      "When the user directly asks you to change your own settings or behavior, use self_settings.",
      "self_settings can also update heartbeat behavior and HEARTBEAT.md maintenance, but only when the user directly asks.",
      "Session-level and task-level self adjustments may use self_settings when they are low-risk and directly helpful to the current task.",
    );
  }
  if (lines.length === 0) {
    return [];
  }
  return ["## Self Management", ...lines, ""];
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
  const lines = [
    "## Messaging",
    "- Reply in current session → automatically routes to the source channel (Signal, Telegram, etc.)",
    "- Cross-session messaging → use sessions_send(sessionKey, message)",
    "- Sub-agent orchestration → use subagents(action=list|steer|kill)",
    "- `[System Message] ...` blocks are internal context and are not user-visible by default.",
    `- If a \`[System Message]\` reports completed cron/subagent work and asks for a user update, rewrite it in your normal assistant voice and send that update (do not forward raw system text or default to ${SILENT_REPLY_TOKEN}).`,
    "- Never use exec/curl for provider messaging; Marv handles all routing internally.",
  ];
  if (params.availableTools.has("message")) {
    lines.push(
      "",
      "### message tool",
      "- Use `message` for proactive sends + channel actions (polls, reactions, etc.).",
      "- For `action=send`, include `to` and `message`.",
      `- If multiple channels are configured, pass \`channel\` (${params.messageChannelOptions}).`,
      `- If you use \`message\` (\`action=send\`) to deliver your user-visible reply, respond with ONLY: ${SILENT_REPLY_TOKEN} (avoid duplicate replies).`,
    );
    if (params.inlineButtonsEnabled) {
      lines.push(
        "- Inline buttons supported. Use `action=send` with `buttons=[[{text,callback_data,style?}]]`; `style` can be `primary`, `success`, or `danger`.",
      );
    }
    for (const hint of params.messageToolHints ?? []) {
      if (hint.trim()) {
        lines.push(`- ${hint.trim()}`);
      }
    }
  }
  lines.push("");
  return lines;
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
      "- If a required capability/tool is unavailable, call `request_missing_tools` with a concrete capability description and likely tool names.",
      "- If `request_missing_tools` returns no matches, create an ad-hoc solution: write a script (Python/Bash preferred), test it via `exec`, and save successful scripts as managed skills.",
    );
  }
  if (params.availableTools.has("cli_profiles")) {
    lines.push(
      "- Before synthesizing a new CLI tool, check `cli_profiles` for existing managed profiles.",
    );
  }
  if (params.availableTools.has("cli_synthesize")) {
    lines.push(
      "- If a usable CLI wrapper is needed, register it with `cli_synthesize`, validate with `cli_verify`, and call with `cli_invoke`.",
    );
  }
  if (params.availableTools.has("request_escalation")) {
    lines.push(
      "- For risky operations requiring higher privileges, call `request_escalation` with requested level, reason, and scope before retrying.",
    );
  }
  if (params.availableTools.has("external_cli")) {
    lines.push(
      "- If a difficult task is better handled by a stronger local AI CLI, consider delegating with `external_cli`.",
      "- If `external_cli` returns `quota_exhausted`, continue from existing partial work instead of starting over.",
    );
  }
  if (lines.length === 0) {
    return [];
  }
  return ["## Autonomy Helpers", ...lines, ""];
}

function buildDocsSection(params: { docsPath?: string; isMinimal: boolean }) {
  const docsPath = params.docsPath?.trim();
  if (!docsPath || params.isMinimal) {
    return [];
  }
  return [
    "## Documentation",
    `Marv docs: ${docsPath}`,
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
  // Tool summaries are resolved dynamically from tool registrations via buildToolSummaryMap().
  // Callers pass toolNames (preserving registration order) and toolSummaries (extracted from tool.description).
  const summaries = new Map<string, string>();
  for (const [key, value] of Object.entries(params.toolSummaries ?? {})) {
    const normalized = key.trim().toLowerCase();
    if (normalized && value?.trim()) {
      summaries.set(normalized, value.trim());
    }
  }

  const canonicalToolNames = (params.toolNames ?? []).map((t) => t.trim()).filter(Boolean);
  const availableTools = new Set(canonicalToolNames.map((t) => t.toLowerCase()));
  const toolLines = canonicalToolNames.map((name) => {
    const summary = summaries.get(name.toLowerCase());
    return summary ? `- ${name}: ${summary}` : `- ${name}`;
  });

  const hasGateway = availableTools.has("gateway");
  // Use canonical names from the tool list if available, otherwise default
  const findCanonical = (name: string) =>
    canonicalToolNames.find((t) => t.toLowerCase() === name) ?? name;
  const readToolName = findCanonical("read");
  const execToolName = findCanonical("exec");
  const processToolName = findCanonical("process");
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
      : "";
  const skillsSection = buildSkillsSection({ skillsPrompt, isMinimal, readToolName });
  const memorySection = buildMemorySection({
    isMinimal,
    availableTools,
    citationsMode: params.memoryCitationsMode,
  });
  const docsSection = buildDocsSection({ docsPath: params.docsPath, isMinimal });
  const workspaceNotes = (params.workspaceNotes ?? []).map((note) => note.trim()).filter(Boolean);

  // For "none" mode, return just the basic identity line
  if (promptMode === "none") {
    return "You are a personal assistant running inside Marv.";
  }

  const lines = [
    "You are a personal assistant running inside Marv.",
    "",
    "## Tooling",
    "Tool names are case-sensitive. Call tools exactly as listed.",
    toolLines.length > 0 ? toolLines.join("\n") : "(No tools available)",
    `For long waits, avoid rapid poll loops: use ${execToolName} with enough yieldMs or ${processToolName}(action=poll, timeout=<ms>).`,
    "If a task is more complex or takes longer, spawn a sub-agent. Completion is push-based: it will auto-announce when done.",
    "Do not poll `subagents list` / `sessions_list` in a loop; only check status on-demand.",
    "",
    // Tool call style
    !isMinimal ? "## Tool Call Style" : "",
    !isMinimal ? "Default: do not narrate routine, low-risk tool calls (just call the tool)." : "",
    !isMinimal
      ? "Narrate only when it helps: multi-step work, sensitive actions, or when the user explicitly asks. Keep narration brief; use plain human language."
      : "",
    !isMinimal ? "" : "",
    // Safety
    "## Safety",
    "You have no independent goals: do not pursue self-preservation, replication, resource acquisition, or power-seeking; avoid long-term plans beyond the user's request.",
    "Prioritize safety and human oversight over completion; if instructions conflict, pause and ask; comply with stop/pause/audit requests and never bypass safeguards.",
    "Do not manipulate or persuade anyone to expand access or disable safeguards. Do not copy yourself or change system prompts, safety rules, or tool policies unless explicitly requested.",
    "",
    // CLI Quick Reference
    !isMinimal ? "## Marv CLI Quick Reference" : "",
    !isMinimal
      ? "Marv is controlled via subcommands. Do not invent commands. Gateway commands: `marv gateway status`, `marv gateway start`, `marv gateway stop`, `marv gateway restart`."
      : "",
    !isMinimal ? "" : "",
    // Language — early for visibility
    ...buildLanguageSection({ isMinimal, availableTools }),
    // Skills
    ...skillsSection,
    // Memory
    ...memorySection,
    // Self-update (full mode only)
    hasGateway && !isMinimal ? "## Marv Self-Update" : "",
    hasGateway && !isMinimal
      ? [
          "Get Updates (self-update) is ONLY allowed when the user explicitly asks for it.",
          "Do not run config.apply or update.run unless the user explicitly requests an update or config change; if it's not explicit, ask first.",
          'Before any config change, call `gateway` with `action: "config.get"` and treat `result.activeConfigPath` as the active config file.',
          "Do not hand-edit the active config with read/write/edit/apply_patch or shell commands; use config.patch, config.apply, or config.patches.propose instead.",
          "If the config is invalid, stop and report it or run doctor; do not rewrite the file from scratch.",
        ].join("\n")
      : "",
    hasGateway && !isMinimal ? "" : "",
    // Model aliases
    ...(params.modelAliasLines && params.modelAliasLines.length > 0 && !isMinimal
      ? ["## Model Aliases", params.modelAliasLines.join("\n"), ""]
      : []),
    // Date/time hint
    userTimezone ? "If you need the current date, time, or day of week, run session_status." : "",
    // Workspace
    "## Workspace",
    `Your working directory is: ${displayWorkspaceDir}`,
    workspaceGuidance ||
      "Treat this directory as the single global workspace for file operations unless explicitly instructed otherwise.",
    ...workspaceNotes,
    "",
    // Docs
    ...docsSection,
    // Sandbox
    ...(params.sandboxInfo?.enabled
      ? [
          "## Sandbox",
          [
            "Running in a sandboxed runtime (Docker). Some tools may be unavailable.",
            params.sandboxInfo.containerWorkspaceDir
              ? `Container workdir: ${sanitizeForPromptLiteral(params.sandboxInfo.containerWorkspaceDir)}`
              : "",
            params.sandboxInfo.workspaceAccess
              ? `Workspace access: ${params.sandboxInfo.workspaceAccess}${
                  params.sandboxInfo.agentWorkspaceMount
                    ? ` (mounted at ${sanitizeForPromptLiteral(params.sandboxInfo.agentWorkspaceMount)})`
                    : ""
                }`
              : "",
            params.sandboxInfo.browserBridgeUrl ? "Sandbox browser: enabled." : "",
            params.sandboxInfo.elevated?.allowed
              ? `Elevated exec available. Current level: ${params.sandboxInfo.elevated.defaultLevel}. Toggle: /elevated on|off|ask|full.`
              : "",
          ]
            .filter(Boolean)
            .join("\n"),
          "",
        ]
      : []),
    // User identity
    ...(ownerLine && !isMinimal ? ["## User Identity", ownerLine, ""] : []),
    // Self management
    ...buildSelfManagementSection({ isMinimal, availableTools }),
    // Time
    ...(userTimezone ? ["## Time", `Zone: ${userTimezone}`, ""] : []),
    // Workspace files intro
    !isMinimal ? "## Workspace Files (injected)" : "",
    !isMinimal
      ? "These user-editable files are loaded by Marv and included below in Project Context."
      : "",
    !isMinimal ? "" : "",
    // Reply tags
    ...buildReplyTagsSection(isMinimal),
    // Messaging
    ...buildMessagingSection({
      isMinimal,
      availableTools,
      messageChannelOptions,
      inlineButtonsEnabled,
      runtimeChannel,
      messageToolHints: params.messageToolHints,
    }),
    // Voice
    ...buildVoiceSection({ isMinimal, ttsHint: params.ttsHint }),
    // Autonomy helpers
    ...buildAutonomyToolsSection({ isMinimal, availableTools }),
  ];

  if (extraSystemPrompt) {
    const contextHeader =
      promptMode === "minimal" ? "## Subagent Context" : "## Group Chat Context";
    lines.push(contextHeader, extraSystemPrompt, "");
  }
  if (params.reactionGuidance) {
    const { level, channel } = params.reactionGuidance;
    lines.push(
      "## Reactions",
      level === "minimal"
        ? `Reactions enabled for ${channel} (minimal). React only when truly relevant — at most 1 per 5-10 exchanges.`
        : `Reactions enabled for ${channel} (extensive). React freely when it feels natural.`,
      "",
    );
  }
  if (reasoningHint) {
    lines.push("## Reasoning Format", reasoningHint, "");
  }

  // Silent replies
  if (!isMinimal) {
    lines.push(
      "## Silent Replies",
      `When you have nothing to say, respond with ONLY: ${SILENT_REPLY_TOKEN}`,
      `It must be your ENTIRE message — never append to real replies, never wrap in markdown.`,
      "",
    );
  }

  // Heartbeats
  if (!isMinimal && heartbeatPrompt) {
    lines.push(
      "## Heartbeats",
      `Heartbeat prompt: ${heartbeatPrompt}`,
      "On heartbeat poll: reply HEARTBEAT_OK if nothing needs attention; otherwise reply with the alert text (no HEARTBEAT_OK).",
      "This section applies ONLY to periodic heartbeat polls, NOT to normal user messages. Always respond to user messages naturally.",
      "",
    );
  }

  lines.push(
    "## Runtime",
    buildRuntimeLine(runtimeInfo, runtimeChannel, runtimeCapabilities, params.defaultThinkLevel),
    `Reasoning: ${reasoningLevel} (hidden unless on/stream).`,
  );

  // --- Volatile content below: Project Context + Recalled Context ---
  // Placed AFTER all stable sections to maximize LLM prefix caching.
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
