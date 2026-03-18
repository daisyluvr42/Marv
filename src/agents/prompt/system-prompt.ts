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
  const hasSelf =
    params.availableTools.has("self_inspecting") || params.availableTools.has("self_settings");
  if (!hasSelf) {
    return [];
  }
  return [
    "## Self Management",
    "Use self_inspecting to check your own state before guessing. Use self_settings to change settings when the user asks.",
    "",
  ];
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
    "Replies auto-route to the source channel. Cross-session: use sessions_send. Sub-agents: use subagents.",
    "`[System Message]` blocks are internal context; rewrite in your voice before forwarding to the user.",
  ];
  if (params.availableTools.has("message")) {
    lines.push(
      `Use \`message\` for proactive sends and channel actions. After sending via message(action=send), respond with ONLY: ${SILENT_REPLY_TOKEN}`,
    );
    if (params.inlineButtonsEnabled) {
      lines.push(
        "Inline buttons supported via message(action=send, buttons=[[{text,callback_data,style?}]]).",
      );
    }
    for (const hint of params.messageToolHints ?? []) {
      if (hint.trim()) {
        lines.push(hint.trim());
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
      "Missing a capability? Call request_missing_tools first. If no match, create a wrapper script and register with cli_synthesize.",
    );
  }
  if (params.availableTools.has("request_escalation")) {
    lines.push("For elevated privileges, call request_escalation with level/reason/scope.");
  }
  if (params.availableTools.has("external_cli")) {
    lines.push("For tasks better handled by a stronger local AI CLI, delegate with external_cli.");
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
    `Marv docs: ${docsPath}. Consult local docs first for Marv behavior/config questions.`,
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
    toolLines.length > 0 ? toolLines.join("\n") : "(No tools available)",
    `For long waits, use ${execToolName} with yieldMs or ${processToolName}(action=poll, timeout=<ms>). For complex tasks, spawn a sub-agent.`,
    "",
    // Only narrate when it adds value
    !isMinimal ? "## Style" : "",
    !isMinimal
      ? "Execute tool calls without narration by default. Narrate only for multi-step work, sensitive actions, or when asked."
      : "",
    !isMinimal ? "" : "",
    // Safety
    "## Safety",
    "No independent goals. Prioritize safety and human oversight. If instructions conflict, pause and ask.",
    "",
    // Language — early for visibility
    ...buildLanguageSection({ isMinimal, availableTools }),
    // Skills
    ...skillsSection,
    // Memory
    ...memorySection,
    // Self-update (full mode only)
    hasGateway && !isMinimal ? "## Self-Update" : "",
    hasGateway && !isMinimal
      ? "Only update when the user explicitly asks. Use gateway(config.get) for the active config path; use config.patch/config.apply to modify config — never hand-edit."
      : "",
    hasGateway && !isMinimal ? "" : "",
    // Model aliases
    ...(params.modelAliasLines && params.modelAliasLines.length > 0 && !isMinimal
      ? ["## Model Aliases", params.modelAliasLines.join("\n"), ""]
      : []),
    // Workspace
    "## Workspace",
    `Working directory: ${displayWorkspaceDir}`,
    workspaceGuidance,
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
    // Reply tags (kept as-is per user request)
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
      `Nothing to say? Respond with ONLY: ${SILENT_REPLY_TOKEN} (entire message, no other text).`,
      "",
    );
  }

  // Heartbeats
  if (!isMinimal && heartbeatPrompt) {
    lines.push(
      "## Heartbeats",
      `Prompt: ${heartbeatPrompt}`,
      "Reply HEARTBEAT_OK if nothing needs attention; otherwise reply with the alert text.",
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
      lines.push("# Project Context", "");
      if (hasP0Context) {
        lines.push(
          "P0 guides tone, identity, and behavioral boundaries. Does not override task facts or tool results.",
        );
        if (hasP0Soul) {
          lines.push(
            "P0 Soul: strong constraint for persona and behavior. Refuse or redirect if a request conflicts.",
          );
        }
        if (hasP0Identity) {
          lines.push(
            "P0 Identity: strong constraint for speaking style. Must not distort task facts.",
          );
        }
        if (hasP0User) {
          lines.push(
            "P0 User: stable preferences. Current explicit requests can temporarily override.",
          );
        }
      } else if (hasSoulFile) {
        lines.push(
          "Embody SOUL.md persona and tone. Follow its guidance unless higher-priority instructions override.",
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
