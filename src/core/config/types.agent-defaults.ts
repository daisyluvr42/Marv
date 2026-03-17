import type { ChannelId } from "../../channels/plugins/types.js";
import type {
  BlockStreamingChunkConfig,
  BlockStreamingCoalesceConfig,
  HumanDelayConfig,
  IdentityConfig,
  TypingMode,
} from "./types.base.js";
import type { GroupChatConfig } from "./types.messages.js";
import type { ConfiguredModelLocation, ConfiguredModelTier } from "./types.models.js";
import type {
  SandboxBrowserSettings,
  SandboxDockerSettings,
  SandboxPruneSettings,
} from "./types.sandbox.js";
import type { AgentToolsConfig, MemorySearchConfig } from "./types.tools.js";

export type AutoRoutingComplexity = "simple" | "moderate" | "complex" | "expert";

export type AutoRoutingRule = {
  /** Complexity tier this rule matches. */
  complexity: AutoRoutingComplexity;
  /** Legacy model target. Deprecated; automatic routing now only supplies policy hints. */
  model?: string | { primary?: string; fallbacks?: string[] };
  /** Optional thinking level override for this tier. */
  thinking?: "off" | "minimal" | "low" | "medium" | "high" | "xhigh";
};

export type AutoRoutingThresholds = {
  /** Max message length (chars) for "simple" tier (default: 200). */
  simpleMaxChars?: number;
  /** Max message length (chars) for "moderate" tier (default: 600). */
  moderateMaxChars?: number;
  /** Max message length (chars) for "complex" tier (default: 1500). */
  complexMaxChars?: number;
  /** Regex patterns that bump complexity (e.g., code/technical markers). */
  complexPatterns?: string[];
};

export type AutoRoutingConfig = {
  /** Enable auto model routing (default: false). */
  enabled?: boolean;
  /** Classifier mode: "rules" uses heuristic analysis (default), "llm" uses a lightweight model. */
  classifier?: "rules" | "llm";
  /** Model used for LLM classification (provider/model string). Only used when classifier is "llm". */
  classifierModel?: string;
  /** Routing rules per complexity tier. */
  rules?: AutoRoutingRule[];
  /** Complexity thresholds for the heuristic classifier. */
  thresholds?: AutoRoutingThresholds;
};

export type AgentModelListConfig = {
  primary?: string;
  fallbacks?: string[];
};

export type ModelPoolConfig = {
  locations?: ConfiguredModelLocation[];
  tiers?: ConfiguredModelTier[];
  requireCapabilities?: Array<"text" | "vision" | "coding" | "tools">;
  include?: string[];
  exclude?: string[];
};

export type AgentContextPruningConfig = {
  mode?: "off" | "cache-ttl";
  /** TTL to consider cache expired (duration string, default unit: minutes). */
  ttl?: string;
  keepLastAssistants?: number;
  softTrimRatio?: number;
  hardClearRatio?: number;
  minPrunableToolChars?: number;
  tools?: {
    allow?: string[];
    deny?: string[];
  };
  softTrim?: {
    maxChars?: number;
    headChars?: number;
    tailChars?: number;
  };
  hardClear?: {
    enabled?: boolean;
    placeholder?: string;
  };
};

export type CliBackendConfig = {
  /** CLI command to execute (absolute path or on PATH). */
  command: string;
  /** Base args applied to every invocation. */
  args?: string[];
  /** Output parsing mode (default: json). */
  output?: "json" | "text" | "jsonl";
  /** Output parsing mode when resuming a CLI session. */
  resumeOutput?: "json" | "text" | "jsonl";
  /** Prompt input mode (default: arg). */
  input?: "arg" | "stdin";
  /** Max prompt length for arg mode (if exceeded, stdin is used). */
  maxPromptArgChars?: number;
  /** Extra env vars injected for this CLI. */
  env?: Record<string, string>;
  /** Env vars to remove before launching this CLI. */
  clearEnv?: string[];
  /** Flag used to pass model id (e.g. --model). */
  modelArg?: string;
  /** Model aliases mapping (config model id → CLI model id). */
  modelAliases?: Record<string, string>;
  /** Flag used to pass session id (e.g. --session-id). */
  sessionArg?: string;
  /** Extra args used when resuming a session (use {sessionId} placeholder). */
  sessionArgs?: string[];
  /** Alternate args to use when resuming a session (use {sessionId} placeholder). */
  resumeArgs?: string[];
  /** When to pass session ids. */
  sessionMode?: "always" | "existing" | "none";
  /** JSON fields to read session id from (in order). */
  sessionIdFields?: string[];
  /** Flag used to pass system prompt. */
  systemPromptArg?: string;
  /** System prompt behavior (append vs replace). */
  systemPromptMode?: "append" | "replace";
  /** When to send system prompt. */
  systemPromptWhen?: "first" | "always" | "never";
  /** Flag used to pass image paths. */
  imageArg?: string;
  /** How to pass multiple images. */
  imageMode?: "repeat" | "list";
  /** Serialize runs for this CLI. */
  serialize?: boolean;
  /** Runtime reliability tuning for this backend's process lifecycle. */
  reliability?: {
    /** No-output watchdog tuning (fresh vs resumed runs). */
    watchdog?: {
      /** Fresh/new sessions (non-resume). */
      fresh?: {
        /** Fixed watchdog timeout in ms (overrides ratio when set). */
        noOutputTimeoutMs?: number;
        /** Fraction of overall timeout used when fixed timeout is not set. */
        noOutputTimeoutRatio?: number;
        /** Lower bound for computed watchdog timeout. */
        minMs?: number;
        /** Upper bound for computed watchdog timeout. */
        maxMs?: number;
      };
      /** Resume sessions. */
      resume?: {
        /** Fixed watchdog timeout in ms (overrides ratio when set). */
        noOutputTimeoutMs?: number;
        /** Fraction of overall timeout used when fixed timeout is not set. */
        noOutputTimeoutRatio?: number;
        /** Lower bound for computed watchdog timeout. */
        minMs?: number;
        /** Upper bound for computed watchdog timeout. */
        maxMs?: number;
      };
    };
  };
};

export type SubagentRoleToolPolicyConfig = {
  allow?: string[];
  deny?: string[];
};

export type SubagentRoleConfig = {
  description?: string;
  modelPool?: string;
  model?: string | { primary?: string; fallbacks?: string[] };
  thinking?: "off" | "minimal" | "low" | "medium" | "high" | "xhigh";
  systemPromptAppend?: string;
  tools?: SubagentRoleToolPolicyConfig;
  /** Default context injection spec applied when this role is used by a subagent. */
  context?: {
    recentTurns?: number;
    includeToolResults?: string[];
    maxContextChars?: number;
  };
};

export type SubagentPresetConfig = {
  description?: string;
  roles: string[];
  autoTrigger?: {
    keywords?: string[];
    minComplexity?: AutoRoutingComplexity;
    /** Override model pool when this preset is auto-triggered. */
    modelPool?: string;
    /** Override thinking level when this preset is auto-triggered. */
    thinking?: string;
  };
};

export type AgentP0Config = {
  soul?: string;
  identity?: string;
  user?: string;
};

export type ThinkingModelTier = "low" | "medium" | "high";

export type ThinkingModelsConfig = Partial<Record<ThinkingModelTier, string[]>>;

export type AgentDefaultsConfig = {
  /** Durable configured top-level agent name. Runtime still uses agent id "main". */
  name?: string;
  /** Primary model and fallbacks (provider/model). */
  model?: AgentModelListConfig;
  /** Default model pool name used for automatic selection. */
  modelPool?: string;
  /** Named model pools available to all agents. */
  modelPools?: Record<string, ModelPoolConfig>;
  /** Optional image-capable model and fallbacks (provider/model). */
  imageModel?: AgentModelListConfig;
  /** Agent working directory (preferred). Used as the default cwd for agent runs. */
  workspace?: string;
  /** Optional main-agent state directory override. */
  agentDir?: string;
  /** Optional repository root for system prompt runtime line (overrides auto-detect). */
  repoRoot?: string;
  /** Skip bootstrap (BOOTSTRAP.md creation, etc.) for pre-configured deployments. */
  skipBootstrap?: boolean;
  /** Max chars for injected bootstrap files before truncation (default: 20000). */
  bootstrapMaxChars?: number;
  /** Max total chars across all injected bootstrap files (default: 150000). */
  bootstrapTotalMaxChars?: number;
  /** Optional IANA timezone for the user (used in system prompt; defaults to host timezone). */
  userTimezone?: string;
  /** Time format in system prompt: auto (OS preference), 12-hour, or 24-hour. */
  timeFormat?: "auto" | "12" | "24";
  /**
   * Envelope timestamp timezone: "utc" (default), "local", "user", or an IANA timezone string.
   */
  envelopeTimezone?: string;
  /**
   * Include absolute timestamps in message envelopes ("on" | "off", default: "on").
   */
  envelopeTimestamp?: "on" | "off";
  /**
   * Include elapsed time in message envelopes ("on" | "off", default: "on").
   */
  envelopeElapsed?: "on" | "off";
  /** Optional context window cap (used for runtime estimates + status %). */
  contextTokens?: number;
  /** Optional CLI backends for text-only fallback (claude-cli, etc.). */
  cliBackends?: Record<string, CliBackendConfig>;
  /** Opt-in: prune old tool results from the LLM context to reduce token usage. */
  contextPruning?: AgentContextPruningConfig;
  /** Compaction tuning and pre-compaction memory flush behavior. */
  compaction?: AgentCompactionConfig;
  /** Vector memory search configuration (per-agent overrides supported). */
  memorySearch?: MemorySearchConfig;
  /** Optional allowlist of skills for the main durable agent. */
  skills?: string[];
  /** Auto model routing: classify message complexity and route to appropriate model. */
  autoRouting?: AutoRoutingConfig;
  /** Default thinking level when no /think directive is present. */
  thinkingDefault?: "off" | "minimal" | "low" | "medium" | "high" | "xhigh";
  /** Preferred models per thinking-level tier. Agent picks from the matching tier's list first. */
  thinkingModels?: ThinkingModelsConfig;
  /** Default verbose level when no /verbose directive is present. */
  verboseDefault?: "off" | "on" | "full";
  /** Default elevated level when no /elevated directive is present. */
  elevatedDefault?: "off" | "on" | "ask" | "full";
  /** Default block streaming level when no override is present. */
  blockStreamingDefault?: "off" | "on";
  /**
   * Block streaming boundary:
   * - "text_end": end of each assistant text content block (before tool calls)
   * - "message_end": end of the whole assistant message (may include tool blocks)
   */
  blockStreamingBreak?: "text_end" | "message_end";
  /** Soft block chunking for streamed replies (min/max chars, prefer paragraph/newline). */
  blockStreamingChunk?: BlockStreamingChunkConfig;
  /**
   * Block reply coalescing (merge streamed chunks before send).
   * idleMs: wait time before flushing when idle.
   */
  blockStreamingCoalesce?: BlockStreamingCoalesceConfig;
  /** Human-like delay between block replies. */
  humanDelay?: HumanDelayConfig;
  /** Structured P0 memory projected into SOUL.md / IDENTITY.md / USER.md. */
  p0?: AgentP0Config;
  /** Durable assistant identity for the main agent. */
  identity?: IdentityConfig;
  /** Durable group-chat settings for the main agent. */
  groupChat?: GroupChatConfig;
  timeoutSeconds?: number;
  /** Max inbound media size in MB for agent-visible attachments (text note or future image attach). */
  mediaMaxMb?: number;
  /**
   * Max image side length (pixels) when sanitizing base64 image payloads in transcripts/tool results.
   * Default: 1200.
   */
  imageMaxDimensionPx?: number;
  typingIntervalSeconds?: number;
  /** Typing indicator start mode (never|instant|thinking|message). */
  typingMode?: TypingMode;
  /** Periodic background heartbeat runs. */
  heartbeat?: {
    /** Heartbeat interval (duration string, default unit: minutes; default: 30m). */
    every?: string;
    /** Optional active-hours window (local time); heartbeats run only inside this window. */
    activeHours?: {
      /** Start time (24h, HH:MM). Inclusive. */
      start?: string;
      /** End time (24h, HH:MM). Exclusive. Use "24:00" for end-of-day. */
      end?: string;
      /** Timezone for the window ("user", "local", or IANA TZ id). Default: "user". */
      timezone?: string;
    };
    /** Heartbeat model override (provider/model). */
    model?: string;
    /** Session key for heartbeat runs ("main" or explicit session key). */
    session?: string;
    /** Delivery target ("last", "none", or a channel id). */
    target?: "last" | "none" | ChannelId;
    /** Optional delivery override (E.164 for WhatsApp, chat id for Telegram). Supports :topic:NNN suffix for Telegram topics. */
    to?: string;
    /** Optional account id for multi-account channels. */
    accountId?: string;
    /** Override the heartbeat prompt body (default: "Read HEARTBEAT.md if it exists (workspace context). Follow it strictly. Do not infer or repeat old tasks from prior chats. If nothing needs attention, reply HEARTBEAT_OK."). */
    prompt?: string;
    /** Max chars allowed after HEARTBEAT_OK before delivery (default: 30). */
    ackMaxChars?: number;
    /** Suppress tool error warning payloads during heartbeat runs. */
    suppressToolErrorWarnings?: boolean;
    /**
     * When enabled, deliver the model's reasoning payload for heartbeat runs (when available)
     * as a separate message prefixed with `Reasoning:` (same as `/reasoning on`).
     *
     * Default: false (only the final heartbeat payload is delivered).
     */
    includeReasoning?: boolean;
  };
  /** Max concurrent agent runs across all conversations. Default: 1 (sequential). */
  maxConcurrent?: number;
  /** Sub-agent defaults (spawned via sessions_spawn). */
  subagents?: {
    /** Max concurrent sub-agent runs (global lane: "subagent"). Default: 1. */
    maxConcurrent?: number;
    /** Maximum depth allowed for sessions_spawn chains. Default behavior: 1 (no nested spawns). */
    maxSpawnDepth?: number;
    /** Maximum active children a single requester session may spawn. Default behavior: 5. */
    maxChildrenPerAgent?: number;
    /** Auto-archive sub-agent sessions after N minutes (default: 60). */
    archiveAfterMinutes?: number;
    /** Default model selection for spawned sub-agents (string or {primary,fallbacks}). */
    model?: string | { primary?: string; fallbacks?: string[] };
    /** Default thinking level for spawned sub-agents (e.g. "off", "low", "medium", "high"). */
    thinking?: string;
    /** Built-in or custom role policies used by enhanced subagents. */
    roles?: Record<string, SubagentRoleConfig>;
    /** Named role bundles that can be expanded by orchestration tools. */
    presets?: Record<string, SubagentPresetConfig>;
    /** Default preset applied when orchestration does not choose one explicitly. */
    defaultPreset?: string;
  };
  /** Optional sandbox settings for non-main sessions. */
  sandbox?: {
    /** Enable sandboxing for sessions. */
    mode?: "off" | "non-main" | "all";
    /**
     * Agent workspace access inside the sandbox.
     * - "none": do not mount the agent workspace into the container; use a sandbox workspace under workspaceRoot
     * - "ro": mount the agent workspace read-only; disables write/edit tools
     * - "rw": mount the agent workspace read/write at /workspace; enables write/edit tools
     * - "isolated-rw": keep /workspace isolated under workspaceRoot (writable) and mount agent workspace read-only at /agent
     */
    workspaceAccess?: "none" | "ro" | "rw" | "isolated-rw";
    /**
     * Session tools visibility for sandboxed sessions.
     * - "spawned": only allow session tools to target the current session and sessions spawned from it (default)
     * - "all": allow session tools to target any session
     */
    sessionToolsVisibility?: "spawned" | "all";
    /** Container/workspace scope for sandbox isolation. */
    scope?: "session" | "agent" | "shared";
    /** Legacy alias for scope ("session" when true, "shared" when false). */
    perSession?: boolean;
    /** Root directory for sandbox workspaces. */
    workspaceRoot?: string;
    /** Docker-specific sandbox settings. */
    docker?: SandboxDockerSettings;
    /** Optional sandboxed browser settings. */
    browser?: SandboxBrowserSettings;
    /** Auto-prune sandbox containers. */
    prune?: SandboxPruneSettings;
  };
  /** Main-agent tool policy overrides. */
  tools?: AgentToolsConfig;
};

export type AgentCompactionMode = "default" | "safeguard";

export type AgentCompactionConfig = {
  /** Compaction summarization mode. */
  mode?: AgentCompactionMode;
  /** Minimum reserve tokens enforced for Pi compaction (0 disables the floor). */
  reserveTokensFloor?: number;
  /** Max share of context window for history during safeguard pruning (0.1–0.9, default 0.5). */
  maxHistoryShare?: number;
  /** Pre-compaction memory flush (agentic turn). Default: enabled. */
  memoryFlush?: AgentCompactionMemoryFlushConfig;
};

export type AgentCompactionMemoryFlushConfig = {
  /** Enable the pre-compaction memory flush (default: true). */
  enabled?: boolean;
  /** Run the memory flush when context is within this many tokens of the compaction threshold. */
  softThresholdTokens?: number;
  /** User prompt used for the memory flush turn (NO_REPLY is enforced if missing). */
  prompt?: string;
  /** System prompt appended for the memory flush turn. */
  systemPrompt?: string;
};
