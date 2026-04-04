import type { MarvConfig } from "../../../core/config/config.js";
import { isCronSessionKey } from "../../../routing/session-key.js";
import { normalizeMessageChannel } from "../../../utils/message-channel.js";
import { resolveAgentConfig } from "../../agent-scope.js";
import { normalizeToolName } from "./tool-policy.js";

export const TOOLSET_SELECTION_MODES = ["off", "observe", "enforce"] as const;
export type ToolsetSelectionMode = (typeof TOOLSET_SELECTION_MODES)[number];

export const TOOLSET_INTENTS = ["coding", "research", "messaging", "operator", "mixed"] as const;
export type ToolsetIntent = (typeof TOOLSET_INTENTS)[number];

export const TOOLSET_INTENT_SIGNAL_KINDS = [
  "explicit_instruction",
  "command",
  "channel",
  "tool_profile",
  "fallback",
] as const;
export type ToolsetIntentSignalKind = (typeof TOOLSET_INTENT_SIGNAL_KINDS)[number];

export type ToolsetIntentSignal = {
  kind: ToolsetIntentSignalKind;
  intent: ToolsetIntent;
  reason: string;
};

export const TOOLSET_INTENT_RESOLUTION_PRIORITY: readonly ToolsetIntentSignalKind[] = [
  "explicit_instruction",
  "command",
  "channel",
  "tool_profile",
  "fallback",
];

export type ToolsetPlan = {
  mode: ToolsetSelectionMode;
  intent: ToolsetIntent;
  reasons: string[];
  suppressedTools: string[];
  suppressedSkills: string[];
  effectiveToolCount: number;
  effectiveSkillCount: number;
};

export const TOOLSET_SUPPRESSION_GROUPS = {
  "gateway-cron": ["gateway", "cron"],
  "browser-canvas": ["browser", "canvas"],
  "messaging-send": ["message", "tts"],
  "code-edit": ["read", "write", "edit", "apply_patch"],
  "code-exec": ["exec", "process", "external_cli", "cli_profiles", "cli_invoke", "cli_synthesize"],
} as const satisfies Record<string, readonly string[]>;

export type ToolsetSuppressionGroup = keyof typeof TOOLSET_SUPPRESSION_GROUPS;

export const TOOLSET_INTENT_SUPPRESSION_GROUPS: Record<
  ToolsetIntent,
  readonly ToolsetSuppressionGroup[]
> = {
  coding: ["gateway-cron", "browser-canvas", "messaging-send"],
  research: ["browser-canvas", "gateway-cron"],
  messaging: ["code-edit", "code-exec", "browser-canvas"],
  operator: [],
  mixed: [],
};

export const SUPPRESSED_TOOL_REQUEST_RATE_WARNING_THRESHOLD = 0.05;

export type ToolsetPlanningContext = {
  cfg?: MarvConfig;
  agentId?: string;
  instruction?: string | null | undefined;
  directUserInstruction?: boolean;
  taskId?: string | null | undefined;
  sessionKey?: string | null | undefined;
  messageProvider?: string | null | undefined;
  toolProfile?: string | null | undefined;
  providerToolProfile?: string | null | undefined;
};

export type ToolsetPlanCounts = {
  toolNames?: string[];
  skillNames?: string[];
  suppressedSkills?: string[];
};

const TOOLSET_MESSAGING_CHANNELS = new Set([
  "telegram",
  "discord",
  "slack",
  "signal",
  "imessage",
  "googlechat",
  "whatsapp",
  "web",
  "matrix",
  "msteams",
  "zalo",
  "zalouser",
  "irc",
  "nostr",
]);

const TOOLSET_INTENT_KEYWORDS: Record<Exclude<ToolsetIntent, "mixed">, readonly string[]> = {
  coding: [
    "code",
    "coding",
    "implement",
    "fix",
    "patch",
    "test",
    "build",
    "compile",
    "repo",
    "repository",
    "workspace",
    "refactor",
    "file",
  ],
  research: [
    "research",
    "investigate",
    "analyze",
    "analysis",
    "search",
    "browse",
    "look up",
    "find out",
    "compare",
    "summarize",
  ],
  messaging: [
    "reply",
    "respond",
    "response",
    "message",
    "dm",
    "text back",
    "answer them",
    "send a message",
    "send back",
  ],
  operator: [
    "deploy",
    "restart",
    "status",
    "health",
    "logs",
    "configure",
    "config",
    "gateway",
    "server",
    "cron",
    "schedule",
  ],
};

const TOOL_REFERENCE_ALIASES = {
  browser: ["browser"],
  canvas: ["canvas"],
  gateway: ["gateway"],
  cron: ["cron", "schedule"],
  message: ["message", "send a message"],
  tts: ["tts", "text to speech"],
  read: ["read"],
  write: ["write"],
  edit: ["edit"],
  apply_patch: ["apply_patch", "apply patch", "patch"],
  exec: ["exec", "bash", "shell command", "run shell"],
  process: ["process", "background process"],
  external_cli: ["external cli"],
  cli_profiles: ["cli profiles"],
  cli_invoke: ["cli invoke"],
  cli_synthesize: ["cli synthesize", "synthesize"],
} as const satisfies Record<string, readonly string[]>;

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function normalizeText(value: string | null | undefined): string {
  return value?.trim().toLowerCase().replace(/\s+/g, " ") ?? "";
}

function hasKeyword(text: string, keyword: string): boolean {
  if (!text || !keyword) {
    return false;
  }
  const normalizedKeyword = normalizeText(keyword);
  if (!normalizedKeyword) {
    return false;
  }
  const pattern = new RegExp(`\\b${escapeRegex(normalizedKeyword).replace(/ /g, "\\s+")}\\b`, "i");
  return pattern.test(text);
}

export function expandSuppressedToolsForIntent(intent: ToolsetIntent): string[] {
  const groups = TOOLSET_INTENT_SUPPRESSION_GROUPS[intent];
  const expanded = groups.flatMap((group) => TOOLSET_SUPPRESSION_GROUPS[group]);
  return Array.from(new Set(expanded));
}

export function detectExplicitToolMentions(text: string | null | undefined): string[] {
  const input = text?.trim().toLowerCase();
  if (!input) {
    return [];
  }

  const matches: string[] = [];
  for (const [toolName, aliases] of Object.entries(TOOL_REFERENCE_ALIASES)) {
    if (
      aliases.some((alias) => {
        const pattern = new RegExp(`\\b${escapeRegex(alias.toLowerCase())}\\b`, "i");
        return pattern.test(input);
      })
    ) {
      matches.push(toolName);
    }
  }

  return matches.toSorted();
}

export function shouldForceMixedIntentFromInstruction(params: {
  instruction: string | null | undefined;
  intent: ToolsetIntent;
}): boolean {
  const suppressed = new Set(expandSuppressedToolsForIntent(params.intent));
  if (suppressed.size === 0) {
    return false;
  }
  return detectExplicitToolMentions(params.instruction).some((toolName) =>
    suppressed.has(toolName),
  );
}

export function createEmptyToolsetPlan(mode: ToolsetSelectionMode = "off"): ToolsetPlan {
  return {
    mode,
    intent: "mixed",
    reasons: [],
    suppressedTools: [],
    suppressedSkills: [],
    effectiveToolCount: 0,
    effectiveSkillCount: 0,
  };
}

function resolveSelectionConfig(params: {
  cfg?: MarvConfig;
  agentId?: string;
}): { enabled?: boolean; mode?: ToolsetSelectionMode } | undefined {
  const cfg = params.cfg;
  if (!cfg) {
    return undefined;
  }
  const agentId = params.agentId?.trim() || "main";
  return resolveAgentConfig(cfg, agentId)?.tools?.selection;
}

export function resolveToolsetSelectionMode(params: {
  cfg?: MarvConfig;
  agentId?: string;
}): ToolsetSelectionMode {
  const selection = resolveSelectionConfig(params);
  if (!selection) {
    return "off";
  }
  if (selection.enabled === false) {
    return "off";
  }
  if (selection.mode) {
    return selection.mode;
  }
  return selection.enabled === true ? "observe" : "off";
}

function collectInstructionSignals(instruction: string): ToolsetIntentSignal[] {
  const normalized = normalizeText(instruction);
  if (!normalized) {
    return [];
  }
  const signals: ToolsetIntentSignal[] = [];
  for (const [intent, keywords] of Object.entries(TOOLSET_INTENT_KEYWORDS)) {
    if (keywords.some((keyword) => hasKeyword(normalized, keyword))) {
      signals.push({
        kind: "explicit_instruction",
        intent: intent as Exclude<ToolsetIntent, "mixed">,
        reason: `instruction suggests ${intent}`,
      });
    }
  }
  return signals;
}

function collectCommandSignals(params: ToolsetPlanningContext): ToolsetIntentSignal[] {
  const signals: ToolsetIntentSignal[] = [];
  if (params.taskId?.trim()) {
    signals.push({
      kind: "command",
      intent: "coding",
      reason: `task context ${params.taskId.trim()}`,
    });
  }
  if (params.sessionKey && isCronSessionKey(params.sessionKey)) {
    signals.push({
      kind: "command",
      intent: "operator",
      reason: "cron session",
    });
  }
  return signals;
}

function collectChannelSignals(messageProvider: string | null | undefined): ToolsetIntentSignal[] {
  const channel = normalizeMessageChannel(messageProvider);
  if (!channel || !TOOLSET_MESSAGING_CHANNELS.has(channel)) {
    return [];
  }
  return [
    {
      kind: "channel",
      intent: "messaging",
      reason: `channel ${channel}`,
    },
  ];
}

function collectToolProfileSignals(params: {
  toolProfile?: string | null | undefined;
  providerToolProfile?: string | null | undefined;
}): ToolsetIntentSignal[] {
  const signals: ToolsetIntentSignal[] = [];
  const pushIfKnown = (profile: string | null | undefined, source: string) => {
    const normalized = profile?.trim().toLowerCase();
    if (normalized === "coding" || normalized === "messaging") {
      signals.push({
        kind: "tool_profile",
        intent: normalized,
        reason: `${source} profile ${normalized}`,
      });
    }
  };
  pushIfKnown(params.toolProfile, "tool");
  pushIfKnown(params.providerToolProfile, "provider");
  return signals;
}

export function resolveToolsetIntentSignals(params: ToolsetPlanningContext): ToolsetIntentSignal[] {
  const instruction = normalizeText(params.instruction);
  return [
    ...collectInstructionSignals(instruction),
    ...collectCommandSignals(params),
    ...collectChannelSignals(params.messageProvider),
    ...collectToolProfileSignals(params),
    {
      kind: "fallback",
      intent: "mixed",
      reason: "fallback to mixed",
    },
  ];
}

export function resolveToolsetIntent(params: ToolsetPlanningContext): {
  intent: ToolsetIntent;
  reasons: string[];
  signals: ToolsetIntentSignal[];
} {
  const signals = resolveToolsetIntentSignals(params);
  for (const kind of TOOLSET_INTENT_RESOLUTION_PRIORITY) {
    const current = signals.filter((signal) => signal.kind === kind);
    if (current.length === 0) {
      continue;
    }
    const intents = Array.from(new Set(current.map((signal) => signal.intent)));
    if (intents.length === 1) {
      const resolved = intents[0] ?? "mixed";
      if (
        resolved !== "mixed" &&
        params.directUserInstruction !== false &&
        shouldForceMixedIntentFromInstruction({
          instruction: params.instruction,
          intent: resolved,
        })
      ) {
        return {
          intent: "mixed",
          reasons: [
            ...current.map((signal) => signal.reason),
            "explicit user tool request forces mixed intent",
          ],
          signals,
        };
      }
      return {
        intent: resolved,
        reasons: current.map((signal) => signal.reason),
        signals,
      };
    }
    return {
      intent: "mixed",
      reasons: [...current.map((signal) => signal.reason), `conflicting ${kind} signals`],
      signals,
    };
  }
  return {
    intent: "mixed",
    reasons: ["fallback to mixed"],
    signals,
  };
}

function resolveSuppressedToolNames(intent: ToolsetIntent, toolNames?: string[]): string[] {
  const suppressed = new Set(expandSuppressedToolsForIntent(intent).map(normalizeToolName));
  if (suppressed.size === 0) {
    return [];
  }
  if (!toolNames || toolNames.length === 0) {
    return Array.from(suppressed);
  }
  const resolved: string[] = [];
  for (const toolName of toolNames) {
    const normalized = normalizeToolName(toolName);
    if (suppressed.has(normalized)) {
      resolved.push(toolName);
    }
  }
  return Array.from(new Set(resolved));
}

export function isToolSuppressedByPlan(toolName: string, plan: ToolsetPlan): boolean {
  if (plan.mode !== "enforce" || plan.suppressedTools.length === 0) {
    return false;
  }
  const normalized = normalizeToolName(toolName);
  return plan.suppressedTools.some((entry) => normalizeToolName(entry) === normalized);
}

export function applyToolsetPlanToToolNames(toolNames: string[], plan: ToolsetPlan): string[] {
  if (plan.mode !== "enforce" || plan.suppressedTools.length === 0) {
    return [...toolNames];
  }
  return toolNames.filter((toolName) => !isToolSuppressedByPlan(toolName, plan));
}

export function applyToolsetPlanToTools<T extends { name: string }>(
  tools: T[],
  plan: ToolsetPlan,
): T[] {
  if (plan.mode !== "enforce" || plan.suppressedTools.length === 0) {
    return [...tools];
  }
  return tools.filter((tool) => !isToolSuppressedByPlan(tool.name, plan));
}

export function withToolsetPlanCounts(plan: ToolsetPlan, counts?: ToolsetPlanCounts): ToolsetPlan {
  const toolNames = counts?.toolNames ?? [];
  const skillNames = counts?.skillNames ?? [];
  const suppressedTools = resolveSuppressedToolNames(plan.intent, toolNames);
  const effectiveToolNames =
    plan.mode === "enforce" && suppressedTools.length > 0
      ? toolNames.filter(
          (toolName) =>
            !suppressedTools.some(
              (suppressedTool) => normalizeToolName(suppressedTool) === normalizeToolName(toolName),
            ),
        )
      : [...toolNames];
  const suppressedSkills = Array.from(new Set(counts?.suppressedSkills ?? []));
  const effectiveSkillCount = Math.max(0, skillNames.length - suppressedSkills.length);
  return {
    ...plan,
    suppressedTools,
    suppressedSkills,
    effectiveToolCount: effectiveToolNames.length,
    effectiveSkillCount,
  };
}

export function createToolsetPlan(
  params: ToolsetPlanningContext & {
    mode?: ToolsetSelectionMode;
    counts?: ToolsetPlanCounts;
  },
): ToolsetPlan {
  const mode = params.mode ?? resolveToolsetSelectionMode(params);
  const resolved = resolveToolsetIntent(params);
  return withToolsetPlanCounts(
    {
      mode,
      intent: resolved.intent,
      reasons: resolved.reasons,
      suppressedTools: [],
      suppressedSkills: [],
      effectiveToolCount: 0,
      effectiveSkillCount: 0,
    },
    params.counts,
  );
}
