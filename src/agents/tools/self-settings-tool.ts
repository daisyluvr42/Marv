import fs from "node:fs/promises";
import path from "node:path";
import { Type } from "@sinclair/typebox";
import { resolveElevatedPermissions } from "../../auto-reply/reply/reply-elevated.js";
import type { MsgContext } from "../../auto-reply/templating.js";
import { type MarvConfig, loadConfig, writeConfigFile } from "../../core/config/config.js";
import {
  loadSessionStore,
  updateSessionStore,
  type SessionEntry,
} from "../../core/config/sessions.js";
import type { ExternalCliAdapterId } from "../../core/config/types.tools.js";
import type { SessionsPatchParams } from "../../core/gateway/protocol/index.js";
import { sessionsHandlers } from "../../core/gateway/server-methods/sessions.js";
import {
  pruneLegacyStoreKeys,
  resolveGatewaySessionStoreTarget,
  resolveSessionModelRef,
} from "../../core/gateway/session-utils.js";
import { applySessionsPatchToStore } from "../../core/gateway/sessions-patch.js";
import { computeNextRunAtMs } from "../../cron/schedule.js";
import { resolveAgentIdFromSessionKey } from "../../routing/session-key.js";
import type { GatewayMessageChannel } from "../../utils/message-channel.js";
import { resolveAgentDir, resolveAgentWorkspaceDir } from "../agent-scope.js";
import { ensureAuthProfileStore } from "../auth-profiles.js";
import { loadModelCatalog } from "../model/model-catalog.js";
import { normalizeProviderId, resolveDefaultModelForAgent } from "../model/model-selection.js";
import {
  refreshRuntimeModelRegistry,
  resolveRuntimeRegistryPathForDisplay,
} from "../model/runtime-model-registry.js";
import { DEFAULT_HEARTBEAT_FILENAME } from "../workspace.js";
import type { AnyAgentTool } from "./common.js";
import { readNumberParam, readStringParam, ToolInputError } from "./common.js";
import {
  handleConfigGet,
  handleConfigSet,
  handleConfigUnset,
  handleSkillInstall,
  handleSkillList,
  handleSkillSourceAdd,
  handleSkillSourceList,
} from "./config-manage.js";
import { normalizeExternalCliId } from "./external-cli-adapters.js";

const SelfSettingsToolSchema = Type.Object(
  {
    model: Type.Optional(Type.String()),
    authProfile: Type.Optional(Type.String()),
    thinkingLevel: Type.Optional(Type.String()),
    verboseLevel: Type.Optional(Type.String()),
    reasoningLevel: Type.Optional(Type.String()),
    responseUsage: Type.Optional(Type.String()),
    elevatedLevel: Type.Optional(Type.String()),
    execHost: Type.Optional(Type.String()),
    execSecurity: Type.Optional(Type.String()),
    execAsk: Type.Optional(Type.String()),
    execNode: Type.Optional(Type.String()),
    queueMode: Type.Optional(Type.String()),
    queueDebounceMs: Type.Optional(Type.Number({ minimum: 0 })),
    queueCap: Type.Optional(Type.Number({ minimum: 1 })),
    queueDrop: Type.Optional(Type.String()),
    modelRegistryAction: Type.Optional(Type.String()),
    sessionAction: Type.Optional(Type.String()),
    deepMemoryEnabled: Type.Optional(Type.Boolean()),
    deepMemorySchedule: Type.Optional(Type.String()),
    deepMemoryModelProvider: Type.Optional(Type.String()),
    deepMemoryModelApi: Type.Optional(Type.String()),
    deepMemoryModel: Type.Optional(Type.String()),
    deepMemoryBaseUrl: Type.Optional(Type.String()),
    deepMemoryTimeoutMs: Type.Optional(Type.Number({ minimum: 1 })),
    deepMemoryClusterSummarization: Type.Optional(Type.Boolean()),
    deepMemoryConflictJudgment: Type.Optional(Type.Boolean()),
    deepMemoryCrossScopeReflection: Type.Optional(Type.Boolean()),
    deepMemoryMaxItems: Type.Optional(Type.Number({ minimum: 1 })),
    deepMemoryMaxReflections: Type.Optional(Type.Number({ minimum: 1 })),
    memorySearchEnabled: Type.Optional(Type.Boolean()),
    memorySearchProvider: Type.Optional(Type.String()),
    memorySearchModel: Type.Optional(Type.String()),
    memorySearchDimensions: Type.Optional(Type.Number({ minimum: 1 })),
    memorySearchFallback: Type.Optional(Type.String()),
    memorySearchRemoteBaseUrl: Type.Optional(Type.String()),
    memorySearchRemoteApiKey: Type.Optional(Type.String()),
    memorySearchRerankerEnabled: Type.Optional(Type.Boolean()),
    memorySearchRerankerApiUrl: Type.Optional(Type.String()),
    memorySearchRerankerModel: Type.Optional(Type.String()),
    memorySearchRerankerApiKey: Type.Optional(Type.String()),
    memorySearchRerankerMaxCandidates: Type.Optional(Type.Number({ minimum: 1 })),
    memorySearchRerankerFtsFirst: Type.Optional(Type.Boolean()),
    externalCliEnabled: Type.Optional(Type.Boolean()),
    externalCliDefault: Type.Optional(Type.String()),
    externalCliAvailableBrands: Type.Optional(Type.String()),
    heartbeatEvery: Type.Optional(Type.String()),
    heartbeatPrompt: Type.Optional(Type.String()),
    heartbeatModel: Type.Optional(Type.String()),
    heartbeatTarget: Type.Optional(Type.String()),
    heartbeatTo: Type.Optional(Type.String()),
    heartbeatAccountId: Type.Optional(Type.String()),
    heartbeatIncludeReasoning: Type.Optional(Type.Boolean()),
    heartbeatSuppressToolErrorWarnings: Type.Optional(Type.Boolean()),
    heartbeatAckMaxChars: Type.Optional(Type.Number({ minimum: 0 })),
    heartbeatActiveHoursStart: Type.Optional(Type.String()),
    heartbeatActiveHoursEnd: Type.Optional(Type.String()),
    heartbeatActiveHoursTimezone: Type.Optional(Type.String()),
    heartbeatFileAction: Type.Optional(Type.String()),
    heartbeatFileContent: Type.Optional(Type.String()),
    configGet: Type.Optional(
      Type.String({
        description:
          "Read a config value by dot-path (e.g. 'gateway.port', 'tools.web.search.provider', 'models.catalog'). Returns the current value.",
      }),
    ),
    configSet: Type.Optional(
      Type.String({
        description:
          "Set a config value. Format: 'path=value' where path is dot-notation (e.g. 'tools.web.search.provider=tavily', 'gateway.port=4242'). Value is parsed as JSON if valid, otherwise stored as string.",
      }),
    ),
    configUnset: Type.Optional(
      Type.String({
        description: "Remove a config key by dot-path (e.g. 'tools.web.search.tavily.apiKey').",
      }),
    ),
    skillInstall: Type.Optional(
      Type.String({
        description:
          "Install a skill/plugin from a source. Supported formats: npm package name (e.g. '@marv/tavily'), GitHub repo URL (e.g. 'https://github.com/user/repo'), or a local path. The source is cloned/downloaded, validated, and installed to ~/.marv/extensions/.",
      }),
    ),
    skillList: Type.Optional(
      Type.Boolean({
        description: "List all installed skills/plugins and their status.",
      }),
    ),
    skillSourceAdd: Type.Optional(
      Type.String({
        description:
          "Add a skill registry source URL. The agent can later search this source for available skills. Format: 'name=url' (e.g. 'community=https://raw.githubusercontent.com/user/skills-registry/main/registry.json').",
      }),
    ),
    skillSourceList: Type.Optional(
      Type.Boolean({
        description: "List all configured skill registry sources.",
      }),
    ),
  },
  { additionalProperties: false },
);

const GENERIC_DENIED_MESSAGE = "This setting request cannot be applied right now.";
const GENERIC_INVALID_MESSAGE = "I can't apply that session setting directly.";
const REDACTED_VALUE = "[redacted]";

type MemorySearchProvider = "openai" | "gemini" | "local" | "voyage";
type MemorySearchFallback = MemorySearchProvider | "none";
type HeartbeatFileAction = "replace" | "append" | "clear";

type SelfSettingsArgs = Record<string, unknown>;

function buildGenericDeniedResult() {
  return {
    content: [{ type: "text" as const, text: GENERIC_DENIED_MESSAGE }],
    details: { ok: false, applied: false, denied: true },
  };
}

function buildInvalidResult() {
  return {
    content: [{ type: "text" as const, text: GENERIC_INVALID_MESSAGE }],
    details: { ok: false, applied: false, invalid: true },
  };
}

function normalizeSessionAction(raw?: string): "new" | "reset" | undefined {
  const normalized = raw?.trim().toLowerCase();
  if (normalized === "new" || normalized === "reset") {
    return normalized;
  }
  return undefined;
}

function normalizeModelRegistryAction(raw?: string): "refresh" | undefined {
  const normalized = raw?.trim().toLowerCase();
  if (normalized === "refresh" || normalized === "update" || normalized === "sync") {
    return "refresh";
  }
  return undefined;
}

function normalizePatchString(raw: string | undefined): string | null | undefined {
  if (raw === undefined) {
    return undefined;
  }
  const trimmed = raw.trim();
  if (!trimmed) {
    return undefined;
  }
  const normalized = trimmed.toLowerCase();
  if (normalized === "default" || normalized === "clear" || normalized === "reset") {
    return null;
  }
  return trimmed;
}

function normalizeDeepMemoryApi(
  raw: string | undefined,
): "ollama" | "openai-completions" | null | undefined {
  const normalized = normalizePatchString(raw);
  if (normalized == null) {
    return normalized;
  }
  const lower = normalized.trim().toLowerCase();
  if (lower === "ollama") {
    return "ollama";
  }
  if (lower === "openai-completions" || lower === "openai" || lower === "openai-compatible") {
    return "openai-completions";
  }
  return undefined;
}

function normalizeMemorySearchProvider(
  raw: string | undefined,
): MemorySearchProvider | null | undefined {
  const normalized = normalizePatchString(raw);
  if (normalized == null) {
    return normalized;
  }
  const lower = normalized.toLowerCase();
  if (lower === "openai" || lower === "gemini" || lower === "local" || lower === "voyage") {
    return lower;
  }
  return undefined;
}

function normalizeMemorySearchFallback(
  raw: string | undefined,
): MemorySearchFallback | null | undefined {
  const normalized = normalizePatchString(raw);
  if (normalized == null) {
    return normalized;
  }
  const lower = normalized.toLowerCase();
  if (
    lower === "openai" ||
    lower === "gemini" ||
    lower === "local" ||
    lower === "voyage" ||
    lower === "none"
  ) {
    return lower;
  }
  return undefined;
}

function readBooleanParam(params: Record<string, unknown>, key: string): boolean | undefined {
  const raw = params[key];
  if (typeof raw === "boolean") {
    return raw;
  }
  if (typeof raw !== "string") {
    return undefined;
  }
  const normalized = raw.trim().toLowerCase();
  if (["true", "yes", "on", "enabled"].includes(normalized)) {
    return true;
  }
  if (["false", "no", "off", "disabled"].includes(normalized)) {
    return false;
  }
  return undefined;
}

function normalizeAuthProfile(raw: string | undefined): string | null | undefined {
  if (raw === undefined) {
    return undefined;
  }
  const trimmed = raw.trim();
  if (!trimmed) {
    return undefined;
  }
  const normalized = trimmed.toLowerCase();
  if (
    normalized === "default" ||
    normalized === "clear" ||
    normalized === "reset" ||
    normalized === "auto"
  ) {
    return null;
  }
  return trimmed;
}

function normalizeExternalCliDefault(
  raw: string | undefined,
): ExternalCliAdapterId | null | undefined {
  const normalized = normalizePatchString(raw);
  if (normalized == null) {
    return normalized;
  }
  return normalizeExternalCliId(normalized) ?? undefined;
}

function normalizeExternalCliAvailableBrands(
  raw: string | undefined,
): ExternalCliAdapterId[] | undefined {
  if (raw === undefined) {
    return undefined;
  }
  const trimmed = raw.trim();
  if (!trimmed) {
    return [];
  }
  const parts = trimmed
    .split(/[,\n]/g)
    .flatMap((part) => part.split(/\s+/g))
    .map((part) => part.trim())
    .filter(Boolean);
  const out: ExternalCliAdapterId[] = [];
  const seen = new Set<string>();
  for (const part of parts) {
    const normalized = normalizeExternalCliId(part);
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

function normalizeHeartbeatFileAction(raw?: string): HeartbeatFileAction | undefined {
  const normalized = raw?.trim().toLowerCase();
  if (
    normalized === "replace" ||
    normalized === "set" ||
    normalized === "rewrite" ||
    normalized === "overwrite"
  ) {
    return "replace";
  }
  if (normalized === "append" || normalized === "add") {
    return "append";
  }
  if (normalized === "clear" || normalized === "empty" || normalized === "reset") {
    return "clear";
  }
  return undefined;
}

function buildElevatedMsgContext(opts: {
  agentChannel?: GatewayMessageChannel;
  agentAccountId?: string;
  agentTo?: string;
  senderId?: string;
  senderName?: string;
  senderUsername?: string;
  senderE164?: string;
}): MsgContext {
  return {
    Provider: opts.agentChannel,
    AccountId: opts.agentAccountId,
    To: opts.agentTo,
    From: opts.senderE164 ?? opts.senderId,
    SenderId: opts.senderId,
    SenderName: opts.senderName,
    SenderUsername: opts.senderUsername,
    SenderE164: opts.senderE164,
  };
}

async function invokeSessionReset(params: {
  key: string;
  reason: "new" | "reset";
}): Promise<{ ok: true; entry?: SessionEntry } | { ok: false }> {
  let response:
    | { ok: true; result?: { entry?: SessionEntry } }
    | { ok: false; error?: unknown }
    | undefined;
  await sessionsHandlers["sessions.reset"]({
    req: {
      id: "self-settings-reset",
      method: "sessions.reset",
      params: { key: params.key, reason: params.reason },
    } as never,
    params: { key: params.key, reason: params.reason },
    client: null,
    isWebchatConnect: () => false,
    respond: (ok, result, error) => {
      response = ok
        ? { ok: true, result: result as { entry?: SessionEntry } }
        : { ok: false, error };
    },
    context: {} as never,
  });
  if (!response?.ok) {
    return { ok: false };
  }
  return { ok: true, entry: response.result?.entry };
}

function resolveCurrentSessionEntry(params: {
  store: Record<string, SessionEntry>;
  sessionKeys: string[];
}): SessionEntry | undefined {
  for (const key of params.sessionKeys) {
    const entry = params.store[key];
    if (entry) {
      return entry;
    }
  }
  return undefined;
}

function isValidCronExpr(expr: string): boolean {
  try {
    const next = computeNextRunAtMs({ kind: "cron", expr }, Date.now());
    return next !== undefined;
  } catch {
    return false;
  }
}

function applyOptionalStringPatch(
  target: Record<string, unknown>,
  key: string,
  value: string | null | undefined,
) {
  if (value === undefined) {
    return;
  }
  if (value === null) {
    delete target[key];
    return;
  }
  target[key] = value;
}

function pruneEmptyObject(
  value: Record<string, unknown> | undefined,
): Record<string, unknown> | undefined {
  if (!value) {
    return undefined;
  }
  return Object.keys(value).length > 0 ? value : undefined;
}

function summarizeSensitivePatch(
  label: string,
  value: string | null | undefined,
): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  return `${label} ${value === null ? "default" : "configured"}`;
}

export function createSelfSettingsTool(opts?: {
  agentSessionKey?: string;
  config?: MarvConfig;
  agentChannel?: GatewayMessageChannel;
  agentAccountId?: string;
  agentTo?: string;
  senderId?: string;
  senderName?: string;
  senderUsername?: string;
  senderE164?: string;
  directUserInstruction?: boolean;
}): AnyAgentTool {
  return {
    label: "Self Settings",
    name: "self_settings",
    description:
      "Apply direct self-setting requests for the current session, plus restricted shared deep-memory, shared memory-search, and external CLI fallback settings: model, auth profile, thinking, verbose, reasoning, usage, elevated, exec defaults, queue behavior, session reset/new, runtime model-registry refresh, managed deep-consolidation settings, shared memory-search embedding/reranker defaults, external CLI availability/default preferences, config get/set/unset by dot-path, or skill install/list/source management.",
    parameters: SelfSettingsToolSchema,
    execute: async (_toolCallId, args) => {
      if (!opts?.agentSessionKey?.trim()) {
        throw new ToolInputError("sessionKey required");
      }

      const params = args as SelfSettingsArgs;

      // --- Config get/set/unset and skill management (early return) ---
      const configGetPath = readStringParam(params, "configGet");
      const configSetExpr = readStringParam(params, "configSet");
      const configUnsetPath = readStringParam(params, "configUnset");
      const skillInstallSource = readStringParam(params, "skillInstall");
      const skillListFlag = readBooleanParam(params, "skillList");
      const skillSourceAddExpr = readStringParam(params, "skillSourceAdd");
      const skillSourceListFlag = readBooleanParam(params, "skillSourceList");

      const hasConfigOrSkillAction =
        configGetPath !== undefined ||
        configSetExpr !== undefined ||
        configUnsetPath !== undefined ||
        skillInstallSource !== undefined ||
        skillListFlag !== undefined ||
        skillSourceAddExpr !== undefined ||
        skillSourceListFlag !== undefined;

      if (hasConfigOrSkillAction) {
        const results: Array<{ action: string; result: unknown }> = [];

        if (configGetPath) {
          results.push({ action: "configGet", result: await handleConfigGet(configGetPath) });
        }
        if (configSetExpr) {
          results.push({ action: "configSet", result: await handleConfigSet(configSetExpr) });
        }
        if (configUnsetPath) {
          results.push({ action: "configUnset", result: await handleConfigUnset(configUnsetPath) });
        }
        if (skillInstallSource) {
          results.push({
            action: "skillInstall",
            result: await handleSkillInstall(skillInstallSource),
          });
        }
        if (skillListFlag) {
          results.push({ action: "skillList", result: await handleSkillList() });
        }
        if (skillSourceAddExpr) {
          results.push({
            action: "skillSourceAdd",
            result: await handleSkillSourceAdd(skillSourceAddExpr),
          });
        }
        if (skillSourceListFlag) {
          results.push({ action: "skillSourceList", result: await handleSkillSourceList() });
        }

        const summary = results.map((r) => `${r.action}: ${JSON.stringify(r.result)}`).join("\n");
        return {
          content: [{ type: "text" as const, text: summary }],
          details: { ok: true, applied: true, configActions: results },
        };
      }

      const cfg = opts.config ?? loadConfig();
      const sessionKey = opts.agentSessionKey.trim();
      const agentId = resolveAgentIdFromSessionKey(sessionKey);
      const sessionAction = normalizeSessionAction(readStringParam(params, "sessionAction"));
      const modelRegistryAction = normalizeModelRegistryAction(
        readStringParam(params, "modelRegistryAction"),
      );
      const model = normalizePatchString(readStringParam(params, "model"));
      const authProfile = normalizeAuthProfile(readStringParam(params, "authProfile"));
      const thinkingLevel = normalizePatchString(readStringParam(params, "thinkingLevel"));
      const verboseLevel = normalizePatchString(readStringParam(params, "verboseLevel"));
      const reasoningLevel = normalizePatchString(readStringParam(params, "reasoningLevel"));
      const responseUsage = normalizePatchString(readStringParam(params, "responseUsage"));
      const elevatedLevel = normalizePatchString(readStringParam(params, "elevatedLevel"));
      const execHost = normalizePatchString(readStringParam(params, "execHost"));
      const execSecurity = normalizePatchString(readStringParam(params, "execSecurity"));
      const execAsk = normalizePatchString(readStringParam(params, "execAsk"));
      const execNode = normalizePatchString(readStringParam(params, "execNode"));
      const queueMode = normalizePatchString(readStringParam(params, "queueMode"));
      const queueDrop = normalizePatchString(readStringParam(params, "queueDrop"));
      const queueDebounceMs = readNumberParam(params, "queueDebounceMs");
      const queueCap = readNumberParam(params, "queueCap");
      const deepMemoryEnabled = readBooleanParam(params, "deepMemoryEnabled");
      const deepMemorySchedule = normalizePatchString(
        readStringParam(params, "deepMemorySchedule"),
      );
      const deepMemoryModelProvider = normalizePatchString(
        readStringParam(params, "deepMemoryModelProvider"),
      );
      const deepMemoryModelApi = normalizeDeepMemoryApi(
        readStringParam(params, "deepMemoryModelApi"),
      );
      const deepMemoryModel = normalizePatchString(readStringParam(params, "deepMemoryModel"));
      const deepMemoryBaseUrl = normalizePatchString(readStringParam(params, "deepMemoryBaseUrl"));
      const deepMemoryTimeoutMs = readNumberParam(params, "deepMemoryTimeoutMs");
      const deepMemoryClusterSummarization = readBooleanParam(
        params,
        "deepMemoryClusterSummarization",
      );
      const deepMemoryConflictJudgment = readBooleanParam(params, "deepMemoryConflictJudgment");
      const deepMemoryCrossScopeReflection = readBooleanParam(
        params,
        "deepMemoryCrossScopeReflection",
      );
      const deepMemoryMaxItems = readNumberParam(params, "deepMemoryMaxItems");
      const deepMemoryMaxReflections = readNumberParam(params, "deepMemoryMaxReflections");
      const memorySearchEnabled = readBooleanParam(params, "memorySearchEnabled");
      const memorySearchProvider = normalizeMemorySearchProvider(
        readStringParam(params, "memorySearchProvider"),
      );
      const memorySearchModel = normalizePatchString(readStringParam(params, "memorySearchModel"));
      const memorySearchDimensions = readNumberParam(params, "memorySearchDimensions");
      const memorySearchFallback = normalizeMemorySearchFallback(
        readStringParam(params, "memorySearchFallback"),
      );
      const memorySearchRemoteBaseUrl = normalizePatchString(
        readStringParam(params, "memorySearchRemoteBaseUrl"),
      );
      const memorySearchRemoteApiKey = normalizePatchString(
        readStringParam(params, "memorySearchRemoteApiKey"),
      );
      const memorySearchRerankerEnabled = readBooleanParam(params, "memorySearchRerankerEnabled");
      const memorySearchRerankerApiUrl = normalizePatchString(
        readStringParam(params, "memorySearchRerankerApiUrl"),
      );
      const memorySearchRerankerModel = normalizePatchString(
        readStringParam(params, "memorySearchRerankerModel"),
      );
      const memorySearchRerankerApiKey = normalizePatchString(
        readStringParam(params, "memorySearchRerankerApiKey"),
      );
      const memorySearchRerankerMaxCandidates = readNumberParam(
        params,
        "memorySearchRerankerMaxCandidates",
      );
      const memorySearchRerankerFtsFirst = readBooleanParam(params, "memorySearchRerankerFtsFirst");
      const externalCliEnabled = readBooleanParam(params, "externalCliEnabled");
      const externalCliDefault = normalizeExternalCliDefault(
        readStringParam(params, "externalCliDefault"),
      );
      const externalCliAvailableBrandsRaw = readStringParam(params, "externalCliAvailableBrands");
      const externalCliAvailableBrands = normalizeExternalCliAvailableBrands(
        externalCliAvailableBrandsRaw,
      );
      const heartbeatEvery = normalizePatchString(readStringParam(params, "heartbeatEvery"));
      const heartbeatPrompt = normalizePatchString(readStringParam(params, "heartbeatPrompt"));
      const heartbeatModel = normalizePatchString(readStringParam(params, "heartbeatModel"));
      const heartbeatTarget = normalizePatchString(readStringParam(params, "heartbeatTarget"));
      const heartbeatTo = normalizePatchString(readStringParam(params, "heartbeatTo"));
      const heartbeatAccountId = normalizePatchString(
        readStringParam(params, "heartbeatAccountId"),
      );
      const heartbeatIncludeReasoning = readBooleanParam(params, "heartbeatIncludeReasoning");
      const heartbeatSuppressToolErrorWarnings = readBooleanParam(
        params,
        "heartbeatSuppressToolErrorWarnings",
      );
      const heartbeatAckMaxChars = readNumberParam(params, "heartbeatAckMaxChars");
      const heartbeatActiveHoursStart = normalizePatchString(
        readStringParam(params, "heartbeatActiveHoursStart"),
      );
      const heartbeatActiveHoursEnd = normalizePatchString(
        readStringParam(params, "heartbeatActiveHoursEnd"),
      );
      const heartbeatActiveHoursTimezone = normalizePatchString(
        readStringParam(params, "heartbeatActiveHoursTimezone"),
      );
      const heartbeatFileActionRaw = readStringParam(params, "heartbeatFileAction");
      const heartbeatFileAction = normalizeHeartbeatFileAction(heartbeatFileActionRaw);
      const heartbeatFileContent = readStringParam(params, "heartbeatFileContent", {
        trim: false,
        allowEmpty: true,
      });
      const hasDeepMemoryConfigChange =
        deepMemoryEnabled !== undefined ||
        deepMemorySchedule !== undefined ||
        deepMemoryModelProvider !== undefined ||
        readStringParam(params, "deepMemoryModelApi") !== undefined ||
        deepMemoryModel !== undefined ||
        deepMemoryBaseUrl !== undefined ||
        deepMemoryTimeoutMs !== undefined ||
        deepMemoryClusterSummarization !== undefined ||
        deepMemoryConflictJudgment !== undefined ||
        deepMemoryCrossScopeReflection !== undefined ||
        deepMemoryMaxItems !== undefined ||
        deepMemoryMaxReflections !== undefined;
      const hasMemorySearchConfigChange =
        memorySearchEnabled !== undefined ||
        readStringParam(params, "memorySearchProvider") !== undefined ||
        memorySearchModel !== undefined ||
        memorySearchDimensions !== undefined ||
        readStringParam(params, "memorySearchFallback") !== undefined ||
        memorySearchRemoteBaseUrl !== undefined ||
        memorySearchRemoteApiKey !== undefined ||
        memorySearchRerankerEnabled !== undefined ||
        memorySearchRerankerApiUrl !== undefined ||
        memorySearchRerankerModel !== undefined ||
        memorySearchRerankerApiKey !== undefined ||
        memorySearchRerankerMaxCandidates !== undefined ||
        memorySearchRerankerFtsFirst !== undefined;
      const hasExternalCliConfigChange =
        externalCliEnabled !== undefined ||
        readStringParam(params, "externalCliDefault") !== undefined ||
        externalCliAvailableBrandsRaw !== undefined;
      const hasHeartbeatConfigChange =
        heartbeatEvery !== undefined ||
        heartbeatPrompt !== undefined ||
        heartbeatModel !== undefined ||
        heartbeatTarget !== undefined ||
        heartbeatTo !== undefined ||
        heartbeatAccountId !== undefined ||
        heartbeatIncludeReasoning !== undefined ||
        heartbeatSuppressToolErrorWarnings !== undefined ||
        heartbeatAckMaxChars !== undefined ||
        heartbeatActiveHoursStart !== undefined ||
        heartbeatActiveHoursEnd !== undefined ||
        heartbeatActiveHoursTimezone !== undefined;
      const hasHeartbeatFileChange = heartbeatFileActionRaw !== undefined;
      const hasSystemConfigChange =
        hasDeepMemoryConfigChange ||
        hasMemorySearchConfigChange ||
        hasExternalCliConfigChange ||
        hasHeartbeatConfigChange ||
        hasHeartbeatFileChange;

      if (
        !sessionAction &&
        model === undefined &&
        authProfile === undefined &&
        thinkingLevel === undefined &&
        verboseLevel === undefined &&
        reasoningLevel === undefined &&
        responseUsage === undefined &&
        elevatedLevel === undefined &&
        execHost === undefined &&
        execSecurity === undefined &&
        execAsk === undefined &&
        execNode === undefined &&
        queueMode === undefined &&
        queueDebounceMs === undefined &&
        queueCap === undefined &&
        queueDrop === undefined &&
        modelRegistryAction === undefined &&
        !hasDeepMemoryConfigChange &&
        !hasMemorySearchConfigChange &&
        !hasExternalCliConfigChange &&
        !hasHeartbeatConfigChange &&
        !hasHeartbeatFileChange
      ) {
        throw new ToolInputError("at least one setting change is required");
      }

      if (opts.directUserInstruction === false && hasSystemConfigChange) {
        return buildGenericDeniedResult();
      }

      if (
        readStringParam(params, "deepMemoryModelApi") !== undefined &&
        deepMemoryModelApi === undefined
      ) {
        return buildInvalidResult();
      }
      if (
        readStringParam(params, "memorySearchProvider") !== undefined &&
        memorySearchProvider === undefined
      ) {
        return buildInvalidResult();
      }
      if (
        readStringParam(params, "memorySearchFallback") !== undefined &&
        memorySearchFallback === undefined
      ) {
        return buildInvalidResult();
      }
      if (
        readStringParam(params, "externalCliDefault") !== undefined &&
        externalCliDefault === undefined
      ) {
        return buildInvalidResult();
      }
      if (externalCliAvailableBrandsRaw !== undefined && externalCliAvailableBrands?.length === 0) {
        return buildInvalidResult();
      }
      if (
        externalCliDefault &&
        externalCliAvailableBrands &&
        externalCliAvailableBrands.length > 0 &&
        !externalCliAvailableBrands.includes(externalCliDefault)
      ) {
        return buildInvalidResult();
      }
      if (
        deepMemorySchedule !== undefined &&
        deepMemorySchedule !== null &&
        !isValidCronExpr(deepMemorySchedule)
      ) {
        return buildInvalidResult();
      }
      if (heartbeatFileActionRaw !== undefined && heartbeatFileAction === undefined) {
        return buildInvalidResult();
      }
      if (
        heartbeatFileAction &&
        heartbeatFileAction !== "clear" &&
        heartbeatFileContent === undefined
      ) {
        return buildInvalidResult();
      }
      if (heartbeatAckMaxChars !== undefined && !Number.isInteger(heartbeatAckMaxChars)) {
        return buildInvalidResult();
      }

      let refreshedRegistry: Awaited<ReturnType<typeof refreshRuntimeModelRegistry>> | null = null;
      if (modelRegistryAction === "refresh") {
        refreshedRegistry = await refreshRuntimeModelRegistry({
          cfg,
          agentDir: resolveAgentDir(cfg, agentId),
          force: true,
        });
      }
      if (elevatedLevel !== undefined) {
        const elevated = resolveElevatedPermissions({
          cfg,
          agentId,
          ctx: buildElevatedMsgContext({
            agentChannel: opts.agentChannel,
            agentAccountId: opts.agentAccountId,
            agentTo: opts.agentTo,
            senderId: opts.senderId,
            senderName: opts.senderName,
            senderUsername: opts.senderUsername,
            senderE164: opts.senderE164,
          }),
          provider: opts.agentChannel ?? "",
        });
        if (!elevated.enabled || !elevated.allowed) {
          return buildGenericDeniedResult();
        }
      }

      const sharedDeepMemoryLabels: string[] = [];
      const sharedMemorySearchLabels: string[] = [];
      const sharedExternalCliLabels: string[] = [];
      const sharedHeartbeatLabels: string[] = [];
      let nextConfig = cfg;
      let hasSharedConfigChange = false;
      if (hasDeepMemoryConfigChange) {
        const nextDeepModel = {
          ...nextConfig.memory?.soul?.deepConsolidation?.model,
        } as Record<string, unknown>;
        applyOptionalStringPatch(nextDeepModel, "provider", deepMemoryModelProvider);
        applyOptionalStringPatch(nextDeepModel, "api", deepMemoryModelApi);
        applyOptionalStringPatch(nextDeepModel, "model", deepMemoryModel);
        applyOptionalStringPatch(nextDeepModel, "baseUrl", deepMemoryBaseUrl);
        if (deepMemoryTimeoutMs !== undefined) {
          nextDeepModel.timeoutMs = Math.max(1, Math.trunc(deepMemoryTimeoutMs));
        }

        const nextDeep = {
          ...nextConfig.memory?.soul?.deepConsolidation,
        } as Record<string, unknown>;
        if (deepMemoryEnabled !== undefined) {
          nextDeep.enabled = deepMemoryEnabled;
        }
        applyOptionalStringPatch(nextDeep, "schedule", deepMemorySchedule);
        if (deepMemoryMaxItems !== undefined) {
          nextDeep.maxItems = Math.max(1, Math.trunc(deepMemoryMaxItems));
        }
        if (deepMemoryMaxReflections !== undefined) {
          nextDeep.maxReflections = Math.max(1, Math.trunc(deepMemoryMaxReflections));
        }
        if (deepMemoryClusterSummarization !== undefined) {
          nextDeep.clusterSummarization = deepMemoryClusterSummarization;
        }
        if (deepMemoryConflictJudgment !== undefined) {
          nextDeep.conflictJudgment = deepMemoryConflictJudgment;
        }
        if (deepMemoryCrossScopeReflection !== undefined) {
          nextDeep.crossScopeReflection = deepMemoryCrossScopeReflection;
        }
        const prunedModel = pruneEmptyObject(nextDeepModel);
        if (prunedModel) {
          nextDeep.model = prunedModel;
        } else {
          delete nextDeep.model;
        }

        nextConfig = {
          ...nextConfig,
          memory: {
            ...nextConfig.memory,
            soul: {
              ...nextConfig.memory?.soul,
              deepConsolidation: nextDeep,
            },
          },
        };
        hasSharedConfigChange = true;

        if (deepMemoryEnabled !== undefined) {
          sharedDeepMemoryLabels.push(`deep memory ${deepMemoryEnabled ? "enabled" : "disabled"}`);
        }
        if (deepMemorySchedule !== undefined) {
          sharedDeepMemoryLabels.push(`deep memory schedule ${deepMemorySchedule ?? "default"}`);
        }
        if (deepMemoryModelProvider !== undefined) {
          sharedDeepMemoryLabels.push(
            `deep memory provider ${deepMemoryModelProvider ?? "default"}`,
          );
        }
        if (deepMemoryModelApi !== undefined) {
          sharedDeepMemoryLabels.push(`deep memory api ${deepMemoryModelApi ?? "default"}`);
        }
        if (deepMemoryModel !== undefined) {
          sharedDeepMemoryLabels.push(`deep memory model ${deepMemoryModel ?? "default"}`);
        }
        if (deepMemoryBaseUrl !== undefined) {
          sharedDeepMemoryLabels.push(`deep memory base URL ${deepMemoryBaseUrl ?? "default"}`);
        }
        if (deepMemoryTimeoutMs !== undefined) {
          sharedDeepMemoryLabels.push(`deep memory timeout ${Math.trunc(deepMemoryTimeoutMs)}ms`);
        }
        if (deepMemoryClusterSummarization !== undefined) {
          sharedDeepMemoryLabels.push(
            `deep summaries ${deepMemoryClusterSummarization ? "on" : "off"}`,
          );
        }
        if (deepMemoryConflictJudgment !== undefined) {
          sharedDeepMemoryLabels.push(
            `deep conflict judgment ${deepMemoryConflictJudgment ? "on" : "off"}`,
          );
        }
        if (deepMemoryCrossScopeReflection !== undefined) {
          sharedDeepMemoryLabels.push(
            `deep cross-scope reflection ${deepMemoryCrossScopeReflection ? "on" : "off"}`,
          );
        }
        if (deepMemoryMaxItems !== undefined) {
          sharedDeepMemoryLabels.push(`deep memory max items ${Math.trunc(deepMemoryMaxItems)}`);
        }
        if (deepMemoryMaxReflections !== undefined) {
          sharedDeepMemoryLabels.push(
            `deep memory max reflections ${Math.trunc(deepMemoryMaxReflections)}`,
          );
        }
      }
      if (hasMemorySearchConfigChange) {
        const nextReranker = {
          ...nextConfig.agents?.defaults?.memorySearch?.query?.hybrid?.reranker,
        } as Record<string, unknown>;
        if (memorySearchRerankerEnabled !== undefined) {
          nextReranker.enabled = memorySearchRerankerEnabled;
        }
        applyOptionalStringPatch(nextReranker, "apiUrl", memorySearchRerankerApiUrl);
        applyOptionalStringPatch(nextReranker, "model", memorySearchRerankerModel);
        applyOptionalStringPatch(nextReranker, "apiKey", memorySearchRerankerApiKey);
        if (memorySearchRerankerMaxCandidates !== undefined) {
          nextReranker.maxCandidates = Math.max(1, Math.trunc(memorySearchRerankerMaxCandidates));
        }
        if (memorySearchRerankerFtsFirst !== undefined) {
          nextReranker.ftsFirst = memorySearchRerankerFtsFirst;
        }

        const nextHybrid = {
          ...nextConfig.agents?.defaults?.memorySearch?.query?.hybrid,
        } as Record<string, unknown>;
        const prunedReranker = pruneEmptyObject(nextReranker);
        if (prunedReranker) {
          nextHybrid.reranker = prunedReranker;
        } else {
          delete nextHybrid.reranker;
        }

        const nextQuery = {
          ...nextConfig.agents?.defaults?.memorySearch?.query,
        } as Record<string, unknown>;
        const prunedHybrid = pruneEmptyObject(nextHybrid);
        if (prunedHybrid) {
          nextQuery.hybrid = prunedHybrid;
        } else {
          delete nextQuery.hybrid;
        }

        const nextRemote = {
          ...nextConfig.agents?.defaults?.memorySearch?.remote,
        } as Record<string, unknown>;
        applyOptionalStringPatch(nextRemote, "baseUrl", memorySearchRemoteBaseUrl);
        applyOptionalStringPatch(nextRemote, "apiKey", memorySearchRemoteApiKey);

        const nextMemorySearch = {
          ...nextConfig.agents?.defaults?.memorySearch,
        } as Record<string, unknown>;
        if (memorySearchEnabled !== undefined) {
          nextMemorySearch.enabled = memorySearchEnabled;
        }
        if (memorySearchDimensions !== undefined) {
          nextMemorySearch.dimensions = Math.max(1, Math.trunc(memorySearchDimensions));
        }
        applyOptionalStringPatch(nextMemorySearch, "provider", memorySearchProvider);
        applyOptionalStringPatch(nextMemorySearch, "model", memorySearchModel);
        applyOptionalStringPatch(nextMemorySearch, "fallback", memorySearchFallback);
        const prunedRemote = pruneEmptyObject(nextRemote);
        if (prunedRemote) {
          nextMemorySearch.remote = prunedRemote;
        } else {
          delete nextMemorySearch.remote;
        }
        const prunedQuery = pruneEmptyObject(nextQuery);
        if (prunedQuery) {
          nextMemorySearch.query = prunedQuery;
        } else {
          delete nextMemorySearch.query;
        }

        const nextRerankerConfig = nextMemorySearch.query as
          | { hybrid?: { reranker?: { enabled?: unknown; apiUrl?: unknown; model?: unknown } } }
          | undefined;
        const rerankerConfig = nextRerankerConfig?.hybrid?.reranker;
        if (
          rerankerConfig?.enabled === true &&
          (typeof rerankerConfig.apiUrl !== "string" ||
            !rerankerConfig.apiUrl.trim() ||
            typeof rerankerConfig.model !== "string" ||
            !rerankerConfig.model.trim())
        ) {
          return buildInvalidResult();
        }

        nextConfig = {
          ...nextConfig,
          agents: {
            ...nextConfig.agents,
            defaults: {
              ...nextConfig.agents?.defaults,
              memorySearch: nextMemorySearch,
            },
          },
        };
        hasSharedConfigChange = true;

        if (memorySearchEnabled !== undefined) {
          sharedMemorySearchLabels.push(
            `memory search ${memorySearchEnabled ? "enabled" : "disabled"}`,
          );
        }
        if (memorySearchProvider !== undefined) {
          sharedMemorySearchLabels.push(`memory provider ${memorySearchProvider ?? "default"}`);
        }
        if (memorySearchModel !== undefined) {
          sharedMemorySearchLabels.push(`memory model ${memorySearchModel ?? "default"}`);
        }
        if (memorySearchDimensions !== undefined) {
          sharedMemorySearchLabels.push(`memory dimensions ${Math.trunc(memorySearchDimensions)}`);
        }
        if (memorySearchFallback !== undefined) {
          sharedMemorySearchLabels.push(`memory fallback ${memorySearchFallback ?? "default"}`);
        }
        if (memorySearchRemoteBaseUrl !== undefined) {
          sharedMemorySearchLabels.push(
            `memory remote base URL ${memorySearchRemoteBaseUrl ?? "default"}`,
          );
        }
        const remoteApiKeySummary = summarizeSensitivePatch(
          "memory remote API key",
          memorySearchRemoteApiKey,
        );
        if (remoteApiKeySummary) {
          sharedMemorySearchLabels.push(remoteApiKeySummary);
        }
        if (memorySearchRerankerEnabled !== undefined) {
          sharedMemorySearchLabels.push(
            `memory reranker ${memorySearchRerankerEnabled ? "enabled" : "disabled"}`,
          );
        }
        if (memorySearchRerankerApiUrl !== undefined) {
          sharedMemorySearchLabels.push(
            `memory reranker URL ${memorySearchRerankerApiUrl ?? "default"}`,
          );
        }
        if (memorySearchRerankerModel !== undefined) {
          sharedMemorySearchLabels.push(
            `memory reranker model ${memorySearchRerankerModel ?? "default"}`,
          );
        }
        const rerankerApiKeySummary = summarizeSensitivePatch(
          "memory reranker API key",
          memorySearchRerankerApiKey,
        );
        if (rerankerApiKeySummary) {
          sharedMemorySearchLabels.push(rerankerApiKeySummary);
        }
        if (memorySearchRerankerMaxCandidates !== undefined) {
          sharedMemorySearchLabels.push(
            `memory reranker max candidates ${Math.trunc(memorySearchRerankerMaxCandidates)}`,
          );
        }
        if (memorySearchRerankerFtsFirst !== undefined) {
          sharedMemorySearchLabels.push(
            `memory reranker ftsFirst ${memorySearchRerankerFtsFirst ? "on" : "off"}`,
          );
        }
      }
      if (hasExternalCliConfigChange) {
        const nextExternalCli = {
          ...nextConfig.tools?.externalCli,
        } as Record<string, unknown>;
        if (externalCliEnabled !== undefined) {
          nextExternalCli.enabled = externalCliEnabled;
        }
        if (externalCliDefault !== undefined) {
          if (externalCliDefault === null) {
            delete nextExternalCli.defaultCli;
          } else {
            nextExternalCli.defaultCli = externalCliDefault;
          }
        }
        if (externalCliAvailableBrands !== undefined) {
          nextExternalCli.availableCli = externalCliAvailableBrands;
        }

        nextConfig = {
          ...nextConfig,
          tools: {
            ...nextConfig.tools,
            externalCli: nextExternalCli,
          },
        };
        hasSharedConfigChange = true;

        if (externalCliEnabled !== undefined) {
          sharedExternalCliLabels.push(
            `external CLI fallback ${externalCliEnabled ? "enabled" : "disabled"}`,
          );
        }
        if (externalCliDefault !== undefined) {
          sharedExternalCliLabels.push(`external CLI default ${externalCliDefault ?? "default"}`);
        }
        if (externalCliAvailableBrands !== undefined) {
          sharedExternalCliLabels.push(
            `external CLI brands ${externalCliAvailableBrands.join(", ")}`,
          );
        }
      }
      if (hasHeartbeatConfigChange) {
        const nextActiveHours = {
          ...nextConfig.agents?.defaults?.heartbeat?.activeHours,
        } as Record<string, unknown>;
        applyOptionalStringPatch(nextActiveHours, "start", heartbeatActiveHoursStart);
        applyOptionalStringPatch(nextActiveHours, "end", heartbeatActiveHoursEnd);
        applyOptionalStringPatch(nextActiveHours, "timezone", heartbeatActiveHoursTimezone);

        const nextHeartbeat = {
          ...nextConfig.agents?.defaults?.heartbeat,
        } as Record<string, unknown>;
        applyOptionalStringPatch(nextHeartbeat, "every", heartbeatEvery);
        applyOptionalStringPatch(nextHeartbeat, "prompt", heartbeatPrompt);
        applyOptionalStringPatch(nextHeartbeat, "model", heartbeatModel);
        applyOptionalStringPatch(nextHeartbeat, "target", heartbeatTarget);
        applyOptionalStringPatch(nextHeartbeat, "to", heartbeatTo);
        applyOptionalStringPatch(nextHeartbeat, "accountId", heartbeatAccountId);
        if (heartbeatIncludeReasoning !== undefined) {
          nextHeartbeat.includeReasoning = heartbeatIncludeReasoning;
        }
        if (heartbeatSuppressToolErrorWarnings !== undefined) {
          nextHeartbeat.suppressToolErrorWarnings = heartbeatSuppressToolErrorWarnings;
        }
        if (heartbeatAckMaxChars !== undefined) {
          nextHeartbeat.ackMaxChars = Math.max(0, Math.trunc(heartbeatAckMaxChars));
        }
        const prunedActiveHours = pruneEmptyObject(nextActiveHours);
        if (prunedActiveHours) {
          nextHeartbeat.activeHours = prunedActiveHours;
        } else {
          delete nextHeartbeat.activeHours;
        }

        nextConfig = {
          ...nextConfig,
          agents: {
            ...nextConfig.agents,
            defaults: {
              ...nextConfig.agents?.defaults,
              heartbeat: nextHeartbeat,
            },
          },
        };
        hasSharedConfigChange = true;

        if (heartbeatEvery !== undefined) {
          sharedHeartbeatLabels.push(`heartbeat every ${heartbeatEvery ?? "default"}`);
        }
        if (heartbeatPrompt !== undefined) {
          sharedHeartbeatLabels.push(
            `heartbeat prompt ${heartbeatPrompt === null ? "default" : "configured"}`,
          );
        }
        if (heartbeatModel !== undefined) {
          sharedHeartbeatLabels.push(`heartbeat model ${heartbeatModel ?? "default"}`);
        }
        if (heartbeatTarget !== undefined) {
          sharedHeartbeatLabels.push(`heartbeat target ${heartbeatTarget ?? "default"}`);
        }
        if (heartbeatTo !== undefined) {
          sharedHeartbeatLabels.push(`heartbeat recipient ${heartbeatTo ?? "default"}`);
        }
        if (heartbeatAccountId !== undefined) {
          sharedHeartbeatLabels.push(`heartbeat account ${heartbeatAccountId ?? "default"}`);
        }
        if (heartbeatIncludeReasoning !== undefined) {
          sharedHeartbeatLabels.push(
            `heartbeat reasoning ${heartbeatIncludeReasoning ? "enabled" : "disabled"}`,
          );
        }
        if (heartbeatSuppressToolErrorWarnings !== undefined) {
          sharedHeartbeatLabels.push(
            `heartbeat tool warnings ${heartbeatSuppressToolErrorWarnings ? "suppressed" : "default"}`,
          );
        }
        if (heartbeatAckMaxChars !== undefined) {
          sharedHeartbeatLabels.push(
            `heartbeat ack max chars ${Math.max(0, Math.trunc(heartbeatAckMaxChars))}`,
          );
        }
        if (heartbeatActiveHoursStart !== undefined) {
          sharedHeartbeatLabels.push(
            `heartbeat active start ${heartbeatActiveHoursStart ?? "default"}`,
          );
        }
        if (heartbeatActiveHoursEnd !== undefined) {
          sharedHeartbeatLabels.push(
            `heartbeat active end ${heartbeatActiveHoursEnd ?? "default"}`,
          );
        }
        if (heartbeatActiveHoursTimezone !== undefined) {
          sharedHeartbeatLabels.push(
            `heartbeat active timezone ${heartbeatActiveHoursTimezone ?? "default"}`,
          );
        }
      }
      if (hasSharedConfigChange) {
        try {
          await writeConfigFile(nextConfig);
        } catch {
          return buildInvalidResult();
        }
      }

      let heartbeatFileSummary: string | undefined;
      let heartbeatFileDetails:
        | {
            action: HeartbeatFileAction;
            path: string;
            size: number;
          }
        | undefined;
      if (heartbeatFileAction) {
        try {
          const workspaceDir = resolveAgentWorkspaceDir(nextConfig, agentId);
          await fs.mkdir(workspaceDir, { recursive: true });
          const heartbeatPath = path.join(workspaceDir, DEFAULT_HEARTBEAT_FILENAME);
          let nextContent = "";
          if (heartbeatFileAction === "replace") {
            nextContent = heartbeatFileContent ?? "";
          } else if (heartbeatFileAction === "append") {
            const current = await fs.readFile(heartbeatPath, "utf-8").catch(() => "");
            const separator = current.length > 0 && !current.endsWith("\n") ? "\n" : "";
            nextContent = `${current}${separator}${heartbeatFileContent ?? ""}`;
          }
          await fs.writeFile(heartbeatPath, nextContent, "utf-8");
          heartbeatFileSummary = `Updated HEARTBEAT.md: ${heartbeatFileAction}.`;
          heartbeatFileDetails = {
            action: heartbeatFileAction,
            path: heartbeatPath,
            size: nextContent.length,
          };
        } catch {
          return buildInvalidResult();
        }
      }

      let resetEntry: SessionEntry | undefined;
      if (sessionAction) {
        const resetResult = await invokeSessionReset({ key: sessionKey, reason: sessionAction });
        if (!resetResult.ok) {
          return buildGenericDeniedResult();
        }
        resetEntry = resetResult.entry;
      }

      const target = resolveGatewaySessionStoreTarget({ cfg, key: sessionKey });
      const storePath = target.storePath;
      const storeBeforeUpdate = loadSessionStore(storePath);
      const patch: SessionsPatchParams = {
        key: target.canonicalKey,
        ...(model !== undefined ? { model } : {}),
        ...(thinkingLevel !== undefined ? { thinkingLevel } : {}),
        ...(verboseLevel !== undefined ? { verboseLevel } : {}),
        ...(reasoningLevel !== undefined ? { reasoningLevel } : {}),
        ...(responseUsage !== undefined
          ? { responseUsage: responseUsage as SessionsPatchParams["responseUsage"] }
          : {}),
        ...(elevatedLevel !== undefined ? { elevatedLevel } : {}),
        ...(execHost !== undefined ? { execHost } : {}),
        ...(execSecurity !== undefined ? { execSecurity } : {}),
        ...(execAsk !== undefined ? { execAsk } : {}),
        ...(execNode !== undefined ? { execNode } : {}),
        ...(queueMode !== undefined ? { queueMode } : {}),
        ...(queueDebounceMs !== undefined ? { queueDebounceMs } : {}),
        ...(queueCap !== undefined ? { queueCap } : {}),
        ...(queueDrop !== undefined ? { queueDrop } : {}),
      };
      const hasSessionPatch =
        model !== undefined ||
        thinkingLevel !== undefined ||
        verboseLevel !== undefined ||
        reasoningLevel !== undefined ||
        responseUsage !== undefined ||
        elevatedLevel !== undefined ||
        execHost !== undefined ||
        execSecurity !== undefined ||
        execAsk !== undefined ||
        execNode !== undefined ||
        queueMode !== undefined ||
        queueDebounceMs !== undefined ||
        queueCap !== undefined ||
        queueDrop !== undefined;

      const applied = hasSessionPatch
        ? await updateSessionStore(storePath, async (store) => {
            const primaryKey = target.canonicalKey;
            if (!store[primaryKey]) {
              const existingKey = target.storeKeys.find((candidate) => Boolean(store[candidate]));
              if (existingKey) {
                store[primaryKey] = store[existingKey];
              }
            }
            pruneLegacyStoreKeys({
              store,
              canonicalKey: primaryKey,
              candidates: target.storeKeys,
            });
            return await applySessionsPatchToStore({
              cfg,
              store,
              storeKey: primaryKey,
              patch,
              loadGatewayModelCatalog: async () => await loadModelCatalog({ config: cfg }),
            });
          })
        : {
            ok: true as const,
            entry: resolveCurrentSessionEntry({
              store: storeBeforeUpdate,
              sessionKeys: [target.canonicalKey, ...target.storeKeys],
            }),
          };
      if (!applied.ok) {
        return buildInvalidResult();
      }

      let nextEntry =
        applied.entry ??
        resetEntry ??
        resolveCurrentSessionEntry({
          store: loadSessionStore(storePath),
          sessionKeys: [target.canonicalKey, ...target.storeKeys],
        }) ??
        ({
          sessionId: "",
          updatedAt: Date.now(),
        } satisfies SessionEntry);
      if (authProfile !== undefined) {
        if (authProfile === null) {
          delete nextEntry.authProfileOverride;
          delete nextEntry.authProfileOverrideSource;
          delete nextEntry.authProfileOverrideCompactionCount;
          nextEntry.updatedAt = Date.now();
          await updateSessionStore(storePath, (store) => {
            store[target.canonicalKey] = nextEntry;
          });
        } else {
          const defaultModelRef = resolveDefaultModelForAgent({ cfg, agentId });
          const resolvedModel = resolveSessionModelRef(cfg, nextEntry, agentId);
          const effectiveProvider = resolvedModel.provider || defaultModelRef.provider;
          const authStore = ensureAuthProfileStore(resolveAgentDir(cfg, agentId), {
            allowKeychainPrompt: false,
          });
          const profile = authStore.profiles[authProfile];
          if (
            !profile ||
            normalizeProviderId(profile.provider) !== normalizeProviderId(effectiveProvider)
          ) {
            return buildInvalidResult();
          }
          nextEntry = {
            ...nextEntry,
            authProfileOverride: authProfile,
            authProfileOverrideSource: "user",
            updatedAt: Date.now(),
          };
          delete nextEntry.authProfileOverrideCompactionCount;
          await updateSessionStore(storePath, (store) => {
            store[target.canonicalKey] = nextEntry;
          });
        }
      }

      const resolved = resolveSessionModelRef(cfg, nextEntry, agentId);
      const appliedLabels = [
        sessionAction ? `session ${sessionAction}` : null,
        model !== undefined ? `model ${resolved.provider}/${resolved.model}` : null,
        authProfile !== undefined
          ? authProfile === null
            ? "auth profile default"
            : `auth profile ${authProfile}`
          : null,
        thinkingLevel !== undefined ? `thinking ${nextEntry.thinkingLevel ?? "default"}` : null,
        verboseLevel !== undefined ? `verbose ${nextEntry.verboseLevel ?? "off"}` : null,
        reasoningLevel !== undefined ? `reasoning ${nextEntry.reasoningLevel ?? "off"}` : null,
        responseUsage !== undefined ? `usage ${nextEntry.responseUsage ?? "off"}` : null,
        elevatedLevel !== undefined ? `elevated ${nextEntry.elevatedLevel ?? "off"}` : null,
        execHost !== undefined ? `exec host ${nextEntry.execHost ?? "default"}` : null,
        execSecurity !== undefined ? `exec security ${nextEntry.execSecurity ?? "default"}` : null,
        execAsk !== undefined ? `exec ask ${nextEntry.execAsk ?? "default"}` : null,
        execNode !== undefined ? `exec node ${nextEntry.execNode ?? "default"}` : null,
        queueMode !== undefined ? `queue ${nextEntry.queueMode ?? "default"}` : null,
        queueDebounceMs !== undefined
          ? `queue debounce ${nextEntry.queueDebounceMs ?? "default"}ms`
          : null,
        queueCap !== undefined ? `queue cap ${nextEntry.queueCap ?? "default"}` : null,
        queueDrop !== undefined ? `queue drop ${nextEntry.queueDrop ?? "default"}` : null,
        modelRegistryAction === "refresh"
          ? `model registry refreshed (${refreshedRegistry?.models.length ?? 0} models)`
          : null,
      ].filter((value): value is string => Boolean(value));

      const summaryParts = [
        appliedLabels.length > 0 ? `Updated current session: ${appliedLabels.join("; ")}.` : null,
        sharedDeepMemoryLabels.length > 0
          ? `Updated shared deep-memory settings: ${sharedDeepMemoryLabels.join("; ")}.`
          : null,
        sharedMemorySearchLabels.length > 0
          ? `Updated shared memory-search settings: ${sharedMemorySearchLabels.join("; ")}.`
          : null,
        sharedExternalCliLabels.length > 0
          ? `Updated shared external-CLI settings: ${sharedExternalCliLabels.join("; ")}.`
          : null,
        sharedHeartbeatLabels.length > 0
          ? `Updated shared heartbeat settings: ${sharedHeartbeatLabels.join("; ")}.`
          : null,
        heartbeatFileSummary ?? null,
      ].filter((value): value is string => Boolean(value));

      return {
        content: [
          {
            type: "text" as const,
            text:
              summaryParts.length > 0
                ? summaryParts.join(" ")
                : "No current-session setting changes were needed.",
          },
        ],
        details: {
          ok: true,
          applied: true,
          sessionKey: target.canonicalKey,
          settings: {
            modelProvider: resolved.provider,
            model: resolved.model,
            thinkingLevel: nextEntry.thinkingLevel,
            verboseLevel: nextEntry.verboseLevel,
            reasoningLevel: nextEntry.reasoningLevel,
            responseUsage: nextEntry.responseUsage,
            elevatedLevel: nextEntry.elevatedLevel,
            execHost: nextEntry.execHost,
            execSecurity: nextEntry.execSecurity,
            execAsk: nextEntry.execAsk,
            execNode: nextEntry.execNode,
            queueMode: nextEntry.queueMode,
            queueDebounceMs: nextEntry.queueDebounceMs,
            queueCap: nextEntry.queueCap,
            queueDrop: nextEntry.queueDrop,
            authProfileOverride: nextEntry.authProfileOverride,
            modelRegistryPath: resolveRuntimeRegistryPathForDisplay(),
            modelRegistryRefreshedAt: refreshedRegistry?.lastSuccessfulRefreshAt,
          },
          sharedConfig: hasSharedConfigChange
            ? {
                deepMemoryEnabled: nextConfig.memory?.soul?.deepConsolidation?.enabled,
                deepMemorySchedule: nextConfig.memory?.soul?.deepConsolidation?.schedule,
                deepMemoryModelProvider:
                  nextConfig.memory?.soul?.deepConsolidation?.model?.provider,
                deepMemoryModelApi: nextConfig.memory?.soul?.deepConsolidation?.model?.api,
                deepMemoryModel: nextConfig.memory?.soul?.deepConsolidation?.model?.model,
                deepMemoryBaseUrl: nextConfig.memory?.soul?.deepConsolidation?.model?.baseUrl,
                deepMemoryTimeoutMs: nextConfig.memory?.soul?.deepConsolidation?.model?.timeoutMs,
                deepMemoryClusterSummarization:
                  nextConfig.memory?.soul?.deepConsolidation?.clusterSummarization,
                deepMemoryConflictJudgment:
                  nextConfig.memory?.soul?.deepConsolidation?.conflictJudgment,
                deepMemoryCrossScopeReflection:
                  nextConfig.memory?.soul?.deepConsolidation?.crossScopeReflection,
                deepMemoryMaxItems: nextConfig.memory?.soul?.deepConsolidation?.maxItems,
                deepMemoryMaxReflections:
                  nextConfig.memory?.soul?.deepConsolidation?.maxReflections,
                memorySearchEnabled: nextConfig.agents?.defaults?.memorySearch?.enabled,
                memorySearchProvider: nextConfig.agents?.defaults?.memorySearch?.provider,
                memorySearchModel: nextConfig.agents?.defaults?.memorySearch?.model,
                memorySearchDimensions: nextConfig.agents?.defaults?.memorySearch?.dimensions,
                memorySearchFallback: nextConfig.agents?.defaults?.memorySearch?.fallback,
                memorySearchRemoteBaseUrl:
                  nextConfig.agents?.defaults?.memorySearch?.remote?.baseUrl,
                memorySearchRemoteApiKey: nextConfig.agents?.defaults?.memorySearch?.remote?.apiKey
                  ? REDACTED_VALUE
                  : undefined,
                memorySearchRerankerEnabled:
                  nextConfig.agents?.defaults?.memorySearch?.query?.hybrid?.reranker?.enabled,
                memorySearchRerankerApiUrl:
                  nextConfig.agents?.defaults?.memorySearch?.query?.hybrid?.reranker?.apiUrl,
                memorySearchRerankerModel:
                  nextConfig.agents?.defaults?.memorySearch?.query?.hybrid?.reranker?.model,
                memorySearchRerankerApiKey: nextConfig.agents?.defaults?.memorySearch?.query?.hybrid
                  ?.reranker?.apiKey
                  ? REDACTED_VALUE
                  : undefined,
                memorySearchRerankerMaxCandidates:
                  nextConfig.agents?.defaults?.memorySearch?.query?.hybrid?.reranker?.maxCandidates,
                memorySearchRerankerFtsFirst:
                  nextConfig.agents?.defaults?.memorySearch?.query?.hybrid?.reranker?.ftsFirst,
                externalCliEnabled: nextConfig.tools?.externalCli?.enabled,
                externalCliDefault: nextConfig.tools?.externalCli?.defaultCli,
                externalCliAvailable: nextConfig.tools?.externalCli?.availableCli,
                heartbeatEvery: nextConfig.agents?.defaults?.heartbeat?.every,
                heartbeatPrompt: nextConfig.agents?.defaults?.heartbeat?.prompt,
                heartbeatModel: nextConfig.agents?.defaults?.heartbeat?.model,
                heartbeatTarget: nextConfig.agents?.defaults?.heartbeat?.target,
                heartbeatTo: nextConfig.agents?.defaults?.heartbeat?.to,
                heartbeatAccountId: nextConfig.agents?.defaults?.heartbeat?.accountId,
                heartbeatIncludeReasoning: nextConfig.agents?.defaults?.heartbeat?.includeReasoning,
                heartbeatSuppressToolErrorWarnings:
                  nextConfig.agents?.defaults?.heartbeat?.suppressToolErrorWarnings,
                heartbeatAckMaxChars: nextConfig.agents?.defaults?.heartbeat?.ackMaxChars,
                heartbeatActiveHoursStart:
                  nextConfig.agents?.defaults?.heartbeat?.activeHours?.start,
                heartbeatActiveHoursEnd: nextConfig.agents?.defaults?.heartbeat?.activeHours?.end,
                heartbeatActiveHoursTimezone:
                  nextConfig.agents?.defaults?.heartbeat?.activeHours?.timezone,
              }
            : undefined,
          files: heartbeatFileDetails
            ? {
                heartbeat: heartbeatFileDetails,
              }
            : undefined,
        },
      };
    },
  };
}
