import { Type } from "@sinclair/typebox";
import { resolveElevatedPermissions } from "../../auto-reply/reply/reply-elevated.js";
import type { MsgContext } from "../../auto-reply/templating.js";
import { type MarvConfig, loadConfig } from "../../core/config/config.js";
import { updateSessionStore, type SessionEntry } from "../../core/config/sessions.js";
import type { SessionsPatchParams } from "../../core/gateway/protocol/index.js";
import { sessionsHandlers } from "../../core/gateway/server-methods/sessions.js";
import {
  pruneLegacyStoreKeys,
  resolveGatewaySessionStoreTarget,
  resolveSessionModelRef,
} from "../../core/gateway/session-utils.js";
import { applySessionsPatchToStore } from "../../core/gateway/sessions-patch.js";
import { resolveAgentIdFromSessionKey } from "../../routing/session-key.js";
import type { GatewayMessageChannel } from "../../utils/message-channel.js";
import { resolveAgentDir } from "../agent-scope.js";
import { ensureAuthProfileStore } from "../auth-profiles.js";
import { loadModelCatalog } from "../model/model-catalog.js";
import { normalizeProviderId, resolveDefaultModelForAgent } from "../model/model-selection.js";
import type { AnyAgentTool } from "./common.js";
import { readNumberParam, readStringParam, ToolInputError } from "./common.js";

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
    sessionAction: Type.Optional(Type.String()),
  },
  { additionalProperties: false },
);

const GENERIC_DENIED_MESSAGE = "This setting request cannot be applied right now.";
const GENERIC_INVALID_MESSAGE = "I can't apply that session setting directly.";

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
      "Apply current-session self-settings for a direct user request: model, auth profile, thinking, verbose, reasoning, usage, elevated, exec defaults, queue behavior, or session reset/new.",
    parameters: SelfSettingsToolSchema,
    execute: async (_toolCallId, args) => {
      if (!opts?.agentSessionKey?.trim()) {
        throw new ToolInputError("sessionKey required");
      }
      if (opts.directUserInstruction === false) {
        return buildGenericDeniedResult();
      }

      const params = args as SelfSettingsArgs;
      const cfg = opts.config ?? loadConfig();
      const sessionKey = opts.agentSessionKey.trim();
      const sessionAction = normalizeSessionAction(readStringParam(params, "sessionAction"));
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
        queueDrop === undefined
      ) {
        throw new ToolInputError("at least one setting change is required");
      }

      const agentId = resolveAgentIdFromSessionKey(sessionKey);
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

      if (sessionAction) {
        const resetResult = await invokeSessionReset({ key: sessionKey, reason: sessionAction });
        if (!resetResult.ok) {
          return buildGenericDeniedResult();
        }
      }

      const target = resolveGatewaySessionStoreTarget({ cfg, key: sessionKey });
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

      const storePath = target.storePath;
      const applied = await updateSessionStore(storePath, async (store) => {
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
      });
      if (!applied.ok) {
        return buildInvalidResult();
      }

      let nextEntry = applied.entry;
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
      ].filter((value): value is string => Boolean(value));

      return {
        content: [
          {
            type: "text" as const,
            text:
              appliedLabels.length > 0
                ? `Updated current session: ${appliedLabels.join("; ")}.`
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
          },
        },
      };
    },
  };
}
