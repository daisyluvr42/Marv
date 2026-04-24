import { Type } from "@sinclair/typebox";
import { resolveElevatedPermissions } from "../../../auto-reply/directives/reply-elevated.js";
import { loadConfig } from "../../../core/config/config.js";
import {
  loadSessionStore,
  updateSessionStore,
  type SessionEntry,
} from "../../../core/config/sessions.js";
import type { SessionsPatchParams } from "../../../core/gateway/protocol/index.js";
import {
  pruneLegacyStoreKeys,
  resolveGatewaySessionStoreTarget,
  resolveSessionModelRef,
} from "../../../core/gateway/session-utils.js";
import { applySessionsPatchToStore } from "../../../core/gateway/sessions-patch.js";
import { resolveAgentIdFromSessionKey } from "../../../routing/session-key.js";
import { resolveAgentDir } from "../../agent-scope.js";
import { ensureAuthProfileStore } from "../../auth-profiles.js";
import { loadModelCatalog } from "../../model/model-catalog.js";
import { normalizeProviderId, resolveDefaultModelForAgent } from "../../model/model-resolve.js";
import {
  refreshRuntimeModelRegistry,
  resolveRuntimeRegistryPathForDisplay,
} from "../../model/runtime-model-registry.js";
import type { AnyAgentTool } from "../common.js";
import { readNumberParam, readStringParam, ToolInputError } from "../common.js";
import {
  buildElevatedTurnContext,
  buildGenericDeniedResult,
  buildInvalidResult,
  invokeSessionReset,
  normalizeAuthProfile,
  normalizeModelRegistryAction,
  normalizePatchString,
  normalizeSessionAction,
  resolveCurrentSessionEntry,
  type SelfSettingsArgs,
  type SelfSettingsToolOpts,
} from "./self-settings-normalize.js";

const SessionSettingsToolSchema = Type.Object(
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
  },
  { additionalProperties: false },
);

export function createSessionSettingsTool(opts?: SelfSettingsToolOpts): AnyAgentTool {
  return {
    label: "Session Settings",
    name: "self_settings_session",
    description:
      "Apply session-level settings: model, auth profile, thinking, verbose, reasoning, usage, elevated, exec defaults, queue behavior, session reset/new, and model registry refresh.",
    parameters: SessionSettingsToolSchema,
    execute: async (_toolCallId, args) => {
      if (!opts?.agentSessionKey?.trim()) {
        throw new ToolInputError("sessionKey required");
      }

      const params = args as SelfSettingsArgs;
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
        modelRegistryAction === undefined
      ) {
        throw new ToolInputError("at least one setting change is required");
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
          ctx: buildElevatedTurnContext({
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
        return buildInvalidResult(applied.error?.message);
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
          if (!profile) {
            return buildInvalidResult(`auth profile not found: ${authProfile}`);
          }
          if (normalizeProviderId(profile.provider) !== normalizeProviderId(effectiveProvider)) {
            return buildInvalidResult(
              `auth profile "${authProfile}" is for provider ${profile.provider}, but current model uses ${effectiveProvider}`,
            );
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

      const summaryText =
        appliedLabels.length > 0
          ? `Updated current session: ${appliedLabels.join("; ")}.`
          : "No current-session setting changes were needed.";

      return {
        content: [{ type: "text" as const, text: summaryText }],
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
        },
      };
    },
  };
}
