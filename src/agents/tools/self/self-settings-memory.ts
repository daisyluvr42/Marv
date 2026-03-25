import { Type } from "@sinclair/typebox";
import { loadConfig, writeConfigFile } from "../../../core/config/config.js";
import type { AnyAgentTool } from "../common.js";
import { readNumberParam, readStringParam, ToolInputError } from "../common.js";
import {
  applyOptionalStringPatch,
  buildGenericDeniedResult,
  buildInvalidResult,
  isValidCronExpr,
  normalizeDeepMemoryApi,
  normalizeExternalCliAvailableBrands,
  normalizeExternalCliDefault,
  normalizeMemorySearchFallback,
  normalizeMemorySearchProvider,
  normalizePatchString,
  pruneEmptyObject,
  readBooleanParam,
  REDACTED_VALUE,
  summarizeSensitivePatch,
  type SelfSettingsArgs,
  type SelfSettingsToolOpts,
} from "./self-settings-normalize.js";

const MemorySettingsToolSchema = Type.Object(
  {
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
  },
  { additionalProperties: false },
);

export function createMemorySettingsTool(opts?: SelfSettingsToolOpts): AnyAgentTool {
  return {
    label: "Memory Settings",
    name: "self_settings_memory",
    description:
      "Apply shared memory and external-CLI settings: deep-memory consolidation (schedule, model, provider, flags), memory-search (provider, model, dimensions, fallback, reranker), and external CLI fallback defaults.",
    parameters: MemorySettingsToolSchema,
    execute: async (_toolCallId, args) => {
      if (!opts?.agentSessionKey?.trim()) {
        throw new ToolInputError("sessionKey required");
      }

      const params = args as SelfSettingsArgs;
      const cfg = opts.config ?? loadConfig();

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

      if (
        !hasDeepMemoryConfigChange &&
        !hasMemorySearchConfigChange &&
        !hasExternalCliConfigChange
      ) {
        throw new ToolInputError("at least one setting change is required");
      }

      if (opts.directUserInstruction === false) {
        return buildGenericDeniedResult();
      }

      // --- Validation ---
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

      // --- Apply config changes ---
      const sharedDeepMemoryLabels: string[] = [];
      const sharedMemorySearchLabels: string[] = [];
      const sharedExternalCliLabels: string[] = [];
      let nextConfig = cfg;

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

      try {
        await writeConfigFile(nextConfig);
      } catch {
        return buildInvalidResult();
      }

      const summaryParts = [
        sharedDeepMemoryLabels.length > 0
          ? `Updated shared deep-memory settings: ${sharedDeepMemoryLabels.join("; ")}.`
          : null,
        sharedMemorySearchLabels.length > 0
          ? `Updated shared memory-search settings: ${sharedMemorySearchLabels.join("; ")}.`
          : null,
        sharedExternalCliLabels.length > 0
          ? `Updated shared external-CLI settings: ${sharedExternalCliLabels.join("; ")}.`
          : null,
      ].filter((value): value is string => Boolean(value));

      return {
        content: [
          {
            type: "text" as const,
            text:
              summaryParts.length > 0
                ? summaryParts.join(" ")
                : "No memory/external-CLI setting changes were needed.",
          },
        ],
        details: {
          ok: true,
          applied: true,
          sharedConfig: {
            deepMemoryEnabled: nextConfig.memory?.soul?.deepConsolidation?.enabled,
            deepMemorySchedule: nextConfig.memory?.soul?.deepConsolidation?.schedule,
            deepMemoryModelProvider: nextConfig.memory?.soul?.deepConsolidation?.model?.provider,
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
            deepMemoryMaxReflections: nextConfig.memory?.soul?.deepConsolidation?.maxReflections,
            memorySearchEnabled: nextConfig.agents?.defaults?.memorySearch?.enabled,
            memorySearchProvider: nextConfig.agents?.defaults?.memorySearch?.provider,
            memorySearchModel: nextConfig.agents?.defaults?.memorySearch?.model,
            memorySearchDimensions: nextConfig.agents?.defaults?.memorySearch?.dimensions,
            memorySearchFallback: nextConfig.agents?.defaults?.memorySearch?.fallback,
            memorySearchRemoteBaseUrl: nextConfig.agents?.defaults?.memorySearch?.remote?.baseUrl,
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
          },
        },
      };
    },
  };
}
