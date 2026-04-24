import type {
  ConfiguredModelLocation,
  ConfiguredModelTier,
  MarvConfig,
} from "../../core/config/types.js";
import { resolveAgentConfig } from "../agent-scope.js";
import { DEFAULT_MODEL, DEFAULT_PROVIDER } from "../defaults.js";
import type { ModelCatalogEntry } from "./model-catalog.js";
import {
  type ModelRef,
  normalizeModelRef,
  normalizeProviderId,
  parseModelRef,
} from "./model-ref.js";
import { resolveSelectedModelRefs } from "./model-selections-store.js";

export type { ModelRef } from "./model-ref.js";
export { normalizeModelRef, normalizeProviderId, parseModelRef } from "./model-ref.js";

export type ThinkLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh";

export type ModelAliasIndex = {
  byAlias: Map<string, { alias: string; ref: ModelRef }>;
  byKey: Map<string, string[]>;
};

type ModelOrderingEntry = {
  provider: string;
  model: string;
  location?: ConfiguredModelLocation;
  tier?: ConfiguredModelTier;
  priority?: number;
  capabilities?: string[];
};

function normalizeAliasKey(value: string): string {
  return value.trim().toLowerCase();
}

export function modelKey(provider: string, model: string) {
  return `${provider}/${model}`;
}

export function findNormalizedProviderValue<T>(
  entries: Record<string, T> | undefined,
  provider: string,
): T | undefined {
  if (!entries) {
    return undefined;
  }
  const providerKey = normalizeProviderId(provider);
  for (const [key, value] of Object.entries(entries)) {
    if (normalizeProviderId(key) === providerKey) {
      return value;
    }
  }
  return undefined;
}

export function findNormalizedProviderKey(
  entries: Record<string, unknown> | undefined,
  provider: string,
): string | undefined {
  if (!entries) {
    return undefined;
  }
  const providerKey = normalizeProviderId(provider);
  return Object.keys(entries).find((key) => normalizeProviderId(key) === providerKey);
}

export function isCliProvider(provider: string, cfg?: MarvConfig): boolean {
  const normalized = normalizeProviderId(provider);
  if (normalized === "claude-cli") {
    return true;
  }
  if (normalized === "codex-cli") {
    return true;
  }
  const backends = cfg?.agents?.defaults?.cliBackends ?? {};
  return Object.keys(backends).some((key) => normalizeProviderId(key) === normalized);
}

function inferProviderForBareModel(raw: string, defaultProvider: string): string {
  const trimmed = raw.trim().toLowerCase();
  if (!trimmed) {
    return defaultProvider;
  }
  if (
    trimmed.startsWith("claude-") ||
    trimmed.startsWith("opus-") ||
    trimmed.startsWith("sonnet-") ||
    trimmed.startsWith("haiku-")
  ) {
    return "anthropic";
  }
  if (trimmed.startsWith("gpt-") || trimmed.startsWith("o1") || trimmed.startsWith("o3")) {
    return "openai";
  }
  if (trimmed.startsWith("gemini")) {
    return "google";
  }
  return defaultProvider;
}

export function normalizeModelSelection(value: unknown): string | undefined {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed || undefined;
  }
  if (!value || typeof value !== "object") {
    return undefined;
  }
  const primary = (value as { primary?: unknown }).primary;
  if (typeof primary === "string") {
    const trimmed = primary.trim();
    return trimmed || undefined;
  }
  return undefined;
}

export function buildConfiguredSelectionKeys(params: {
  cfg: MarvConfig | undefined;
  defaultProvider: string;
}): Set<string> | null {
  if (!params.cfg) {
    return null;
  }
  const keys = resolveSelectedModelRefs({
    cfg: params.cfg,
    defaultProvider: params.defaultProvider,
  });
  return keys.size > 0 ? keys : null;
}

function poolTierWeight(tier?: ConfiguredModelTier): number {
  switch (tier) {
    case "high":
      return 2;
    case "standard":
      return 1;
    case "low":
    default:
      return 0;
  }
}

function defaultSessionTierWeight(tier?: ConfiguredModelTier): number {
  switch (tier) {
    case "low":
      return 0;
    case "standard":
      return 1;
    case "high":
      return 2;
    default:
      return 1;
  }
}

function locationWeight(location?: ConfiguredModelLocation): number {
  return location === "local" ? 0 : 1;
}

function capabilityWeight(capabilities?: string[]): number {
  return capabilities?.length ?? 0;
}

export function comparePoolModelPreference(a: ModelOrderingEntry, b: ModelOrderingEntry): number {
  const byTier = poolTierWeight(b.tier) - poolTierWeight(a.tier);
  if (byTier !== 0) {
    return byTier;
  }
  const byCapabilities = capabilityWeight(b.capabilities) - capabilityWeight(a.capabilities);
  if (byCapabilities !== 0) {
    return byCapabilities;
  }
  const byPriority = (a.priority ?? 0) - (b.priority ?? 0);
  if (byPriority !== 0) {
    return byPriority;
  }
  const byLocation = locationWeight(a.location) - locationWeight(b.location);
  if (byLocation !== 0) {
    return byLocation;
  }
  return modelKey(a.provider, a.model).localeCompare(modelKey(b.provider, b.model));
}

export function compareDefaultSessionModelPreference(
  a: ModelOrderingEntry,
  b: ModelOrderingEntry,
): number {
  const byTier = defaultSessionTierWeight(a.tier) - defaultSessionTierWeight(b.tier);
  if (byTier !== 0) {
    return byTier;
  }
  const byLocation = locationWeight(a.location) - locationWeight(b.location);
  if (byLocation !== 0) {
    return byLocation;
  }
  const byCapabilities = capabilityWeight(a.capabilities) - capabilityWeight(b.capabilities);
  if (byCapabilities !== 0) {
    return byCapabilities;
  }
  const byPriority = (a.priority ?? 0) - (b.priority ?? 0);
  if (byPriority !== 0) {
    return byPriority;
  }
  return modelKey(a.provider, a.model).localeCompare(modelKey(b.provider, b.model));
}

export function buildModelAliasIndex(params: {
  cfg: MarvConfig;
  defaultProvider: string;
}): ModelAliasIndex {
  const byAlias = new Map<string, { alias: string; ref: ModelRef }>();
  const byKey = new Map<string, string[]>();

  const rawModels = params.cfg.models?.metadata ?? {};
  for (const [keyRaw, entryRaw] of Object.entries(rawModels)) {
    const parsed = parseModelRef(String(keyRaw ?? ""), params.defaultProvider);
    if (!parsed) {
      continue;
    }
    const alias = String((entryRaw as { alias?: string } | undefined)?.alias ?? "").trim();
    if (!alias) {
      continue;
    }
    const aliasKey = normalizeAliasKey(alias);
    byAlias.set(aliasKey, { alias, ref: parsed });
    const key = modelKey(parsed.provider, parsed.model);
    const existing = byKey.get(key) ?? [];
    existing.push(alias);
    byKey.set(key, existing);
  }

  return { byAlias, byKey };
}

export function resolveModelRefFromString(params: {
  raw: string;
  defaultProvider: string;
  aliasIndex?: ModelAliasIndex;
}): { ref: ModelRef; alias?: string } | null {
  const trimmed = params.raw.trim();
  if (!trimmed) {
    return null;
  }
  if (!trimmed.includes("/")) {
    const aliasKey = normalizeAliasKey(trimmed);
    const aliasMatch = params.aliasIndex?.byAlias.get(aliasKey);
    if (aliasMatch) {
      return { ref: aliasMatch.ref, alias: aliasMatch.alias };
    }
  }
  const parsed = parseModelRef(trimmed, params.defaultProvider);
  if (!parsed) {
    return null;
  }
  return { ref: parsed };
}

export function resolveConfiguredModelRef(params: {
  cfg: MarvConfig;
  defaultProvider: string;
  defaultModel: string;
}): ModelRef {
  const configuredPrimary = normalizeModelSelection(params.cfg.agents?.defaults?.model);
  if (configuredPrimary) {
    const configuredProvider = configuredPrimary.includes("/")
      ? params.defaultProvider
      : inferProviderForBareModel(configuredPrimary, params.defaultProvider);
    const parsed = parseModelRef(configuredPrimary, configuredProvider);
    if (parsed) {
      if (!configuredPrimary.includes("/") && configuredProvider !== params.defaultProvider) {
        console.warn(
          `Falling back to "${parsed.provider}/${parsed.model}" because "${configuredPrimary}" did not include a provider prefix.`,
        );
      }
      return parsed;
    }
  }

  const selectedRefs = resolveSelectedModelRefs({
    cfg: params.cfg,
    defaultProvider: params.defaultProvider,
  });
  const catalog = params.cfg.models?.catalog ?? {};
  const poolName = params.cfg.agents?.defaults?.modelPool?.trim() || "default";
  const pool =
    params.cfg.agents?.modelPools?.[poolName] ??
    params.cfg.agents?.defaults?.modelPools?.[poolName];
  const candidateRawRefs = selectedRefs.size > 0 ? [...selectedRefs] : Object.keys(catalog);
  const refs = candidateRawRefs
    .map((raw) => parseModelRef(raw, params.defaultProvider))
    .filter((ref): ref is ModelRef => Boolean(ref))
    .filter((ref) => {
      const entry = catalog[`${ref.provider}/${ref.model}`] ?? catalog[ref.model];
      if (!pool) {
        return true;
      }
      if (pool.include?.length && !pool.include.includes(`${ref.provider}/${ref.model}`)) {
        return false;
      }
      if (pool.exclude?.includes(`${ref.provider}/${ref.model}`)) {
        return false;
      }
      if (pool.locations?.length && entry?.location && !pool.locations.includes(entry.location)) {
        return false;
      }
      if (pool.tiers?.length && entry?.tier && !pool.tiers.includes(entry.tier)) {
        return false;
      }
      if (pool.requireCapabilities?.length) {
        const capabilities = entry?.capabilities ?? ["text"];
        for (const capability of pool.requireCapabilities) {
          if (!capabilities.includes(capability)) {
            return false;
          }
        }
      }
      return true;
    })
    .toSorted((a, b) => {
      const entryA = catalog[`${a.provider}/${a.model}`] ?? catalog[a.model] ?? {};
      const entryB = catalog[`${b.provider}/${b.model}`] ?? catalog[b.model] ?? {};
      return compareDefaultSessionModelPreference(
        {
          provider: a.provider,
          model: a.model,
          location: entryA.location,
          tier: entryA.tier,
          priority: entryA.priority,
          capabilities: entryA.capabilities,
        },
        {
          provider: b.provider,
          model: b.model,
          location: entryB.location,
          tier: entryB.tier,
          priority: entryB.priority,
          capabilities: entryB.capabilities,
        },
      );
    });
  if (refs.length > 0) {
    return refs[0];
  }
  return { provider: params.defaultProvider, model: params.defaultModel };
}

export function resolveDefaultModelForAgent(params: {
  cfg: MarvConfig;
  agentId?: string;
}): ModelRef {
  const agentConfig = params.agentId ? resolveAgentConfig(params.cfg, params.agentId) : undefined;
  const poolName = agentConfig?.modelPool?.trim() || params.cfg.agents?.defaults?.modelPool?.trim();
  const agentModel =
    typeof agentConfig?.model === "string"
      ? { primary: agentConfig.model }
      : agentConfig?.model
        ? {
            primary: agentConfig.model.primary,
            fallbacks: agentConfig.model.fallbacks,
          }
        : undefined;
  const shouldOverrideDefaults = Boolean(
    agentModel || (poolName && poolName !== params.cfg.agents?.defaults?.modelPool),
  );
  const cfg = shouldOverrideDefaults
    ? {
        ...params.cfg,
        agents: {
          ...params.cfg.agents,
          defaults: {
            ...params.cfg.agents?.defaults,
            ...(agentModel ? { model: agentModel } : {}),
            ...(poolName ? { modelPool: poolName } : {}),
          },
        },
      }
    : params.cfg;
  return resolveConfiguredModelRef({
    cfg,
    defaultProvider: DEFAULT_PROVIDER,
    defaultModel: DEFAULT_MODEL,
  });
}

export function resolveSubagentConfiguredModelSelection(params: {
  cfg: MarvConfig;
  agentId: string;
}): string | undefined {
  const agentConfig = resolveAgentConfig(params.cfg, params.agentId);
  return (
    normalizeModelSelection(agentConfig?.subagents?.model) ??
    normalizeModelSelection(params.cfg.agents?.defaults?.subagents?.model) ??
    normalizeModelSelection(agentConfig?.model)
  );
}

export function resolveSubagentRoleModelSelection(params: {
  cfg: MarvConfig;
  agentId: string;
  role?: string;
  modelOverride?: unknown;
}): string | undefined {
  const explicit = normalizeModelSelection(params.modelOverride);
  if (explicit) {
    return explicit;
  }

  const roleKey = params.role?.trim();
  const roleConfig = roleKey ? params.cfg.agents?.defaults?.subagents?.roles?.[roleKey] : undefined;
  const roleModel = normalizeModelSelection(roleConfig?.model);
  if (roleModel) {
    return roleModel;
  }

  const rolePoolName = roleConfig?.modelPool?.trim();
  if (rolePoolName) {
    const cfgWithRolePool: MarvConfig = {
      ...params.cfg,
      agents: {
        ...params.cfg.agents,
        defaults: {
          ...params.cfg.agents?.defaults,
          modelPool: rolePoolName,
        },
      },
    };
    const resolved = resolveConfiguredModelRef({
      cfg: cfgWithRolePool,
      defaultProvider: DEFAULT_PROVIDER,
      defaultModel: DEFAULT_MODEL,
    });
    return `${resolved.provider}/${resolved.model}`;
  }

  return resolveSubagentConfiguredModelSelection({
    cfg: params.cfg,
    agentId: params.agentId,
  });
}

export function resolveSubagentSpawnModelSelection(params: {
  cfg: MarvConfig;
  agentId: string;
  modelOverride?: unknown;
  role?: string;
}): string {
  const runtimeDefault = resolveDefaultModelForAgent({
    cfg: params.cfg,
    agentId: params.agentId,
  });
  return (
    resolveSubagentRoleModelSelection({
      cfg: params.cfg,
      agentId: params.agentId,
      role: params.role,
      modelOverride: params.modelOverride,
    }) ?? `${runtimeDefault.provider}/${runtimeDefault.model}`
  );
}

export function buildAllowedModelSet(params: {
  cfg: MarvConfig;
  catalog: ModelCatalogEntry[];
  defaultProvider: string;
  defaultModel?: string;
}): {
  allowAny: boolean;
  allowedCatalog: ModelCatalogEntry[];
  allowedKeys: Set<string>;
} {
  const selectedRefs = resolveSelectedModelRefs({
    cfg: params.cfg,
    defaultProvider: params.defaultProvider,
  });
  const configuredProviders = new Set<string>([
    ...Object.keys(params.cfg.models?.providers ?? {}).map((provider) =>
      normalizeProviderId(provider),
    ),
    ...Object.values(params.cfg.auth?.profiles ?? {})
      .map((profile) => normalizeProviderId(String(profile?.provider ?? "")))
      .filter(Boolean),
  ]);
  const defaultModel = params.defaultModel?.trim();
  const defaultRef =
    defaultModel && params.defaultProvider
      ? parseModelRef(defaultModel, params.defaultProvider)
      : null;
  const defaultKey = defaultRef ? modelKey(defaultRef.provider, defaultRef.model) : undefined;
  const allowedKeys = new Set<string>();
  if (selectedRefs.size > 0) {
    for (const ref of selectedRefs) {
      allowedKeys.add(ref);
    }
  } else {
    for (const entry of params.catalog) {
      if (configuredProviders.has(normalizeProviderId(entry.provider))) {
        allowedKeys.add(modelKey(entry.provider, entry.id));
      }
    }
    for (const [provider, config] of Object.entries(params.cfg.models?.providers ?? {})) {
      for (const model of config.models ?? []) {
        const parsed = normalizeModelRef(provider, model.id);
        allowedKeys.add(modelKey(parsed.provider, parsed.model));
      }
    }
  }

  if (allowedKeys.size === 0) {
    const allKeys = new Set(params.catalog.map((entry) => modelKey(entry.provider, entry.id)));
    if (defaultKey) {
      allKeys.add(defaultKey);
    }
    return {
      allowAny: true,
      allowedCatalog: params.catalog,
      allowedKeys: allKeys,
    };
  }

  if (defaultKey) {
    allowedKeys.add(defaultKey);
  }

  const allowedCatalog = params.catalog.filter((entry) =>
    allowedKeys.has(modelKey(entry.provider, entry.id)),
  );

  return { allowAny: false, allowedCatalog, allowedKeys };
}

export type ModelRefStatus = {
  key: string;
  inCatalog: boolean;
  allowAny: boolean;
  allowed: boolean;
};

export function getModelRefStatus(params: {
  cfg: MarvConfig;
  catalog: ModelCatalogEntry[];
  ref: ModelRef;
  defaultProvider: string;
  defaultModel?: string;
}): ModelRefStatus {
  const allowed = buildAllowedModelSet({
    cfg: params.cfg,
    catalog: params.catalog,
    defaultProvider: params.defaultProvider,
    defaultModel: params.defaultModel,
  });
  const key = modelKey(params.ref.provider, params.ref.model);
  return {
    key,
    inCatalog: params.catalog.some((entry) => modelKey(entry.provider, entry.id) === key),
    allowAny: allowed.allowAny,
    allowed: allowed.allowAny || allowed.allowedKeys.has(key),
  };
}

export function resolveAllowedModelRef(params: {
  cfg: MarvConfig;
  catalog: ModelCatalogEntry[];
  raw: string;
  defaultProvider: string;
  defaultModel?: string;
}):
  | { ref: ModelRef; key: string }
  | {
      error: string;
    } {
  const trimmed = params.raw.trim();
  if (!trimmed) {
    return { error: "invalid model: empty" };
  }

  const aliasIndex = buildModelAliasIndex({
    cfg: params.cfg,
    defaultProvider: params.defaultProvider,
  });
  const resolved = resolveModelRefFromString({
    raw: trimmed,
    defaultProvider: params.defaultProvider,
    aliasIndex,
  });
  if (!resolved) {
    return { error: `invalid model: ${trimmed}` };
  }

  const status = getModelRefStatus({
    cfg: params.cfg,
    catalog: params.catalog,
    ref: resolved.ref,
    defaultProvider: params.defaultProvider,
    defaultModel: params.defaultModel,
  });
  if (!status.allowed) {
    return { error: `model not allowed: ${status.key}` };
  }

  return { ref: resolved.ref, key: status.key };
}

export function resolveThinkingDefault(params: {
  cfg: MarvConfig;
  provider: string;
  model: string;
  catalog?: ModelCatalogEntry[];
}): ThinkLevel {
  const configured = params.cfg.agents?.defaults?.thinkingDefault;
  if (configured) {
    return configured;
  }
  const candidate = params.catalog?.find(
    (entry) => entry.provider === params.provider && entry.id === params.model,
  );
  if (candidate?.reasoning) {
    return "low";
  }
  return "off";
}

/**
 * Resolve the model configured for Gmail hook processing.
 * Returns null if hooks.gmail.model is not set.
 */
export function resolveHooksGmailModel(params: {
  cfg: MarvConfig;
  defaultProvider: string;
}): ModelRef | null {
  const hooksModel = params.cfg.hooks?.gmail?.model;
  if (!hooksModel?.trim()) {
    return null;
  }

  const aliasIndex = buildModelAliasIndex({
    cfg: params.cfg,
    defaultProvider: params.defaultProvider,
  });

  const resolved = resolveModelRefFromString({
    raw: hooksModel,
    defaultProvider: params.defaultProvider,
    aliasIndex,
  });

  return resolved?.ref ?? null;
}
