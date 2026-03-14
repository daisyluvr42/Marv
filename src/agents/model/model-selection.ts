import type { MarvConfig } from "../../core/config/config.js";
import { resolveAgentConfig } from "../agent-scope.js";
import { DEFAULT_MODEL, DEFAULT_PROVIDER } from "../defaults.js";
import type { ModelCatalogEntry } from "./model-catalog.js";
import { resolveSelectedModelRefs } from "./model-selections.js";
import { normalizeGoogleModelId } from "./models-config.providers.js";

export type ModelRef = {
  provider: string;
  model: string;
};

export type ThinkLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh";

export type ModelAliasIndex = {
  byAlias: Map<string, { alias: string; ref: ModelRef }>;
  byKey: Map<string, string[]>;
};

const ANTHROPIC_MODEL_ALIASES: Record<string, string> = {
  "opus-4.6": "claude-opus-4-6",
  "opus-4.5": "claude-opus-4-5",
  "sonnet-4.6": "claude-sonnet-4-6",
  "sonnet-4.5": "claude-sonnet-4-5",
};
const OPENAI_CODEX_OAUTH_MODEL_PREFIXES = ["gpt-5.4-codex", "gpt-5.3-codex"] as const;

function normalizeAliasKey(value: string): string {
  return value.trim().toLowerCase();
}

export function modelKey(provider: string, model: string) {
  return `${provider}/${model}`;
}

export function normalizeProviderId(provider: string): string {
  const normalized = provider.trim().toLowerCase();
  if (normalized === "z.ai" || normalized === "z-ai") {
    return "zai";
  }
  if (normalized === "opencode-zen") {
    return "opencode";
  }
  if (normalized === "qwen") {
    return "qwen-portal";
  }
  if (normalized === "kimi-code") {
    return "kimi-coding";
  }
  return normalized;
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

function normalizeAnthropicModelId(model: string): string {
  const trimmed = model.trim();
  if (!trimmed) {
    return trimmed;
  }
  const lower = trimmed.toLowerCase();
  return ANTHROPIC_MODEL_ALIASES[lower] ?? trimmed;
}

function normalizeProviderModelId(provider: string, model: string): string {
  if (provider === "anthropic") {
    return normalizeAnthropicModelId(model);
  }
  if (provider === "google") {
    return normalizeGoogleModelId(model);
  }
  return model;
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

function shouldUseOpenAICodexProvider(provider: string, model: string): boolean {
  if (provider !== "openai") {
    return false;
  }
  const normalized = model.trim().toLowerCase();
  if (!normalized) {
    return false;
  }
  return OPENAI_CODEX_OAUTH_MODEL_PREFIXES.some(
    (prefix) => normalized === prefix || normalized.startsWith(`${prefix}-`),
  );
}

export function normalizeModelRef(provider: string, model: string): ModelRef {
  const normalizedProvider = normalizeProviderId(provider);
  const normalizedModel = normalizeProviderModelId(normalizedProvider, model.trim());
  if (shouldUseOpenAICodexProvider(normalizedProvider, normalizedModel)) {
    return { provider: "openai-codex", model: normalizedModel };
  }
  return { provider: normalizedProvider, model: normalizedModel };
}

export function parseModelRef(raw: string, defaultProvider: string): ModelRef | null {
  const trimmed = raw.trim();
  if (!trimmed) {
    return null;
  }
  const slash = trimmed.indexOf("/");
  if (slash === -1) {
    return normalizeModelRef(defaultProvider, trimmed);
  }
  const providerRaw = trimmed.slice(0, slash).trim();
  const model = trimmed.slice(slash + 1).trim();
  if (!providerRaw || !model) {
    return null;
  }
  return normalizeModelRef(providerRaw, model);
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
      const entryA = catalog[`${a.provider}/${a.model}`] ?? {};
      const entryB = catalog[`${b.provider}/${b.model}`] ?? {};
      const locationWeight = (entry: { location?: string }) => (entry.location === "cloud" ? 1 : 0);
      const tierWeight = (entry: { tier?: string }) =>
        entry.tier === "high" ? 2 : entry.tier === "standard" ? 1 : 0;
      const byLocation = locationWeight(entryA) - locationWeight(entryB);
      if (byLocation !== 0) {
        return byLocation;
      }
      const byTier = tierWeight(entryA) - tierWeight(entryB);
      if (byTier !== 0) {
        return byTier;
      }
      const byPriority = (entryA.priority ?? 0) - (entryB.priority ?? 0);
      if (byPriority !== 0) {
        return byPriority;
      }
      return modelKey(a.provider, a.model).localeCompare(modelKey(b.provider, b.model));
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
