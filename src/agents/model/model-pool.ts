import type { MarvConfig } from "../../core/config/config.js";
import type {
  ConfiguredModelCatalogEntry,
  ConfiguredModelLocation,
  ConfiguredModelTier,
  ModelPoolConfig,
} from "../../core/config/types.js";
import { resolveAgentConfig, resolveDefaultAgentId } from "../agent-scope.js";
import { ensureAuthProfileStore, listProfilesForProvider } from "../auth-profiles.js";
import { getCustomProviderApiKey, resolveEnvApiKey } from "./model-auth.js";
import { getRuntimeModelAvailability } from "./model-availability-state.js";
import { buildModelAliasIndex, normalizeProviderId, parseModelRef } from "./model-selection.js";
import { resolveProviderFamilyProviders, resolveSelectedModelRefs } from "./model-selections.js";

const LOCAL_PROVIDERS = new Set(["ollama", "vllm", "lmstudio", "localai", "llamacpp"]);

export type RuntimeModelCapability = "text" | "vision" | "coding" | "tools";

export type RuntimeConfiguredModel = {
  ref: string;
  provider: string;
  model: string;
  location: ConfiguredModelLocation;
  tier: ConfiguredModelTier;
  capabilities: RuntimeModelCapability[];
  priority: number;
  enabled: boolean;
  available: boolean;
  availabilityReason?: string;
  aliases: string[];
};

export type RuntimeModelPlan = {
  poolName: string;
  configured: RuntimeConfiguredModel[];
  candidates: RuntimeConfiguredModel[];
};

export type RuntimeModelRequirements = {
  requiredCapabilities?: RuntimeModelCapability[];
};

function inferLocation(
  provider: string,
  entry?: ConfiguredModelCatalogEntry,
): ConfiguredModelLocation {
  if (entry?.location) {
    return entry.location;
  }
  return LOCAL_PROVIDERS.has(normalizeProviderId(provider)) ? "local" : "cloud";
}

function inferTier(entry?: ConfiguredModelCatalogEntry): ConfiguredModelTier {
  return entry?.tier ?? "standard";
}

function inferCapabilities(
  entry?: ConfiguredModelCatalogEntry,
  model?: string,
): RuntimeModelCapability[] {
  const configured = entry?.capabilities;
  if (configured && configured.length > 0) {
    return [...new Set(configured)];
  }
  const inferred = new Set<RuntimeModelCapability>(["text"]);
  const lower = (model ?? "").toLowerCase();
  if (
    lower.includes("vision") ||
    lower.includes("vl") ||
    lower.includes("4o") ||
    lower.includes("gemini") ||
    lower.includes("gpt-4.1") ||
    lower.includes("gpt-5")
  ) {
    inferred.add("vision");
  }
  if (
    lower.includes("coder") ||
    lower.includes("codex") ||
    lower.includes("code") ||
    lower.includes("kimi-k2") ||
    lower.includes("claude")
  ) {
    inferred.add("coding");
    inferred.add("tools");
  }
  return [...inferred];
}

function isModelAvailable(params: { cfg: MarvConfig; provider: string; agentDir?: string }): {
  available: boolean;
  reason?: string;
} {
  const provider = normalizeProviderId(params.provider);
  if (LOCAL_PROVIDERS.has(provider)) {
    return { available: true };
  }
  const store = ensureAuthProfileStore(params.agentDir, { allowKeychainPrompt: false });
  const providerFamily = [...resolveProviderFamilyProviders(provider)];
  for (const familyProvider of providerFamily) {
    if (listProfilesForProvider(store, familyProvider).length > 0) {
      return { available: true };
    }
    if (resolveEnvApiKey(familyProvider) || getCustomProviderApiKey(params.cfg, familyProvider)) {
      return { available: true };
    }
  }
  return { available: false, reason: "missing_auth" };
}

function resolvePoolConfig(
  cfg: MarvConfig,
  agentId?: string,
): { poolName: string; pool?: ModelPoolConfig } {
  const agentCfg = agentId ? resolveAgentConfig(cfg, agentId) : undefined;
  const poolName =
    agentCfg?.modelPool?.trim() || cfg.agents?.defaults?.modelPool?.trim() || "default";
  return {
    poolName,
    pool: cfg.agents?.modelPools?.[poolName] ?? cfg.agents?.defaults?.modelPools?.[poolName],
  };
}

function buildConfiguredModelList(params: {
  cfg: MarvConfig;
  agentDir?: string;
  pool?: ModelPoolConfig;
}): RuntimeConfiguredModel[] {
  const refs = new Set<string>();
  const explicitCatalog = params.cfg.models?.catalog ?? {};
  const selectedRefs = resolveSelectedModelRefs({
    cfg: params.cfg,
    defaultProvider: "anthropic",
  });

  if (selectedRefs.size > 0) {
    for (const ref of selectedRefs) {
      refs.add(ref);
    }
  } else {
    for (const raw of Object.keys(explicitCatalog)) {
      refs.add(raw);
    }
    for (const raw of params.pool?.include ?? []) {
      refs.add(raw);
    }
  }

  const defaultProvider = "anthropic";
  const aliasIndex = buildModelAliasIndex({
    cfg: params.cfg,
    defaultProvider,
  });
  const configuredModels: RuntimeConfiguredModel[] = [];
  for (const rawRef of refs) {
    const parsed = parseModelRef(rawRef, defaultProvider);
    if (!parsed) {
      continue;
    }
    const ref = `${parsed.provider}/${parsed.model}`;
    const catalogEntry = explicitCatalog[ref] ?? explicitCatalog[rawRef];
    const runtimeAvailability = getRuntimeModelAvailability(ref);
    const availability = isModelAvailable({
      cfg: params.cfg,
      provider: parsed.provider,
      agentDir: params.agentDir,
    });
    const enabledBySelection = selectedRefs.size === 0 || selectedRefs.has(ref);
    const blockedByUnsupportedState = runtimeAvailability?.status === "unsupported";
    configuredModels.push({
      ref,
      provider: parsed.provider,
      model: parsed.model,
      location: inferLocation(parsed.provider, catalogEntry),
      tier: inferTier(catalogEntry),
      capabilities: inferCapabilities(catalogEntry, parsed.model),
      priority: catalogEntry?.priority ?? 0,
      enabled: catalogEntry?.enabled !== false && enabledBySelection,
      available:
        catalogEntry?.enabled === false
          ? false
          : blockedByUnsupportedState
            ? false
            : availability.available,
      availabilityReason:
        catalogEntry?.enabled === false
          ? "disabled"
          : blockedByUnsupportedState
            ? "unsupported"
            : runtimeAvailability?.status === "auth_invalid"
              ? "auth_invalid"
              : availability.reason,
      aliases: aliasIndex.byKey.get(ref) ?? [],
    });
  }
  return configuredModels;
}

function matchesPool(entry: RuntimeConfiguredModel, pool?: ModelPoolConfig): boolean {
  if (!pool) {
    return true;
  }
  if (pool.locations?.length && !pool.locations.includes(entry.location)) {
    return false;
  }
  if (pool.tiers?.length && !pool.tiers.includes(entry.tier)) {
    return false;
  }
  if (pool.requireCapabilities?.length) {
    for (const capability of pool.requireCapabilities) {
      if (!entry.capabilities.includes(capability)) {
        return false;
      }
    }
  }
  if (pool.include?.length && !pool.include.includes(entry.ref)) {
    return false;
  }
  if (pool.exclude?.includes(entry.ref)) {
    return false;
  }
  return true;
}

function matchesRequirements(
  entry: RuntimeConfiguredModel,
  requirements?: RuntimeModelRequirements,
): boolean {
  if (!requirements?.requiredCapabilities?.length) {
    return true;
  }
  for (const capability of requirements.requiredCapabilities) {
    if (!entry.capabilities.includes(capability)) {
      return false;
    }
  }
  return true;
}

function tierWeight(tier: ConfiguredModelTier): number {
  switch (tier) {
    case "low":
      return 0;
    case "standard":
      return 1;
    case "high":
      return 2;
  }
}

function locationWeight(location: ConfiguredModelLocation): number {
  return location === "local" ? 0 : 1;
}

export function resolveRuntimeModelPlan(params: {
  cfg: MarvConfig;
  agentId?: string;
  agentDir?: string;
  requirements?: RuntimeModelRequirements;
}): RuntimeModelPlan {
  const poolConfig = resolvePoolConfig(params.cfg, params.agentId);
  const configured = buildConfiguredModelList({
    cfg: params.cfg,
    agentDir: params.agentDir,
    pool: poolConfig.pool,
  });
  const candidates = configured
    .filter((entry) => entry.enabled && entry.available)
    .filter((entry) => matchesPool(entry, poolConfig.pool))
    .filter((entry) => matchesRequirements(entry, params.requirements))
    .toSorted((a, b) => {
      const byLocation = locationWeight(a.location) - locationWeight(b.location);
      if (byLocation !== 0) {
        return byLocation;
      }
      const byTier = tierWeight(a.tier) - tierWeight(b.tier);
      if (byTier !== 0) {
        return byTier;
      }
      const byPriority = (a.priority ?? 0) - (b.priority ?? 0);
      if (byPriority !== 0) {
        return byPriority;
      }
      return a.ref.localeCompare(b.ref);
    });

  return {
    poolName: poolConfig.poolName,
    configured,
    candidates,
  };
}

export function resolveAgentModelPoolName(params: { cfg: MarvConfig; agentId?: string }): string {
  return resolvePoolConfig(params.cfg, params.agentId ?? resolveDefaultAgentId(params.cfg))
    .poolName;
}
