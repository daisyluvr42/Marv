import {
  resolveApiKeyForProvider,
  type ResolvedProviderAuth,
} from "../../agents/model/model-auth.js";
import {
  resolveRuntimeModelPlan,
  type RuntimeConfiguredModel,
} from "../../agents/model/model-pool.js";
import { normalizeProviderId, parseModelRef } from "../../agents/model/model-ref.js";
import type { MarvConfig } from "../../core/config/config.js";
import type { ConfiguredModelLocation, ConfiguredModelTier } from "../../core/config/types.js";

/**
 * Experience pipeline role. Each role has a preferred model tier/location.
 *
 * - distiller: standard/high cloud model (e.g. gemini-2.0-flash)
 * - calibration: highest-tier model (agent's primary)
 * - context: cheap local/low model
 * - attribution: cheap local/low model
 */
export type ExperienceRole = "distiller" | "calibration" | "context" | "attribution";

/** How to call the resolved model's API. */
export type ExperienceApiType = "ollama" | "openai-compat" | "google" | "anthropic";

export type ResolvedExperienceModel = {
  provider: string;
  model: string;
  ref: string;
  location: ConfiguredModelLocation;
  tier: ConfiguredModelTier;
  apiType: ExperienceApiType;
  baseUrl: string;
  auth?: ResolvedProviderAuth;
};

// ---------------------------------------------------------------------------
// Provider classification
// ---------------------------------------------------------------------------

const LOCAL_PROVIDERS = new Set(["ollama", "vllm", "lmstudio", "localai", "llamacpp"]);

const PROVIDER_API_TYPES: Record<string, ExperienceApiType> = {
  ollama: "ollama",
  google: "google",
  "google-vertex": "google",
  anthropic: "anthropic",
};

const DEFAULT_BASE_URLS: Record<string, string> = {
  openai: "https://api.openai.com/v1",
  google: "https://generativelanguage.googleapis.com/v1beta",
  anthropic: "https://api.anthropic.com/v1",
  groq: "https://api.groq.com/openai/v1",
  together: "https://api.together.xyz/v1",
  openrouter: "https://openrouter.ai/api/v1",
  xai: "https://api.x.ai/v1",
  cerebras: "https://api.cerebras.ai/v1",
  ollama: "http://127.0.0.1:11434",
  vllm: "http://127.0.0.1:8000/v1",
  lmstudio: "http://127.0.0.1:1234/v1",
};

function resolveApiType(provider: string): ExperienceApiType {
  return PROVIDER_API_TYPES[provider] ?? "openai-compat";
}

function resolveBaseUrl(cfg: MarvConfig, provider: string): string {
  // Honour explicit provider config first
  const providerEntry = cfg.models?.providers?.[provider];
  if (providerEntry && typeof providerEntry === "object" && providerEntry.baseUrl) {
    return providerEntry.baseUrl.replace(/\/+$/, "");
  }
  return DEFAULT_BASE_URLS[provider] ?? "";
}

// ---------------------------------------------------------------------------
// Role → tier/location preferences
// ---------------------------------------------------------------------------

function roleTierPreference(role: ExperienceRole): ConfiguredModelTier[] {
  switch (role) {
    case "calibration":
      return ["high", "standard", "low"];
    case "distiller":
      return ["standard", "high", "low"];
    case "context":
      return ["low", "standard", "high"];
    case "attribution":
      return ["low", "standard", "high"];
  }
}

function roleLocationPreference(role: ExperienceRole): ConfiguredModelLocation[] {
  switch (role) {
    case "calibration":
      return ["cloud", "local"];
    case "distiller":
      return ["cloud", "local"];
    case "context":
      return ["local", "cloud"];
    case "attribution":
      return ["local", "cloud"];
  }
}

// ---------------------------------------------------------------------------
// Main resolver
// ---------------------------------------------------------------------------

/**
 * Resolve the best available model for an experience pipeline role.
 *
 * Resolution order:
 * 1. Explicit per-role config in ExperienceConfig (e.g. `calibrationModel`)
 * 2. Legacy `deepConsolidation.model` config
 * 3. Runtime model pool, scored by role tier/location preference
 *
 * Returns null when no suitable model is available.
 */
export async function resolveExperienceModel(params: {
  cfg: MarvConfig;
  role: ExperienceRole;
  agentId?: string;
  agentDir?: string;
}): Promise<ResolvedExperienceModel | null> {
  const { cfg, role } = params;
  const experienceConfig = cfg.memory?.experience;

  // 1. Explicit per-role model ref (e.g. "google/gemini-2.0-flash")
  const roleModelKeys: Record<ExperienceRole, string> = {
    distiller: "distillerModel",
    calibration: "calibrationModel",
    context: "contextModel",
    attribution: "attributionModel",
  };
  const explicitRef = experienceConfig?.[roleModelKeys[role] as keyof typeof experienceConfig] as
    | string
    | undefined;
  if (explicitRef?.trim()) {
    const resolved = await resolveFromRef(cfg, explicitRef.trim(), params.agentDir);
    if (resolved) {
      return resolved;
    }
  }

  // 2. Legacy deepConsolidation.model
  const legacyModel = cfg.memory?.soul?.deepConsolidation?.model;
  if (legacyModel && (legacyModel.model || legacyModel.provider)) {
    const resolved = resolveFromLegacyConfig(cfg, legacyModel);
    if (resolved) {
      return resolved;
    }
  }

  // 3. Runtime model pool with tier/location scoring
  const plan = resolveRuntimeModelPlan({
    cfg,
    agentId: params.agentId,
    agentDir: params.agentDir,
  });

  if (plan.candidates.length === 0) {
    return null;
  }

  const tierPref = roleTierPreference(role);
  const locationPref = roleLocationPreference(role);

  const scored = plan.candidates.map((candidate) => {
    const tierIdx = tierPref.indexOf(candidate.tier);
    const locIdx = locationPref.indexOf(candidate.location);
    // Lower score = better match. Unknown positions pushed to the end.
    const tierScore = tierIdx >= 0 ? tierIdx : tierPref.length;
    const locScore = locIdx >= 0 ? locIdx : locationPref.length;
    return { model: candidate, score: locScore * 10 + tierScore };
  });
  scored.sort((a, b) => a.score - b.score);

  const best = scored[0].model;
  return await resolveFromRuntimeModel(cfg, best, params.agentDir);
}

// ---------------------------------------------------------------------------
// Internal resolution helpers
// ---------------------------------------------------------------------------

async function resolveFromRef(
  cfg: MarvConfig,
  ref: string,
  agentDir?: string,
): Promise<ResolvedExperienceModel | null> {
  const parsed = parseModelRef(ref, "openai");
  if (!parsed) {
    return null;
  }

  const provider = normalizeProviderId(parsed.provider);
  const isLocal = LOCAL_PROVIDERS.has(provider);
  const apiType = resolveApiType(provider);
  const baseUrl = resolveBaseUrl(cfg, provider);

  let auth: ResolvedProviderAuth | undefined;
  if (!isLocal) {
    try {
      auth = await resolveApiKeyForProvider({ provider, cfg, agentDir });
    } catch {
      // Cannot authenticate — skip this model
      return null;
    }
  }

  return {
    provider,
    model: parsed.model,
    ref: `${provider}/${parsed.model}`,
    location: isLocal ? "local" : "cloud",
    tier: "standard",
    apiType,
    baseUrl,
    auth,
  };
}

function resolveFromLegacyConfig(
  cfg: MarvConfig,
  legacyModel: { provider?: string; api?: string; model?: string; baseUrl?: string },
): ResolvedExperienceModel | null {
  const provider = legacyModel.provider?.trim() || "ollama";
  const model = legacyModel.model?.trim();
  if (!model) {
    return null;
  }

  const normalized = normalizeProviderId(provider);
  const isLocal = LOCAL_PROVIDERS.has(normalized);
  const apiType = legacyModel.api === "ollama" ? "ollama" : resolveApiType(normalized);
  const baseUrl =
    legacyModel.baseUrl?.trim().replace(/\/+$/, "") || resolveBaseUrl(cfg, normalized);

  return {
    provider: normalized,
    model,
    ref: `${normalized}/${model}`,
    location: isLocal ? "local" : "cloud",
    tier: "standard",
    apiType,
    baseUrl,
  };
}

async function resolveFromRuntimeModel(
  cfg: MarvConfig,
  model: RuntimeConfiguredModel,
  agentDir?: string,
): Promise<ResolvedExperienceModel | null> {
  const provider = normalizeProviderId(model.provider);
  const isLocal = LOCAL_PROVIDERS.has(provider) || model.location === "local";
  const apiType = resolveApiType(provider);
  const baseUrl = resolveBaseUrl(cfg, provider);

  let auth: ResolvedProviderAuth | undefined;
  if (!isLocal) {
    try {
      auth = await resolveApiKeyForProvider({ provider, cfg, agentDir });
    } catch {
      return null;
    }
  }

  return {
    provider,
    model: model.model,
    ref: model.ref,
    location: model.location,
    tier: model.tier,
    apiType,
    baseUrl,
    auth,
  };
}
