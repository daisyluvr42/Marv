import type { Api, Model } from "@mariozechner/pi-ai";
import { resolveMarvAgentDir } from "../../agents/agent-paths.js";
import type { AuthProfileStore } from "../../agents/auth-profiles.js";
import { listProfilesForProvider } from "../../agents/auth-profiles.js";
import {
  getCustomProviderApiKey,
  resolveAwsSdkEnvVarName,
  resolveEnvApiKey,
} from "../../agents/model/model-auth.js";
import {
  ANTIGRAVITY_OPUS_46_FORWARD_COMPAT_CANDIDATES,
  resolveForwardCompatModel,
} from "../../agents/model/model-forward-compat.js";
import { normalizeProviderId } from "../../agents/model/model-resolve.js";
import { ensureMarvModelsJson } from "../../agents/model/models-config.js";
import type { ModelRegistry } from "../../agents/model/pi-model-discovery.js";
import { discoverAuthStorage, discoverModels } from "../../agents/model/pi-model-discovery.js";
import { ensurePiAuthJsonFromAuthProfiles } from "../../agents/runner/pi-auth-json.js";
import type { MarvConfig } from "../../core/config/config.js";
import {
  formatErrorWithStack,
  MODEL_AVAILABILITY_UNAVAILABLE_CODE,
  shouldFallbackToAuthHeuristics,
} from "./list.errors.js";
import type { ModelRow } from "./list.types.js";
import { isLocalBaseUrl, modelKey } from "./shared.js";

type ConfiguredProviderEntry = NonNullable<NonNullable<MarvConfig["models"]>["providers"]>[string];

function resolveConfiguredProviderEntry(
  cfg: MarvConfig | undefined,
  provider: string,
): ConfiguredProviderEntry | undefined {
  const providers = cfg?.models?.providers ?? {};
  const direct = providers[provider];
  if (direct) {
    return direct;
  }

  const normalized = normalizeProviderId(provider);
  return Object.entries(providers).find(([key]) => normalizeProviderId(key) === normalized)?.[1];
}

function isKeylessBaseUrlProvider(cfg: MarvConfig | undefined, provider: string): boolean {
  const entry = resolveConfiguredProviderEntry(cfg, provider);
  return Boolean(entry?.baseUrl?.trim() && !getCustomProviderApiKey(cfg, provider));
}

const hasAuthForProvider = (provider: string, cfg?: MarvConfig, authStore?: AuthProfileStore) => {
  if (!cfg || !authStore) {
    return false;
  }
  if (listProfilesForProvider(authStore, provider).length > 0) {
    return true;
  }
  if (provider === "amazon-bedrock" && resolveAwsSdkEnvVarName()) {
    return true;
  }
  if (resolveEnvApiKey(provider)) {
    return true;
  }
  if (getCustomProviderApiKey(cfg, provider)) {
    return true;
  }
  if (isKeylessBaseUrlProvider(cfg, provider)) {
    return true;
  }
  return false;
};

function createAvailabilityUnavailableError(message: string): Error {
  const err = new Error(message);
  (err as { code?: string }).code = MODEL_AVAILABILITY_UNAVAILABLE_CODE;
  return err;
}

function normalizeAvailabilityError(err: unknown): Error {
  if (shouldFallbackToAuthHeuristics(err) && err instanceof Error) {
    return err;
  }
  return createAvailabilityUnavailableError(
    `Model availability unavailable: getAvailable() failed.\n${formatErrorWithStack(err)}`,
  );
}

function validateAvailableModels(availableModels: unknown): Model<Api>[] {
  if (!Array.isArray(availableModels)) {
    throw createAvailabilityUnavailableError(
      "Model availability unavailable: getAvailable() returned a non-array value.",
    );
  }

  for (const model of availableModels) {
    if (
      !model ||
      typeof model !== "object" ||
      typeof (model as { provider?: unknown }).provider !== "string" ||
      typeof (model as { id?: unknown }).id !== "string"
    ) {
      throw createAvailabilityUnavailableError(
        "Model availability unavailable: getAvailable() returned invalid model entries.",
      );
    }
  }

  return availableModels as Model<Api>[];
}

function loadAvailableModels(registry: ModelRegistry): Model<Api>[] {
  let availableModels: unknown;
  try {
    availableModels = registry.getAvailable();
  } catch (err) {
    throw normalizeAvailabilityError(err);
  }
  try {
    return validateAvailableModels(availableModels);
  } catch (err) {
    throw normalizeAvailabilityError(err);
  }
}

export async function loadModelRegistry(cfg: MarvConfig) {
  await ensureMarvModelsJson(cfg);
  const agentDir = resolveMarvAgentDir();
  await ensurePiAuthJsonFromAuthProfiles(agentDir);
  const authStorage = discoverAuthStorage(agentDir);
  const registry = discoverModels(authStorage, agentDir);
  const appended = appendAntigravityForwardCompatModels(registry.getAll(), registry);
  const models = mergeConfiguredBaseUrlModels(cfg, appended.models);
  const synthesizedForwardCompat = appended.synthesizedForwardCompat;
  let availableKeys: Set<string> | undefined;
  let availabilityErrorMessage: string | undefined;

  try {
    const availableModels = loadAvailableModels(registry);
    availableKeys = new Set(availableModels.map((model) => modelKey(model.provider, model.id)));
    for (const model of models) {
      if (isKeylessBaseUrlProvider(cfg, model.provider)) {
        availableKeys.add(modelKey(model.provider, model.id));
      }
    }
    for (const synthesized of synthesizedForwardCompat) {
      if (hasAvailableTemplate(availableKeys, synthesized.templatePrefixes)) {
        availableKeys.add(synthesized.key);
      }
    }
  } catch (err) {
    if (!shouldFallbackToAuthHeuristics(err)) {
      throw err;
    }

    // Some providers can report model-level availability as unavailable.
    // Fall back to provider-level auth heuristics when availability is undefined.
    availableKeys = undefined;
    if (!availabilityErrorMessage) {
      availabilityErrorMessage = formatErrorWithStack(err);
    }
  }
  return { registry, models, availableKeys, availabilityErrorMessage };
}

function mergeConfiguredBaseUrlModels(
  cfg: MarvConfig,
  discoveredModels: Model<Api>[],
): Model<Api>[] {
  const merged = [...discoveredModels];
  const seen = new Set(discoveredModels.map((model) => modelKey(model.provider, model.id)));

  for (const [provider, providerEntry] of Object.entries(cfg.models?.providers ?? {})) {
    if (!providerEntry?.baseUrl?.trim() || getCustomProviderApiKey(cfg, provider)) {
      continue;
    }

    for (const modelEntry of providerEntry.models ?? []) {
      const modelId = modelEntry.id?.trim();
      if (!modelId) {
        continue;
      }
      const modelName = (modelEntry as { name?: string }).name?.trim() || modelId;

      const key = modelKey(provider, modelId);
      if (seen.has(key)) {
        continue;
      }

      seen.add(key);
      merged.push({
        id: modelId,
        name: modelName,
        provider,
        baseUrl: providerEntry.baseUrl,
        api: (modelEntry.api ?? providerEntry.api ?? "openai-completions") as Api,
        reasoning: modelEntry.reasoning,
        input: modelEntry.input,
        cost: modelEntry.cost,
        contextWindow: modelEntry.contextWindow,
        maxTokens: modelEntry.maxTokens,
        headers: modelEntry.headers,
        compat: modelEntry.compat as never,
      });
    }
  }

  return merged;
}

type SynthesizedForwardCompat = {
  key: string;
  templatePrefixes: readonly string[];
};

function appendAntigravityForwardCompatModels(
  models: Model<Api>[],
  modelRegistry: ModelRegistry,
): { models: Model<Api>[]; synthesizedForwardCompat: SynthesizedForwardCompat[] } {
  const nextModels = [...models];
  const synthesizedForwardCompat: SynthesizedForwardCompat[] = [];

  for (const candidate of ANTIGRAVITY_OPUS_46_FORWARD_COMPAT_CANDIDATES) {
    const key = modelKey("google-antigravity", candidate.id);
    const hasForwardCompat = nextModels.some((model) => modelKey(model.provider, model.id) === key);
    if (hasForwardCompat) {
      continue;
    }

    const fallback = resolveForwardCompatModel("google-antigravity", candidate.id, modelRegistry);
    if (!fallback) {
      continue;
    }

    nextModels.push(fallback);
    synthesizedForwardCompat.push({
      key,
      templatePrefixes: candidate.templatePrefixes,
    });
  }

  return { models: nextModels, synthesizedForwardCompat };
}

function hasAvailableTemplate(
  availableKeys: Set<string>,
  templatePrefixes: readonly string[],
): boolean {
  for (const key of availableKeys) {
    if (templatePrefixes.some((prefix) => key.startsWith(prefix))) {
      return true;
    }
  }
  return false;
}

export function toModelRow(params: {
  model?: Model<Api>;
  key: string;
  tags: string[];
  aliases?: string[];
  availableKeys?: Set<string>;
  cfg?: MarvConfig;
  authStore?: AuthProfileStore;
}): ModelRow {
  const { model, key, tags, aliases = [], availableKeys, cfg, authStore } = params;
  if (!model) {
    return {
      key,
      name: key,
      input: "-",
      contextWindow: null,
      local: null,
      available: null,
      tags: [...tags, "missing"],
      missing: true,
    };
  }

  const input = model.input.join("+") || "text";
  const local = isLocalBaseUrl(model.baseUrl);
  // Prefer model-level registry availability when present.
  // Fall back to provider-level auth heuristics only if registry availability isn't available.
  const available =
    availableKeys !== undefined
      ? availableKeys.has(modelKey(model.provider, model.id))
      : cfg && authStore
        ? hasAuthForProvider(model.provider, cfg, authStore)
        : false;
  const aliasTags = aliases.length > 0 ? [`alias:${aliases.join(",")}`] : [];
  const mergedTags = new Set(tags);
  if (aliasTags.length > 0) {
    for (const tag of mergedTags) {
      if (tag === "alias" || tag.startsWith("alias:")) {
        mergedTags.delete(tag);
      }
    }
    for (const tag of aliasTags) {
      mergedTags.add(tag);
    }
  }

  return {
    key,
    name: model.name || model.id,
    input,
    contextWindow: model.contextWindow ?? null,
    local,
    available,
    tags: Array.from(mergedTags),
    missing: false,
  };
}
