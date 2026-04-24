import { syncProviderSelectionsFromProviderConfig } from "../agents/model/model-selections-store.js";
import type { MarvConfig } from "../core/config/config.js";
import type {
  ModelApi,
  ModelMetadataConfig,
  ModelDefinitionConfig,
  ModelProviderConfig,
} from "../core/config/types.models.js";

function extractAgentDefaultModelFallbacks(model: unknown): string[] | undefined {
  if (!model || typeof model !== "object") {
    return undefined;
  }
  if (!("fallbacks" in model)) {
    return undefined;
  }
  const fallbacks = (model as { fallbacks?: unknown }).fallbacks;
  return Array.isArray(fallbacks) ? fallbacks.map((v) => String(v)) : undefined;
}

export function applyOnboardAuthAgentModelsAndProviders(
  cfg: MarvConfig,
  params: {
    modelMetadata: Record<string, ModelMetadataConfig>;
    providers: Record<string, ModelProviderConfig>;
  },
): MarvConfig {
  return {
    ...cfg,
    models: {
      mode: cfg.models?.mode ?? "merge",
      ...cfg.models,
      metadata: params.modelMetadata,
      selections: cfg.models?.selections,
      providers: params.providers,
    },
  };
}

export function applyAgentDefaultModelPrimary(cfg: MarvConfig, primary: string): MarvConfig {
  const existingFallbacks = extractAgentDefaultModelFallbacks(cfg.agents?.defaults?.model);
  return {
    ...cfg,
    agents: {
      ...cfg.agents,
      defaults: {
        ...cfg.agents?.defaults,
        model: {
          ...(existingFallbacks ? { fallbacks: existingFallbacks } : undefined),
          primary,
        },
      },
    },
  };
}

export function applyProviderConfigWithDefaultModels(
  cfg: MarvConfig,
  params: {
    modelMetadata: Record<string, ModelMetadataConfig>;
    providerId: string;
    api: ModelApi;
    baseUrl: string;
    defaultModels: ModelDefinitionConfig[];
    defaultModelId?: string;
  },
): MarvConfig {
  const providers = { ...cfg.models?.providers } as Record<string, ModelProviderConfig>;
  const existingProvider = providers[params.providerId] as ModelProviderConfig | undefined;

  const existingModels: ModelDefinitionConfig[] = Array.isArray(existingProvider?.models)
    ? existingProvider.models
    : [];

  const defaultModels = params.defaultModels;
  const defaultModelId = params.defaultModelId ?? defaultModels[0]?.id;
  const hasDefaultModel = defaultModelId
    ? existingModels.some((model) => model.id === defaultModelId)
    : true;
  const mergedModels =
    existingModels.length > 0
      ? hasDefaultModel || defaultModels.length === 0
        ? existingModels
        : [...existingModels, ...defaultModels]
      : defaultModels;
  providers[params.providerId] = buildProviderConfig({
    existingProvider,
    api: params.api,
    baseUrl: params.baseUrl,
    mergedModels,
    fallbackModels: defaultModels,
  });

  return syncProviderSelectionsFromProviderConfig({
    cfg: applyOnboardAuthAgentModelsAndProviders(cfg, {
      modelMetadata: params.modelMetadata,
      providers,
    }),
    providerId: params.providerId,
  });
}

export function applyProviderConfigWithDefaultModel(
  cfg: MarvConfig,
  params: {
    modelMetadata: Record<string, ModelMetadataConfig>;
    providerId: string;
    api: ModelApi;
    baseUrl: string;
    defaultModel: ModelDefinitionConfig;
    defaultModelId?: string;
  },
): MarvConfig {
  return applyProviderConfigWithDefaultModels(cfg, {
    modelMetadata: params.modelMetadata,
    providerId: params.providerId,
    api: params.api,
    baseUrl: params.baseUrl,
    defaultModels: [params.defaultModel],
    defaultModelId: params.defaultModelId ?? params.defaultModel.id,
  });
}

export function applyProviderConfigWithModelCatalog(
  cfg: MarvConfig,
  params: {
    modelMetadata: Record<string, ModelMetadataConfig>;
    providerId: string;
    api: ModelApi;
    baseUrl: string;
    catalogModels: ModelDefinitionConfig[];
  },
): MarvConfig {
  const providers = { ...cfg.models?.providers } as Record<string, ModelProviderConfig>;
  const existingProvider = providers[params.providerId] as ModelProviderConfig | undefined;
  const existingModels: ModelDefinitionConfig[] = Array.isArray(existingProvider?.models)
    ? existingProvider.models
    : [];

  const catalogModels = params.catalogModels;
  const mergedModels =
    existingModels.length > 0
      ? [
          ...existingModels,
          ...catalogModels.filter(
            (model) => !existingModels.some((existing) => existing.id === model.id),
          ),
        ]
      : catalogModels;
  providers[params.providerId] = buildProviderConfig({
    existingProvider,
    api: params.api,
    baseUrl: params.baseUrl,
    mergedModels,
    fallbackModels: catalogModels,
  });

  return syncProviderSelectionsFromProviderConfig({
    cfg: applyOnboardAuthAgentModelsAndProviders(cfg, {
      modelMetadata: params.modelMetadata,
      providers,
    }),
    providerId: params.providerId,
  });
}

function buildProviderConfig(params: {
  existingProvider: ModelProviderConfig | undefined;
  api: ModelApi;
  baseUrl: string;
  mergedModels: ModelDefinitionConfig[];
  fallbackModels: ModelDefinitionConfig[];
}): ModelProviderConfig {
  const { apiKey: existingApiKey, ...existingProviderRest } = (params.existingProvider ?? {}) as {
    apiKey?: string;
  };
  const normalizedApiKey = typeof existingApiKey === "string" ? existingApiKey.trim() : undefined;

  return {
    ...existingProviderRest,
    baseUrl: params.baseUrl,
    api: params.api,
    ...(normalizedApiKey ? { apiKey: normalizedApiKey } : {}),
    models: params.mergedModels.length > 0 ? params.mergedModels : params.fallbackModels,
  };
}
