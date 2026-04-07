import { DEFAULT_PROVIDER } from "../agents/defaults.js";
import { clearProviderFailureStates } from "../agents/model/model-availability-state.js";
import { loadModelCatalog } from "../agents/model/model-catalog.js";
import { normalizeProviderId, parseModelRef } from "../agents/model/model-selection.js";
import {
  replaceSelectedModelRefsForProviderFamily,
  resolveProviderFamilyProviders,
} from "../agents/model/model-selections.js";
import type { MarvConfig } from "../core/config/config.js";
import type { WizardPrompter } from "../wizard/prompts.js";

export async function syncProviderSelectionsAfterAuth(
  config: MarvConfig,
  defaultModel: string,
): Promise<MarvConfig> {
  const parsed = parseModelRef(defaultModel, DEFAULT_PROVIDER);
  if (!parsed) {
    return config;
  }

  const providerFamily = resolveProviderFamilyProviders(parsed.provider);

  // Clear unsupported/auth_invalid failure states for this provider family
  // so models can re-enter candidates with the new auth credentials.
  for (const familyProvider of providerFamily) {
    clearProviderFailureStates(familyProvider);
  }

  const catalog = await loadModelCatalog({ config, useCache: false });
  const refs = catalog
    .filter((entry) => providerFamily.has(normalizeProviderId(entry.provider)))
    .map((entry) => `${entry.provider}/${entry.id}`);

  refs.push(`${parsed.provider}/${parsed.model}`);
  return replaceSelectedModelRefsForProviderFamily({
    cfg: config,
    provider: parsed.provider,
    sourceKey: parsed.provider,
    refs,
    defaultProvider: parsed.provider,
  });
}

export async function applyDefaultModelChoice(params: {
  config: MarvConfig;
  setDefaultModel: boolean;
  defaultModel: string;
  applyDefaultConfig: (config: MarvConfig) => MarvConfig;
  applyProviderConfig: (config: MarvConfig) => MarvConfig;
  noteDefault?: string;
  noteAgentModel: (model: string) => Promise<void>;
  prompter: WizardPrompter;
}): Promise<{ config: MarvConfig; agentModelOverride?: string }> {
  if (params.setDefaultModel) {
    const next = await syncProviderSelectionsAfterAuth(
      params.applyDefaultConfig(params.config),
      params.defaultModel,
    );
    if (params.noteDefault) {
      await params.prompter.note(`Default model set to ${params.noteDefault}`, "Model configured");
    }
    return { config: next };
  }

  const next = await syncProviderSelectionsAfterAuth(
    params.applyProviderConfig(params.config),
    params.defaultModel,
  );
  await params.noteAgentModel(params.defaultModel);
  return { config: next, agentModelOverride: params.defaultModel };
}
