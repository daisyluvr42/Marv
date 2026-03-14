import type { MarvConfig } from "../core/config/config.js";

export const OPENAI_DEFAULT_MODEL = "openai/gpt-5.1-codex";

export function applyOpenAIProviderConfig(cfg: MarvConfig): MarvConfig {
  const modelMetadata = { ...cfg.models?.metadata };
  modelMetadata[OPENAI_DEFAULT_MODEL] = {
    ...modelMetadata[OPENAI_DEFAULT_MODEL],
    alias: modelMetadata[OPENAI_DEFAULT_MODEL]?.alias ?? "GPT",
  };

  return {
    ...cfg,
    models: {
      ...cfg.models,
      metadata: modelMetadata,
    },
  };
}

export function applyOpenAIConfig(cfg: MarvConfig): MarvConfig {
  const next = applyOpenAIProviderConfig(cfg);
  return {
    ...next,
    agents: {
      ...next.agents,
      defaults: {
        ...next.agents?.defaults,
        model:
          next.agents?.defaults?.model && typeof next.agents.defaults.model === "object"
            ? {
                ...next.agents.defaults.model,
                primary: OPENAI_DEFAULT_MODEL,
              }
            : { primary: OPENAI_DEFAULT_MODEL },
      },
    },
  };
}
