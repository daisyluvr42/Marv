import { OPENCODE_ZEN_DEFAULT_MODEL_REF } from "../agents/model/opencode-zen-models.js";
import type { MarvConfig } from "../core/config/config.js";
import { applyAgentDefaultModelPrimary } from "./onboard-auth.config-shared.js";

export function applyOpencodeZenProviderConfig(cfg: MarvConfig): MarvConfig {
  const modelMetadata = { ...cfg.models?.metadata };
  modelMetadata[OPENCODE_ZEN_DEFAULT_MODEL_REF] = {
    ...modelMetadata[OPENCODE_ZEN_DEFAULT_MODEL_REF],
    alias: modelMetadata[OPENCODE_ZEN_DEFAULT_MODEL_REF]?.alias ?? "Opus",
  };

  return {
    ...cfg,
    models: {
      ...cfg.models,
      metadata: modelMetadata,
    },
  };
}

export function applyOpencodeZenConfig(cfg: MarvConfig): MarvConfig {
  const next = applyOpencodeZenProviderConfig(cfg);
  return applyAgentDefaultModelPrimary(next, OPENCODE_ZEN_DEFAULT_MODEL_REF);
}
