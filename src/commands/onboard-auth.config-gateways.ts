import {
  buildCloudflareAiGatewayModelDefinition,
  resolveCloudflareAiGatewayBaseUrl,
} from "../agents/cloudflare-ai-gateway.js";
import type { MarvConfig } from "../core/config/config.js";
import {
  applyAgentDefaultModelPrimary,
  applyProviderConfigWithDefaultModel,
} from "./onboard-auth.config-shared.js";
import {
  CLOUDFLARE_AI_GATEWAY_DEFAULT_MODEL_REF,
  VERCEL_AI_GATEWAY_DEFAULT_MODEL_REF,
} from "./onboard-auth.credentials.js";

export function applyVercelAiGatewayProviderConfig(cfg: MarvConfig): MarvConfig {
  const modelMetadata = { ...cfg.models?.metadata };
  modelMetadata[VERCEL_AI_GATEWAY_DEFAULT_MODEL_REF] = {
    ...modelMetadata[VERCEL_AI_GATEWAY_DEFAULT_MODEL_REF],
    alias: modelMetadata[VERCEL_AI_GATEWAY_DEFAULT_MODEL_REF]?.alias ?? "Vercel AI Gateway",
  };

  return {
    ...cfg,
    models: {
      ...cfg.models,
      metadata: modelMetadata,
    },
  };
}

export function applyCloudflareAiGatewayProviderConfig(
  cfg: MarvConfig,
  params?: { accountId?: string; gatewayId?: string },
): MarvConfig {
  const modelMetadata = { ...cfg.models?.metadata };
  modelMetadata[CLOUDFLARE_AI_GATEWAY_DEFAULT_MODEL_REF] = {
    ...modelMetadata[CLOUDFLARE_AI_GATEWAY_DEFAULT_MODEL_REF],
    alias: modelMetadata[CLOUDFLARE_AI_GATEWAY_DEFAULT_MODEL_REF]?.alias ?? "Cloudflare AI Gateway",
  };

  const defaultModel = buildCloudflareAiGatewayModelDefinition();
  const existingProvider = cfg.models?.providers?.["cloudflare-ai-gateway"] as
    | { baseUrl?: unknown }
    | undefined;
  const baseUrl =
    params?.accountId && params?.gatewayId
      ? resolveCloudflareAiGatewayBaseUrl({
          accountId: params.accountId,
          gatewayId: params.gatewayId,
        })
      : typeof existingProvider?.baseUrl === "string"
        ? existingProvider.baseUrl
        : undefined;

  if (!baseUrl) {
    return {
      ...cfg,
      models: {
        ...cfg.models,
        metadata: modelMetadata,
      },
    };
  }

  return applyProviderConfigWithDefaultModel(cfg, {
    modelMetadata,
    providerId: "cloudflare-ai-gateway",
    api: "anthropic-messages",
    baseUrl,
    defaultModel,
  });
}

export function applyVercelAiGatewayConfig(cfg: MarvConfig): MarvConfig {
  const next = applyVercelAiGatewayProviderConfig(cfg);
  return applyAgentDefaultModelPrimary(next, VERCEL_AI_GATEWAY_DEFAULT_MODEL_REF);
}

export function applyCloudflareAiGatewayConfig(
  cfg: MarvConfig,
  params?: { accountId?: string; gatewayId?: string },
): MarvConfig {
  const next = applyCloudflareAiGatewayProviderConfig(cfg, params);
  return applyAgentDefaultModelPrimary(next, CLOUDFLARE_AI_GATEWAY_DEFAULT_MODEL_REF);
}
