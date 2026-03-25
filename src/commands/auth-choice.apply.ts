import { refreshRuntimeModelRegistry } from "../agents/model/runtime-model-registry.js";
import { invalidateGatewayModelCatalogCache } from "../core/gateway/server-model-catalog.js";
import type { ApplyAuthChoiceParams, ApplyAuthChoiceResult } from "./auth-choice.apply-types.js";
import { applyAuthChoiceAnthropic } from "./auth-choice.apply.anthropic.js";
import { applyAuthChoiceApiProviders } from "./auth-choice.apply.api-providers.js";
import { applyAuthChoiceCopilotProxy } from "./auth-choice.apply.copilot-proxy.js";
import { applyAuthChoiceCustomApi } from "./auth-choice.apply.custom-api.js";
import { applyAuthChoiceGitHubCopilot } from "./auth-choice.apply.github-copilot.js";
import { applyAuthChoiceGoogleAntigravity } from "./auth-choice.apply.google-antigravity.js";
import { applyAuthChoiceGoogleGeminiCli } from "./auth-choice.apply.google-gemini-cli.js";
import { applyAuthChoiceMiniMax } from "./auth-choice.apply.minimax.js";
import { applyAuthChoiceOAuth } from "./auth-choice.apply.oauth.js";
import { applyAuthChoiceOpenAI } from "./auth-choice.apply.openai.js";
import { applyAuthChoiceQwenPortal } from "./auth-choice.apply.qwen-portal.js";
import { applyAuthChoiceVllm } from "./auth-choice.apply.vllm.js";
import { applyAuthChoiceXAI } from "./auth-choice.apply.xai.js";

export type { ApplyAuthChoiceParams, ApplyAuthChoiceResult } from "./auth-choice.apply-types.js";

export async function applyAuthChoice(
  params: ApplyAuthChoiceParams,
): Promise<ApplyAuthChoiceResult> {
  const handlers: Array<(p: ApplyAuthChoiceParams) => Promise<ApplyAuthChoiceResult | null>> = [
    applyAuthChoiceAnthropic,
    applyAuthChoiceVllm,
    applyAuthChoiceOpenAI,
    applyAuthChoiceCustomApi,
    applyAuthChoiceOAuth,
    applyAuthChoiceApiProviders,
    applyAuthChoiceMiniMax,
    applyAuthChoiceGitHubCopilot,
    applyAuthChoiceGoogleAntigravity,
    applyAuthChoiceGoogleGeminiCli,
    applyAuthChoiceCopilotProxy,
    applyAuthChoiceQwenPortal,
    applyAuthChoiceXAI,
  ];

  for (const handler of handlers) {
    const result = await handler(params);
    if (result) {
      // After auth credentials are stored, invalidate the gateway catalog cache
      // and refresh the runtime model registry so the model pool picks up all
      // available models for the newly configured provider.
      invalidateGatewayModelCatalogCache();
      try {
        await refreshRuntimeModelRegistry({
          cfg: result.config,
          agentDir: params.agentDir,
          force: true,
        });
      } catch {
        // Best-effort; registry will be refreshed on next gateway start.
      }
      return result;
    }
  }

  return { config: params.config };
}
