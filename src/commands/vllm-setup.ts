import { upsertAuthProfileWithLock } from "../agents/auth-profiles.js";
import type { MarvConfig } from "../core/config/config.js";
import { fetchWithPrivateNetworkAccess } from "../infra/net/private-network-fetch.js";
import type { WizardPrompter } from "../wizard/prompts.js";

export const VLLM_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1";
export const VLLM_DEFAULT_CONTEXT_WINDOW = 128000;
export const VLLM_DEFAULT_MAX_TOKENS = 8192;
export const VLLM_DEFAULT_COST = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
};
const VLLM_VERIFY_TIMEOUT_MS = 5000;

type VllmModelsResponse = {
  data?: Array<{
    id?: string;
  }>;
};

async function verifyVllmModel(params: {
  baseUrl: string;
  apiKey: string;
  modelId: string;
}): Promise<void> {
  const url = `${params.baseUrl}/models`;
  const { response, release } = await fetchWithPrivateNetworkAccess({
    url,
    init: {
      headers: {
        Authorization: `Bearer ${params.apiKey}`,
      },
    },
    timeoutMs: VLLM_VERIFY_TIMEOUT_MS,
    auditContext: "onboard.vllm.verify",
  });
  try {
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(
        `vLLM endpoint verification failed (${response.status})${detail ? `: ${detail}` : ""}`,
      );
    }
    const payload = (await response.json()) as VllmModelsResponse;
    const availableModels = (payload.data ?? [])
      .map((entry) => String(entry.id ?? "").trim())
      .filter(Boolean);
    if (availableModels.length === 0) {
      throw new Error(`No models found at ${params.baseUrl}.`);
    }
    if (!availableModels.includes(params.modelId)) {
      throw new Error(
        `vLLM model "${params.modelId}" not found at ${params.baseUrl}. Available models: ${availableModels.join(", ")}`,
      );
    }
  } finally {
    await release();
  }
}

export async function promptAndConfigureVllm(params: {
  cfg: MarvConfig;
  prompter: WizardPrompter;
  agentDir?: string;
}): Promise<{ config: MarvConfig; modelId: string; modelRef: string }> {
  const baseUrlRaw = await params.prompter.text({
    message: "vLLM base URL",
    initialValue: VLLM_DEFAULT_BASE_URL,
    placeholder: VLLM_DEFAULT_BASE_URL,
    validate: (value) => (value?.trim() ? undefined : "Required"),
  });
  const apiKeyRaw = await params.prompter.text({
    message: "vLLM API key",
    placeholder: "sk-... (or any non-empty string)",
    validate: (value) => (value?.trim() ? undefined : "Required"),
  });
  const modelIdRaw = await params.prompter.text({
    message: "vLLM model",
    placeholder: "meta-llama/Meta-Llama-3-8B-Instruct",
    validate: (value) => (value?.trim() ? undefined : "Required"),
  });

  const baseUrl = String(baseUrlRaw ?? "")
    .trim()
    .replace(/\/+$/, "");
  const apiKey = String(apiKeyRaw ?? "").trim();
  const modelId = String(modelIdRaw ?? "").trim();
  const modelRef = `vllm/${modelId}`;

  const progress = params.prompter.progress("Verifying vLLM endpoint...");
  try {
    await verifyVllmModel({ baseUrl, apiKey, modelId });
    progress.stop("vLLM endpoint verified.");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    progress.stop(`vLLM verification failed: ${message}`);
    throw error;
  }

  await upsertAuthProfileWithLock({
    profileId: "vllm:default",
    credential: { type: "api_key", provider: "vllm", key: apiKey },
    agentDir: params.agentDir,
  });

  const nextConfig: MarvConfig = {
    ...params.cfg,
    models: {
      ...params.cfg.models,
      mode: params.cfg.models?.mode ?? "merge",
      providers: {
        ...params.cfg.models?.providers,
        vllm: {
          baseUrl,
          api: "openai-completions",
          apiKey: "VLLM_API_KEY",
          models: [
            {
              id: modelId,
              name: modelId,
              reasoning: false,
              input: ["text"],
              cost: VLLM_DEFAULT_COST,
              contextWindow: VLLM_DEFAULT_CONTEXT_WINDOW,
              maxTokens: VLLM_DEFAULT_MAX_TOKENS,
            },
          ],
        },
      },
    },
  };

  return { config: nextConfig, modelId, modelRef };
}
