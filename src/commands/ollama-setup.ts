import { resolveOllamaApiBase } from "../agents/model/models-config.providers.js";
import type { MarvConfig } from "../core/config/config.js";
import { fetchWithPrivateNetworkAccess } from "../infra/net/private-network-fetch.js";
import type { WizardPrompter } from "../wizard/prompts.js";

const OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434";
const OLLAMA_DEFAULT_CONTEXT_WINDOW = 128000;
const OLLAMA_DEFAULT_MAX_TOKENS = 8192;
const OLLAMA_DEFAULT_COST = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
};
const OLLAMA_VERIFY_TIMEOUT_MS = 5000;

type OllamaTagsResponse = {
  models?: Array<{
    name?: string;
  }>;
};

async function verifyOllamaModel(params: {
  baseUrl: string;
  apiKey?: string;
  modelId: string;
}): Promise<void> {
  const apiBase = resolveOllamaApiBase(params.baseUrl);
  const { response, release } = await fetchWithPrivateNetworkAccess({
    url: `${apiBase}/api/tags`,
    init: {
      headers: params.apiKey?.trim()
        ? {
            Authorization: `Bearer ${params.apiKey.trim()}`,
          }
        : undefined,
    },
    timeoutMs: OLLAMA_VERIFY_TIMEOUT_MS,
    auditContext: "onboard.ollama.verify",
  });
  try {
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(
        `Ollama endpoint verification failed (${response.status})${detail ? `: ${detail}` : ""}`,
      );
    }
    const payload = (await response.json()) as OllamaTagsResponse;
    const availableModels = (payload.models ?? [])
      .map((entry) => String(entry.name ?? "").trim())
      .filter(Boolean);
    if (availableModels.length === 0) {
      throw new Error(`No Ollama models found at ${apiBase}.`);
    }
    if (!availableModels.includes(params.modelId)) {
      throw new Error(
        `Ollama model "${params.modelId}" not found at ${apiBase}. Available models: ${availableModels.join(", ")}`,
      );
    }
  } finally {
    await release();
  }
}

export async function promptAndConfigureOllama(params: {
  cfg: MarvConfig;
  prompter: WizardPrompter;
}): Promise<{ config: MarvConfig; modelId: string; modelRef: string }> {
  const baseUrlRaw = await params.prompter.text({
    message: "Ollama base URL",
    initialValue: OLLAMA_DEFAULT_BASE_URL,
    placeholder: OLLAMA_DEFAULT_BASE_URL,
    validate: (value) => (value?.trim() ? undefined : "Required"),
  });
  const apiKeyRaw = await params.prompter.text({
    message: "Ollama API key (leave blank if not required)",
    placeholder: "Optional",
    initialValue: "",
  });
  const modelIdRaw = await params.prompter.text({
    message: "Ollama model",
    placeholder: "qwen2.5-coder:latest",
    validate: (value) => (value?.trim() ? undefined : "Required"),
  });

  const baseUrl = resolveOllamaApiBase(String(baseUrlRaw ?? "").trim());
  const apiKey = String(apiKeyRaw ?? "").trim();
  const modelId = String(modelIdRaw ?? "").trim();
  const modelRef = `ollama/${modelId}`;

  const progress = params.prompter.progress("Verifying Ollama endpoint...");
  try {
    await verifyOllamaModel({ baseUrl, apiKey, modelId });
    progress.stop("Ollama endpoint verified.");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    progress.stop(`Ollama verification failed: ${message}`);
    throw error;
  }

  const nextConfig: MarvConfig = {
    ...params.cfg,
    models: {
      ...params.cfg.models,
      mode: params.cfg.models?.mode ?? "merge",
      providers: {
        ...params.cfg.models?.providers,
        ollama: {
          baseUrl,
          api: "ollama",
          ...(apiKey ? { apiKey } : {}),
          models: [
            {
              id: modelId,
              name: modelId,
              reasoning: false,
              input: ["text"],
              cost: OLLAMA_DEFAULT_COST,
              contextWindow: OLLAMA_DEFAULT_CONTEXT_WINDOW,
              maxTokens: OLLAMA_DEFAULT_MAX_TOKENS,
            },
          ],
        },
      },
    },
  };

  return { config: nextConfig, modelId, modelRef };
}
