import { resolveOllamaApiBase } from "../../agents/model/models-config.providers.js";
import type { MarvConfig } from "../../core/config/config.js";
import type {
  DeepConsolidationModelApi,
  DeepConsolidationModelConfig,
} from "../../core/config/types.memory.js";
import type { ModelProviderConfig } from "../../core/config/types.models.js";
import { fetchWithSsrFGuard } from "../../infra/net/fetch-guard.js";
import type { SsrFPolicy } from "../../infra/net/ssrf.js";
import { normalizeSecretInput } from "../../utils/normalize-secret-input.js";

const DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434";
const DEFAULT_OPENAI_COMPAT_BASE_URL = "http://127.0.0.1:8000/v1";
const DEFAULT_MODEL_ID = "qwen2.5:3b";
const DEFAULT_TIMEOUT_MS = 30_000;
const PRIVATE_NETWORK_POLICY: SsrFPolicy = { allowPrivateNetwork: true };

export type LocalLlmApi = DeepConsolidationModelApi;
export type LocalLlmConfig = DeepConsolidationModelConfig;

export type ResolvedLocalLlmConfig = {
  api: LocalLlmApi;
  provider?: string;
  model: string;
  baseUrl: string;
  timeoutMs: number;
  headers: Record<string, string>;
};

export type InferenceResult = { ok: true; text: string } | { ok: false; error: string };

export function resolveLocalLlmConfig(params: {
  cfg: MarvConfig;
  model?: LocalLlmConfig;
}): ResolvedLocalLlmConfig {
  const model = params.model ?? {};
  const provider =
    typeof model.provider === "string" && model.provider.trim() ? model.provider.trim() : undefined;
  const providerEntry = provider ? resolveProviderConfig(params.cfg, provider) : undefined;
  const api = resolveApi({
    explicitApi: model.api,
    provider,
    providerEntry,
  });
  if (!api) {
    throw new Error("deep-consolidation model api must be ollama or openai-completions");
  }
  const rawBaseUrl = model.baseUrl?.trim() || providerEntry?.baseUrl?.trim() || "";
  const baseUrl =
    api === "ollama"
      ? resolveOllamaApiBase(rawBaseUrl || DEFAULT_OLLAMA_BASE_URL)
      : normalizeBaseUrl(rawBaseUrl || DEFAULT_OPENAI_COMPAT_BASE_URL);
  const timeoutMs =
    typeof model.timeoutMs === "number" && Number.isFinite(model.timeoutMs) && model.timeoutMs > 0
      ? Math.floor(model.timeoutMs)
      : DEFAULT_TIMEOUT_MS;
  const modelId =
    model.model?.trim() ||
    providerEntry?.models.find((entry) => entry.id.trim())?.id ||
    DEFAULT_MODEL_ID;
  const headers = buildHeaders(providerEntry);

  return {
    api,
    provider,
    model: modelId,
    baseUrl,
    timeoutMs,
    headers,
  };
}

export async function probeLocalModel(params: {
  cfg: MarvConfig;
  model?: LocalLlmConfig;
}): Promise<{ ok: true; resolved: ResolvedLocalLlmConfig } | { ok: false; reason: string }> {
  let resolved: ResolvedLocalLlmConfig;
  try {
    resolved = resolveLocalLlmConfig(params);
  } catch (err) {
    return { ok: false, reason: err instanceof Error ? err.message : String(err) };
  }

  const url =
    resolved.api === "ollama"
      ? joinApiPath(resolved.baseUrl, "api/tags")
      : joinApiPath(resolved.baseUrl, "models");
  try {
    const { response, release } = await fetchWithSsrFGuard({
      url,
      init: {
        method: "GET",
        headers: resolved.headers,
      },
      timeoutMs: resolved.timeoutMs,
      policy: PRIVATE_NETWORK_POLICY,
      auditContext: "deep-consolidation.probe",
    });
    try {
      if (!response.ok) {
        const detail = await readErrorResponse(response);
        return {
          ok: false,
          reason: `model probe failed (HTTP ${response.status})${detail ? `: ${detail}` : ""}`,
        };
      }
      return { ok: true, resolved };
    } finally {
      await release();
    }
  } catch (err) {
    return { ok: false, reason: err instanceof Error ? err.message : String(err) };
  }
}

export async function inferLocal(params: {
  cfg: MarvConfig;
  model?: LocalLlmConfig;
  system: string;
  prompt: string;
}): Promise<InferenceResult> {
  let resolved: ResolvedLocalLlmConfig;
  try {
    resolved = resolveLocalLlmConfig(params);
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }

  try {
    if (resolved.api === "ollama") {
      return await inferWithOllama(resolved, params.system, params.prompt);
    }
    return await inferWithOpenAiCompatible(resolved, params.system, params.prompt);
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}

function resolveProviderConfig(cfg: MarvConfig, provider: string): ModelProviderConfig | undefined {
  const providers = cfg.models?.providers ?? {};
  if (providers[provider]) {
    return providers[provider];
  }
  const normalized = provider.trim().toLowerCase();
  return Object.entries(providers).find(([key]) => key.trim().toLowerCase() === normalized)?.[1];
}

function resolveApi(params: {
  explicitApi?: string;
  provider?: string;
  providerEntry?: ModelProviderConfig;
}): LocalLlmApi | null {
  if (params.explicitApi === "ollama" || params.explicitApi === "openai-completions") {
    return params.explicitApi;
  }
  if (
    params.providerEntry?.api === "ollama" ||
    params.providerEntry?.api === "openai-completions"
  ) {
    return params.providerEntry.api;
  }
  if (params.providerEntry) {
    return null;
  }
  if (params.provider?.trim().toLowerCase() === "ollama") {
    return "ollama";
  }
  return "openai-completions";
}

function buildHeaders(providerEntry?: ModelProviderConfig): Record<string, string> {
  const headers: Record<string, string> = { ...providerEntry?.headers };
  const apiKey = resolveApiKey(providerEntry?.apiKey);
  const hasAuthorizationHeader = Object.keys(headers).some(
    (key) => key.trim().toLowerCase() === "authorization",
  );
  if (apiKey && providerEntry?.authHeader !== false && !hasAuthorizationHeader) {
    headers.Authorization = `Bearer ${apiKey}`;
  }
  return headers;
}

function resolveApiKey(raw: unknown): string | undefined {
  const normalized = normalizeSecretInput(raw);
  if (!normalized) {
    return undefined;
  }
  if (/^[A-Z][A-Z0-9_]*$/.test(normalized)) {
    const envValue = normalizeSecretInput(process.env[normalized]);
    return envValue || normalized;
  }
  return normalized;
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}

function joinApiPath(baseUrl: string, path: string): string {
  const normalizedBase = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  return new URL(path.replace(/^\/+/, ""), normalizedBase).toString();
}

async function inferWithOllama(
  resolved: ResolvedLocalLlmConfig,
  system: string,
  prompt: string,
): Promise<InferenceResult> {
  const url = joinApiPath(resolved.baseUrl, "api/chat");
  const { response, release } = await fetchWithSsrFGuard({
    url,
    init: {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...resolved.headers,
      },
      body: JSON.stringify({
        model: resolved.model,
        stream: false,
        messages: [
          { role: "system", content: system },
          { role: "user", content: prompt },
        ],
        options: {
          temperature: 0.2,
          num_ctx: 8192,
        },
      }),
    },
    timeoutMs: resolved.timeoutMs,
    policy: PRIVATE_NETWORK_POLICY,
    auditContext: "deep-consolidation.ollama",
  });
  try {
    if (!response.ok) {
      const detail = await readErrorResponse(response);
      return {
        ok: false,
        error: `ollama inference failed (HTTP ${response.status})${detail ? `: ${detail}` : ""}`,
      };
    }
    const payload = (await response.json()) as {
      message?: {
        content?: string;
        reasoning?: string;
      };
    };
    const text = normalizeModelText(payload.message?.content || payload.message?.reasoning || "");
    return text ? { ok: true, text } : { ok: false, error: "ollama inference returned empty text" };
  } finally {
    await release();
  }
}

async function inferWithOpenAiCompatible(
  resolved: ResolvedLocalLlmConfig,
  system: string,
  prompt: string,
): Promise<InferenceResult> {
  const url = joinApiPath(resolved.baseUrl, "chat/completions");
  const { response, release } = await fetchWithSsrFGuard({
    url,
    init: {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...resolved.headers,
      },
      body: JSON.stringify({
        model: resolved.model,
        temperature: 0.2,
        max_tokens: 256,
        messages: [
          { role: "system", content: system },
          { role: "user", content: prompt },
        ],
      }),
    },
    timeoutMs: resolved.timeoutMs,
    policy: PRIVATE_NETWORK_POLICY,
    auditContext: "deep-consolidation.openai-compatible",
  });
  try {
    if (!response.ok) {
      const detail = await readErrorResponse(response);
      return {
        ok: false,
        error: `openai-compatible inference failed (HTTP ${response.status})${detail ? `: ${detail}` : ""}`,
      };
    }
    const payload = (await response.json()) as {
      choices?: Array<{
        message?: {
          content?: unknown;
        };
      }>;
    };
    const text = normalizeModelText(extractOpenAiCompatibleText(payload));
    return text
      ? { ok: true, text }
      : { ok: false, error: "openai-compatible inference returned empty text" };
  } finally {
    await release();
  }
}

function extractOpenAiCompatibleText(payload: {
  choices?: Array<{
    message?: {
      content?: unknown;
    };
  }>;
}): string {
  const content = payload.choices?.[0]?.message?.content;
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  return content
    .map((part) => {
      if (!part || typeof part !== "object") {
        return "";
      }
      const text = (part as { text?: unknown }).text;
      return typeof text === "string" ? text : "";
    })
    .filter(Boolean)
    .join("");
}

function normalizeModelText(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

async function readErrorResponse(response: Response): Promise<string | undefined> {
  try {
    const text = (await response.text()).replace(/\s+/g, " ").trim();
    if (!text) {
      return undefined;
    }
    return text.length <= 300 ? text : `${text.slice(0, 300)}…`;
  } catch {
    return undefined;
  }
}
