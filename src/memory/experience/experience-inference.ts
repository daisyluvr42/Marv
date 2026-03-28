import type { MarvConfig } from "../../core/config/config.js";
import type { InferenceResult } from "../storage/local-llm-client.js";
import {
  resolveExperienceModel,
  type ExperienceRole,
  type ResolvedExperienceModel,
} from "./experience-model-resolver.js";

export type { ExperienceRole } from "./experience-model-resolver.js";

/**
 * Run inference for an experience pipeline role using the best available model.
 *
 * Resolution order (via experience-model-resolver):
 * 1. Explicit per-role model from ExperienceConfig (distillerModel, calibrationModel, etc.)
 * 2. Legacy deepConsolidation.model config
 * 3. Runtime model pool, scored by role-specific tier/location preference
 * 4. Fallback to inferLocal() (preserves existing Ollama path)
 */
export async function experienceInfer(params: {
  cfg: MarvConfig;
  role: ExperienceRole;
  system: string;
  prompt: string;
  agentId?: string;
  agentDir?: string;
}): Promise<InferenceResult> {
  const model = await resolveExperienceModel({
    cfg: params.cfg,
    role: params.role,
    agentId: params.agentId,
    agentDir: params.agentDir,
  });

  if (!model) {
    // Last resort: delegate to inferLocal() which handles legacy deepConsolidation config
    return fallbackToInferLocal(params);
  }

  try {
    switch (model.apiType) {
      case "ollama":
        return await inferOllama(model, params.system, params.prompt);
      case "openai-compat":
        return await inferOpenAiCompat(model, params.system, params.prompt);
      case "google":
        return await inferGoogle(model, params.system, params.prompt);
      case "anthropic":
        return await inferAnthropic(model, params.system, params.prompt);
    }
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}

// ---------------------------------------------------------------------------
// Timeout & limits
// ---------------------------------------------------------------------------

const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_TOKENS = 1024;

// ---------------------------------------------------------------------------
// Fallback
// ---------------------------------------------------------------------------

async function fallbackToInferLocal(params: {
  cfg: MarvConfig;
  role: ExperienceRole;
  system: string;
  prompt: string;
}): Promise<InferenceResult> {
  try {
    const { inferLocal } = await import("../storage/local-llm-client.js");
    const modelConfig = params.cfg.memory?.soul?.deepConsolidation?.model;
    return await inferLocal({
      cfg: params.cfg,
      model: modelConfig,
      system: params.system,
      prompt: params.prompt,
    });
  } catch (err) {
    return {
      ok: false,
      error: `no model available for experience role "${params.role}": ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

// ---------------------------------------------------------------------------
// Provider-specific inference implementations
// ---------------------------------------------------------------------------

async function inferOllama(
  model: ResolvedExperienceModel,
  system: string,
  prompt: string,
): Promise<InferenceResult> {
  const baseUrl = model.baseUrl || "http://127.0.0.1:11434";
  // Ollama native API (not /v1). Strip /v1 suffix if present.
  const apiBase = baseUrl.replace(/\/v1\/?$/i, "");
  const url = joinPath(apiBase, "api/chat");

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: model.model,
      stream: false,
      messages: [
        { role: "system", content: system },
        { role: "user", content: prompt },
      ],
      options: { temperature: 0.2, num_ctx: 8192 },
    }),
    signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
  });

  if (!response.ok) {
    const detail = await readErrorText(response);
    return {
      ok: false,
      error: `ollama inference failed (HTTP ${response.status})${detail ? `: ${detail}` : ""}`,
    };
  }

  const payload = (await response.json()) as {
    message?: { content?: string; reasoning?: string };
  };
  const text = normalizeText(payload.message?.content || payload.message?.reasoning || "");
  return text ? { ok: true, text } : { ok: false, error: "ollama returned empty text" };
}

async function inferOpenAiCompat(
  model: ResolvedExperienceModel,
  system: string,
  prompt: string,
): Promise<InferenceResult> {
  const baseUrl = model.baseUrl || "https://api.openai.com/v1";
  const url = joinPath(baseUrl.replace(/\/+$/, ""), "chat/completions");

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (model.auth?.apiKey) {
    headers.Authorization = `Bearer ${model.auth.apiKey}`;
  }

  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: model.model,
      temperature: 0.2,
      max_tokens: DEFAULT_MAX_TOKENS,
      messages: [
        { role: "system", content: system },
        { role: "user", content: prompt },
      ],
    }),
    signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
  });

  if (!response.ok) {
    const detail = await readErrorText(response);
    return {
      ok: false,
      error: `openai-compat inference failed (HTTP ${response.status})${detail ? `: ${detail}` : ""}`,
    };
  }

  const payload = (await response.json()) as {
    choices?: Array<{ message?: { content?: unknown } }>;
  };
  const text = normalizeText(extractOpenAiText(payload));
  return text ? { ok: true, text } : { ok: false, error: "openai-compat returned empty text" };
}

async function inferGoogle(
  model: ResolvedExperienceModel,
  system: string,
  prompt: string,
): Promise<InferenceResult> {
  const baseUrl = model.baseUrl || "https://generativelanguage.googleapis.com/v1beta";
  const apiKey = model.auth?.apiKey;

  // API-key auth: pass key as query param. OAuth/token: use Authorization header.
  const useQueryKey = apiKey && model.auth?.mode !== "oauth" && model.auth?.mode !== "token";
  const queryParam = useQueryKey ? `?key=${encodeURIComponent(apiKey)}` : "";
  const url = joinPath(baseUrl, `models/${model.model}:generateContent`) + queryParam;

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey && (model.auth?.mode === "oauth" || model.auth?.mode === "token")) {
    headers.Authorization = `Bearer ${apiKey}`;
  }

  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: system }] },
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: {
        temperature: 0.2,
        maxOutputTokens: DEFAULT_MAX_TOKENS,
      },
    }),
    signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
  });

  if (!response.ok) {
    const detail = await readErrorText(response);
    return {
      ok: false,
      error: `google inference failed (HTTP ${response.status})${detail ? `: ${detail}` : ""}`,
    };
  }

  const payload = (await response.json()) as {
    candidates?: Array<{
      content?: { parts?: Array<{ text?: string }> };
    }>;
  };
  const text = normalizeText(payload.candidates?.[0]?.content?.parts?.[0]?.text ?? "");
  return text ? { ok: true, text } : { ok: false, error: "google returned empty text" };
}

async function inferAnthropic(
  model: ResolvedExperienceModel,
  system: string,
  prompt: string,
): Promise<InferenceResult> {
  const baseUrl = model.baseUrl || "https://api.anthropic.com/v1";
  const url = joinPath(baseUrl.replace(/\/+$/, ""), "messages");

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",
  };
  if (model.auth?.apiKey) {
    if (model.auth.mode === "oauth" || model.auth.mode === "token") {
      headers.Authorization = `Bearer ${model.auth.apiKey}`;
    } else {
      headers["x-api-key"] = model.auth.apiKey;
    }
  }

  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: model.model,
      max_tokens: DEFAULT_MAX_TOKENS,
      system,
      messages: [{ role: "user", content: prompt }],
    }),
    signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
  });

  if (!response.ok) {
    const detail = await readErrorText(response);
    return {
      ok: false,
      error: `anthropic inference failed (HTTP ${response.status})${detail ? `: ${detail}` : ""}`,
    };
  }

  const payload = (await response.json()) as {
    content?: Array<{ type?: string; text?: string }>;
  };
  const text = normalizeText(payload.content?.find((block) => block.type === "text")?.text ?? "");
  return text ? { ok: true, text } : { ok: false, error: "anthropic returned empty text" };
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function extractOpenAiText(payload: {
  choices?: Array<{ message?: { content?: unknown } }>;
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

function normalizeText(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function joinPath(base: string, segment: string): string {
  const normalizedBase = base.endsWith("/") ? base : `${base}/`;
  return new URL(segment.replace(/^\/+/, ""), normalizedBase).toString();
}

async function readErrorText(response: Response): Promise<string | undefined> {
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
