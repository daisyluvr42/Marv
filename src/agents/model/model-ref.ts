/**
 * Low-level model reference parsing utilities.
 * Extracted to break circular dependencies in the model module graph.
 */

export function normalizeGoogleModelId(id: string): string {
  if (id === "gemini-3-pro" || id === "gemini-3-pro-preview") {
    // gemini-3-pro-preview was shut down 2026-03-09; map to 3.1
    return "gemini-3.1-pro-preview";
  }
  if (id === "gemini-3-flash") {
    return "gemini-3-flash-preview";
  }
  return id;
}

export type ModelRef = {
  provider: string;
  model: string;
};

const ANTHROPIC_MODEL_ALIASES: Record<string, string> = {
  "opus-4.6": "claude-opus-4-6",
  "opus-4.5": "claude-opus-4-5",
  "sonnet-4.6": "claude-sonnet-4-6",
  "sonnet-4.5": "claude-sonnet-4-5",
};

export function normalizeProviderId(provider: string): string {
  const normalized = provider.trim().toLowerCase();
  if (normalized === "z.ai" || normalized === "z-ai") {
    return "zai";
  }
  if (normalized === "opencode-zen") {
    return "opencode";
  }
  if (normalized === "qwen") {
    return "qwen-portal";
  }
  if (normalized === "kimi-code") {
    return "kimi-coding";
  }
  return normalized;
}

function normalizeAnthropicModelId(model: string): string {
  const trimmed = model.trim();
  if (!trimmed) {
    return trimmed;
  }
  const lower = trimmed.toLowerCase();
  return ANTHROPIC_MODEL_ALIASES[lower] ?? trimmed;
}

function normalizeProviderModelId(provider: string, model: string): string {
  if (provider === "anthropic") {
    return normalizeAnthropicModelId(model);
  }
  if (provider === "google") {
    return normalizeGoogleModelId(model);
  }
  return model;
}

export function normalizeModelRef(provider: string, model: string): ModelRef {
  const normalizedProvider = normalizeProviderId(provider);
  const normalizedModel = normalizeProviderModelId(normalizedProvider, model.trim());
  return { provider: normalizedProvider, model: normalizedModel };
}

export function parseModelRef(raw: string, defaultProvider: string): ModelRef | null {
  const trimmed = raw.trim();
  if (!trimmed) {
    return null;
  }
  const slash = trimmed.indexOf("/");
  if (slash === -1) {
    return normalizeModelRef(defaultProvider, trimmed);
  }
  const providerRaw = trimmed.slice(0, slash).trim();
  const model = trimmed.slice(slash + 1).trim();
  if (!providerRaw || !model) {
    return null;
  }
  return normalizeModelRef(providerRaw, model);
}
