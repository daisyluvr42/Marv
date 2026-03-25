/**
 * Hardcoded baseline model catalog per provider.
 *
 * These serve as the initial seed before Runtime Registry API-fetches
 * live model lists from providers. For providers without a list API
 * (e.g. Anthropic), these baselines are the only source.
 *
 * Update periodically by checking official provider docs:
 * - Google:    https://ai.google.dev/gemini-api/docs/models
 * - OpenAI:    https://platform.openai.com/docs/models
 * - Anthropic: https://docs.anthropic.com/en/docs/about-claude/models/all-models
 *
 * Last updated: 2026-03-20
 */

import type { ModelCatalogEntry } from "./model-types.js";

type BaselineModel = Omit<ModelCatalogEntry, "provider">;

// ---------------------------------------------------------------------------
// Google Gemini
// ---------------------------------------------------------------------------
const GOOGLE_MODELS: BaselineModel[] = [
  // Gemini 3.1 (latest preview)
  {
    id: "gemini-3.1-pro-preview",
    name: "Gemini 3.1 Pro",
    contextWindow: 1_048_576,
    reasoning: true,
    input: ["text", "image"],
  },
  {
    id: "gemini-3.1-flash-lite-preview",
    name: "Gemini 3.1 Flash-Lite",
    contextWindow: 1_048_576,
    input: ["text", "image"],
  },
  // Gemini 3 (preview)
  {
    id: "gemini-3-flash-preview",
    name: "Gemini 3 Flash",
    contextWindow: 1_048_576,
    reasoning: true,
    input: ["text", "image"],
  },
  // Gemini 2.5 (stable)
  {
    id: "gemini-2.5-pro",
    name: "Gemini 2.5 Pro",
    contextWindow: 1_048_576,
    reasoning: true,
    input: ["text", "image"],
  },
  {
    id: "gemini-2.5-flash",
    name: "Gemini 2.5 Flash",
    contextWindow: 1_048_576,
    reasoning: true,
    input: ["text", "image"],
  },
  {
    id: "gemini-2.5-flash-lite",
    name: "Gemini 2.5 Flash-Lite",
    contextWindow: 1_048_576,
    input: ["text", "image"],
  },
];

// ---------------------------------------------------------------------------
// Anthropic
// ---------------------------------------------------------------------------
const ANTHROPIC_MODELS: BaselineModel[] = [
  // Current generation (4.6)
  {
    id: "claude-opus-4-6",
    name: "Claude Opus 4.6",
    contextWindow: 1_000_000,
    reasoning: true,
    input: ["text", "image"],
  },
  {
    id: "claude-sonnet-4-6",
    name: "Claude Sonnet 4.6",
    contextWindow: 1_000_000,
    reasoning: true,
    input: ["text", "image"],
  },
  // 4.5 generation
  {
    id: "claude-haiku-4-5",
    name: "Claude Haiku 4.5",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text", "image"],
  },
  {
    id: "claude-opus-4-5",
    name: "Claude Opus 4.5",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text", "image"],
  },
  {
    id: "claude-sonnet-4-5",
    name: "Claude Sonnet 4.5",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text", "image"],
  },
  // 4.1 / 4.0 (legacy, still available)
  {
    id: "claude-opus-4-1",
    name: "Claude Opus 4.1",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text", "image"],
  },
  {
    id: "claude-sonnet-4-0",
    name: "Claude Sonnet 4",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text", "image"],
  },
  {
    id: "claude-opus-4-0",
    name: "Claude Opus 4",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text", "image"],
  },
];

// ---------------------------------------------------------------------------
// OpenAI
// ---------------------------------------------------------------------------
const OPENAI_MODELS: BaselineModel[] = [
  // GPT-5.4 (latest)
  {
    id: "gpt-5.4",
    name: "GPT-5.4",
    contextWindow: 1_050_000,
    reasoning: true,
    input: ["text", "image"],
  },
  {
    id: "gpt-5.4-mini",
    name: "GPT-5.4 Mini",
    contextWindow: 400_000,
    reasoning: true,
    input: ["text", "image"],
  },
  {
    id: "gpt-5.4-nano",
    name: "GPT-5.4 Nano",
    contextWindow: 400_000,
    reasoning: true,
    input: ["text", "image"],
  },
  // GPT-5.x series
  {
    id: "gpt-5.3-codex",
    name: "GPT-5.3 Codex",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text"],
  },
  {
    id: "gpt-5.2",
    name: "GPT-5.2",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text", "image"],
  },
  {
    id: "gpt-5-mini",
    name: "GPT-5 Mini",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text", "image"],
  },
  // GPT-4.1 series
  {
    id: "gpt-4.1",
    name: "GPT-4.1",
    contextWindow: 1_000_000,
    input: ["text", "image"],
  },
  {
    id: "gpt-4.1-mini",
    name: "GPT-4.1 Mini",
    contextWindow: 1_000_000,
    input: ["text", "image"],
  },
  {
    id: "gpt-4.1-nano",
    name: "GPT-4.1 Nano",
    contextWindow: 1_000_000,
    input: ["text", "image"],
  },
  // GPT-4o (still available via API)
  {
    id: "gpt-4o",
    name: "GPT-4o",
    contextWindow: 128_000,
    input: ["text", "image"],
  },
  {
    id: "gpt-4o-mini",
    name: "GPT-4o Mini",
    contextWindow: 128_000,
    input: ["text", "image"],
  },
  // O-series reasoning
  {
    id: "o3",
    name: "o3",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text", "image"],
  },
  {
    id: "o4-mini",
    name: "o4-mini",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text", "image"],
  },
  {
    id: "o3-mini",
    name: "o3-mini",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text"],
  },
];

// ---------------------------------------------------------------------------
// OpenAI Codex (Responses API)
// ---------------------------------------------------------------------------
const OPENAI_CODEX_MODELS: BaselineModel[] = [
  {
    id: "gpt-5.4-codex",
    name: "GPT-5.4 Codex",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text"],
  },
  {
    id: "gpt-5.3-codex",
    name: "GPT-5.3 Codex",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text"],
  },
  {
    id: "gpt-5.3-codex-spark",
    name: "GPT-5.3 Codex Spark",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text"],
  },
  {
    id: "gpt-5.2-codex",
    name: "GPT-5.2 Codex",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text"],
  },
  {
    id: "gpt-5.1-codex-max",
    name: "GPT-5.1 Codex Max",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text"],
  },
  {
    id: "gpt-5.1-codex-mini",
    name: "GPT-5.1 Codex Mini",
    contextWindow: 200_000,
    reasoning: true,
    input: ["text"],
  },
];

// ---------------------------------------------------------------------------
// xAI (Grok)
// ---------------------------------------------------------------------------
const XAI_MODELS: BaselineModel[] = [
  {
    id: "grok-3",
    name: "Grok 3",
    contextWindow: 131_072,
    reasoning: true,
    input: ["text", "image"],
  },
  {
    id: "grok-3-mini",
    name: "Grok 3 Mini",
    contextWindow: 131_072,
    reasoning: true,
    input: ["text"],
  },
];

// ---------------------------------------------------------------------------
// Mistral
// ---------------------------------------------------------------------------
const MISTRAL_MODELS: BaselineModel[] = [
  {
    id: "mistral-large-latest",
    name: "Mistral Large",
    contextWindow: 128_000,
    input: ["text", "image"],
  },
  {
    id: "codestral-latest",
    name: "Codestral",
    contextWindow: 256_000,
    input: ["text"],
  },
];

// ---------------------------------------------------------------------------
// All baselines by provider
// ---------------------------------------------------------------------------
const BASELINE_PROVIDERS: Record<string, BaselineModel[]> = {
  google: GOOGLE_MODELS,
  anthropic: ANTHROPIC_MODELS,
  openai: OPENAI_MODELS,
  "openai-codex": OPENAI_CODEX_MODELS,
  xai: XAI_MODELS,
  mistral: MISTRAL_MODELS,
};

/**
 * Return baseline models for all known providers.
 * Each entry has `provider` set from the provider key.
 */
export function getBaselineModels(): ModelCatalogEntry[] {
  const models: ModelCatalogEntry[] = [];
  for (const [provider, baselines] of Object.entries(BASELINE_PROVIDERS)) {
    for (const baseline of baselines) {
      models.push({ ...baseline, provider });
    }
  }
  return models;
}

/**
 * Return baseline model IDs for a specific provider (for dedup checks).
 */
export function getBaselineModelIds(provider: string): Set<string> {
  const baselines = BASELINE_PROVIDERS[provider];
  if (!baselines) {
    return new Set();
  }
  return new Set(baselines.map((m) => m.id));
}
