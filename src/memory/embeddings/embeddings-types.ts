import type { MarvConfig } from "../../core/config/config.js";

export type EmbeddingProvider = {
  id: string;
  model: string;
  dimensions?: number;
  maxInputTokens?: number;
  embedQuery: (text: string) => Promise<number[]>;
  embedBatch: (texts: string[]) => Promise<number[][]>;
};

export type EmbeddingProviderId = "openai" | "local" | "gemini" | "voyage" | "script";
export type EmbeddingProviderRequest = EmbeddingProviderId | "auto";
export type EmbeddingProviderFallback = EmbeddingProviderId | "none";

export type EmbeddingProviderOptions = {
  config: MarvConfig;
  agentDir?: string;
  provider: EmbeddingProviderRequest;
  remote?: {
    baseUrl?: string;
    apiKey?: string;
    headers?: Record<string, string>;
  };
  model: string;
  dimensions?: number;
  fallback: EmbeddingProviderFallback;
  local?: {
    modelPath?: string;
    modelCacheDir?: string;
  };
};
