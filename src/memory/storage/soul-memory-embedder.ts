import type { MarvConfig } from "../../core/config/config.js";
import type { EmbeddingProvider } from "../embeddings/embeddings-types.js";
import { EMBEDDING_DIMS } from "./soul-memory-types.js";

/**
 * Soul Memory ML embedder interface.
 * Wraps an EmbeddingProvider from src/memory/embeddings and caches per agent.
 */
export type SoulEmbedder = {
  embed: (text: string) => Promise<number[]>;
  embedBatch: (texts: string[]) => Promise<number[][]>;
  dimensions: number;
  providerId: string;
};

// Cache per agentId to avoid re-creating providers
const embedderCache = new Map<string, SoulEmbedder | null>();

/**
 * Resolve the best available ML embedding provider for Soul Memory.
 *
 * Reads `agents.defaults.memorySearch` config to determine provider settings,
 * then delegates to `createEmbeddingProvider()` from the embeddings module.
 *
 * Returns null when no ML embedding provider is available (hash fallback will be used).
 */
export async function getSoulEmbedder(
  cfg: MarvConfig,
  agentId: string,
): Promise<SoulEmbedder | null> {
  const cacheKey = agentId;
  if (embedderCache.has(cacheKey)) {
    return embedderCache.get(cacheKey) ?? null;
  }

  try {
    const { createEmbeddingProvider } = await import("../embeddings/embeddings.js");
    const { resolveMemorySearchConfig } = await import("../../agents/memory-search.js");

    const searchConfig = resolveMemorySearchConfig(cfg, agentId);
    if (!searchConfig) {
      embedderCache.set(cacheKey, null);
      return null;
    }

    const result = await createEmbeddingProvider({
      config: cfg,
      provider: searchConfig.provider,
      model: searchConfig.model,
      dimensions: searchConfig.dimensions,
      fallback: searchConfig.fallback,
      local: searchConfig.local,
      remote: searchConfig.remote,
    });

    if (!result.provider) {
      embedderCache.set(cacheKey, null);
      return null;
    }

    const provider: EmbeddingProvider = result.provider;

    // Detect dimensions: try a probe embed to get actual vector size
    const probeDims = provider.dimensions;
    let dimensions = probeDims ?? 0;
    if (!dimensions) {
      try {
        const probeVec = await provider.embedQuery("test");
        dimensions = probeVec.length;
      } catch {
        // Probe failed — cannot determine dimensions
        embedderCache.set(cacheKey, null);
        return null;
      }
    }

    if (dimensions <= 0 || dimensions === EMBEDDING_DIMS) {
      // Same as legacy hash dimensions or invalid — not useful
      embedderCache.set(cacheKey, null);
      return null;
    }

    const embedder: SoulEmbedder = {
      embed: (text: string) => provider.embedQuery(text),
      embedBatch: (texts: string[]) => provider.embedBatch(texts),
      dimensions,
      providerId: provider.id,
    };

    embedderCache.set(cacheKey, embedder);
    return embedder;
  } catch {
    embedderCache.set(cacheKey, null);
    return null;
  }
}

/** Clear the embedder cache (useful for testing or config changes). */
export function clearSoulEmbedderCache(): void {
  embedderCache.clear();
}
