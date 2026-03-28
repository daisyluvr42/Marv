import type { MarvConfig } from "../../core/config/config.js";
import type { SoulEmbedder } from "./soul-memory-embedder.js";
import { resolveSoulMemoryDbPath } from "./soul-memory-path.js";
import {
  getStoredEmbeddingDims,
  migrateSoulVecDimensions,
  openSoulMemoryDb,
  setStoredEmbeddingProvider,
} from "./soul-memory-schema.js";

/**
 * Resolve the ML query embedding for a Soul Memory search.
 * Returns the ML embedding vector, or null if no ML embedder is available.
 *
 * Also ensures the vec table dimensions match the embedder (migrates if needed).
 */
export async function resolveMlQueryEmbedding(params: {
  cfg: MarvConfig;
  agentId: string;
  query: string;
}): Promise<number[] | null> {
  try {
    const { getSoulEmbedder } = await import("./soul-memory-embedder.js");
    const embedder = await getSoulEmbedder(params.cfg, params.agentId);
    if (!embedder) {
      return null;
    }
    // Ensure vec table dimensions match
    ensureVecDimensionsForEmbedder(params.agentId, embedder);
    return await embedder.embed(params.query);
  } catch {
    return null;
  }
}

/**
 * Re-embed a batch of legacy items with the ML embedder.
 * Returns the number of items re-embedded.
 */
export async function reembedLegacyBatch(params: {
  cfg: MarvConfig;
  agentId: string;
  batchSize?: number;
}): Promise<number> {
  const { getSoulEmbedder } = await import("./soul-memory-embedder.js");
  const embedder = await getSoulEmbedder(params.cfg, params.agentId);
  if (!embedder) {
    return 0;
  }
  // Ensure vec table matches embedder dims
  ensureVecDimensionsForEmbedder(params.agentId, embedder);

  const { loadLegacyEmbeddingItems, reembedMemoryItem } = await import("./soul-memory-crud.js");
  const batchSize = params.batchSize ?? 50;
  const items = loadLegacyEmbeddingItems(params.agentId, batchSize);
  if (items.length === 0) {
    return 0;
  }

  // Batch embed all texts at once
  const texts = items.map((item) => item.content);
  const embeddings = await embedder.embedBatch(texts);

  const db = openSoulMemoryDb(params.agentId);
  try {
    let count = 0;
    for (let i = 0; i < items.length; i += 1) {
      const item = items[i];
      const embedding = embeddings[i];
      if (!item || !embedding || embedding.length === 0) {
        continue;
      }
      reembedMemoryItem(db, item.id, embedding);
      count += 1;
    }
    return count;
  } finally {
    db.close();
  }
}

/**
 * Check if the vec table dimensions match the embedder; migrate if not.
 * Safe to call repeatedly — skips if dims already match.
 */
function ensureVecDimensionsForEmbedder(agentId: string, embedder: SoulEmbedder): void {
  const db = openSoulMemoryDb(agentId);
  try {
    const currentDims = getStoredEmbeddingDims(db);
    if (currentDims === embedder.dimensions) {
      return;
    }
    const dbPath = resolveSoulMemoryDbPath(agentId);
    migrateSoulVecDimensions(db, dbPath, embedder.dimensions);
    setStoredEmbeddingProvider(db, embedder.providerId);
  } finally {
    db.close();
  }
}
