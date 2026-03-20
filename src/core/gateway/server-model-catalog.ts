import {
  loadModelCatalog,
  type ModelCatalogEntry,
  resetModelCatalogCacheForTest,
} from "../../agents/model/model-catalog.js";
import { loadConfig } from "../config/config.js";

export type GatewayModelChoice = ModelCatalogEntry;

// Catalog discovery is expensive (Pi SDK calls provider APIs to list models).
// Provider model lists rarely change, so cache aggressively (14 days).
// The model pool (what users/agents pick from) is recomputed cheaply each time.
// Auth/provider changes invalidate via invalidateGatewayModelCatalogCache().
const CATALOG_TTL_MS = 14 * 24 * 60 * 60 * 1000;
let catalogCache: { entries: GatewayModelChoice[]; loadedAt: number } | null = null;

/** Invalidate the gateway catalog cache so the next load re-discovers models. */
export function invalidateGatewayModelCatalogCache() {
  catalogCache = null;
}

// Test-only escape hatch.
export function __resetModelCatalogCacheForTest() {
  catalogCache = null;
  resetModelCatalogCacheForTest();
}

export async function loadGatewayModelCatalog(): Promise<GatewayModelChoice[]> {
  const now = Date.now();
  if (catalogCache && now - catalogCache.loadedAt < CATALOG_TTL_MS) {
    return catalogCache.entries;
  }
  const entries = await loadModelCatalog({ config: loadConfig(), useCache: false });
  if (entries.length > 0) {
    catalogCache = { entries, loadedAt: now };
  }
  return entries;
}
