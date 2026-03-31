import type { DatabaseSync } from "node:sqlite";
import { SOURCE_RUNTIME_EVENT } from "../storage/soul-memory-types.js";

export const REINFORCEMENT_LOG_WEIGHT = 0.2;
export const SOUL_MEMORY_SCOPE_HITS_TABLE = "memory_scope_hits";

/** Maximum confidence that runtime_event items can reach through reinforcement. */
const RUNTIME_EVENT_CONFIDENCE_CAP = 0.65;

const MEMORY_ITEMS_TABLE = "memory_items";

type ReinforcementConfig = { reinforcementLogWeight: number };

function normalizeScopeValue(value: string): string {
  return value.trim().toLowerCase();
}

export function computeReinforcementFactor(
  reinforcementCount: number,
  config: ReinforcementConfig,
): number {
  const normalizedCount = Math.max(1, Math.floor(reinforcementCount));
  if (config.reinforcementLogWeight <= 0) {
    return 1;
  }
  const logBoost = Math.log1p(Math.max(0, normalizedCount - 1));
  const factor = 1 + logBoost * config.reinforcementLogWeight;
  return Math.max(1, factor);
}

export function reinforceRetrievedItems(
  db: DatabaseSync,
  results: Array<{ id: string; wasRecallBoosted: boolean; source?: string }>,
  nowMs: number,
  options?: { memoryItemsTable?: string },
): void {
  if (results.length === 0) {
    return;
  }
  const memoryItemsTable = options?.memoryItemsTable ?? MEMORY_ITEMS_TABLE;
  const boosted = db.prepare(
    `UPDATE ${memoryItemsTable} SET ` +
      "last_accessed_at = ?, last_reinforced_at = ?, reinforcement_count = reinforcement_count + 1, " +
      "confidence = 1.0 WHERE id = ?",
  );
  const boostedCapped = db.prepare(
    `UPDATE ${memoryItemsTable} SET ` +
      "last_accessed_at = ?, last_reinforced_at = ?, reinforcement_count = reinforcement_count + 1, " +
      "confidence = MIN(?, confidence + 0.05) WHERE id = ?",
  );
  const reinforce = db.prepare(
    `UPDATE ${memoryItemsTable} SET ` +
      "last_accessed_at = ?, last_reinforced_at = ?, reinforcement_count = reinforcement_count + 1, " +
      "confidence = MIN(1.0, confidence + 0.05) WHERE id = ?",
  );
  const reinforceCapped = db.prepare(
    `UPDATE ${memoryItemsTable} SET ` +
      "last_accessed_at = ?, last_reinforced_at = ?, reinforcement_count = reinforcement_count + 1, " +
      "confidence = MIN(?, confidence + 0.05) WHERE id = ?",
  );
  const handled = new Set<string>();
  for (const result of results) {
    const itemId = result.id;
    if (!itemId || handled.has(itemId)) {
      continue;
    }
    handled.add(itemId);
    const isRuntimeEvent = result.source === SOURCE_RUNTIME_EVENT;
    if (result.wasRecallBoosted) {
      if (isRuntimeEvent) {
        boostedCapped.run(nowMs, nowMs, RUNTIME_EVENT_CONFIDENCE_CAP, itemId);
      } else {
        boosted.run(nowMs, nowMs, itemId);
      }
      continue;
    }
    if (isRuntimeEvent) {
      reinforceCapped.run(nowMs, nowMs, RUNTIME_EVENT_CONFIDENCE_CAP, itemId);
    } else {
      reinforce.run(nowMs, nowMs, itemId);
    }
  }
}

export function recordScopeHits(
  db: DatabaseSync,
  results: Array<{ id: string }>,
  scopeId: string,
  nowMs: number,
  options?: { scopeHitsTable?: string },
): void {
  if (results.length === 0) {
    return;
  }
  const normalizedScopeId = normalizeScopeValue(scopeId);
  if (!normalizedScopeId) {
    return;
  }
  const scopeHitsTable = options?.scopeHitsTable ?? SOUL_MEMORY_SCOPE_HITS_TABLE;
  const upsert = db.prepare(
    `INSERT INTO ${scopeHitsTable} (memory_id, scope_id, hit_count, first_hit_at, last_hit_at) ` +
      "VALUES (?, ?, 1, ?, ?) " +
      "ON CONFLICT(memory_id, scope_id) DO UPDATE SET " +
      "hit_count = hit_count + 1, " +
      "last_hit_at = excluded.last_hit_at",
  );
  const seen = new Set<string>();
  for (const result of results) {
    const memoryId = result.id.trim();
    if (!memoryId || seen.has(memoryId)) {
      continue;
    }
    seen.add(memoryId);
    upsert.run(memoryId, normalizedScopeId, nowMs, nowMs);
  }
}
