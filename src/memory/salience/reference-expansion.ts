import type { DatabaseSync } from "node:sqlite";
import type { ReferenceExpansionConfig } from "./salience-types.js";

export const SOUL_MEMORY_REF_TABLE = "memory_item_refs";
export const REFERENCE_EXPANSION_ENABLED = true;
export const REFERENCE_MAX_HOPS = 2;
export const REFERENCE_EDGE_DECAY = 0.8;
export const REFERENCE_BOOST_WEIGHT = 0.3;
export const REFERENCE_MAX_BOOST = 0.6;
export const REFERENCE_SEED_TOPK_MULTIPLIER = 2;

const MEMORY_REF_TOKEN_RE = /\[ref:(mem_[a-z0-9]+)\]/gi;

export type ReferenceExpansionScoredItem = {
  id: string;
  score: number;
  referenceBoost: number;
};

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, value));
}

function extractMemoryReferenceIds(value: string): string[] {
  const refs = new Set<string>();
  const regex = new RegExp(MEMORY_REF_TOKEN_RE.source, MEMORY_REF_TOKEN_RE.flags);
  for (const match of value.matchAll(regex)) {
    const itemId = String(match[1] ?? "")
      .trim()
      .toLowerCase();
    if (/^mem_[a-z0-9]+$/.test(itemId)) {
      refs.add(itemId);
    }
  }
  return [...refs];
}

export function upsertItemReferences(
  db: DatabaseSync,
  sourceMemoryId: string,
  content: string,
  nowMs: number,
  options?: { refTable?: string },
): void {
  const refTable = options?.refTable ?? SOUL_MEMORY_REF_TABLE;
  db.prepare(`DELETE FROM ${refTable} WHERE source_memory_id = ?`).run(sourceMemoryId);
  const refs = extractMemoryReferenceIds(content).filter((targetId) => targetId !== sourceMemoryId);
  if (refs.length === 0) {
    return;
  }
  const stmt = db.prepare(
    `INSERT OR IGNORE INTO ${refTable} (source_memory_id, target_memory_id, created_at) ` +
      "VALUES (?, ?, ?)",
  );
  for (const targetId of refs) {
    stmt.run(sourceMemoryId, targetId, nowMs);
  }
}

export function loadReferencesBySourceIds(
  db: DatabaseSync,
  sourceIds: string[],
  options?: { refTable?: string },
): Map<string, string[]> {
  const out = new Map<string, string[]>();
  if (sourceIds.length === 0) {
    return out;
  }
  const refTable = options?.refTable ?? SOUL_MEMORY_REF_TABLE;
  const placeholders = sourceIds.map(() => "?").join(", ");
  type Row = { source_memory_id?: string; target_memory_id?: string };
  const rows = db
    .prepare(
      `SELECT source_memory_id, target_memory_id FROM ${refTable} ` +
        `WHERE source_memory_id IN (${placeholders}) ` +
        "ORDER BY source_memory_id ASC, target_memory_id ASC",
    )
    .all(...sourceIds) as Row[];
  for (const row of rows) {
    const sourceId = String(row.source_memory_id ?? "").trim();
    const targetId = String(row.target_memory_id ?? "").trim();
    if (!sourceId || !targetId) {
      continue;
    }
    const bucket = out.get(sourceId);
    if (!bucket) {
      out.set(sourceId, [targetId]);
      continue;
    }
    if (!bucket.includes(targetId)) {
      bucket.push(targetId);
    }
  }
  return out;
}

export function applyReferenceExpansion(params: {
  db: DatabaseSync;
  scoredById: Map<string, ReferenceExpansionScoredItem>;
  topK: number;
  soulConfig: ReferenceExpansionConfig;
  refTable?: string;
}): void {
  if (
    !params.soulConfig.referenceExpansionEnabled ||
    params.soulConfig.referenceMaxHops <= 0 ||
    params.soulConfig.referenceBoostWeight <= 0 ||
    params.soulConfig.referenceEdgeDecay <= 0 ||
    params.scoredById.size === 0
  ) {
    return;
  }

  const sorted = [...params.scoredById.values()].toSorted((a, b) => b.score - a.score);
  const seedLimit = Math.max(
    params.topK,
    Math.min(
      sorted.length,
      Math.floor(params.topK * params.soulConfig.referenceSeedTopKMultiplier),
    ),
  );
  const seeds = sorted.slice(0, seedLimit).filter((entry) => entry.score > 0);
  if (seeds.length === 0) {
    return;
  }

  let frontier = new Map<string, number>();
  for (const seed of seeds) {
    const existing = frontier.get(seed.id) ?? 0;
    if (seed.score > existing) {
      frontier.set(seed.id, seed.score);
    }
  }

  const boostById = new Map<string, number>();
  for (let hop = 1; hop <= params.soulConfig.referenceMaxHops && frontier.size > 0; hop += 1) {
    const refsBySource = loadReferencesBySourceIds(params.db, [...frontier.keys()], {
      refTable: params.refTable,
    });
    const nextFrontier = new Map<string, number>();
    for (const [sourceId, sourceInfluence] of frontier) {
      const targets = refsBySource.get(sourceId) ?? [];
      for (const targetId of targets) {
        if (!targetId || targetId === sourceId) {
          continue;
        }
        const propagated = sourceInfluence * params.soulConfig.referenceEdgeDecay;
        if (propagated <= 0) {
          continue;
        }
        const boostDelta = propagated * params.soulConfig.referenceBoostWeight;
        if (boostDelta > 0 && params.scoredById.has(targetId)) {
          const previousBoost = boostById.get(targetId) ?? 0;
          boostById.set(
            targetId,
            Math.min(params.soulConfig.referenceMaxBoost, previousBoost + boostDelta),
          );
        }
        const previousFrontier = nextFrontier.get(targetId) ?? 0;
        if (propagated > previousFrontier) {
          nextFrontier.set(targetId, propagated);
        }
      }
    }
    frontier = nextFrontier;
  }

  for (const [itemId, boost] of boostById) {
    if (boost <= 0) {
      continue;
    }
    const result = params.scoredById.get(itemId);
    if (!result) {
      continue;
    }
    result.referenceBoost = boost;
    result.score = clamp(result.score + boost, 0, 1.5);
  }
}
