import fsSync from "node:fs";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { resolveSoulMemoryDbPath } from "./soul-memory-store.js";
import { requireNodeSqlite } from "./sqlite.js";

const SOUL_MEMORY_VEC_TABLE = "memory_items_vec";

type MemoryRow = {
  id: string;
  scope_type: string;
  scope_id: string;
  kind: string;
  content: string;
  confidence: number;
  tier: string;
  reinforcement_count: number;
  created_at: number;
};

type MutableMemoryItem = {
  id: string;
  scopeType: string;
  scopeId: string;
  kind: string;
  content: string;
  confidence: number;
  tier: string;
  reinforcementCount: number;
  createdAt: number;
};

export type SoulMemoryDedupeResult = {
  mergedPairs: number;
  removedIds: string[];
};

export function dedupeSoulMemories(params: {
  agentId: string;
  similarityThreshold?: number;
  maxItems?: number;
  nowMs?: number;
  compactionEnabled?: boolean;
}): SoulMemoryDedupeResult {
  const threshold = clamp(params.similarityThreshold ?? 0.9, 0.5, 1);
  const maxItems = Number.isFinite(params.maxItems)
    ? Math.max(10, Math.floor(params.maxItems as number))
    : 4000;
  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  const db = openSoulMemoryDb(params.agentId);
  try {
    const items = loadMemoryItems(db, maxItems, params.compactionEnabled ?? false);
    if (items.length < 2) {
      return { mergedPairs: 0, removedIds: [] };
    }

    const itemsById = new Map(items.map((item) => [item.id, item]));
    const removed = new Set<string>();
    const merges: Array<{ keeperId: string; duplicateId: string }> = [];

    const groups = groupByScopeAndKind(items);
    for (const group of groups.values()) {
      for (let i = 0; i < group.length; i += 1) {
        const left = group[i];
        if (!left || removed.has(left.id)) {
          continue;
        }
        for (let j = i + 1; j < group.length; j += 1) {
          const right = group[j];
          if (!right || removed.has(right.id)) {
            continue;
          }
          const similarity = semanticSimilarity(left.content, right.content);
          if (similarity < threshold) {
            continue;
          }
          const keeper = pickKeeper(left, right);
          const duplicate = keeper.id === left.id ? right : left;
          const currentKeeper = itemsById.get(keeper.id);
          const currentDuplicate = itemsById.get(duplicate.id);
          if (!currentKeeper || !currentDuplicate || removed.has(currentDuplicate.id)) {
            continue;
          }

          currentKeeper.reinforcementCount += Math.max(1, currentDuplicate.reinforcementCount);
          currentKeeper.confidence = Math.max(
            currentKeeper.confidence,
            currentDuplicate.confidence,
          );
          currentKeeper.createdAt = Math.min(currentKeeper.createdAt, currentDuplicate.createdAt);

          removed.add(currentDuplicate.id);
          merges.push({ keeperId: currentKeeper.id, duplicateId: currentDuplicate.id });
        }
      }
    }

    if (merges.length === 0) {
      return { mergedPairs: 0, removedIds: [] };
    }

    db.exec("BEGIN IMMEDIATE TRANSACTION");
    try {
      const seenKeepers = new Set<string>();
      const updateStmt = db.prepare(
        "UPDATE memory_items SET reinforcement_count = ?, confidence = ?, tier = ?, created_at = ?, last_reinforced_at = ? WHERE id = ?",
      );
      for (const merge of merges) {
        if (seenKeepers.has(merge.keeperId)) {
          continue;
        }
        seenKeepers.add(merge.keeperId);
        const keeper = itemsById.get(merge.keeperId);
        if (!keeper) {
          continue;
        }
        updateStmt.run(
          Math.max(1, Math.floor(keeper.reinforcementCount)),
          clamp(keeper.confidence, 0, 1),
          keeper.tier,
          Math.max(0, Math.floor(keeper.createdAt)),
          nowMs,
          keeper.id,
        );
      }

      const deleteStmt = db.prepare("DELETE FROM memory_items WHERE id = ?");
      const deleteVecStmt = prepareDeleteVectorStmt(db);
      for (const merge of merges) {
        deleteStmt.run(merge.duplicateId);
        deleteVecStmt?.run(merge.duplicateId);
      }

      db.exec("COMMIT");
    } catch (err) {
      db.exec("ROLLBACK");
      throw err;
    }

    return {
      mergedPairs: merges.length,
      removedIds: merges.map((entry) => entry.duplicateId),
    };
  } finally {
    db.close();
  }
}

function groupByScopeAndKind(items: MutableMemoryItem[]): Map<string, MutableMemoryItem[]> {
  const groups = new Map<string, MutableMemoryItem[]>();
  for (const item of items) {
    const key = `${item.scopeType}::${item.scopeId}::${item.kind}`;
    const bucket = groups.get(key) ?? [];
    bucket.push(item);
    groups.set(key, bucket);
  }
  for (const bucket of groups.values()) {
    bucket.sort((a, b) => a.createdAt - b.createdAt || a.id.localeCompare(b.id));
  }
  return groups;
}

function pickKeeper(a: MutableMemoryItem, b: MutableMemoryItem): MutableMemoryItem {
  if (a.reinforcementCount !== b.reinforcementCount) {
    return a.reinforcementCount > b.reinforcementCount ? a : b;
  }
  if (a.createdAt !== b.createdAt) {
    return a.createdAt < b.createdAt ? a : b;
  }
  return a.id.localeCompare(b.id) <= 0 ? a : b;
}

function semanticSimilarity(a: string, b: string): number {
  const left = normalizeText(a);
  const right = normalizeText(b);
  if (!left || !right) {
    return 0;
  }
  if (left === right) {
    return 1;
  }
  const leftTokens = tokenize(left);
  const rightTokens = tokenize(right);
  if (leftTokens.size === 0 || rightTokens.size === 0) {
    return 0;
  }
  let intersection = 0;
  for (const token of leftTokens) {
    if (rightTokens.has(token)) {
      intersection += 1;
    }
  }
  const union = leftTokens.size + rightTokens.size - intersection;
  const jaccard = union > 0 ? intersection / union : 0;
  const containment = Math.max(
    intersection / Math.max(1, leftTokens.size),
    intersection / Math.max(1, rightTokens.size),
  );
  return Math.max(jaccard, containment * 0.95);
}

function tokenize(value: string): Set<string> {
  const out = new Set<string>();
  const matches = value.match(/[a-z0-9_]+|[\u4e00-\u9fff]+/g) ?? [];
  for (const token of matches) {
    if (token) {
      out.add(token);
    }
  }
  return out;
}

function normalizeText(value: string): string {
  return value.toLowerCase().replace(/\s+/g, " ").trim();
}

function loadMemoryItems(
  db: DatabaseSync,
  limit: number,
  excludeEpisodic: boolean,
): MutableMemoryItem[] {
  // When compaction is enabled, exclude episodic items from dedupe
  const whereClause = excludeEpisodic
    ? " WHERE NOT ((memory_type = 'episodic' OR memory_type IS NULL)) "
    : " ";
  const rows = db
    .prepare(
      "SELECT id, scope_type, scope_id, kind, content, confidence, tier, reinforcement_count, created_at " +
        `FROM memory_items${whereClause}ORDER BY created_at ASC LIMIT ?`,
    )
    .all(limit) as MemoryRow[];
  return rows.map((row) => ({
    id: String(row.id),
    scopeType: String(row.scope_type),
    scopeId: String(row.scope_id),
    kind: String(row.kind),
    content: String(row.content),
    confidence: Number(row.confidence ?? 0),
    tier: normalizeTier(row.tier),
    reinforcementCount: Math.max(1, Math.floor(Number(row.reinforcement_count ?? 1))),
    createdAt: Math.max(0, Math.floor(Number(row.created_at ?? 0))),
  }));
}

function normalizeTier(value: string): string {
  const normalized = value.trim().toLowerCase();
  if (normalized === "palace") {
    return normalized;
  }
  return "palace";
}

function openSoulMemoryDb(agentId: string): DatabaseSync {
  const dbPath = resolveSoulMemoryDbPath(agentId);
  fsSync.mkdirSync(path.dirname(dbPath), { recursive: true });
  const { DatabaseSync } = requireNodeSqlite();
  return new DatabaseSync(dbPath);
}

function prepareDeleteVectorStmt(db: DatabaseSync): { run: (memoryId: string) => void } | null {
  try {
    db.prepare(`SELECT id FROM ${SOUL_MEMORY_VEC_TABLE} LIMIT 1`).get();
    const stmt = db.prepare(`DELETE FROM ${SOUL_MEMORY_VEC_TABLE} WHERE id = ?`);
    return {
      run: (memoryId) => {
        try {
          stmt.run(memoryId);
        } catch {
          // Ignore vector cleanup failures to keep core maintenance resilient.
        }
      },
    };
  } catch {
    return null;
  }
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, value));
}
