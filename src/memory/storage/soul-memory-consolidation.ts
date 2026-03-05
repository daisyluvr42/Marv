import fsSync from "node:fs";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { resolveSoulMemoryDbPath, writeSoulMemory } from "./soul-memory-store.js";
import { requireNodeSqlite } from "./sqlite.js";

type MemoryRow = {
  id: string;
  scope_type: string;
  scope_id: string;
  kind: string;
  content: string;
};

type ConsolidationItem = {
  id: string;
  scopeType: string;
  scopeId: string;
  kind: string;
  content: string;
};

export type SoulMemoryConsolidationResult = {
  generalizedCount: number;
  consolidatedIds: string[];
};

export function consolidateSoulMemories(params: {
  agentId: string;
  minSimilarity?: number;
  maxSimilarity?: number;
  minClusterSize?: number;
  duplicateThreshold?: number;
  maxItems?: number;
  summarizeCluster?: (input: {
    kind: string;
    scopeType: string;
    scopeId: string;
    items: ConsolidationItem[];
  }) => string;
  nowMs?: number;
}): SoulMemoryConsolidationResult {
  const minSimilarity = clamp(params.minSimilarity ?? 0.6, 0.1, 1);
  const maxSimilarity = clamp(params.maxSimilarity ?? 0.9, minSimilarity, 1);
  const minClusterSize = Number.isFinite(params.minClusterSize)
    ? Math.max(3, Math.floor(params.minClusterSize as number))
    : 3;
  const duplicateThreshold = clamp(params.duplicateThreshold ?? 0.85, 0.5, 1);
  const maxItems = Number.isFinite(params.maxItems)
    ? Math.max(10, Math.floor(params.maxItems as number))
    : 2000;
  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();

  const db = openSoulMemoryDb(params.agentId);
  try {
    const items = loadMemoryItems(db, maxItems);
    if (items.length < minClusterSize) {
      return { generalizedCount: 0, consolidatedIds: [] };
    }

    const grouped = groupByScopeAndKind(items);
    const existingTexts = items.map((item) => item.content);
    const consolidatedIds: string[] = [];

    for (const group of grouped.values()) {
      if (group.length < minClusterSize) {
        continue;
      }
      const clusters = buildSimilarityClusters(group, { minSimilarity, maxSimilarity });
      for (const cluster of clusters) {
        if (cluster.length < minClusterSize) {
          continue;
        }
        const head = cluster[0];
        if (!head) {
          continue;
        }
        const summary = normalizeText(
          params.summarizeCluster
            ? params.summarizeCluster({
                kind: head.kind,
                scopeType: head.scopeType,
                scopeId: head.scopeId,
                items: cluster,
              })
            : buildHeuristicConsolidation(cluster),
        );
        if (!summary) {
          continue;
        }
        if (existingTexts.some((text) => semanticSimilarity(summary, text) >= duplicateThreshold)) {
          continue;
        }
        const inserted = writeSoulMemory({
          agentId: params.agentId,
          scopeType: head.scopeType,
          scopeId: head.scopeId,
          kind: head.kind,
          content: summary,
          confidence: 0.52,
          source: "auto_extraction",
          nowMs,
        });
        if (!inserted) {
          continue;
        }
        consolidatedIds.push(inserted.id);
        existingTexts.push(summary);
      }
    }

    return {
      generalizedCount: consolidatedIds.length,
      consolidatedIds,
    };
  } finally {
    db.close();
  }
}

function buildSimilarityClusters(
  items: ConsolidationItem[],
  params: { minSimilarity: number; maxSimilarity: number },
): ConsolidationItem[][] {
  const n = items.length;
  const neighbors = Array.from({ length: n }, () => new Set<number>());
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const left = items[i];
      const right = items[j];
      if (!left || !right) {
        continue;
      }
      const sim = semanticSimilarity(left.content, right.content);
      if (sim < params.minSimilarity || sim > params.maxSimilarity) {
        continue;
      }
      neighbors[i]?.add(j);
      neighbors[j]?.add(i);
    }
  }

  const visited = new Set<number>();
  const clusters: ConsolidationItem[][] = [];
  for (let i = 0; i < n; i += 1) {
    if (visited.has(i)) {
      continue;
    }
    const queue = [i];
    visited.add(i);
    const cluster: ConsolidationItem[] = [];
    while (queue.length > 0) {
      const idx = queue.shift();
      if (idx == null) {
        continue;
      }
      const item = items[idx];
      if (item) {
        cluster.push(item);
      }
      for (const neighbor of neighbors[idx] ?? []) {
        if (visited.has(neighbor)) {
          continue;
        }
        visited.add(neighbor);
        queue.push(neighbor);
      }
    }
    if (cluster.length > 0) {
      clusters.push(cluster);
    }
  }
  return clusters;
}

function buildHeuristicConsolidation(items: ConsolidationItem[]): string {
  const kind = items[0]?.kind || "knowledge";
  const snippets = items
    .slice(0, 3)
    .map((item) => compactLine(item.content))
    .filter(Boolean)
    .map((line) => `"${line}"`);
  if (snippets.length === 0) {
    return "";
  }
  return `Generalized ${kind} pattern observed across multiple cases: ${snippets.join("; ")}.`;
}

function compactLine(value: string): string {
  const normalized = normalizeText(value);
  if (!normalized) {
    return "";
  }
  if (normalized.length <= 120) {
    return normalized;
  }
  return `${normalized.slice(0, 117)}...`;
}

function groupByScopeAndKind(items: ConsolidationItem[]): Map<string, ConsolidationItem[]> {
  const grouped = new Map<string, ConsolidationItem[]>();
  for (const item of items) {
    const key = `${item.scopeType}::${item.scopeId}::${item.kind}`;
    const bucket = grouped.get(key) ?? [];
    bucket.push(item);
    grouped.set(key, bucket);
  }
  return grouped;
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
  return Math.max(jaccard, containment * 0.9);
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

function loadMemoryItems(db: DatabaseSync, limit: number): ConsolidationItem[] {
  const rows = db
    .prepare(
      "SELECT id, scope_type, scope_id, kind, content " +
        "FROM memory_items ORDER BY created_at DESC LIMIT ?",
    )
    .all(limit) as MemoryRow[];
  return rows.map((row) => ({
    id: String(row.id),
    scopeType: String(row.scope_type),
    scopeId: String(row.scope_id),
    kind: String(row.kind),
    content: String(row.content),
  }));
}

function openSoulMemoryDb(agentId: string): DatabaseSync {
  const dbPath = resolveSoulMemoryDbPath(agentId);
  fsSync.mkdirSync(path.dirname(dbPath), { recursive: true });
  const { DatabaseSync } = requireNodeSqlite();
  return new DatabaseSync(dbPath);
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, value));
}
