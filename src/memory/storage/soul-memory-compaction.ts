import crypto from "node:crypto";
import fsSync from "node:fs";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import {
  buildSimilarityClusters,
  groupByScopeAndKind,
  type ConsolidationItem,
} from "./soul-memory-consolidation.js";
import {
  type ResolvedCompactionConfig,
  resolveSoulMemoryDbPath,
  writeSoulMemory,
} from "./soul-memory-store.js";
import { requireNodeSqlite } from "./sqlite.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type CompactionResult = {
  /** Number of new semantic index nodes created from clusters. */
  compactedClusters: number;
  /** Number of episodic items marked is_compacted = 1. */
  compactedEpisodic: number;
  /** @deprecated Archival disabled. Always 0. */
  archivedCompacted: number;
  /** @deprecated Archival disabled. Always 0. */
  archivedOrphans: number;
  /** IDs of newly created semantic memories. */
  semanticIds: string[];
  /** Number of semantic nodes evolved (old retired, new created). */
  evolvedSemantics: number;
  /** Number of evolution conflicts deferred to user (inferred evidence vs existing). */
  evolutionConflicts: number;
};

type EpisodicRow = {
  id: string;
  scope_type: string;
  scope_id: string;
  kind: string;
  content: string;
  record_kind: string;
  source: string;
  source_detail: string;
  confidence: number;
  created_at: number;
  metadata_json: string | null;
};

// ---------------------------------------------------------------------------
// Stopwords for semantic key extraction
// ---------------------------------------------------------------------------

const STOPWORDS = new Set([
  // English
  "the",
  "a",
  "an",
  "is",
  "are",
  "was",
  "were",
  "be",
  "been",
  "being",
  "have",
  "has",
  "had",
  "do",
  "does",
  "did",
  "will",
  "would",
  "could",
  "should",
  "may",
  "might",
  "shall",
  "can",
  "to",
  "of",
  "in",
  "for",
  "on",
  "with",
  "at",
  "by",
  "from",
  "as",
  "into",
  "about",
  "like",
  "through",
  "after",
  "over",
  "between",
  "out",
  "up",
  "down",
  "off",
  "then",
  "than",
  "so",
  "no",
  "not",
  "only",
  "very",
  "just",
  "also",
  "and",
  "but",
  "or",
  "if",
  "that",
  "this",
  "it",
  "its",
  "they",
  "them",
  "their",
  "we",
  "our",
  "you",
  "your",
  "he",
  "she",
  "his",
  "her",
  "i",
  "me",
  "my",
  // Common memory prefixes
  "user",
  "mentioned",
  "said",
  "told",
  "asked",
  "noted",
  "prefers",
  "likes",
  "wants",
  "uses",
  "seems",
]);

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export function compactEpisodic(params: {
  agentId: string;
  config: ResolvedCompactionConfig;
  nowMs?: number;
  summarizeCluster?: (items: ConsolidationItem[]) => string;
}): CompactionResult {
  if (!params.config.enabled) {
    return {
      compactedClusters: 0,
      compactedEpisodic: 0,
      archivedCompacted: 0,
      archivedOrphans: 0,
      semanticIds: [],
      evolvedSemantics: 0,
      evolutionConflicts: 0,
    };
  }

  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  const db = openSoulMemoryDb(params.agentId);
  try {
    const result = runCompaction(db, params.agentId, params.config, nowMs, params.summarizeCluster);
    // Archival disabled: Memory Palace is an append-only store.
    // Compaction generates semantic index nodes but never deletes episodic originals.
    return {
      ...result,
      archivedCompacted: 0,
      archivedOrphans: 0,
    };
  } finally {
    db.close();
  }
}

// ---------------------------------------------------------------------------
// Step 1: Compaction — cluster episodic items, produce semantic nodes
// ---------------------------------------------------------------------------

function runCompaction(
  db: DatabaseSync,
  agentId: string,
  config: ResolvedCompactionConfig,
  nowMs: number,
  summarizeCluster?: (items: ConsolidationItem[]) => string,
): Omit<CompactionResult, "archivedCompacted" | "archivedOrphans"> {
  // Ensure conflict table exists for evolution conflict insertion
  ensureConflictSchemaCompact(db);

  const retireSemanticStmt = db.prepare("UPDATE memory_items SET valid_until = ? WHERE id = ?");
  const insertSupersedesStmt = db.prepare(
    "INSERT OR IGNORE INTO memory_lineage (source_id, target_id, edge_type, created_at) " +
      "VALUES (?, ?, 'supersedes', ?)",
  );
  const insertConflictStmt = db.prepare(
    "INSERT OR IGNORE INTO memory_conflicts (" +
      "id, memory_id_a, memory_id_b, content_a, content_b, conflict_reason, " +
      "detected_at, resolved_at, resolution, resolved_by, resolution_strategy" +
      ") VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, 'ask_user')",
  );

  // Load uncompacted episodic items (batch-limited, oldest first for fairness)
  const rows = db
    .prepare(
      "SELECT id, scope_type, scope_id, kind, content, record_kind, source, source_detail, " +
        "confidence, created_at, metadata_json " +
        "FROM memory_items WHERE memory_type = 'episodic' AND is_compacted = 0 " +
        "ORDER BY created_at ASC LIMIT ?",
    )
    .all(config.batchLimit) as EpisodicRow[];

  if (rows.length < config.minClusterSize) {
    return {
      compactedClusters: 0,
      compactedEpisodic: 0,
      semanticIds: [],
      evolvedSemantics: 0,
      evolutionConflicts: 0,
    };
  }

  // Convert to ConsolidationItem for clustering
  const items: Array<ConsolidationItem & { _row: EpisodicRow }> = rows.map((row) => ({
    id: String(row.id),
    scopeType: String(row.scope_type),
    scopeId: String(row.scope_id),
    kind: String(row.kind),
    content: String(row.content),
    _row: row,
  }));

  // Group by scope+kind, then cluster within each group
  const grouped = groupByScopeAndKind(items);
  let compactedClusters = 0;
  let compactedEpisodic = 0;
  let evolvedSemantics = 0;
  let evolutionConflicts = 0;
  const semanticIds: string[] = [];

  const markCompactedStmt = db.prepare("UPDATE memory_items SET is_compacted = 1 WHERE id = ?");
  const insertLineageStmt = db.prepare(
    "INSERT OR IGNORE INTO memory_lineage (source_id, target_id, edge_type, created_at) " +
      "VALUES (?, ?, 'compacted_from', ?)",
  );

  for (const group of grouped.values()) {
    if (group.length < config.minClusterSize) {
      continue;
    }

    const clusters = buildSimilarityClusters(group, {
      minSimilarity: config.similarityMin,
      maxSimilarity: config.similarityMax,
    });

    for (const cluster of clusters) {
      if (cluster.length < config.minClusterSize) {
        continue;
      }

      const head = cluster[0];
      if (!head) {
        continue;
      }

      // Extract semantic key (conservative v1)
      const headRow = (head as ConsolidationItem & { _row: EpisodicRow })._row;
      const semanticKey = extractSemanticKey(
        cluster,
        headRow.record_kind,
        head.scopeType,
        head.scopeId,
      );

      // Check if active semantic with same key already exists
      let existingSemanticId: string | null = null;
      if (semanticKey) {
        const existing = db
          .prepare(
            "SELECT id, content FROM memory_items WHERE semantic_key = ? AND memory_type = 'semantic' " +
              "AND valid_until IS NULL LIMIT 1",
          )
          .get(semanticKey) as { id: string; content: string } | undefined;
        if (existing) {
          // Determine dominant source_detail in this cluster
          const dominantDetail = dominantSourceDetail(cluster, rows);
          if (dominantDetail !== "explicit") {
            // Inferred evidence: mark conflict for user confirmation, skip evolution
            insertConflictStmt.run(
              `mcf_evo_${crypto.randomUUID().replace(/-/g, "")}`,
              existing.id,
              cluster.map((c) => c.id).join(","),
              existing.content,
              cluster
                .map((c) => c.content)
                .slice(0, 3)
                .join("; "),
              "evolution conflict: new inferred evidence contradicts existing semantic",
              nowMs,
            );
            evolutionConflicts++;
            // Still mark episodic as compacted (they were consumed)
            for (const member of cluster) {
              markCompactedStmt.run(member.id);
            }
            compactedEpisodic += cluster.length;
            continue;
          }
          existingSemanticId = existing.id;
        }
      }

      // Generate summary for the semantic node
      const summary = summarizeCluster ? summarizeCluster(cluster) : buildHeuristicSummary(cluster);

      if (!summary) {
        continue;
      }

      // Write the semantic node via writeSoulMemory (all items are in the Memory Palace)
      const semanticItem = writeSoulMemory({
        agentId,
        scopeType: head.scopeType,
        scopeId: head.scopeId,
        kind: head.kind,
        content: summary,
        confidence: 0.7,
        source: "auto_extraction",
        tier: "palace",
        recordKind: normalizeRecordKind(headRow.record_kind),
        metadata: {
          compactedFrom: cluster.map((c) => c.id),
          compactionSource: "compaction",
          ...(existingSemanticId ? { supersedes: existingSemanticId } : {}),
        },
        nowMs,
      });

      if (!semanticItem) {
        continue;
      }

      // Update the semantic node with compaction-specific fields
      db.prepare(
        "UPDATE memory_items SET memory_type = 'semantic', source_detail = 'system', " +
          "valid_from = ?, semantic_key = ? WHERE id = ?",
      ).run(nowMs, semanticKey, semanticItem.id);

      // Evolution: retire old semantic and write supersedes lineage
      if (existingSemanticId) {
        retireSemanticStmt.run(nowMs, existingSemanticId);
        insertSupersedesStmt.run(semanticItem.id, existingSemanticId, nowMs);
        evolvedSemantics++;
      }

      // Mark source episodic items as compacted + write lineage
      for (const member of cluster) {
        markCompactedStmt.run(member.id);
        insertLineageStmt.run(semanticItem.id, member.id, nowMs);
      }

      compactedClusters++;
      compactedEpisodic += cluster.length;
      semanticIds.push(semanticItem.id);
    }
  }

  return {
    compactedClusters,
    compactedEpisodic,
    semanticIds,
    evolvedSemantics,
    evolutionConflicts,
  };
}

// Archival removed: Memory Palace is an append-only store. Compaction generates
// semantic index nodes but never deletes or archives original episodic items.

// ---------------------------------------------------------------------------
// Semantic key extraction (conservative v1)
// ---------------------------------------------------------------------------

const KEYABLE_RECORD_KINDS = new Set(["preference", "fact"]);

function extractSemanticKey(
  cluster: ConsolidationItem[],
  recordKind: string,
  scopeType: string,
  scopeId: string,
): string | null {
  const normalized = recordKind.trim().toLowerCase();
  if (!KEYABLE_RECORD_KINDS.has(normalized)) {
    return null;
  }

  // Tokenize all items, count frequency
  const tokenFreq = new Map<string, number>();
  for (const item of cluster) {
    const tokens = tokenize(item.content);
    for (const token of tokens) {
      if (STOPWORDS.has(token)) {
        continue;
      }
      tokenFreq.set(token, (tokenFreq.get(token) ?? 0) + 1);
    }
  }

  // Sort by frequency, take top 2
  const sorted = [...tokenFreq.entries()].toSorted((a, b) => b[1] - a[1]);
  const top2 = sorted.slice(0, 2);

  if (top2.length < 1) {
    return null;
  }

  // Check coverage: top-2 tokens must appear in >= 60% of cluster members
  const threshold = Math.ceil(cluster.length * 0.6);
  const coverageOk = top2.every(([, count]) => count >= threshold);
  if (!coverageOk) {
    return null;
  }

  const prefix = normalized === "preference" ? "pref" : "fact";
  const topTokens = top2.map(([token]) => token).join("_");
  return `${scopeType}:${scopeId}:${prefix}:${topTokens}`;
}

function tokenize(text: string): Set<string> {
  const tokens = text.toLowerCase().match(/[a-z0-9_]+|[\u4e00-\u9fff]+/g);
  return new Set(tokens ?? []);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Determine the dominant source_detail across a cluster's episodic items.
 * If any item has explicit source_detail, the cluster is considered explicit.
 */
function dominantSourceDetail(
  cluster: ConsolidationItem[],
  allRows: EpisodicRow[],
): "explicit" | "inferred" | "system" {
  const rowById = new Map(allRows.map((r) => [String(r.id), r]));
  for (const item of cluster) {
    const row = rowById.get(item.id);
    if (row && String(row.source_detail).trim().toLowerCase() === "explicit") {
      return "explicit";
    }
  }
  return "inferred";
}

function buildHeuristicSummary(items: ConsolidationItem[]): string {
  const snippets = items
    .slice(0, 3)
    .map((item) => {
      const text = item.content.trim();
      return text.length <= 100 ? text : `${text.slice(0, 97)}...`;
    })
    .filter(Boolean);
  if (snippets.length === 0) {
    return "";
  }

  const countNote = items.length > 3 ? ` (and ${items.length - 3} more)` : "";
  return `Recurring pattern across ${items.length} observations${countNote}: ${snippets.join("; ")}`;
}

function normalizeRecordKind(value: string): "fact" | "relationship" | "experience" | "soul" {
  const normalized = value.trim().toLowerCase();
  if (normalized === "fact" || normalized === "relationship" || normalized === "soul") {
    return normalized;
  }
  return "experience";
}

function openSoulMemoryDb(agentId: string): DatabaseSync {
  const dbPath = resolveSoulMemoryDbPath(agentId);
  fsSync.mkdirSync(path.dirname(dbPath), { recursive: true });
  const { DatabaseSync } = requireNodeSqlite();
  const db = new DatabaseSync(dbPath);
  db.exec("PRAGMA busy_timeout = 5000;");
  return db;
}

// Minimal conflict schema bootstrap for evolution conflict insertion.
// Full schema is managed by soul-memory-conflict.ts; this just ensures the table exists.
function ensureConflictSchemaCompact(db: DatabaseSync): void {
  db.exec(
    "CREATE TABLE IF NOT EXISTS memory_conflicts (" +
      "id TEXT PRIMARY KEY, " +
      "memory_id_a TEXT NOT NULL, " +
      "memory_id_b TEXT NOT NULL, " +
      "content_a TEXT NOT NULL, " +
      "content_b TEXT NOT NULL, " +
      "conflict_reason TEXT NOT NULL, " +
      "detected_at INTEGER NOT NULL, " +
      "resolved_at INTEGER, " +
      "resolution TEXT, " +
      "resolved_by TEXT, " +
      "resolution_strategy TEXT NOT NULL DEFAULT 'auto'" +
      ");",
  );
}
