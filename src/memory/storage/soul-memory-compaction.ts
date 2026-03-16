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
  type ResolvedP3CompactionConfig,
  resolveSoulMemoryDbPath,
  writeSoulMemory,
} from "./soul-memory-store.js";
import { requireNodeSqlite } from "./sqlite.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type P3CompactionResult = {
  /** Number of new P2 semantic nodes created from clusters. */
  compactedClusters: number;
  /** Number of episodic items marked is_compacted = 1. */
  compactedEpisodic: number;
  /** Number of compacted episodic items archived (moved to memory_archive). */
  archivedCompacted: number;
  /** Number of orphan episodic items archived via safety valve. */
  archivedOrphans: number;
  /** IDs of newly created semantic memories. */
  semanticIds: string[];
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

export function compactP3Episodic(params: {
  agentId: string;
  config: ResolvedP3CompactionConfig;
  nowMs?: number;
  summarizeCluster?: (items: ConsolidationItem[]) => string;
}): P3CompactionResult {
  if (!params.config.enabled) {
    return {
      compactedClusters: 0,
      compactedEpisodic: 0,
      archivedCompacted: 0,
      archivedOrphans: 0,
      semanticIds: [],
    };
  }

  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  const db = openSoulMemoryDb(params.agentId);
  try {
    const result = runCompaction(db, params.agentId, params.config, nowMs, params.summarizeCluster);
    const archiveResult = runArchival(db, params.config, nowMs);
    return {
      ...result,
      archivedCompacted: archiveResult.archivedCompacted,
      archivedOrphans: archiveResult.archivedOrphans,
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
  config: ResolvedP3CompactionConfig,
  nowMs: number,
  summarizeCluster?: (items: ConsolidationItem[]) => string,
): Omit<P3CompactionResult, "archivedCompacted" | "archivedOrphans"> {
  // Load uncompacted P3 episodic items
  const rows = db
    .prepare(
      "SELECT id, scope_type, scope_id, kind, content, record_kind, source, source_detail, " +
        "confidence, created_at, metadata_json " +
        "FROM memory_items WHERE tier = 'P3' AND memory_type = 'episodic' AND is_compacted = 0",
    )
    .all() as EpisodicRow[];

  if (rows.length < config.minClusterSize) {
    return { compactedClusters: 0, compactedEpisodic: 0, semanticIds: [] };
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

      // Check if active semantic with same key already exists (v1: skip if so)
      if (semanticKey) {
        const existing = db
          .prepare(
            "SELECT id FROM memory_items WHERE semantic_key = ? AND memory_type = 'semantic' " +
              "AND valid_until IS NULL LIMIT 1",
          )
          .get(semanticKey) as { id: string } | undefined;
        if (existing) {
          continue; // v1: skip evolution, handled in v2
        }
      }

      // Generate summary for the semantic node
      const summary = summarizeCluster ? summarizeCluster(cluster) : buildHeuristicSummary(cluster);

      if (!summary) {
        continue;
      }

      // Write the new P2 semantic node via writeSoulMemory
      const semanticItem = writeSoulMemory({
        agentId,
        scopeType: head.scopeType,
        scopeId: head.scopeId,
        kind: head.kind,
        content: summary,
        confidence: 0.7,
        source: "auto_extraction",
        tier: "P2",
        recordKind: normalizeRecordKind(headRow.record_kind),
        metadata: {
          compactedFrom: cluster.map((c) => c.id),
          compactionSource: "p3_compaction",
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

  return { compactedClusters, compactedEpisodic, semanticIds };
}

// ---------------------------------------------------------------------------
// Step 2: Archival — move compacted + orphan episodic items to archive
// ---------------------------------------------------------------------------

function runArchival(
  db: DatabaseSync,
  config: ResolvedP3CompactionConfig,
  nowMs: number,
): { archivedCompacted: number; archivedOrphans: number } {
  const archiveAgeCutoff = nowMs - config.archiveAgeDays * 86_400_000;
  const orphanAgeCutoff = nowMs - config.orphanAgeDays * 86_400_000;

  // Archive compacted episodic items past archiveAgeDays
  const compactedRows = db
    .prepare(
      "SELECT id, scope_type, scope_id, kind, record_kind, content, summary, embedding_json, " +
        "source, created_at, metadata_json " +
        "FROM memory_items WHERE tier = 'P3' AND memory_type = 'episodic' " +
        "AND is_compacted = 1 AND created_at < ?",
    )
    .all(archiveAgeCutoff) as ArchiveSourceRow[];

  const archivedCompacted = archiveEpisodicItems(db, compactedRows, nowMs, "compacted");

  // Archive orphan episodic items past orphanAgeDays (safety valve)
  const orphanRows = db
    .prepare(
      "SELECT id, scope_type, scope_id, kind, record_kind, content, summary, embedding_json, " +
        "source, created_at, metadata_json " +
        "FROM memory_items WHERE tier = 'P3' AND memory_type = 'episodic' " +
        "AND is_compacted = 0 AND created_at < ?",
    )
    .all(orphanAgeCutoff) as ArchiveSourceRow[];

  const archivedOrphans = archiveEpisodicItems(db, orphanRows, nowMs, "orphan_timeout");

  return { archivedCompacted, archivedOrphans };
}

type ArchiveSourceRow = {
  id: string;
  scope_type: string;
  scope_id: string;
  kind: string;
  record_kind: string;
  content: string;
  summary: string | null;
  embedding_json: string;
  source: string;
  created_at: number;
  metadata_json: string | null;
};

/**
 * Archive episodic items one-by-one (no sessionKey merge).
 * Preserves sessionKey in metadata for session-level queries.
 */
function archiveEpisodicItems(
  db: DatabaseSync,
  rows: ArchiveSourceRow[],
  nowMs: number,
  archiveReason: string,
): number {
  if (rows.length === 0) {
    return 0;
  }

  const insertStmt = db.prepare(
    "INSERT INTO memory_archive (" +
      "id, scope_type, scope_id, kind, record_kind, content, summary, embedding_json, " +
      "source, created_at, active_memory_id, metadata_json" +
      ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
  );
  const deleteStmt = db.prepare("DELETE FROM memory_items WHERE id = ?");

  // Load lineage for compacted items (batch)
  const rowIds = rows.map((r) => String(r.id));
  const lineageMap = loadLineageForTargets(db, rowIds);

  let archived = 0;
  for (const row of rows) {
    const archiveId = `arch_${crypto.randomUUID().replace(/-/g, "")}`;
    const originalMeta = parseMetadata(row.metadata_json);
    const archiveMeta: Record<string, unknown> = {
      ...originalMeta,
      original_memory_id: row.id,
      archive_reason: archiveReason,
    };

    // Preserve lineage: which semantic node was this compacted into
    const lineageTargets = lineageMap.get(String(row.id));
    if (lineageTargets && lineageTargets.length > 0) {
      archiveMeta.compacted_into = lineageTargets;
      archiveMeta.lineage_type = "compacted_from";
    }

    const summary = row.summary ?? "";

    insertStmt.run(
      archiveId,
      String(row.scope_type),
      String(row.scope_id),
      String(row.kind),
      String(row.record_kind),
      String(row.content),
      summary,
      String(row.embedding_json),
      String(row.source),
      Number(row.created_at),
      null, // active_memory_id: null since source row is being deleted
      JSON.stringify(archiveMeta),
    );
    deleteStmt.run(row.id);
    archived++;
  }

  return archived;
}

/**
 * Load lineage edges where target_id is in the given set.
 * Returns map of target_id -> [source_id] (i.e., which semantic nodes consumed each episodic item).
 */
function loadLineageForTargets(db: DatabaseSync, targetIds: string[]): Map<string, string[]> {
  const result = new Map<string, string[]>();
  if (targetIds.length === 0) {
    return result;
  }

  const placeholders = targetIds.map(() => "?").join(",");
  const rows = db
    .prepare(
      `SELECT source_id, target_id FROM memory_lineage WHERE target_id IN (${placeholders}) ` +
        "AND edge_type = 'compacted_from'",
    )
    .all(...targetIds) as Array<{ source_id: string; target_id: string }>;

  for (const row of rows) {
    const list = result.get(String(row.target_id)) ?? [];
    list.push(String(row.source_id));
    result.set(String(row.target_id), list);
  }
  return result;
}

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

function parseMetadata(json: string | null): Record<string, unknown> {
  if (!json) {
    return {};
  }
  try {
    const parsed = JSON.parse(json);
    return typeof parsed === "object" && parsed !== null ? parsed : {};
  } catch {
    return {};
  }
}

function openSoulMemoryDb(agentId: string): DatabaseSync {
  const dbPath = resolveSoulMemoryDbPath(agentId);
  fsSync.mkdirSync(path.dirname(dbPath), { recursive: true });
  const { DatabaseSync } = requireNodeSqlite();
  return new DatabaseSync(dbPath);
}
