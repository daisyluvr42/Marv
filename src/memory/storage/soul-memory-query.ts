import type { DatabaseSync } from "node:sqlite";
import {
  applyReferenceExpansion,
  loadReferencesBySourceIds,
} from "../salience/reference-expansion.js";
import {
  computeReinforcementFactor,
  recordScopeHits,
  reinforceRetrievedItems,
} from "../salience/reinforcement.js";
import {
  computeFusionSemanticMatch,
  computeWeightedScore,
  resolveScopePenalty,
} from "../salience/salience-compute.js";
import {
  cosineSimilarity,
  embedTextLegacy,
  lexicalOverlap,
  parseEmbedding,
} from "./soul-memory-embedding.js";
import { openSoulMemoryDb } from "./soul-memory-schema.js";
import {
  combineRrf,
  computeClusterScores,
  computeGraphScores,
  normalizeBm25Score,
  rankByScore,
  searchByBm25,
  searchByVector,
} from "./soul-memory-scoring.js";
import {
  CANDIDATE_FULL_SCAN_MAX_ROWS,
  CANDIDATE_MAX_LIMIT,
  CANDIDATE_MIN_LIMIT,
  CANDIDATE_PER_TOPK_MULTIPLIER,
  DEFAULT_INJECT_THRESHOLD,
  MEMORY_ITEM_SELECT_COLUMNS,
  MILLIS_PER_DAY,
  RECENT_CANDIDATE_SHARE,
  SOUL_ARCHIVE_TABLE,
  type ArchiveRow,
  type MemoryItemRow,
  type SoulArchiveQueryResult,
  type SoulMemoryConfig,
  type SoulMemoryItem,
  type SoulMemoryQueryResult,
  type SoulMemoryScope,
  buildScopedAliasClause,
  clamp,
  dedupeScopes,
  normalizeText,
  resolveSoulMemoryConfig,
  rowToArchiveEvent,
  rowToMemoryItem,
  scopeKey,
  summarizeArchiveContent,
} from "./soul-memory-types.js";

export function querySoulMemoryMulti(params: {
  agentId: string;
  scopes: SoulMemoryScope[];
  query: string;
  topK: number;
  minScore?: number;
  ttlDays?: number;
  nowMs?: number;
  soulConfig?: SoulMemoryConfig;
  /** Point-in-time query: only return semantic nodes valid at this timestamp.
   *  When omitted, retired semantics (valid_until IS NOT NULL) are excluded. */
  temporalMs?: number;
  /** Pre-computed ML embedding for the query. When provided, used for vec0 search
   *  and as the primary vectorScore for items with matching-dimension embeddings.
   *  The legacy hash embedding is used as fallback for items without ML embeddings. */
  queryEmbedding?: number[];
}): SoulMemoryQueryResult[] {
  const topK = Math.max(0, Math.floor(params.topK));
  if (topK <= 0) {
    return [];
  }
  const cleanedQuery = params.query.trim();
  if (!cleanedQuery) {
    return [];
  }
  const uniqueScopes = dedupeScopes(params.scopes);
  if (uniqueScopes.length === 0) {
    return [];
  }
  const minScore = clamp(params.minScore ?? DEFAULT_INJECT_THRESHOLD, 0, 1.5);
  const ttlDays = Number.isFinite(params.ttlDays)
    ? Math.max(0, Math.floor(params.ttlDays as number))
    : 0;
  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  const temporalMs = Number.isFinite(params.temporalMs)
    ? Math.floor(params.temporalMs as number)
    : null;
  const soulConfig = resolveSoulMemoryConfig(params.soulConfig);
  const legacyQueryVec = embedTextLegacy(cleanedQuery);
  // Use ML query embedding for vec0 search if provided; fall back to legacy hash
  const queryVec =
    params.queryEmbedding && params.queryEmbedding.length > 0
      ? params.queryEmbedding
      : legacyQueryVec;
  const activeScopeKeySet = new Set<string>(
    uniqueScopes.map((scope) => scopeKey(scope.scopeType, scope.scopeId)),
  );
  const activeScopeId = scopeKey(
    uniqueScopes[0]?.scopeType ?? "agent",
    uniqueScopes[0]?.scopeId ?? "main",
  );

  const db = openSoulMemoryDb(params.agentId);
  try {
    const bm25ById = searchByBm25({
      db,
      query: cleanedQuery,
      limit: Math.max(topK * 8, 40),
    });
    // Vec0 search — also used for direct distance-based vectorScore
    const vec0ScoresById = searchByVector({
      db,
      queryVec,
      limit: Math.max(topK * 8, 40),
    });
    const totalItems = countMemoryItemsInternal(db);
    const useCandidateQuery = totalItems > CANDIDATE_FULL_SCAN_MAX_ROWS;
    const rows = useCandidateQuery
      ? loadMemoryRowsByIds(
          db,
          buildCandidateMemoryIds({
            db,
            queryVec,
            bm25ById,
            scopes: uniqueScopes,
            topK,
          }),
        )
      : (db
          .prepare(`SELECT ${MEMORY_ITEM_SELECT_COLUMNS} FROM memory_items`)
          .all() as MemoryItemRow[]);
    if (rows.length === 0) {
      return [];
    }

    type Candidate = {
      item: SoulMemoryItem;
      vectorScore: number;
      lexicalScore: number;
      scopePenalty: number;
      clarityScore: number;
      decayFactor: number;
      ageDays: number;
    };

    const candidatesById = new Map<string, Candidate>();
    for (const row of rows) {
      const item = rowToMemoryItem(row);
      const effectiveTs = item.lastAccessedAt ?? item.createdAt;
      const ageDays = Math.max(0, (nowMs - effectiveTs) / MILLIS_PER_DAY);
      // No tier-based TTL exemption: all items are in the Memory Palace. TTL applies uniformly.
      if (ttlDays > 0 && ageDays > ttlDays) {
        continue;
      }

      // Temporal filtering for semantic evolution
      if (item.memoryType === "semantic") {
        if (item.validUntil != null) {
          // Retired semantic
          if (temporalMs != null) {
            // Point-in-time: include only if valid at that moment
            const from = item.validFrom ?? 0;
            if (temporalMs < from || temporalMs >= item.validUntil) {
              continue;
            }
          } else {
            // Default: exclude retired semantics
            continue;
          }
        } else if (temporalMs != null && item.validFrom != null) {
          // Active semantic with a known birth time: exclude if it didn't exist yet
          if (temporalMs < item.validFrom) {
            continue;
          }
        }
      }

      const decayFactor = 1;
      const clarityScore = clamp(item.confidence, 0, 1);

      // Use vec0 distance score when available (ML or legacy vec0 match).
      // Fall back to inline cosine with legacy hash vectors for items not in vec0.
      const vec0Result = vec0ScoresById.get(item.id);
      let vectorScore: number;
      if (vec0Result) {
        vectorScore = vec0Result.score;
      } else {
        const itemVec = parseEmbedding(row.embedding_json);
        vectorScore = cosineSimilarity(legacyQueryVec, itemVec);
      }
      const lexicalScore = lexicalOverlap(cleanedQuery, item.content);
      candidatesById.set(item.id, {
        item,
        vectorScore,
        lexicalScore,
        scopePenalty: resolveScopePenalty(
          {
            item,
            activeScopeKeySet,
          },
          soulConfig,
        ),
        clarityScore,
        decayFactor,
        ageDays,
      });
    }

    if (candidatesById.size === 0) {
      return [];
    }

    const vectorRankById = rankByScore(
      [...candidatesById.values()].map((candidate) => ({
        id: candidate.item.id,
        score: candidate.vectorScore * 0.9 + candidate.lexicalScore * 0.1,
      })),
    );
    const graphById = computeGraphScores({
      db,
      query: cleanedQuery,
      candidateIds: [...candidatesById.keys()],
    });
    const clusterById = computeClusterScores([...candidatesById.values()]);

    const scoredById = new Map<string, SoulMemoryQueryResult>();
    for (const candidate of candidatesById.values()) {
      const bm25 = bm25ById.get(candidate.item.id);
      const vectorRank = vectorRankById.get(candidate.item.id) ?? Number.POSITIVE_INFINITY;
      const bm25Rank = bm25?.rank ?? Number.POSITIVE_INFINITY;
      const rrfScore = combineRrf(vectorRank, bm25Rank);
      const bm25Score = bm25 == null ? 0 : normalizeBm25Score(bm25.score);
      const graphScore = graphById.get(candidate.item.id) ?? 0;
      const clusterScore = clusterById.get(candidate.item.id) ?? 0;
      const fusionSemanticMatch = computeFusionSemanticMatch({
        vectorScore: candidate.vectorScore,
        lexicalScore: candidate.lexicalScore,
        bm25Score,
        graphScore,
        clusterScore,
      });
      const relevanceScore = clamp(fusionSemanticMatch * candidate.scopePenalty, 0, 1);
      const clarityScore = candidate.clarityScore;
      const wasRecallBoosted = false;
      const decayFactor = 1; // No decay in the new architecture
      const similarityScore = computeWeightedScore(
        relevanceScore,
        soulConfig.scoreSimilarityWeight,
      );
      const decayScore = 1; // No decay
      const reinforcementFactor = computeReinforcementFactor(
        candidate.item.reinforcementCount,
        soulConfig,
      );
      const salienceDecay = decayScore;
      const salienceReinforcement = reinforcementFactor;
      const salienceScore = salienceReinforcement; // score = relevance x scope x reinforcement
      const tierMultiplier = 1; // All items are in the Memory Palace
      // Discount compacted episodic items so semantic distillations rank higher
      const compactedFactor =
        candidate.item.isCompacted && candidate.item.memoryType === "episodic"
          ? soulConfig.compaction.compactedDiscount
          : 1;
      // Simplified scoring: relevance x scope x reinforcement
      const score = clamp(
        compactedFactor * candidate.item.confidence * similarityScore * salienceScore,
        0,
        1.5,
      );
      const result: SoulMemoryQueryResult = {
        ...candidate.item,
        score,
        vectorScore: candidate.vectorScore,
        lexicalScore: candidate.lexicalScore,
        bm25Score,
        rrfScore,
        graphScore,
        clusterScore,
        relevanceScore,
        scopePenalty: candidate.scopePenalty,
        clarityScore,
        tierMultiplier,
        wasRecallBoosted,
        timeDecay: decayFactor,
        salienceScore,
        salienceDecay,
        salienceReinforcement,
        reinforcementFactor,
        referenceBoost: 0,
        references: [],
        ageDays: candidate.ageDays,
        conflictIds: [],
      };
      scoredById.set(candidate.item.id, result);
    }

    if (scoredById.size === 0) {
      return [];
    }
    applyReferenceExpansion({
      db,
      scoredById,
      topK,
      soulConfig,
    });

    const dedup = new Map<string, SoulMemoryQueryResult>();
    for (const result of scoredById.values()) {
      if (result.score < minScore) {
        continue;
      }
      const dedupKey = normalizeText(result.content);
      const existing = dedup.get(dedupKey);
      if (!existing || result.score > existing.score) {
        dedup.set(dedupKey, result);
      }
    }

    const ranked = [...dedup.values()].toSorted((a, b) => b.score - a.score).slice(0, topK);
    const referencesByMemoryId = loadReferencesBySourceIds(
      db,
      ranked.map((entry) => entry.id),
    );
    for (const entry of ranked) {
      entry.references = referencesByMemoryId.get(entry.id) ?? [];
    }
    reinforceRetrievedItems(db, ranked, nowMs);
    recordScopeHits(db, ranked, activeScopeId, nowMs);

    // Annotate results with unresolved conflict IDs
    annotateConflictIds(db, ranked);

    return ranked;
  } finally {
    db.close();
  }
}

export function querySoulArchive(params: {
  agentId: string;
  scopes: SoulMemoryScope[];
  query: string;
  topK: number;
  minScore?: number;
  nowMs?: number;
  /** Pre-computed ML embedding for the query (same as querySoulMemoryMulti). */
  queryEmbedding?: number[];
}): SoulArchiveQueryResult[] {
  const topK = Math.max(0, Math.floor(params.topK));
  if (topK <= 0) {
    return [];
  }
  const cleanedQuery = params.query.trim();
  if (!cleanedQuery) {
    return [];
  }
  const uniqueScopes = dedupeScopes(params.scopes);
  // Use ML embedding for inline cosine if provided; otherwise legacy hash
  const queryVec =
    params.queryEmbedding && params.queryEmbedding.length > 0
      ? params.queryEmbedding
      : embedTextLegacy(cleanedQuery);
  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  const minScore = clamp(params.minScore ?? 0.15, 0, 1.5);
  const db = openSoulMemoryDb(params.agentId);
  try {
    const rows = loadArchiveRowsByScope(db, uniqueScopes, Math.max(topK * 10, 80));
    const results: SoulArchiveQueryResult[] = [];
    for (const row of rows) {
      const content = row.content;
      const summary = row.summary || summarizeArchiveContent(content);
      const vectorScore = cosineSimilarity(queryVec, parseEmbedding(row.embedding_json));
      const lexicalScore = Math.max(
        lexicalOverlap(cleanedQuery, content),
        lexicalOverlap(cleanedQuery, summary),
      );
      const score = clamp(vectorScore * 0.55 + lexicalScore * 0.45, 0, 1.5);
      if (score < minScore) {
        continue;
      }
      const createdAt = Number(row.created_at ?? 0);
      const ageDays = Math.max(0, (nowMs - createdAt) / MILLIS_PER_DAY);
      results.push({
        ...rowToArchiveEvent(row),
        score,
        vectorScore,
        lexicalScore,
        ageDays,
      });
    }
    return results.toSorted((a, b) => b.score - a.score).slice(0, topK);
  } finally {
    db.close();
  }
}

export function countMemoryItemsInternal(db: DatabaseSync): number {
  const row = db.prepare("SELECT COUNT(*) as c FROM memory_items").get() as
    | { c?: number }
    | undefined;
  return Number(row?.c ?? 0);
}

export function resolveCandidateLimit(topK: number): number {
  const scaled = Math.max(CANDIDATE_MIN_LIMIT, Math.floor(topK * CANDIDATE_PER_TOPK_MULTIPLIER));
  return Math.min(CANDIDATE_MAX_LIMIT, scaled);
}

export function addCandidateIds(target: Set<string>, ids: Iterable<string>, maxSize: number): void {
  for (const id of ids) {
    const normalized = id.trim();
    if (!normalized || target.has(normalized)) {
      continue;
    }
    target.add(normalized);
    if (target.size >= maxSize) {
      return;
    }
  }
}

export function buildCandidateMemoryIds(params: {
  db: DatabaseSync;
  queryVec: number[];
  bm25ById: Map<string, { rank: number; score: number }>;
  scopes: SoulMemoryScope[];
  topK: number;
}): string[] {
  const candidateLimit = resolveCandidateLimit(params.topK);
  const candidateIds = new Set<string>();
  const vectorById = searchByVector({
    db: params.db,
    queryVec: params.queryVec,
    limit: candidateLimit,
  });
  addCandidateIds(candidateIds, vectorById.keys(), candidateLimit);
  addCandidateIds(candidateIds, params.bm25ById.keys(), candidateLimit);

  if (params.scopes.length > 0 && candidateIds.size < candidateLimit) {
    const scopedLimit = Math.max(1, Math.floor(candidateLimit * RECENT_CANDIDATE_SHARE));
    addCandidateIds(
      candidateIds,
      listRecentMemoryIds({
        db: params.db,
        scopes: params.scopes,
        limit: scopedLimit,
      }),
      candidateLimit,
    );
  }
  if (candidateIds.size < candidateLimit) {
    addCandidateIds(
      candidateIds,
      listRecentMemoryIds({
        db: params.db,
        limit: candidateLimit - candidateIds.size,
      }),
      candidateLimit,
    );
  }
  return [...candidateIds];
}

export function loadMemoryRowsByIds(db: DatabaseSync, memoryIds: string[]): MemoryItemRow[] {
  if (memoryIds.length === 0) {
    return [];
  }
  const placeholders = memoryIds.map(() => "?").join(", ");
  return db
    .prepare(`SELECT ${MEMORY_ITEM_SELECT_COLUMNS} FROM memory_items WHERE id IN (${placeholders})`)
    .all(...memoryIds) as MemoryItemRow[];
}

export function loadArchiveRowsByScope(
  db: DatabaseSync,
  scopes: SoulMemoryScope[],
  limit: number,
): ArchiveRow[] {
  const columns =
    "id, scope_type, scope_id, kind, record_kind, content, summary, embedding_json, source, created_at, active_memory_id, metadata_json";
  if (scopes.length === 0) {
    return db
      .prepare(`SELECT ${columns} FROM ${SOUL_ARCHIVE_TABLE} ORDER BY created_at DESC LIMIT ?`)
      .all(limit) as ArchiveRow[];
  }
  const scoped = buildScopedAliasClause({ scopes, alias: "" });
  return db
    .prepare(
      `SELECT ${columns} FROM ${SOUL_ARCHIVE_TABLE} WHERE ${scoped.clause} ORDER BY created_at DESC LIMIT ?`,
    )
    .all(...scoped.values, limit) as ArchiveRow[];
}

export function listRecentMemoryIds(params: {
  db: DatabaseSync;
  limit: number;
  scopes?: SoulMemoryScope[];
}): string[] {
  const limit = Math.max(0, Math.floor(params.limit));
  if (limit <= 0) {
    return [];
  }
  let where = "";
  let values: string[] = [];
  if (params.scopes && params.scopes.length > 0) {
    const scoped = buildScopedAliasClause({ scopes: params.scopes, alias: "" });
    where = `WHERE ${scoped.clause}`;
    values = scoped.values;
  }
  type Row = { id?: string };
  const rows = params.db
    .prepare(
      "SELECT id FROM memory_items " +
        where +
        " ORDER BY COALESCE(last_accessed_at, created_at) DESC LIMIT ?",
    )
    .all(...values, limit) as Row[];
  const out: string[] = [];
  for (const row of rows) {
    const id = String(row.id ?? "").trim();
    if (id) {
      out.push(id);
    }
  }
  return out;
}

/**
 * Batch-annotate query results with unresolved conflict IDs from memory_conflicts.
 * Lightweight: single query, no scoring impact.
 */
export function annotateConflictIds(db: DatabaseSync, results: SoulMemoryQueryResult[]): void {
  if (results.length === 0) {
    return;
  }
  // Check if memory_conflicts table exists
  const tableCheck = db
    .prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_conflicts' LIMIT 1",
    )
    .get() as { name?: string } | undefined;
  if (!tableCheck?.name) {
    return;
  }
  const ids = results.map((r) => r.id);
  const placeholders = ids.map(() => "?").join(", ");
  type ConflictHit = { id: string; memory_id_a: string; memory_id_b: string };
  const rows = db
    .prepare(
      "SELECT id, memory_id_a, memory_id_b FROM memory_conflicts " +
        `WHERE resolved_at IS NULL AND (memory_id_a IN (${placeholders}) OR memory_id_b IN (${placeholders}))`,
    )
    .all(...ids, ...ids) as ConflictHit[];
  if (rows.length === 0) {
    return;
  }
  // Build map: memory_id -> conflict IDs
  const conflictMap = new Map<string, string[]>();
  for (const row of rows) {
    for (const mid of [row.memory_id_a, row.memory_id_b]) {
      const list = conflictMap.get(mid) ?? [];
      list.push(row.id);
      conflictMap.set(mid, list);
    }
  }
  for (const result of results) {
    result.conflictIds = conflictMap.get(result.id) ?? [];
  }
}
