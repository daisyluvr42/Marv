import crypto from "node:crypto";
import fsSync from "node:fs";
import os from "node:os";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { resolveStateDir } from "../config/paths.js";
import { requireNodeSqlite } from "./sqlite.js";

const MILLIS_PER_DAY = 24 * 60 * 60 * 1000;
const SOURCE_CORE_PREFERENCE = "core_preference";
const SOURCE_MANUAL_LOG = "manual_log";
const SOURCE_MIGRATION = "migration";
const SOURCE_AUTO_EXTRACTION = "auto_extraction";

const TIER_P0 = "P0";
const TIER_P1 = "P1";
const TIER_P2 = "P2";

const FORGET_CONFIDENCE_THRESHOLD = 0.05;
const FORGET_STREAK_HALF_LIVES = 5;
const DEFAULT_INJECT_THRESHOLD = 0.65;
const P0_CLARITY_HALF_LIFE_DAYS = 365;
const P1_CLARITY_HALF_LIFE_DAYS = 60;
const P2_CLARITY_HALF_LIFE_DAYS = 15;
const P0_RECALL_RELEVANCE_THRESHOLD = 0.8;
const P2_TO_P1_MIN_CLARITY = 0.9;
const P2_TO_P1_MIN_AGE_DAYS = 14;
const P2_TO_P1_MIN_SCOPE_COUNT = 3;
const P1_TO_P0_MIN_CLARITY = 0.8;
const P1_TO_P0_MIN_AGE_DAYS = 300;
const P0_SCOPE_PENALTY = 0.8;
const CROSS_SCOPE_PENALTY = 0.2;
const MATCH_SCOPE_PENALTY = 1;
const P0_TIER_MULTIPLIER = 1.25;
const P1_TIER_MULTIPLIER = 1;
const P2_TIER_MULTIPLIER = 0.85;
const FUSION_VECTOR_WEIGHT = 0.32;
const FUSION_LEXICAL_WEIGHT = 0.15;
const FUSION_BM25_WEIGHT = 0.15;
const FUSION_GRAPH_WEIGHT = 0.16;
const FUSION_CLUSTER_WEIGHT = 0.06;
const FUSION_WEIGHT_SUM =
  FUSION_VECTOR_WEIGHT +
  FUSION_LEXICAL_WEIGHT +
  FUSION_BM25_WEIGHT +
  FUSION_GRAPH_WEIGHT +
  FUSION_CLUSTER_WEIGHT;
const EMBEDDING_DIMS = 128;
const SOUL_MEMORY_PATH_PREFIX = "soul-memory/";
const SOUL_MEMORY_FTS_TABLE = "memory_items_fts";
const SOUL_MEMORY_ENTITY_TABLE = "memory_item_entities";
const SOUL_MEMORY_SCOPE_HITS_TABLE = "memory_scope_hits";
const RRF_RANK_CONSTANT = 40;
const VECTOR_RRF_WEIGHT = 0.6;
const BM25_RRF_WEIGHT = 0.4;

type SourceProfile = {
  confidence: number;
  tier: SoulMemoryTier;
};

const SOURCE_PROFILE: Record<SoulMemorySource, SourceProfile> = {
  [SOURCE_CORE_PREFERENCE]: { confidence: 0.95, tier: TIER_P0 },
  [SOURCE_MANUAL_LOG]: { confidence: 0.85, tier: TIER_P1 },
  [SOURCE_MIGRATION]: { confidence: 0.85, tier: TIER_P1 },
  [SOURCE_AUTO_EXTRACTION]: { confidence: 0.5, tier: TIER_P2 },
};

const ENTITY_EN_STOPWORDS = new Set<string>([
  "this",
  "that",
  "with",
  "from",
  "have",
  "been",
  "will",
  "would",
  "could",
  "should",
  "what",
  "when",
  "where",
  "which",
  "there",
  "their",
  "about",
  "after",
  "before",
  "today",
  "tomorrow",
]);

const ENTITY_ZH_STOPWORDS = new Set<string>([
  "我们",
  "你们",
  "他们",
  "这个",
  "那个",
  "今天",
  "明天",
  "昨天",
  "因为",
  "所以",
  "然后",
  "以及",
  "如果",
  "但是",
]);

type MemoryItemRow = {
  id: string;
  scope_type: string;
  scope_id: string;
  kind: string;
  content: string;
  embedding_json: string;
  confidence: number;
  tier: string;
  source: string;
  created_at: number;
  last_accessed_at: number | null;
};

export type SoulMemorySource =
  | typeof SOURCE_CORE_PREFERENCE
  | typeof SOURCE_MANUAL_LOG
  | typeof SOURCE_MIGRATION
  | typeof SOURCE_AUTO_EXTRACTION;

export type SoulMemoryTier = typeof TIER_P0 | typeof TIER_P1 | typeof TIER_P2;

export type SoulMemoryScope = {
  scopeType: string;
  scopeId: string;
  weight: number;
};

export type SoulMemoryItem = {
  id: string;
  scopeType: string;
  scopeId: string;
  kind: string;
  content: string;
  confidence: number;
  tier: SoulMemoryTier;
  source: SoulMemorySource;
  createdAt: number;
  lastAccessedAt: number | null;
};

export type SoulMemoryQueryResult = SoulMemoryItem & {
  score: number;
  vectorScore: number;
  lexicalScore: number;
  bm25Score: number;
  rrfScore: number;
  graphScore: number;
  clusterScore: number;
  relevanceScore: number;
  scopePenalty: number;
  clarityScore: number;
  tierMultiplier: number;
  wasRecallBoosted: boolean;
  timeDecay: number;
  ageDays: number;
};

export type SoulMemoryPromotionCandidate = SoulMemoryItem & {
  ageDays: number;
  clarityScore: number;
  distinctScopeHits: number;
};

export type SoulMemoryPromotionSummary = {
  promotedToP1: number;
  promotedToP0: number;
  p1PromotionIds: string[];
  p0PromotionIds: string[];
  p0ApprovalCandidates: SoulMemoryPromotionCandidate[];
  skillExtractionCandidates: string[];
};

export function buildSoulMemoryPath(itemId: string): string {
  return `${SOUL_MEMORY_PATH_PREFIX}${itemId}`;
}

export function parseSoulMemoryPath(input: string): string | null {
  const normalized = input.trim().replace(/^\/+/, "");
  if (!normalized.toLowerCase().startsWith(SOUL_MEMORY_PATH_PREFIX)) {
    return null;
  }
  const itemId = normalized.slice(SOUL_MEMORY_PATH_PREFIX.length).trim();
  if (!/^mem_[a-z0-9]+$/i.test(itemId)) {
    return null;
  }
  return itemId;
}

export function resolveSoulMemoryDbPath(agentId: string): string {
  const stateDir = resolveStateDir(process.env, os.homedir);
  return path.join(stateDir, "memory", "soul", `${normalizeScopeValue(agentId)}.sqlite`);
}

export function writeSoulMemory(params: {
  agentId: string;
  scopeType: string;
  scopeId: string;
  kind: string;
  content: string;
  confidence?: number;
  source?: string;
  nowMs?: number;
}): SoulMemoryItem | null {
  const content = params.content.trim();
  if (!content) {
    return null;
  }
  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  const source = resolveMemorySource({
    source: params.source,
    inputConfidence: params.confidence ?? SOURCE_PROFILE[SOURCE_MANUAL_LOG].confidence,
  });
  const sourceProfile = SOURCE_PROFILE[source];
  const scopeType = normalizeScopeValue(params.scopeType);
  const scopeId = normalizeScopeValue(params.scopeId);
  const kind = normalizeScopeValue(params.kind);
  const normalizedContent = normalizeText(content);

  const db = openSoulMemoryDb(params.agentId);
  try {
    const existing = findExistingMemoryItem({
      db,
      scopeType,
      scopeId,
      kind,
      normalizedContent,
    });
    if (existing) {
      const nextConfidence = Math.max(existing.confidence, sourceProfile.confidence);
      const nextTier = sourceProfile.tier;
      const nextSource = source;
      const changed =
        nextConfidence !== existing.confidence ||
        nextTier !== existing.tier ||
        nextSource !== existing.source;
      if (!changed) {
        return existing;
      }
      db.prepare("UPDATE memory_items SET confidence = ?, tier = ?, source = ? WHERE id = ?").run(
        nextConfidence,
        nextTier,
        nextSource,
        existing.id,
      );
      return getSoulMemoryItemInternal(db, existing.id);
    }

    const embedding = JSON.stringify(embedText(content));
    const id = `mem_${crypto.randomUUID().replace(/-/g, "")}`;
    db.prepare(
      "INSERT INTO memory_items (" +
        "id, scope_type, scope_id, kind, content, embedding_json, confidence, tier, source, created_at, last_accessed_at" +
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
    ).run(
      id,
      scopeType,
      scopeId,
      kind,
      content,
      embedding,
      sourceProfile.confidence,
      sourceProfile.tier,
      source,
      nowMs,
    );
    upsertItemEntities(db, id, content);
    return getSoulMemoryItemInternal(db, id);
  } finally {
    db.close();
  }
}

export function getSoulMemoryItem(params: {
  agentId: string;
  itemId: string;
}): SoulMemoryItem | null {
  const db = openSoulMemoryDb(params.agentId);
  try {
    return getSoulMemoryItemInternal(db, params.itemId);
  } finally {
    db.close();
  }
}

export function listSoulMemoryItems(params: {
  agentId: string;
  scopeType?: string;
  scopeId?: string;
  kind?: string;
  limit?: number;
}): SoulMemoryItem[] {
  const db = openSoulMemoryDb(params.agentId);
  try {
    const clauses: string[] = [];
    const values: Array<string | number> = [];
    if (params.scopeType?.trim()) {
      clauses.push("scope_type = ?");
      values.push(normalizeScopeValue(params.scopeType));
    }
    if (params.scopeId?.trim()) {
      clauses.push("scope_id = ?");
      values.push(normalizeScopeValue(params.scopeId));
    }
    if (params.kind?.trim()) {
      clauses.push("kind = ?");
      values.push(normalizeScopeValue(params.kind));
    }
    const where = clauses.length > 0 ? `WHERE ${clauses.join(" AND ")}` : "";
    const limit = Math.max(1, Math.min(500, Math.floor(params.limit ?? 200)));
    const rows = db
      .prepare(
        "SELECT id, scope_type, scope_id, kind, content, embedding_json, confidence, tier, source, " +
          "created_at, last_accessed_at FROM memory_items " +
          `${where} ORDER BY created_at DESC LIMIT ?`,
      )
      .all(...values, limit) as MemoryItemRow[];
    return rows.map((row) => rowToMemoryItem(row));
  } finally {
    db.close();
  }
}

export function querySoulMemoryMulti(params: {
  agentId: string;
  scopes: SoulMemoryScope[];
  query: string;
  topK: number;
  minScore?: number;
  ttlDays?: number;
  nowMs?: number;
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
  const queryVec = embedText(cleanedQuery);
  const activeScopeKeySet = new Set<string>(
    uniqueScopes.map((scope) => scopeKey(scope.scopeType, scope.scopeId)),
  );
  const activeScopeId = scopeKey(
    uniqueScopes[0]?.scopeType ?? "agent",
    uniqueScopes[0]?.scopeId ?? "main",
  );

  const db = openSoulMemoryDb(params.agentId);
  try {
    const rows = db
      .prepare(
        "SELECT id, scope_type, scope_id, kind, content, embedding_json, confidence, tier, source, created_at, last_accessed_at " +
          "FROM memory_items",
      )
      .all() as MemoryItemRow[];

    type Candidate = {
      item: SoulMemoryItem;
      vectorScore: number;
      lexicalScore: number;
      scopePenalty: number;
      clarityScore: number;
      ageDays: number;
    };

    const candidatesById = new Map<string, Candidate>();
    const staleMemoryIds: string[] = [];
    for (const row of rows) {
      const item = rowToMemoryItem(row);
      const effectiveTs = item.lastAccessedAt ?? item.createdAt;
      const ageDays = Math.max(0, (nowMs - effectiveTs) / MILLIS_PER_DAY);
      if (ttlDays > 0 && ageDays > ttlDays && item.tier !== TIER_P0) {
        continue;
      }

      const clarityScore = computeCurrentClarity(item, ageDays);
      if (shouldPruneMemoryItem(item, ageDays)) {
        staleMemoryIds.push(item.id);
        continue;
      }

      const itemVec = parseEmbedding(row.embedding_json);
      const vectorScore = cosineSimilarity(queryVec, itemVec);
      const lexicalScore = lexicalOverlap(cleanedQuery, item.content);
      candidatesById.set(item.id, {
        item,
        vectorScore,
        lexicalScore,
        scopePenalty: resolveScopePenalty({
          item,
          activeScopeKeySet,
        }),
        clarityScore,
        ageDays,
      });
    }

    if (staleMemoryIds.length > 0) {
      pruneSoulMemoryItems(db, staleMemoryIds);
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
    const bm25ById = searchByBm25({
      db,
      query: cleanedQuery,
      limit: Math.max(topK * 8, 40),
    });
    const graphById = computeGraphScores({
      db,
      query: cleanedQuery,
      candidateIds: [...candidatesById.keys()],
    });
    const clusterById = computeClusterScores([...candidatesById.values()]);

    const dedup = new Map<string, SoulMemoryQueryResult>();
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
      let clarityScore = candidate.clarityScore;
      let wasRecallBoosted = false;
      if (candidate.item.tier === TIER_P0 && relevanceScore >= P0_RECALL_RELEVANCE_THRESHOLD) {
        clarityScore = 1;
        wasRecallBoosted = true;
      }
      const tierMultiplier = tierPriorityFactor(candidate.item.tier);
      const score = clamp(tierMultiplier * clarityScore * relevanceScore, 0, 1.5);
      if (score < minScore) {
        continue;
      }
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
        timeDecay: clarityDecayFactor(candidate.item.tier, candidate.ageDays),
        ageDays: candidate.ageDays,
      };
      const dedupKey = normalizeText(candidate.item.content);
      const existing = dedup.get(dedupKey);
      if (!existing || result.score > existing.score) {
        dedup.set(dedupKey, result);
      }
    }

    const ranked = [...dedup.values()].toSorted((a, b) => b.score - a.score).slice(0, topK);
    reinforceRetrievedItems(db, ranked, nowMs);
    recordScopeHits(db, ranked, activeScopeId, nowMs);
    return ranked;
  } finally {
    db.close();
  }
}

export function countSoulMemoryItems(params: {
  agentId: string;
  scopeType?: string;
  scopeId?: string;
}): number {
  const db = openSoulMemoryDb(params.agentId);
  try {
    const clauses: string[] = [];
    const values: string[] = [];
    if (params.scopeType?.trim()) {
      clauses.push("scope_type = ?");
      values.push(normalizeScopeValue(params.scopeType));
    }
    if (params.scopeId?.trim()) {
      clauses.push("scope_id = ?");
      values.push(normalizeScopeValue(params.scopeId));
    }
    const where = clauses.length > 0 ? ` WHERE ${clauses.join(" AND ")}` : "";
    const row = db.prepare(`SELECT COUNT(*) as c FROM memory_items${where}`).get(...values) as
      | { c?: number }
      | undefined;
    return row?.c ?? 0;
  } finally {
    db.close();
  }
}

function openSoulMemoryDb(agentId: string): DatabaseSync {
  const dbPath = resolveSoulMemoryDbPath(agentId);
  fsSync.mkdirSync(path.dirname(dbPath), { recursive: true });
  const { DatabaseSync } = requireNodeSqlite();
  const db = new DatabaseSync(dbPath);
  db.exec("PRAGMA foreign_keys = ON;");
  ensureSoulMemorySchema(db);
  return db;
}

function ensureSoulMemorySchema(db: DatabaseSync): void {
  db.exec(
    "CREATE TABLE IF NOT EXISTS memory_items (" +
      "id TEXT PRIMARY KEY, " +
      "scope_type TEXT NOT NULL, " +
      "scope_id TEXT NOT NULL, " +
      "kind TEXT NOT NULL, " +
      "content TEXT NOT NULL, " +
      "embedding_json TEXT NOT NULL, " +
      "confidence REAL NOT NULL, " +
      "tier TEXT NOT NULL DEFAULT 'P1', " +
      "source TEXT NOT NULL DEFAULT 'manual_log', " +
      "created_at INTEGER NOT NULL, " +
      "last_accessed_at INTEGER" +
      ");",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_soul_memory_scope ON memory_items (scope_type, scope_id);",
  );
  db.exec("CREATE INDEX IF NOT EXISTS idx_soul_memory_tier ON memory_items (tier);");
  db.exec("CREATE INDEX IF NOT EXISTS idx_soul_memory_source ON memory_items (source);");
  db.exec(
    `CREATE TABLE IF NOT EXISTS ${SOUL_MEMORY_SCOPE_HITS_TABLE} (` +
      "memory_id TEXT NOT NULL, " +
      "scope_id TEXT NOT NULL, " +
      "hit_count INTEGER NOT NULL DEFAULT 0, " +
      "first_hit_at INTEGER NOT NULL, " +
      "last_hit_at INTEGER NOT NULL, " +
      "PRIMARY KEY (memory_id, scope_id), " +
      "FOREIGN KEY (memory_id) REFERENCES memory_items(id) ON DELETE CASCADE" +
      ");",
  );
  db.exec(
    `CREATE INDEX IF NOT EXISTS idx_soul_memory_scope_hits_scope ON ${SOUL_MEMORY_SCOPE_HITS_TABLE} (scope_id);`,
  );
  db.exec(
    `CREATE INDEX IF NOT EXISTS idx_soul_memory_scope_hits_memory ON ${SOUL_MEMORY_SCOPE_HITS_TABLE} (memory_id);`,
  );
  db.exec(
    `CREATE TABLE IF NOT EXISTS ${SOUL_MEMORY_ENTITY_TABLE} (` +
      "memory_id TEXT NOT NULL, " +
      "entity TEXT NOT NULL, " +
      "PRIMARY KEY (memory_id, entity), " +
      "FOREIGN KEY (memory_id) REFERENCES memory_items(id) ON DELETE CASCADE" +
      ");",
  );
  db.exec(
    `CREATE INDEX IF NOT EXISTS idx_soul_memory_entities_entity ON ${SOUL_MEMORY_ENTITY_TABLE} (entity);`,
  );
  ensureSoulMemoryFts(db);
}

function ensureSoulMemoryFts(db: DatabaseSync): void {
  try {
    const tableExists = hasSoulMemoryFtsTable(db);
    if (!tableExists) {
      db.exec(
        `CREATE VIRTUAL TABLE ${SOUL_MEMORY_FTS_TABLE} USING fts5(` +
          "id UNINDEXED, " +
          "content, " +
          "content='memory_items', " +
          "content_rowid='rowid', " +
          "tokenize='unicode61'" +
          ");",
      );
    }
    db.exec(
      "CREATE TRIGGER IF NOT EXISTS trg_soul_memory_items_ai " +
        "AFTER INSERT ON memory_items BEGIN " +
        `INSERT INTO ${SOUL_MEMORY_FTS_TABLE}(rowid, id, content) VALUES (new.rowid, new.id, new.content); ` +
        "END;",
    );
    db.exec(
      "CREATE TRIGGER IF NOT EXISTS trg_soul_memory_items_ad " +
        "AFTER DELETE ON memory_items BEGIN " +
        `INSERT INTO ${SOUL_MEMORY_FTS_TABLE}(${SOUL_MEMORY_FTS_TABLE}, rowid, id, content) VALUES ('delete', old.rowid, old.id, old.content); ` +
        "END;",
    );
    db.exec(
      "CREATE TRIGGER IF NOT EXISTS trg_soul_memory_items_au " +
        "AFTER UPDATE OF id, content ON memory_items BEGIN " +
        `INSERT INTO ${SOUL_MEMORY_FTS_TABLE}(${SOUL_MEMORY_FTS_TABLE}, rowid, id, content) VALUES ('delete', old.rowid, old.id, old.content); ` +
        `INSERT INTO ${SOUL_MEMORY_FTS_TABLE}(rowid, id, content) VALUES (new.rowid, new.id, new.content); ` +
        "END;",
    );
    if (!tableExists) {
      db.exec(`INSERT INTO ${SOUL_MEMORY_FTS_TABLE}(${SOUL_MEMORY_FTS_TABLE}) VALUES ('rebuild');`);
    }
  } catch {
    // Some Node runtimes can ship SQLite without FTS5; keep vector-only retrieval in that case.
  }
}

function hasSoulMemoryFtsTable(db: DatabaseSync): boolean {
  const row = db
    .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?")
    .get(SOUL_MEMORY_FTS_TABLE) as { name?: string } | undefined;
  return row?.name === SOUL_MEMORY_FTS_TABLE;
}

function upsertItemEntities(db: DatabaseSync, memoryId: string, content: string): void {
  const entities = [...extractEntities(content)];
  db.prepare(`DELETE FROM ${SOUL_MEMORY_ENTITY_TABLE} WHERE memory_id = ?`).run(memoryId);
  if (entities.length === 0) {
    return;
  }
  const stmt = db.prepare(
    `INSERT OR IGNORE INTO ${SOUL_MEMORY_ENTITY_TABLE} (memory_id, entity) VALUES (?, ?)`,
  );
  for (const entity of entities) {
    stmt.run(memoryId, entity);
  }
}

function getSoulMemoryItemInternal(db: DatabaseSync, itemId: string): SoulMemoryItem | null {
  const row = db
    .prepare(
      "SELECT id, scope_type, scope_id, kind, content, embedding_json, confidence, tier, source, created_at, last_accessed_at " +
        "FROM memory_items WHERE id = ?",
    )
    .get(itemId) as MemoryItemRow | undefined;
  return row ? rowToMemoryItem(row) : null;
}

function findExistingMemoryItem(params: {
  db: DatabaseSync;
  scopeType: string;
  scopeId: string;
  kind: string;
  normalizedContent: string;
}): SoulMemoryItem | null {
  const rows = params.db
    .prepare(
      "SELECT id, scope_type, scope_id, kind, content, embedding_json, confidence, tier, source, created_at, last_accessed_at " +
        "FROM memory_items WHERE scope_type = ? AND scope_id = ? AND kind = ?",
    )
    .all(params.scopeType, params.scopeId, params.kind) as MemoryItemRow[];
  for (const row of rows) {
    if (normalizeText(row.content) === params.normalizedContent) {
      return rowToMemoryItem(row);
    }
  }
  return null;
}

function reinforceRetrievedItems(
  db: DatabaseSync,
  results: Array<{ id: string; wasRecallBoosted: boolean }>,
  nowMs: number,
): void {
  if (results.length === 0) {
    return;
  }
  const boosted = db.prepare(
    "UPDATE memory_items SET last_accessed_at = ?, confidence = 1.0 WHERE id = ?",
  );
  const reinforce = db.prepare(
    "UPDATE memory_items SET last_accessed_at = ?, confidence = MIN(1.0, confidence + 0.05) WHERE id = ?",
  );
  const handled = new Set<string>();
  for (const result of results) {
    const itemId = result.id;
    if (!itemId || handled.has(itemId)) {
      continue;
    }
    handled.add(itemId);
    if (result.wasRecallBoosted) {
      boosted.run(nowMs, itemId);
      continue;
    }
    reinforce.run(nowMs, itemId);
  }
}

function recordScopeHits(
  db: DatabaseSync,
  results: Array<{ id: string }>,
  scopeId: string,
  nowMs: number,
): void {
  if (results.length === 0) {
    return;
  }
  const normalizedScopeId = normalizeScopeValue(scopeId);
  if (!normalizedScopeId) {
    return;
  }
  const upsert = db.prepare(
    `INSERT INTO ${SOUL_MEMORY_SCOPE_HITS_TABLE} (memory_id, scope_id, hit_count, first_hit_at, last_hit_at) ` +
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

function pruneSoulMemoryItems(db: DatabaseSync, memoryIds: string[]): number {
  if (memoryIds.length === 0) {
    return 0;
  }
  const seen = new Set<string>();
  const deleteStmt = db.prepare("DELETE FROM memory_items WHERE id = ?");
  let deleted = 0;
  for (const memoryId of memoryIds) {
    const trimmed = memoryId.trim();
    if (!trimmed || seen.has(trimmed)) {
      continue;
    }
    seen.add(trimmed);
    const result = deleteStmt.run(trimmed) as { changes?: number };
    deleted += Number(result.changes ?? 0);
  }
  return deleted;
}

function rowToMemoryItem(row: MemoryItemRow): SoulMemoryItem {
  return {
    id: String(row.id),
    scopeType: String(row.scope_type),
    scopeId: String(row.scope_id),
    kind: String(row.kind),
    content: String(row.content),
    confidence: Number(row.confidence ?? 0),
    tier: normalizeTier(String(row.tier)),
    source: normalizeSource(String(row.source)),
    createdAt: Number(row.created_at ?? 0),
    lastAccessedAt: row.last_accessed_at == null ? null : Number(row.last_accessed_at),
  };
}

function normalizeTier(value: string): SoulMemoryTier {
  const normalized = value.trim().toUpperCase();
  if (normalized === TIER_P0 || normalized === TIER_P1 || normalized === TIER_P2) {
    return normalized;
  }
  return TIER_P1;
}

function normalizeSource(value: string): SoulMemorySource {
  const normalized = value.trim().toLowerCase();
  if (
    normalized === SOURCE_CORE_PREFERENCE ||
    normalized === SOURCE_MANUAL_LOG ||
    normalized === SOURCE_MIGRATION ||
    normalized === SOURCE_AUTO_EXTRACTION
  ) {
    return normalized;
  }
  return SOURCE_MANUAL_LOG;
}

function normalizeScopeValue(value: string): string {
  return value.trim().toLowerCase();
}

function normalizeText(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, value));
}

function resolveMemorySource(params: {
  source?: string;
  inputConfidence: number;
}): SoulMemorySource {
  const normalized = (params.source ?? "").trim().toLowerCase();
  if (normalized === "p0" || normalized === SOURCE_CORE_PREFERENCE || normalized === "explicit") {
    return SOURCE_CORE_PREFERENCE;
  }
  if (normalized === "p2" || normalized === SOURCE_AUTO_EXTRACTION || normalized === "auto") {
    return SOURCE_AUTO_EXTRACTION;
  }
  if (normalized === SOURCE_MIGRATION || normalized === "migration") {
    return SOURCE_MIGRATION;
  }
  if (normalized === "p1" || normalized === SOURCE_MANUAL_LOG || normalized === "manual") {
    return SOURCE_MANUAL_LOG;
  }
  const confidence = Number.isFinite(params.inputConfidence)
    ? Number(params.inputConfidence)
    : SOURCE_PROFILE[SOURCE_MANUAL_LOG].confidence;
  if (confidence >= 0.9) {
    return SOURCE_CORE_PREFERENCE;
  }
  if (confidence <= 0.55) {
    return SOURCE_AUTO_EXTRACTION;
  }
  return SOURCE_MANUAL_LOG;
}

function clarityDecayFactor(tier: SoulMemoryTier, ageDays: number): number {
  const normalizedAgeDays = Math.max(0, ageDays);
  const halfLifeDays =
    tier === TIER_P0
      ? P0_CLARITY_HALF_LIFE_DAYS
      : tier === TIER_P2
        ? P2_CLARITY_HALF_LIFE_DAYS
        : P1_CLARITY_HALF_LIFE_DAYS;
  const factor = 0.5 ** (normalizedAgeDays / halfLifeDays);
  return clamp(factor, 0, 1);
}

function computeCurrentClarity(item: SoulMemoryItem, ageDays: number): number {
  return clamp(item.confidence * clarityDecayFactor(item.tier, ageDays), 0, 1);
}

function resolveTierHalfLifeDays(tier: SoulMemoryTier): number | null {
  if (tier === TIER_P1) {
    return P1_CLARITY_HALF_LIFE_DAYS;
  }
  if (tier === TIER_P2) {
    return P2_CLARITY_HALF_LIFE_DAYS;
  }
  return null;
}

function computeBelowThresholdDurationDays(params: {
  item: SoulMemoryItem;
  ageDays: number;
  threshold: number;
  halfLifeDays: number;
}): number {
  const ageDays = Math.max(0, params.ageDays);
  const threshold = clamp(params.threshold, 0, 1);
  const baseConfidence = clamp(params.item.confidence, 0, 1);
  if (baseConfidence <= 0 || baseConfidence <= threshold) {
    return ageDays;
  }
  const crossingAgeDays = params.halfLifeDays * Math.log2(baseConfidence / threshold);
  if (!Number.isFinite(crossingAgeDays) || crossingAgeDays <= 0) {
    return ageDays;
  }
  return Math.max(0, ageDays - crossingAgeDays);
}

function shouldPruneMemoryItem(item: SoulMemoryItem, ageDays: number): boolean {
  const halfLifeDays = resolveTierHalfLifeDays(item.tier);
  if (!halfLifeDays || !Number.isFinite(halfLifeDays) || halfLifeDays <= 0) {
    return false;
  }
  const clarity = computeCurrentClarity(item, ageDays);
  if (clarity >= FORGET_CONFIDENCE_THRESHOLD) {
    return false;
  }
  const belowThresholdDays = computeBelowThresholdDurationDays({
    item,
    ageDays,
    threshold: FORGET_CONFIDENCE_THRESHOLD,
    halfLifeDays,
  });
  return belowThresholdDays >= halfLifeDays * FORGET_STREAK_HALF_LIVES;
}

function tierPriorityFactor(tier: SoulMemoryTier): number {
  if (tier === TIER_P0) {
    return P0_TIER_MULTIPLIER;
  }
  if (tier === TIER_P2) {
    return P2_TIER_MULTIPLIER;
  }
  return P1_TIER_MULTIPLIER;
}

function resolveScopePenalty(params: {
  item: SoulMemoryItem;
  activeScopeKeySet: Set<string>;
}): number {
  if (params.activeScopeKeySet.has(scopeKey(params.item.scopeType, params.item.scopeId))) {
    return MATCH_SCOPE_PENALTY;
  }
  if (params.item.scopeType === "global" || params.item.scopeType === "user") {
    return P0_SCOPE_PENALTY;
  }
  return CROSS_SCOPE_PENALTY;
}

function computeFusionSemanticMatch(params: {
  vectorScore: number;
  lexicalScore: number;
  bm25Score: number;
  graphScore: number;
  clusterScore: number;
}): number {
  const weighted =
    params.vectorScore * FUSION_VECTOR_WEIGHT +
    params.lexicalScore * FUSION_LEXICAL_WEIGHT +
    params.bm25Score * FUSION_BM25_WEIGHT +
    params.graphScore * FUSION_GRAPH_WEIGHT +
    params.clusterScore * FUSION_CLUSTER_WEIGHT;
  if (FUSION_WEIGHT_SUM <= 0) {
    return 0;
  }
  return clamp(weighted / FUSION_WEIGHT_SUM, 0, 1);
}

function scopeKey(scopeType: string, scopeId: string): string {
  return `${scopeType}:${scopeId}`;
}

function dedupeScopes(scopes: SoulMemoryScope[]): SoulMemoryScope[] {
  const dedup = new Map<string, SoulMemoryScope>();
  for (const scope of scopes) {
    const scopeType = normalizeScopeValue(scope.scopeType);
    const scopeId = normalizeScopeValue(scope.scopeId);
    if (!scopeType || !scopeId) {
      continue;
    }
    const key = scopeKey(scopeType, scopeId);
    const existing = dedup.get(key);
    const normalized: SoulMemoryScope = {
      scopeType,
      scopeId,
      weight: Number.isFinite(scope.weight) ? scope.weight : 1,
    };
    if (!existing || normalized.weight > existing.weight) {
      dedup.set(key, normalized);
    }
  }
  return [...dedup.values()];
}

function parseEmbedding(raw: string): number[] {
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.map((entry) => Number(entry)).filter((entry) => Number.isFinite(entry));
  } catch {
    return [];
  }
}

function computeGraphScores(params: {
  db: DatabaseSync;
  query: string;
  candidateIds: string[];
}): Map<string, number> {
  const out = new Map<string, number>();
  if (params.candidateIds.length === 0) {
    return out;
  }
  const entitiesById = loadEntitiesForMemoryIds(params.db, params.candidateIds);
  const queryEntities = extractEntities(params.query);
  if (queryEntities.size === 0) {
    for (const id of params.candidateIds) {
      out.set(id, 0);
    }
    return out;
  }
  const relatedWeights = loadRelatedEntityWeights({
    db: params.db,
    queryEntities,
  });
  const maxRelatedWeight = Math.max(1, ...relatedWeights.values());

  for (const id of params.candidateIds) {
    const entities = entitiesById.get(id) ?? new Set<string>();
    if (entities.size === 0) {
      out.set(id, 0);
      continue;
    }
    let directMatches = 0;
    for (const queryEntity of queryEntities) {
      if (entities.has(queryEntity)) {
        directMatches += 1;
      }
    }
    const directScore = queryEntities.size > 0 ? directMatches / queryEntities.size : 0;
    let propagatedScore = 0;
    for (const entity of entities) {
      const related = (relatedWeights.get(entity) ?? 0) / maxRelatedWeight;
      if (related > propagatedScore) {
        propagatedScore = related;
      }
    }
    out.set(id, clamp(directScore * 0.75 + propagatedScore * 0.25, 0, 1));
  }
  return out;
}

function computeClusterScores(
  candidates: Array<{ item: { id: string; kind: string } }>,
): Map<string, number> {
  const out = new Map<string, number>();
  if (candidates.length === 0) {
    return out;
  }
  const kindCounts = new Map<string, number>();
  for (const candidate of candidates) {
    const kind = candidate.item.kind;
    kindCounts.set(kind, (kindCounts.get(kind) ?? 0) + 1);
  }
  const maxCount = Math.max(1, ...kindCounts.values());
  for (const candidate of candidates) {
    const count = kindCounts.get(candidate.item.kind) ?? 0;
    out.set(candidate.item.id, clamp(count / maxCount, 0, 1));
  }
  return out;
}

function loadEntitiesForMemoryIds(db: DatabaseSync, memoryIds: string[]): Map<string, Set<string>> {
  const out = new Map<string, Set<string>>();
  if (memoryIds.length === 0) {
    return out;
  }
  const placeholders = memoryIds.map(() => "?").join(", ");
  type Row = { memory_id?: string; entity?: string };
  const rows = db
    .prepare(
      `SELECT memory_id, entity FROM ${SOUL_MEMORY_ENTITY_TABLE} WHERE memory_id IN (${placeholders})`,
    )
    .all(...memoryIds) as Row[];
  for (const row of rows) {
    const memoryId = String(row.memory_id ?? "").trim();
    const entity = String(row.entity ?? "").trim();
    if (!memoryId || !entity) {
      continue;
    }
    let bucket = out.get(memoryId);
    if (!bucket) {
      bucket = new Set<string>();
      out.set(memoryId, bucket);
    }
    bucket.add(entity);
  }
  return out;
}

function loadRelatedEntityWeights(params: {
  db: DatabaseSync;
  scopes?: SoulMemoryScope[];
  queryEntities: Set<string>;
}): Map<string, number> {
  const out = new Map<string, number>();
  const entities = [...params.queryEntities];
  if (entities.length === 0) {
    return out;
  }
  const entityPlaceholders = entities.map(() => "?").join(", ");
  let scopeClause = "";
  let scopeValues: string[] = [];
  if (params.scopes && params.scopes.length > 0) {
    const scoped = buildScopedAliasClause({ scopes: params.scopes, alias: "m" });
    scopeClause = ` AND (${scoped.clause})`;
    scopeValues = scoped.values;
  }
  type Row = { entity?: string; co_count?: number };
  const rows = params.db
    .prepare(
      `SELECT rel.entity AS entity, COUNT(*) AS co_count ` +
        `FROM ${SOUL_MEMORY_ENTITY_TABLE} src ` +
        `JOIN ${SOUL_MEMORY_ENTITY_TABLE} rel ON src.memory_id = rel.memory_id AND rel.entity != src.entity ` +
        "JOIN memory_items m ON m.id = src.memory_id " +
        `WHERE src.entity IN (${entityPlaceholders})${scopeClause} ` +
        "GROUP BY rel.entity ORDER BY co_count DESC LIMIT ?",
    )
    .all(...entities, ...scopeValues, 256) as Row[];
  for (const row of rows) {
    const entity = String(row.entity ?? "").trim();
    if (!entity) {
      continue;
    }
    out.set(entity, Number(row.co_count ?? 0));
  }
  return out;
}

function buildScopedAliasClause(params: { scopes: SoulMemoryScope[]; alias: string }): {
  clause: string;
  values: string[];
} {
  const prefix = params.alias ? `${params.alias}.` : "";
  const clause = params.scopes
    .map(() => `(${prefix}scope_type = ? AND ${prefix}scope_id = ?)`)
    .join(" OR ");
  const values: string[] = [];
  for (const scope of params.scopes) {
    values.push(scope.scopeType, scope.scopeId);
  }
  return { clause, values };
}

function rankByScore(entries: Array<{ id: string; score: number }>): Map<string, number> {
  const ranked = entries
    .slice()
    .toSorted((a, b) => b.score - a.score)
    .map((entry, index) => [entry.id, index + 1] as const);
  return new Map<string, number>(ranked);
}

function combineRrf(vectorRank: number, bm25Rank: number): number {
  const raw =
    VECTOR_RRF_WEIGHT * reciprocalRank(vectorRank) + BM25_RRF_WEIGHT * reciprocalRank(bm25Rank);
  const topRaw = VECTOR_RRF_WEIGHT * reciprocalRank(1) + BM25_RRF_WEIGHT * reciprocalRank(1);
  if (topRaw <= 0) {
    return 0;
  }
  return clamp(raw / topRaw, 0, 1);
}

function reciprocalRank(rank: number): number {
  if (!Number.isFinite(rank) || rank <= 0) {
    return 0;
  }
  return 1 / (RRF_RANK_CONSTANT + rank);
}

function normalizeBm25Score(rawScore: number): number {
  if (!Number.isFinite(rawScore)) {
    return 0;
  }
  // SQLite bm25() lower is better; clamp negatives to strongest match.
  const nonNegative = Math.max(0, rawScore);
  return clamp(1 / (1 + nonNegative), 0, 1);
}

function buildFtsMatchQuery(query: string): string | null {
  const tokens = [...tokenize(query)].slice(0, 12);
  if (tokens.length === 0) {
    return null;
  }
  return tokens.map((token) => `"${token.replaceAll('"', '""')}"`).join(" OR ");
}

function searchByBm25(params: {
  db: DatabaseSync;
  query: string;
  scopes?: SoulMemoryScope[];
  limit: number;
}): Map<string, { rank: number; score: number }> {
  if (!hasSoulMemoryFtsTable(params.db)) {
    return new Map();
  }
  const ftsQuery = buildFtsMatchQuery(params.query);
  if (!ftsQuery) {
    return new Map();
  }
  let scopeClause = "";
  let scopeValues: string[] = [];
  if (params.scopes && params.scopes.length > 0) {
    const scoped = buildScopedAliasClause({ scopes: params.scopes, alias: "m" });
    scopeClause = ` AND (${scoped.clause})`;
    scopeValues = scoped.values;
  }
  type Bm25Row = { id?: string; bm25_score?: number | null };
  try {
    const rows = params.db
      .prepare(
        "SELECT m.id AS id, bm25(memory_items_fts) AS bm25_score " +
          "FROM memory_items_fts " +
          "JOIN memory_items m ON m.id = memory_items_fts.id " +
          "WHERE memory_items_fts MATCH ?" +
          scopeClause +
          " ORDER BY bm25_score ASC LIMIT ?",
      )
      .all(ftsQuery, ...scopeValues, params.limit) as Bm25Row[];
    const out = new Map<string, { rank: number; score: number }>();
    for (const row of rows) {
      const id = String(row.id ?? "").trim();
      if (!id || out.has(id)) {
        continue;
      }
      out.set(id, {
        rank: out.size + 1,
        score: Number(row.bm25_score ?? 0),
      });
    }
    return out;
  } catch {
    return new Map();
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) {
    return 0;
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    normA += x * x;
    normB += y * y;
  }
  if (normA <= 0 || normB <= 0) {
    return 0;
  }
  return dot / Math.sqrt(normA * normB);
}

function lexicalOverlap(a: string, b: string): number {
  const aTokens = tokenize(a);
  const bTokens = tokenize(b);
  if (aTokens.size === 0 || bTokens.size === 0) {
    return 0;
  }
  let intersection = 0;
  for (const token of aTokens) {
    if (bTokens.has(token)) {
      intersection += 1;
    }
  }
  const union = new Set([...aTokens, ...bTokens]).size;
  return union > 0 ? intersection / union : 0;
}

function extractEntities(value: string): Set<string> {
  const out = new Set<string>();
  const rawTokens =
    value.match(
      /[A-Z][A-Z0-9_-]{2,}|[A-Z][a-z]+(?:[A-Z][a-z]+)+|[\u4e00-\u9fff]{2,12}|[a-z0-9_]{4,}/g,
    ) ?? [];
  for (const raw of rawTokens) {
    const trimmed = raw.trim();
    if (!trimmed) {
      continue;
    }
    const hasChinese = /[\u4e00-\u9fff]/.test(trimmed);
    if (hasChinese) {
      if (ENTITY_ZH_STOPWORDS.has(trimmed)) {
        continue;
      }
      out.add(trimmed);
      continue;
    }
    const normalized = trimmed.toLowerCase();
    if (ENTITY_EN_STOPWORDS.has(normalized)) {
      continue;
    }
    if (/^[0-9]+$/.test(normalized)) {
      continue;
    }
    out.add(normalized);
  }
  return out;
}

function tokenize(value: string): Set<string> {
  const out = new Set<string>();
  const matches = value.toLowerCase().match(/[a-z0-9_]+|[\u4e00-\u9fff]+/g) ?? [];
  for (const token of matches) {
    if (token) {
      out.add(token);
    }
  }
  return out;
}

function embedText(value: string): number[] {
  const tokens = tokenize(value);
  if (tokens.size === 0) {
    return [];
  }
  const vec = Array.from({ length: EMBEDDING_DIMS }, () => 0);
  for (const token of tokens) {
    const hash = crypto.createHash("sha256").update(token).digest();
    const indexA = hash[0] % EMBEDDING_DIMS;
    const indexB = hash[3] % EMBEDDING_DIMS;
    const signA = hash[1] % 2 === 0 ? 1 : -1;
    const signB = hash[4] % 2 === 0 ? 1 : -1;
    const weightA = 0.5 + hash[2] / 255;
    const weightB = 0.25 + hash[5] / 255;
    vec[indexA] = (vec[indexA] ?? 0) + signA * weightA;
    vec[indexB] = (vec[indexB] ?? 0) + signB * weightB;
  }
  const magnitude = Math.sqrt(vec.reduce((sum, entry) => sum + entry * entry, 0));
  if (magnitude <= 1e-10) {
    return vec;
  }
  return vec.map((entry) => entry / magnitude);
}

function loadScopeHitSummary(
  db: DatabaseSync,
): Map<string, { distinctScopeHits: number; totalHits: number }> {
  const out = new Map<string, { distinctScopeHits: number; totalHits: number }>();
  type Row = { memory_id?: string; scope_count?: number; total_hits?: number };
  const rows = db
    .prepare(
      `SELECT memory_id, COUNT(*) AS scope_count, COALESCE(SUM(hit_count), 0) AS total_hits ` +
        `FROM ${SOUL_MEMORY_SCOPE_HITS_TABLE} GROUP BY memory_id`,
    )
    .all() as Row[];
  for (const row of rows) {
    const memoryId = String(row.memory_id ?? "").trim();
    if (!memoryId) {
      continue;
    }
    out.set(memoryId, {
      distinctScopeHits: Number(row.scope_count ?? 0),
      totalHits: Number(row.total_hits ?? 0),
    });
  }
  return out;
}

export function promoteSoulMemories(params: {
  agentId: string;
  nowMs?: number;
  autoApproveP0?: boolean;
  approvedP0Ids?: string[];
}): SoulMemoryPromotionSummary {
  const db = openSoulMemoryDb(params.agentId);
  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  const autoApproveP0 = params.autoApproveP0 === true;
  const approvedP0Ids = new Set((params.approvedP0Ids ?? []).map((entry) => entry.trim()));
  try {
    const rows = db
      .prepare(
        "SELECT id, scope_type, scope_id, kind, content, embedding_json, confidence, tier, source, created_at, last_accessed_at " +
          "FROM memory_items",
      )
      .all() as MemoryItemRow[];
    if (rows.length === 0) {
      return {
        promotedToP1: 0,
        promotedToP0: 0,
        p1PromotionIds: [],
        p0PromotionIds: [],
        p0ApprovalCandidates: [],
        skillExtractionCandidates: [],
      };
    }

    const scopeHitSummary = loadScopeHitSummary(db);
    const promoteP1Stmt = db.prepare("UPDATE memory_items SET tier = ? WHERE id = ?");
    const promoteP0Stmt = db.prepare("UPDATE memory_items SET tier = ?, source = ? WHERE id = ?");
    const p1PromotionIds: string[] = [];
    const p0PromotionIds: string[] = [];
    const skillExtractionCandidates: string[] = [];
    const promotedInRun = new Set<string>();

    for (const row of rows) {
      const item = rowToMemoryItem(row);
      if (item.tier !== TIER_P2) {
        continue;
      }
      const hitSummary = scopeHitSummary.get(item.id);
      const distinctScopeHits = hitSummary?.distinctScopeHits ?? 0;
      if (distinctScopeHits < P2_TO_P1_MIN_SCOPE_COUNT) {
        continue;
      }
      const survivalAgeDays = Math.max(0, (nowMs - item.createdAt) / MILLIS_PER_DAY);
      const effectiveTs = item.lastAccessedAt ?? item.createdAt;
      const clarityAgeDays = Math.max(0, (nowMs - effectiveTs) / MILLIS_PER_DAY);
      const clarityScore = computeCurrentClarity(item, clarityAgeDays);
      if (survivalAgeDays < P2_TO_P1_MIN_AGE_DAYS || clarityScore < P2_TO_P1_MIN_CLARITY) {
        continue;
      }
      promoteP1Stmt.run(TIER_P1, item.id);
      p1PromotionIds.push(item.id);
      skillExtractionCandidates.push(item.id);
      promotedInRun.add(item.id);
    }

    const p0ApprovalCandidates: SoulMemoryPromotionCandidate[] = [];
    for (const row of rows) {
      const item = rowToMemoryItem(row);
      if (item.tier !== TIER_P1 || promotedInRun.has(item.id)) {
        continue;
      }
      const survivalAgeDays = Math.max(0, (nowMs - item.createdAt) / MILLIS_PER_DAY);
      const effectiveTs = item.lastAccessedAt ?? item.createdAt;
      const clarityAgeDays = Math.max(0, (nowMs - effectiveTs) / MILLIS_PER_DAY);
      const clarityScore = computeCurrentClarity(item, clarityAgeDays);
      if (survivalAgeDays < P1_TO_P0_MIN_AGE_DAYS || clarityScore < P1_TO_P0_MIN_CLARITY) {
        continue;
      }
      const distinctScopeHits = scopeHitSummary.get(item.id)?.distinctScopeHits ?? 0;
      const candidate: SoulMemoryPromotionCandidate = {
        ...item,
        ageDays: survivalAgeDays,
        clarityScore,
        distinctScopeHits,
      };
      const approved = autoApproveP0 || approvedP0Ids.has(item.id);
      if (approved) {
        promoteP0Stmt.run(TIER_P0, SOURCE_CORE_PREFERENCE, item.id);
        p0PromotionIds.push(item.id);
        continue;
      }
      p0ApprovalCandidates.push(candidate);
    }

    p0ApprovalCandidates.sort((a, b) => {
      const clarityDelta = b.clarityScore - a.clarityScore;
      if (Math.abs(clarityDelta) > 1e-8) {
        return clarityDelta;
      }
      return b.ageDays - a.ageDays;
    });

    return {
      promotedToP1: p1PromotionIds.length,
      promotedToP0: p0PromotionIds.length,
      p1PromotionIds,
      p0PromotionIds,
      p0ApprovalCandidates,
      skillExtractionCandidates,
    };
  } finally {
    db.close();
  }
}

export function applySoulMemoryConfidenceDecay(params: {
  agentId: string;
  scopeType?: string;
  scopeId?: string;
  nowMs?: number;
}): { updated: number; deleted: number } {
  const db = openSoulMemoryDb(params.agentId);
  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  try {
    const clauses: string[] = [];
    const values: string[] = [];
    if (params.scopeType?.trim()) {
      clauses.push("scope_type = ?");
      values.push(normalizeScopeValue(params.scopeType));
    }
    if (params.scopeId?.trim()) {
      clauses.push("scope_id = ?");
      values.push(normalizeScopeValue(params.scopeId));
    }
    const where = clauses.length > 0 ? ` WHERE ${clauses.join(" AND ")}` : "";
    const rows = db
      .prepare(
        "SELECT id, scope_type, scope_id, kind, content, embedding_json, confidence, tier, source, created_at, last_accessed_at " +
          `FROM memory_items${where}`,
      )
      .all(...values) as MemoryItemRow[];

    let updated = 0;
    let deleted = 0;
    for (const row of rows) {
      const item = rowToMemoryItem(row);
      const effectiveTs = item.lastAccessedAt ?? item.createdAt;
      const ageDays = Math.max(0, (nowMs - effectiveTs) / MILLIS_PER_DAY);
      const decayed = computeCurrentClarity(item, ageDays);
      if (shouldPruneMemoryItem(item, ageDays)) {
        db.prepare("DELETE FROM memory_items WHERE id = ?").run(item.id);
        deleted += 1;
        continue;
      }
      if (Math.abs(decayed - item.confidence) <= 1e-8) {
        continue;
      }
      db.prepare("UPDATE memory_items SET confidence = ? WHERE id = ?").run(decayed, item.id);
      updated += 1;
    }
    return { updated, deleted };
  } finally {
    db.close();
  }
}
