import crypto from "node:crypto";
import fsSync from "node:fs";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { resolveStateDir } from "../../core/config/paths.js";
import { truncateUtf16Safe } from "../../utils.js";
import {
  REFERENCE_BOOST_WEIGHT,
  REFERENCE_EDGE_DECAY,
  REFERENCE_EXPANSION_ENABLED,
  REFERENCE_MAX_BOOST,
  REFERENCE_MAX_HOPS,
  REFERENCE_SEED_TOPK_MULTIPLIER,
  SOUL_MEMORY_REF_TABLE,
  applyReferenceExpansion,
  loadReferencesBySourceIds,
  upsertItemReferences,
} from "../salience/reference-expansion.js";
import {
  REINFORCEMENT_LOG_WEIGHT,
  SOUL_MEMORY_SCOPE_HITS_TABLE,
  computeReinforcementFactor,
  recordScopeHits,
  reinforceRetrievedItems,
} from "../salience/reinforcement.js";
import {
  CROSS_SCOPE_PENALTY,
  FORGET_CONFIDENCE_THRESHOLD,
  FORGET_STREAK_HALF_LIVES,
  MATCH_SCOPE_PENALTY,
  P0_SCOPE_PENALTY,
  SCORE_DECAY_WEIGHT,
  SCORE_SIMILARITY_WEIGHT,
  clarityDecayFactor,
  computeCurrentClarity,
  computeFusionSemanticMatch,
  computeWeightedScore,
  resolveScopePenalty,
  shouldPruneMemoryItem,
} from "../salience/salience-compute.js";
import { requireNodeSqlite } from "./sqlite.js";

const MILLIS_PER_DAY = 24 * 60 * 60 * 1000;
const SOURCE_CORE_PREFERENCE = "core_preference";
const SOURCE_MANUAL_LOG = "manual_log";
const SOURCE_MIGRATION = "migration";
const SOURCE_AUTO_EXTRACTION = "auto_extraction";
const SOURCE_RUNTIME_EVENT = "runtime_event";

const TIER_P0 = "P0";
const TIER_P1 = "P1";
const TIER_P2 = "P2";
const TIER_P3 = "P3";

const RECORD_KIND_FACT = "fact";
const RECORD_KIND_RELATIONSHIP = "relationship";
const RECORD_KIND_EXPERIENCE = "experience";
const RECORD_KIND_SOUL = "soul";

const DEFAULT_INJECT_THRESHOLD = 0.65;
const EMBEDDING_DIMS = 128;
const SOUL_MEMORY_PATH_PREFIX = "soul-memory/";
const SOUL_ARCHIVE_PATH_PREFIX = "soul-archive/";
const SOUL_MEMORY_FTS_TABLE = "memory_items_fts";
const SOUL_MEMORY_VEC_TABLE = "memory_items_vec";
const SOUL_MEMORY_ENTITY_TABLE = "memory_item_entities";
const MEMORY_ITEM_SELECT_COLUMNS =
  "id, scope_type, scope_id, kind, content, embedding_json, confidence, tier, source, " +
  "record_kind, summary, metadata_json, created_at, last_accessed_at, reinforcement_count, last_reinforced_at, " +
  "memory_type, valid_from, valid_until, source_detail, is_compacted, semantic_key";
const SOUL_ARCHIVE_TABLE = "memory_archive";
const SOUL_ARCHIVE_FTS_TABLE = "memory_archive_fts";
const RRF_RANK_CONSTANT = 40;
const VECTOR_RRF_WEIGHT = 0.6;
const BM25_RRF_WEIGHT = 0.4;
const CANDIDATE_FULL_SCAN_MAX_ROWS = 2000;
const CANDIDATE_MIN_LIMIT = 160;
const CANDIDATE_MAX_LIMIT = 960;
const CANDIDATE_PER_TOPK_MULTIPLIER = 24;
const RECENT_CANDIDATE_SHARE = 0.35;

const require = createRequire(import.meta.url);

type SoulVectorState = {
  available: boolean;
  attempted: boolean;
  dims?: number;
  extensionPath?: string;
  loadError?: string;
};

const soulVectorStateByDbPath = new Map<string, SoulVectorState>();

type SourceProfile = {
  confidence: number;
  tier: SoulMemoryTier;
};

// All sources now map to P3. Tier promotion is replaced by EXPERIENCE.md distillation.
const SOURCE_PROFILE: Record<SoulMemorySource, SourceProfile> = {
  [SOURCE_CORE_PREFERENCE]: { confidence: 0.95, tier: TIER_P3 },
  [SOURCE_MANUAL_LOG]: { confidence: 0.85, tier: TIER_P3 },
  [SOURCE_MIGRATION]: { confidence: 0.85, tier: TIER_P3 },
  [SOURCE_AUTO_EXTRACTION]: { confidence: 0.5, tier: TIER_P3 },
  [SOURCE_RUNTIME_EVENT]: { confidence: 0.5, tier: TIER_P3 },
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
  record_kind: string;
  summary: string | null;
  metadata_json: string | null;
  created_at: number;
  last_accessed_at: number | null;
  reinforcement_count: number;
  last_reinforced_at: number | null;
  // P3 compaction fields
  memory_type: string | null;
  valid_from: number | null;
  valid_until: number | null;
  source_detail: string | null;
  is_compacted: number | null;
  semantic_key: string | null;
};

type ArchiveRow = {
  id: string;
  scope_type: string;
  scope_id: string;
  kind: string;
  record_kind: string;
  content: string;
  summary: string;
  embedding_json: string;
  source: string;
  created_at: number;
  active_memory_id: string | null;
  metadata_json: string | null;
};

export type SoulMemorySource =
  | typeof SOURCE_CORE_PREFERENCE
  | typeof SOURCE_MANUAL_LOG
  | typeof SOURCE_MIGRATION
  | typeof SOURCE_AUTO_EXTRACTION
  | typeof SOURCE_RUNTIME_EVENT;

export type SoulMemoryTier = typeof TIER_P0 | typeof TIER_P1 | typeof TIER_P2 | typeof TIER_P3;

export type SoulMemoryRecordKind =
  | typeof RECORD_KIND_FACT
  | typeof RECORD_KIND_RELATIONSHIP
  | typeof RECORD_KIND_EXPERIENCE
  | typeof RECORD_KIND_SOUL;

export type SoulMemoryScope = {
  scopeType: string;
  scopeId: string;
  weight: number;
};

export type SoulMemoryType = "episodic" | "semantic";
export type SoulMemorySourceDetail = "explicit" | "inferred" | "system";

export type SoulMemoryItem = {
  id: string;
  scopeType: string;
  scopeId: string;
  kind: string;
  content: string;
  summary?: string;
  confidence: number;
  tier: SoulMemoryTier;
  source: SoulMemorySource;
  recordKind: SoulMemoryRecordKind;
  metadata?: Record<string, unknown>;
  createdAt: number;
  lastAccessedAt: number | null;
  reinforcementCount: number;
  lastReinforcedAt: number | null;
  // P3 compaction fields
  memoryType: SoulMemoryType;
  validFrom: number | null;
  validUntil: number | null;
  sourceDetail: SoulMemorySourceDetail;
  isCompacted: boolean;
  semanticKey: string | null;
};

export type SoulArchiveEvent = {
  id: string;
  scopeType: string;
  scopeId: string;
  kind: string;
  recordKind: SoulMemoryRecordKind;
  content: string;
  summary: string;
  source: SoulMemorySource;
  createdAt: number;
  activeMemoryId?: string;
  metadata?: Record<string, unknown>;
};

export type SoulArchiveQueryResult = SoulArchiveEvent & {
  score: number;
  vectorScore: number;
  lexicalScore: number;
  ageDays: number;
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
  salienceScore: number;
  salienceDecay: number;
  salienceReinforcement: number;
  reinforcementFactor: number;
  referenceBoost: number;
  references: string[];
  ageDays: number;
  /** IDs of unresolved memory_conflicts involving this item. Empty if none. */
  conflictIds: string[];
};

export type SoulMemoryConfig = {
  forgetConfidenceThreshold?: number;
  forgetStreakHalfLives?: number;
  crossScopePenalty?: number;
  matchScopePenalty?: number;
  scoreSimilarityWeight?: number;
  scoreDecayWeight?: number;
  reinforcementLogWeight?: number;
  referenceExpansionEnabled?: boolean;
  referenceMaxHops?: number;
  referenceEdgeDecay?: number;
  referenceBoostWeight?: number;
  referenceMaxBoost?: number;
  referenceSeedTopKMultiplier?: number;
  p3Compaction?: {
    enabled?: boolean;
    minClusterSize?: number;
    similarityMin?: number;
    similarityMax?: number;
    archiveAgeDays?: number;
    orphanAgeDays?: number;
    compactedDiscount?: number;
    batchLimit?: number;
  };
};

export type ResolvedP3CompactionConfig = {
  enabled: boolean;
  minClusterSize: number;
  similarityMin: number;
  similarityMax: number;
  archiveAgeDays: number;
  orphanAgeDays: number;
  compactedDiscount: number;
  /** Max episodic items to load per compaction run (prevents unbounded memory). */
  batchLimit: number;
};

type ResolvedSoulMemoryConfig = {
  forgetConfidenceThreshold: number;
  forgetStreakHalfLives: number;
  // Scope penalties (kept for scope-aware ranking)
  p0ScopePenalty: number;
  crossScopePenalty: number;
  matchScopePenalty: number;
  // Clarity half-life fields (all Infinity — no decay; kept for ClarityDecayConfig compat)
  p0ClarityHalfLifeDays: number;
  p1ClarityHalfLifeDays: number;
  p2ClarityHalfLifeDays: number;
  p3ClarityHalfLifeDays: number;
  scoreSimilarityWeight: number;
  scoreDecayWeight: number;
  reinforcementLogWeight: number;
  referenceExpansionEnabled: boolean;
  referenceMaxHops: number;
  referenceEdgeDecay: number;
  referenceBoostWeight: number;
  referenceMaxBoost: number;
  referenceSeedTopKMultiplier: number;
  p3Compaction: ResolvedP3CompactionConfig;
};

export function buildSoulMemoryPath(itemId: string): string {
  return `${SOUL_MEMORY_PATH_PREFIX}${itemId}`;
}

export function buildSoulArchivePath(eventId: string): string {
  return `${SOUL_ARCHIVE_PATH_PREFIX}${eventId}`;
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

export function parseSoulArchivePath(input: string): string | null {
  const normalized = input.trim().replace(/^\/+/, "");
  if (!normalized.toLowerCase().startsWith(SOUL_ARCHIVE_PATH_PREFIX)) {
    return null;
  }
  const eventId = normalized.slice(SOUL_ARCHIVE_PATH_PREFIX.length).trim();
  if (!/^arch_[a-z0-9]+$/i.test(eventId)) {
    return null;
  }
  return eventId;
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
  summary?: string;
  confidence?: number;
  source?: string;
  tier?: SoulMemoryTier;
  recordKind?: SoulMemoryRecordKind;
  metadata?: Record<string, unknown>;
  nowMs?: number;
  soulConfig?: SoulMemoryConfig;
  /** Memory type: 'episodic' (default), 'semantic', or 'knowledge'. Knowledge items are exempt from P3 compaction/archival. */
  memoryType?: "episodic" | "semantic" | "knowledge";
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
  const scopeType = normalizeScopeValue(params.scopeType);
  const scopeId = normalizeScopeValue(params.scopeId);
  const kind = normalizeScopeValue(params.kind);
  const summary = normalizeOptionalSummary(params.summary, content);
  const recordKind = resolveRecordKind(params.recordKind, kind, source);
  const metadataJson = stringifyMetadata(params.metadata);
  const sourceProfile = SOURCE_PROFILE[source];
  const explicitTier = params.tier ? normalizeTier(params.tier) : null;
  const resolvedTier = explicitTier ?? sourceProfile.tier;
  const resolvedConfidence = Math.max(
    sourceProfile.confidence,
    clamp(params.confidence ?? sourceProfile.confidence, 0, 1),
  );
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
      const nextConfidence = Math.max(existing.confidence, resolvedConfidence);
      const nextTier = resolvedTier;
      const nextSource = source;
      db.prepare(
        "UPDATE memory_items SET confidence = ?, tier = ?, source = ?, record_kind = ?, summary = ?, metadata_json = ?, " +
          "reinforcement_count = COALESCE(reinforcement_count, 1) + 1, last_reinforced_at = ? " +
          "WHERE id = ?",
      ).run(
        nextConfidence,
        nextTier,
        nextSource,
        recordKind,
        summary,
        metadataJson,
        nowMs,
        existing.id,
      );
      return getSoulMemoryItemInternal(db, existing.id);
    }

    const embeddingVec = embedText(content);
    const embedding = JSON.stringify(embeddingVec);
    const id = `mem_${crypto.randomUUID().replace(/-/g, "")}`;
    const sourceDetail = deriveSourceDetail(source);
    const resolvedMemoryType = params.memoryType ?? "episodic";
    db.prepare(
      "INSERT INTO memory_items (" +
        "id, scope_type, scope_id, kind, content, embedding_json, confidence, tier, source, record_kind, summary, metadata_json, " +
        "created_at, last_accessed_at, reinforcement_count, last_reinforced_at, " +
        "memory_type, source_detail" +
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 1, ?, ?, ?)",
    ).run(
      id,
      scopeType,
      scopeId,
      kind,
      content,
      embedding,
      resolvedConfidence,
      resolvedTier,
      source,
      recordKind,
      summary,
      metadataJson,
      nowMs,
      nowMs,
      resolvedMemoryType,
      sourceDetail,
    );
    upsertItemEntities(db, id, content);
    upsertItemReferences(db, id, content, nowMs);
    upsertSoulMemoryVector(db, id, embeddingVec);
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

export function deleteSoulMemoryScopeItems(params: {
  agentId: string;
  scopeType: string;
  scopeId: string;
}): number {
  const db = openSoulMemoryDb(params.agentId);
  try {
    const rows = db
      .prepare("SELECT id FROM memory_items WHERE scope_type = ? AND scope_id = ?")
      .all(normalizeScopeValue(params.scopeType), normalizeScopeValue(params.scopeId)) as Array<{
      id?: string;
    }>;
    const ids = rows
      .map((row) => (typeof row.id === "string" ? row.id.trim() : ""))
      .filter(Boolean);
    return pruneSoulMemoryItems(db, ids);
  } finally {
    db.close();
  }
}

export function getSoulArchiveEvent(params: {
  agentId: string;
  eventId: string;
}): SoulArchiveEvent | null {
  const db = openSoulMemoryDb(params.agentId);
  try {
    const row = db
      .prepare(
        `SELECT id, scope_type, scope_id, kind, record_kind, content, summary, embedding_json, source, created_at, active_memory_id, metadata_json ` +
          `FROM ${SOUL_ARCHIVE_TABLE} WHERE id = ?`,
      )
      .get(params.eventId) as ArchiveRow | undefined;
    return row ? rowToArchiveEvent(row) : null;
  } finally {
    db.close();
  }
}

export function listSoulMemoryItems(params: {
  agentId: string;
  scopeType?: string;
  scopeId?: string;
  kind?: string;
  tier?: SoulMemoryTier;
  recordKind?: SoulMemoryRecordKind;
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
    if (params.tier?.trim()) {
      clauses.push("tier = ?");
      values.push(normalizeTier(params.tier));
    }
    if (params.recordKind?.trim()) {
      clauses.push("record_kind = ?");
      values.push(normalizeRecordKind(params.recordKind));
    }
    const where = clauses.length > 0 ? `WHERE ${clauses.join(" AND ")}` : "";
    const limit = Math.max(1, Math.min(500, Math.floor(params.limit ?? 200)));
    const rows = db
      .prepare(
        `SELECT ${MEMORY_ITEM_SELECT_COLUMNS} FROM memory_items ` +
          `${where} ORDER BY created_at DESC LIMIT ?`,
      )
      .all(...values, limit) as MemoryItemRow[];
    return rows.map((row) => rowToMemoryItem(row));
  } finally {
    db.close();
  }
}

export function countSoulMemoryItemsByTier(params: {
  agentId: string;
}): Record<SoulMemoryTier, number> {
  const db = openSoulMemoryDb(params.agentId);
  try {
    type Row = { tier?: string; count?: number };
    const rows = db
      .prepare("SELECT tier, COUNT(*) as count FROM memory_items GROUP BY tier")
      .all() as Row[];
    const counts: Record<SoulMemoryTier, number> = {
      P0: 0,
      P1: 0,
      P2: 0,
      P3: 0,
    };
    for (const row of rows) {
      counts[normalizeTier(String(row.tier ?? "P1"))] = Number(row.count ?? 0);
    }
    return counts;
  } finally {
    db.close();
  }
}

export function countSoulMemoryItemsByRecordKind(params: {
  agentId: string;
}): Record<SoulMemoryRecordKind, number> {
  const db = openSoulMemoryDb(params.agentId);
  try {
    type Row = { record_kind?: string; count?: number };
    const rows = db
      .prepare("SELECT record_kind, COUNT(*) as count FROM memory_items GROUP BY record_kind")
      .all() as Row[];
    const counts: Record<SoulMemoryRecordKind, number> = {
      fact: 0,
      relationship: 0,
      experience: 0,
      soul: 0,
    };
    for (const row of rows) {
      counts[normalizeRecordKind(String(row.record_kind ?? RECORD_KIND_EXPERIENCE))] = Number(
        row.count ?? 0,
      );
    }
    return counts;
  } finally {
    db.close();
  }
}

export function countSoulArchiveEvents(params: { agentId: string }): number {
  const db = openSoulMemoryDb(params.agentId);
  try {
    const row = db.prepare(`SELECT COUNT(*) as c FROM ${SOUL_ARCHIVE_TABLE}`).get() as
      | { c?: number }
      | undefined;
    return Number(row?.c ?? 0);
  } finally {
    db.close();
  }
}

export function listSoulMemoryReferences(params: { agentId: string; itemId: string }): string[] {
  const db = openSoulMemoryDb(params.agentId);
  try {
    type Row = { target_memory_id?: string };
    const rows = db
      .prepare(
        `SELECT target_memory_id FROM ${SOUL_MEMORY_REF_TABLE} ` +
          "WHERE source_memory_id = ? ORDER BY target_memory_id ASC",
      )
      .all(params.itemId) as Row[];
    return rows
      .map((row) => String(row.target_memory_id ?? "").trim())
      .filter((value) => Boolean(value));
  } finally {
    db.close();
  }
}

export function ingestSoulMemoryEvent(params: {
  agentId: string;
  scopeType: string;
  scopeId: string;
  archiveScopeType?: string;
  archiveScopeId?: string;
  kind: string;
  content: string;
  summary?: string;
  source?: string;
  metadata?: Record<string, unknown>;
  nowMs?: number;
  recordKind?: SoulMemoryRecordKind;
  /** When true, skip the immediate archive write. P3 items will be archived
   *  later when they are pruned from active memory. */
  skipArchive?: boolean;
}): { activeItem: SoulMemoryItem; archiveEvent?: SoulArchiveEvent } | null {
  const activeItem = writeSoulMemory({
    agentId: params.agentId,
    scopeType: params.scopeType,
    scopeId: params.scopeId,
    kind: params.kind,
    content: params.content,
    summary: params.summary,
    source: params.source ?? SOURCE_RUNTIME_EVENT,
    tier: TIER_P3,
    recordKind: params.recordKind,
    metadata: params.metadata,
    confidence: SOURCE_PROFILE[SOURCE_RUNTIME_EVENT].confidence,
    nowMs: params.nowMs,
  });
  if (!activeItem) {
    return null;
  }
  if (params.skipArchive) {
    return { activeItem };
  }
  const archiveEvent = writeSoulArchiveEvent({
    agentId: params.agentId,
    scopeType: params.archiveScopeType ?? params.scopeType,
    scopeId: params.archiveScopeId ?? params.scopeId,
    kind: params.kind,
    content: params.content,
    summary: params.summary,
    source: normalizeSource(params.source ?? SOURCE_RUNTIME_EVENT),
    recordKind: params.recordKind ?? activeItem.recordKind,
    metadata: params.metadata,
    activeMemoryId: activeItem.id,
    nowMs: params.nowMs,
  });
  return { activeItem, archiveEvent };
}

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
    const bm25ById = searchByBm25({
      db,
      query: cleanedQuery,
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
    const staleMemoryIds: string[] = [];
    for (const row of rows) {
      const item = rowToMemoryItem(row);
      const effectiveTs = item.lastAccessedAt ?? item.createdAt;
      const ageDays = Math.max(0, (nowMs - effectiveTs) / MILLIS_PER_DAY);
      // No tier-based TTL exemption: all items are P3. TTL applies uniformly.
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

      const decayFactor = clarityDecayFactor(item.tier, ageDays, soulConfig);
      const clarityScore = computeCurrentClarity(item, ageDays, soulConfig);
      if (shouldPruneMemoryItem(item, ageDays, soulConfig)) {
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
      const salienceScore = salienceReinforcement; // score = relevance × scope × reinforcement
      const tierMultiplier = 1; // All items are P3
      // Discount compacted episodic items so semantic distillations rank higher
      const compactedFactor =
        candidate.item.isCompacted && candidate.item.memoryType === "episodic"
          ? soulConfig.p3Compaction.compactedDiscount
          : 1;
      // Simplified scoring: relevance × scope × reinforcement
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
  const queryVec = embedText(cleanedQuery);
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

function countMemoryItemsInternal(db: DatabaseSync): number {
  const row = db.prepare("SELECT COUNT(*) as c FROM memory_items").get() as
    | { c?: number }
    | undefined;
  return Number(row?.c ?? 0);
}

function resolveCandidateLimit(topK: number): number {
  const scaled = Math.max(CANDIDATE_MIN_LIMIT, Math.floor(topK * CANDIDATE_PER_TOPK_MULTIPLIER));
  return Math.min(CANDIDATE_MAX_LIMIT, scaled);
}

function addCandidateIds(target: Set<string>, ids: Iterable<string>, maxSize: number): void {
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

function buildCandidateMemoryIds(params: {
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

function loadMemoryRowsByIds(db: DatabaseSync, memoryIds: string[]): MemoryItemRow[] {
  if (memoryIds.length === 0) {
    return [];
  }
  const placeholders = memoryIds.map(() => "?").join(", ");
  return db
    .prepare(`SELECT ${MEMORY_ITEM_SELECT_COLUMNS} FROM memory_items WHERE id IN (${placeholders})`)
    .all(...memoryIds) as MemoryItemRow[];
}

function writeSoulArchiveEvent(params: {
  agentId: string;
  scopeType: string;
  scopeId: string;
  kind: string;
  content: string;
  summary?: string;
  source: SoulMemorySource;
  recordKind: SoulMemoryRecordKind;
  metadata?: Record<string, unknown>;
  activeMemoryId?: string;
  nowMs?: number;
}): SoulArchiveEvent {
  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  const db = openSoulMemoryDb(params.agentId);
  try {
    const id = `arch_${crypto.randomUUID().replace(/-/g, "")}`;
    const normalizedSummary = normalizeOptionalSummary(params.summary, params.content) ?? "";
    const embedding = JSON.stringify(embedText(`${normalizedSummary}\n${params.content}`));
    db.prepare(
      `INSERT INTO ${SOUL_ARCHIVE_TABLE} (` +
        "id, scope_type, scope_id, kind, record_kind, content, summary, embedding_json, source, created_at, active_memory_id, metadata_json" +
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    ).run(
      id,
      normalizeScopeValue(params.scopeType),
      normalizeScopeValue(params.scopeId),
      normalizeScopeValue(params.kind),
      normalizeRecordKind(params.recordKind),
      params.content.trim(),
      normalizedSummary,
      embedding,
      normalizeSource(params.source),
      nowMs,
      params.activeMemoryId?.trim() || null,
      stringifyMetadata(params.metadata),
    );
    const row = db
      .prepare(
        `SELECT id, scope_type, scope_id, kind, record_kind, content, summary, embedding_json, source, created_at, active_memory_id, metadata_json ` +
          `FROM ${SOUL_ARCHIVE_TABLE} WHERE id = ?`,
      )
      .get(id) as ArchiveRow | undefined;
    return rowToArchiveEvent(
      row ?? {
        id,
        scope_type: normalizeScopeValue(params.scopeType),
        scope_id: normalizeScopeValue(params.scopeId),
        kind: normalizeScopeValue(params.kind),
        record_kind: params.recordKind,
        content: params.content.trim(),
        summary: normalizedSummary,
        embedding_json: embedding,
        source: params.source,
        created_at: nowMs,
        active_memory_id: params.activeMemoryId ?? null,
        metadata_json: stringifyMetadata(params.metadata),
      },
    );
  } finally {
    db.close();
  }
}

function loadArchiveRowsByScope(
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

function listRecentMemoryIds(params: {
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
  const db = new DatabaseSync(dbPath, { allowExtension: true });
  db.exec("PRAGMA foreign_keys = ON;");
  ensureSoulMemorySchema(db, dbPath);
  return db;
}

function ensureSoulMemorySchema(db: DatabaseSync, dbPath: string): void {
  db.exec(
    "CREATE TABLE IF NOT EXISTS memory_items (" +
      "id TEXT PRIMARY KEY, " +
      "scope_type TEXT NOT NULL, " +
      "scope_id TEXT NOT NULL, " +
      "kind TEXT NOT NULL, " +
      "content TEXT NOT NULL, " +
      "embedding_json TEXT NOT NULL, " +
      "confidence REAL NOT NULL, " +
      "tier TEXT NOT NULL DEFAULT 'P3', " +
      "source TEXT NOT NULL DEFAULT 'manual_log', " +
      "record_kind TEXT NOT NULL DEFAULT 'experience', " +
      "summary TEXT, " +
      "metadata_json TEXT, " +
      "created_at INTEGER NOT NULL, " +
      "last_accessed_at INTEGER, " +
      "reinforcement_count INTEGER NOT NULL DEFAULT 1, " +
      "last_reinforced_at INTEGER" +
      ");",
  );
  ensureMemoryItemsColumn(db, "reinforcement_count", "INTEGER NOT NULL DEFAULT 1");
  ensureMemoryItemsColumn(db, "last_reinforced_at", "INTEGER");
  ensureMemoryItemsColumn(db, "record_kind", "TEXT NOT NULL DEFAULT 'experience'");
  ensureMemoryItemsColumn(db, "summary", "TEXT");
  ensureMemoryItemsColumn(db, "metadata_json", "TEXT");
  // P3 compaction columns
  ensureMemoryItemsColumn(db, "memory_type", "TEXT NOT NULL DEFAULT 'episodic'");
  ensureMemoryItemsColumn(db, "valid_from", "INTEGER");
  ensureMemoryItemsColumn(db, "valid_until", "INTEGER");
  ensureMemoryItemsColumn(db, "source_detail", "TEXT NOT NULL DEFAULT 'inferred'");
  ensureMemoryItemsColumn(db, "is_compacted", "INTEGER NOT NULL DEFAULT 0");
  ensureMemoryItemsColumn(db, "semantic_key", "TEXT");
  db.exec(
    "UPDATE memory_items SET reinforcement_count = 1 " +
      "WHERE reinforcement_count IS NULL OR reinforcement_count < 1",
  );
  db.exec(
    "UPDATE memory_items SET record_kind = 'experience' " +
      "WHERE record_kind IS NULL OR TRIM(record_kind) = ''",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_soul_memory_scope ON memory_items (scope_type, scope_id);",
  );
  db.exec("CREATE INDEX IF NOT EXISTS idx_soul_memory_tier ON memory_items (tier);");
  db.exec("CREATE INDEX IF NOT EXISTS idx_soul_memory_source ON memory_items (source);");
  db.exec("CREATE INDEX IF NOT EXISTS idx_soul_memory_record_kind ON memory_items (record_kind);");
  // P3 compaction indexes
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_items (memory_type) WHERE tier = 'P3';",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_memory_compacted ON memory_items (is_compacted) " +
      "WHERE tier = 'P3' AND memory_type = 'episodic';",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_memory_semantic_key ON memory_items (semantic_key) " +
      "WHERE memory_type = 'semantic' AND valid_until IS NULL;",
  );
  // Memory lineage table (append-only, not managed by upsertItemReferences)
  db.exec(
    "CREATE TABLE IF NOT EXISTS memory_lineage (" +
      "id INTEGER PRIMARY KEY AUTOINCREMENT, " +
      "source_id TEXT NOT NULL, " +
      "target_id TEXT NOT NULL, " +
      "edge_type TEXT NOT NULL, " +
      "created_at INTEGER NOT NULL" +
      ");",
  );
  db.exec(
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_lineage_edge ON memory_lineage (source_id, target_id, edge_type);",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_lineage_target ON memory_lineage (target_id, edge_type);",
  );
  // Backfill source_detail from source column for existing rows
  backfillSourceDetail(db);
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
  db.exec(
    `CREATE TABLE IF NOT EXISTS ${SOUL_MEMORY_REF_TABLE} (` +
      "source_memory_id TEXT NOT NULL, " +
      "target_memory_id TEXT NOT NULL, " +
      "created_at INTEGER NOT NULL, " +
      "PRIMARY KEY (source_memory_id, target_memory_id), " +
      "FOREIGN KEY (source_memory_id) REFERENCES memory_items(id) ON DELETE CASCADE" +
      ");",
  );
  db.exec(
    `CREATE INDEX IF NOT EXISTS idx_soul_memory_refs_target ON ${SOUL_MEMORY_REF_TABLE} (target_memory_id);`,
  );
  db.exec(
    `CREATE TABLE IF NOT EXISTS ${SOUL_ARCHIVE_TABLE} (` +
      "id TEXT PRIMARY KEY, " +
      "scope_type TEXT NOT NULL, " +
      "scope_id TEXT NOT NULL, " +
      "kind TEXT NOT NULL, " +
      "record_kind TEXT NOT NULL DEFAULT 'fact', " +
      "content TEXT NOT NULL, " +
      "summary TEXT NOT NULL, " +
      "embedding_json TEXT NOT NULL, " +
      "source TEXT NOT NULL DEFAULT 'runtime_event', " +
      "created_at INTEGER NOT NULL, " +
      "active_memory_id TEXT, " +
      "metadata_json TEXT, " +
      "FOREIGN KEY (active_memory_id) REFERENCES memory_items(id) ON DELETE SET NULL" +
      ");",
  );
  db.exec(
    `CREATE INDEX IF NOT EXISTS idx_soul_archive_scope ON ${SOUL_ARCHIVE_TABLE} (scope_type, scope_id);`,
  );
  db.exec(`CREATE INDEX IF NOT EXISTS idx_soul_archive_kind ON ${SOUL_ARCHIVE_TABLE} (kind);`);
  db.exec(
    `CREATE INDEX IF NOT EXISTS idx_soul_archive_record_kind ON ${SOUL_ARCHIVE_TABLE} (record_kind);`,
  );
  db.exec(
    `CREATE INDEX IF NOT EXISTS idx_soul_archive_created ON ${SOUL_ARCHIVE_TABLE} (created_at DESC);`,
  );
  ensureSoulMemoryFts(db);
  ensureSoulArchiveFts(db);
  ensureSoulMemoryVec(db, dbPath);
}

function ensureMemoryItemsColumn(db: DatabaseSync, column: string, definition: string): void {
  const rows = db.prepare("PRAGMA table_info(memory_items)").all() as Array<{ name?: string }>;
  if (rows.some((row) => row.name === column)) {
    return;
  }
  db.exec(`ALTER TABLE memory_items ADD COLUMN ${column} ${definition}`);
}

/** One-time backfill: derive source_detail from existing source column. */
function backfillSourceDetail(db: DatabaseSync): void {
  // Only run if there are rows still at default 'inferred' that should be 'explicit' or 'system'
  const needsBackfill = db
    .prepare(
      "SELECT COUNT(*) as cnt FROM memory_items " +
        "WHERE source_detail = 'inferred' AND source IN ('manual_log', 'core_preference', 'migration')",
    )
    .get() as { cnt: number } | undefined;
  if (!needsBackfill || needsBackfill.cnt === 0) {
    return;
  }
  db.exec(
    "UPDATE memory_items SET source_detail = 'explicit' " +
      "WHERE source IN ('manual_log', 'core_preference') AND source_detail = 'inferred'",
  );
  db.exec(
    "UPDATE memory_items SET source_detail = 'system' " +
      "WHERE source = 'migration' AND source_detail = 'inferred'",
  );
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

function ensureSoulArchiveFts(db: DatabaseSync): void {
  try {
    const tableExists = hasTable(db, SOUL_ARCHIVE_FTS_TABLE);
    if (!tableExists) {
      db.exec(
        `CREATE VIRTUAL TABLE ${SOUL_ARCHIVE_FTS_TABLE} USING fts5(` +
          "id UNINDEXED, " +
          "summary, " +
          "content, " +
          `content='${SOUL_ARCHIVE_TABLE}', ` +
          "content_rowid='rowid', " +
          "tokenize='unicode61'" +
          ");",
      );
    }
    db.exec(
      `CREATE TRIGGER IF NOT EXISTS trg_${SOUL_ARCHIVE_TABLE}_ai ` +
        `AFTER INSERT ON ${SOUL_ARCHIVE_TABLE} BEGIN ` +
        `INSERT INTO ${SOUL_ARCHIVE_FTS_TABLE}(rowid, id, summary, content) VALUES (new.rowid, new.id, new.summary, new.content); ` +
        "END;",
    );
    db.exec(
      `CREATE TRIGGER IF NOT EXISTS trg_${SOUL_ARCHIVE_TABLE}_ad ` +
        `AFTER DELETE ON ${SOUL_ARCHIVE_TABLE} BEGIN ` +
        `INSERT INTO ${SOUL_ARCHIVE_FTS_TABLE}(${SOUL_ARCHIVE_FTS_TABLE}, rowid, id, summary, content) VALUES ('delete', old.rowid, old.id, old.summary, old.content); ` +
        "END;",
    );
    db.exec(
      `CREATE TRIGGER IF NOT EXISTS trg_${SOUL_ARCHIVE_TABLE}_au ` +
        `AFTER UPDATE OF id, summary, content ON ${SOUL_ARCHIVE_TABLE} BEGIN ` +
        `INSERT INTO ${SOUL_ARCHIVE_FTS_TABLE}(${SOUL_ARCHIVE_FTS_TABLE}, rowid, id, summary, content) VALUES ('delete', old.rowid, old.id, old.summary, old.content); ` +
        `INSERT INTO ${SOUL_ARCHIVE_FTS_TABLE}(rowid, id, summary, content) VALUES (new.rowid, new.id, new.summary, new.content); ` +
        "END;",
    );
    if (!tableExists) {
      db.exec(
        `INSERT INTO ${SOUL_ARCHIVE_FTS_TABLE}(${SOUL_ARCHIVE_FTS_TABLE}) VALUES ('rebuild');`,
      );
    }
  } catch {
    // Keep archive retrieval available via structural scan when FTS is unavailable.
  }
}

function ensureSoulMemoryVec(db: DatabaseSync, dbPath: string): void {
  if (!ensureSoulVectorReady(db, dbPath, EMBEDDING_DIMS)) {
    return;
  }
  const hadTable = hasSoulMemoryVecTable(db);
  db.exec(
    `CREATE VIRTUAL TABLE IF NOT EXISTS ${SOUL_MEMORY_VEC_TABLE} USING vec0(` +
      "id TEXT PRIMARY KEY, " +
      `embedding FLOAT[${EMBEDDING_DIMS}]` +
      ")",
  );
  if (!hadTable) {
    rebuildSoulMemoryVectors(db);
  }
}

function vectorToBlob(embedding: number[]): Buffer {
  return Buffer.from(new Float32Array(embedding).buffer);
}

function ensureSoulVectorReady(db: DatabaseSync, dbPath: string, dimensions: number): boolean {
  const current = soulVectorStateByDbPath.get(dbPath) ?? {
    available: false,
    attempted: false,
  };
  if (current.attempted && !current.available) {
    return false;
  }

  try {
    db.enableLoadExtension(true);
    if (current.available && current.extensionPath) {
      db.loadExtension(current.extensionPath);
      soulVectorStateByDbPath.set(dbPath, {
        ...current,
        attempted: true,
        available: true,
        dims: dimensions,
      });
      return true;
    }

    const sqliteVec = require("sqlite-vec") as {
      load: (database: DatabaseSync) => void;
      getLoadablePath?: () => string;
    };
    const extensionPath =
      typeof sqliteVec.getLoadablePath === "function" ? sqliteVec.getLoadablePath() : undefined;
    sqliteVec.load(db);
    soulVectorStateByDbPath.set(dbPath, {
      attempted: true,
      available: true,
      dims: dimensions,
      extensionPath,
    });
    return true;
  } catch (err) {
    soulVectorStateByDbPath.set(dbPath, {
      attempted: true,
      available: false,
      loadError: err instanceof Error ? err.message : String(err),
    });
    return false;
  }
}

function rebuildSoulMemoryVectors(db: DatabaseSync): void {
  if (!hasSoulMemoryVecTable(db)) {
    return;
  }
  type Row = { id?: string; embedding_json?: string };
  const rows = db.prepare("SELECT id, embedding_json FROM memory_items").all() as Row[];
  if (rows.length === 0) {
    return;
  }
  const deleteStmt = db.prepare(`DELETE FROM ${SOUL_MEMORY_VEC_TABLE} WHERE id = ?`);
  const insertStmt = db.prepare(
    `INSERT INTO ${SOUL_MEMORY_VEC_TABLE} (id, embedding) VALUES (?, ?)`,
  );
  for (const row of rows) {
    const id = String(row.id ?? "").trim();
    if (!id) {
      continue;
    }
    const embedding = parseEmbedding(String(row.embedding_json ?? ""));
    if (embedding.length === 0) {
      continue;
    }
    try {
      deleteStmt.run(id);
      insertStmt.run(id, vectorToBlob(embedding));
    } catch {
      // Keep memory table usable if vector row sync fails for individual records.
    }
  }
}

function hasSoulMemoryFtsTable(db: DatabaseSync): boolean {
  return hasTable(db, SOUL_MEMORY_FTS_TABLE);
}

function hasSoulMemoryVecTable(db: DatabaseSync): boolean {
  return hasTable(db, SOUL_MEMORY_VEC_TABLE);
}

function hasTable(db: DatabaseSync, tableName: string): boolean {
  const row = db
    .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?")
    .get(tableName) as { name?: string } | undefined;
  return row?.name === tableName;
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

function upsertSoulMemoryVector(db: DatabaseSync, memoryId: string, embedding: number[]): void {
  if (!memoryId || embedding.length === 0 || !hasSoulMemoryVecTable(db)) {
    return;
  }
  try {
    db.prepare(`DELETE FROM ${SOUL_MEMORY_VEC_TABLE} WHERE id = ?`).run(memoryId);
    db.prepare(`INSERT INTO ${SOUL_MEMORY_VEC_TABLE} (id, embedding) VALUES (?, ?)`).run(
      memoryId,
      vectorToBlob(embedding),
    );
  } catch {
    // Keep retrieval functional even when sqlite-vec is unavailable at runtime.
  }
}

function deleteSoulMemoryVectorRows(db: DatabaseSync, memoryIds: string[]): void {
  if (memoryIds.length === 0 || !hasSoulMemoryVecTable(db)) {
    return;
  }
  try {
    const stmt = db.prepare(`DELETE FROM ${SOUL_MEMORY_VEC_TABLE} WHERE id = ?`);
    const seen = new Set<string>();
    for (const memoryId of memoryIds) {
      const normalized = memoryId.trim();
      if (!normalized || seen.has(normalized)) {
        continue;
      }
      seen.add(normalized);
      stmt.run(normalized);
    }
  } catch {
    // Ignore vector cleanup failures; primary row deletion still happens on memory_items.
  }
}

function getSoulMemoryItemInternal(db: DatabaseSync, itemId: string): SoulMemoryItem | null {
  const row = db
    .prepare(`SELECT ${MEMORY_ITEM_SELECT_COLUMNS} FROM memory_items WHERE id = ?`)
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
      `SELECT ${MEMORY_ITEM_SELECT_COLUMNS} FROM memory_items ` +
        "WHERE scope_type = ? AND scope_id = ? AND kind = ?",
    )
    .all(params.scopeType, params.scopeId, params.kind) as MemoryItemRow[];
  for (const row of rows) {
    if (normalizeText(row.content) === params.normalizedContent) {
      return rowToMemoryItem(row);
    }
  }
  return null;
}

function pruneSoulMemoryItems(db: DatabaseSync, memoryIds: string[]): number {
  if (memoryIds.length === 0) {
    return 0;
  }
  const seen = new Set<string>();
  const unique: string[] = [];
  for (const memoryId of memoryIds) {
    const trimmed = memoryId.trim();
    if (!trimmed || seen.has(trimmed)) {
      continue;
    }
    seen.add(trimmed);
    unique.push(trimmed);
  }
  if (unique.length === 0) {
    return 0;
  }
  // Archive P3 items before deletion so they remain recallable.
  archiveP3BeforeDelete(db, unique);
  deleteSoulMemoryVectorRows(db, unique);
  const deleteStmt = db.prepare("DELETE FROM memory_items WHERE id = ?");
  let deleted = 0;
  for (const memoryId of unique) {
    const result = deleteStmt.run(memoryId) as { changes?: number };
    deleted += Number(result.changes ?? 0);
  }
  return deleted;
}

/** Migrate P3 items from active memory into the archive table before pruning.
 *  Items sharing the same sessionKey are grouped into a single episode-level
 *  archive event instead of being archived one-by-one. */
function archiveP3BeforeDelete(db: DatabaseSync, memoryIds: string[]): void {
  const selectStmt = db.prepare(
    `SELECT ${MEMORY_ITEM_SELECT_COLUMNS} FROM memory_items WHERE id = ? AND tier = 'P3'`,
  );

  // Collect P3 rows and group by sessionKey.
  type P3Row = MemoryItemRow & { _memoryId: string };
  const sessionGroups = new Map<string, P3Row[]>();
  const ungrouped: P3Row[] = [];
  for (const memoryId of memoryIds) {
    const row = selectStmt.get(memoryId) as MemoryItemRow | undefined;
    if (!row) {
      continue;
    }
    const meta = parseMetadataJson(row.metadata_json);
    const sessionKey = typeof meta?.sessionKey === "string" ? meta.sessionKey.trim() : "";
    const tagged: P3Row = { ...row, _memoryId: memoryId };
    if (sessionKey) {
      const group = sessionGroups.get(sessionKey) ?? [];
      group.push(tagged);
      sessionGroups.set(sessionKey, group);
    } else {
      ungrouped.push(tagged);
    }
  }

  const insertStmt = db.prepare(
    `INSERT OR IGNORE INTO ${SOUL_ARCHIVE_TABLE} (` +
      "id, scope_type, scope_id, kind, record_kind, content, summary, embedding_json, source, created_at, active_memory_id, metadata_json" +
      ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
  );

  // Archive each session group as a single episode event.
  for (const [sessionKey, rows] of sessionGroups) {
    if (rows.length === 0) {
      continue;
    }
    const sorted = rows.toSorted((a, b) => Number(a.created_at) - Number(b.created_at));
    const episode = buildEpisodeContent(sorted);
    const head = sorted[0];
    const archiveId = `arch_${crypto.randomUUID().replace(/-/g, "")}`;
    const memberIds = sorted.map((r) => r._memoryId);
    const episodeMeta = buildEpisodeMetadata(sorted, sessionKey);
    const embedding = JSON.stringify(embedText(episode.content));
    insertStmt.run(
      archiveId,
      String(head.scope_type),
      String(head.scope_id),
      "episode",
      RECORD_KIND_EXPERIENCE,
      episode.content,
      episode.summary,
      embedding,
      SOURCE_RUNTIME_EVENT,
      Number(head.created_at),
      memberIds[0] ?? null,
      stringifyMetadata(episodeMeta),
    );
  }

  // Archive ungrouped P3 items individually (fallback for items without sessionKey).
  for (const row of ungrouped) {
    const archiveId = `arch_${crypto.randomUUID().replace(/-/g, "")}`;
    const summary = normalizeOptionalSummary(row.summary ?? undefined, String(row.content)) ?? "";
    insertStmt.run(
      archiveId,
      String(row.scope_type),
      String(row.scope_id),
      String(row.kind),
      String(row.record_kind ?? RECORD_KIND_EXPERIENCE),
      String(row.content).trim(),
      summary,
      String(row.embedding_json),
      String(row.source),
      Number(row.created_at),
      row._memoryId,
      row.metadata_json ?? null,
    );
  }
}

function buildEpisodeContent(rows: MemoryItemRow[]): { content: string; summary: string } {
  const lines: string[] = [];
  for (const row of rows) {
    const meta = parseMetadataJson(row.metadata_json);
    const role = typeof meta?.role === "string" ? meta.role : String(row.kind);
    lines.push(`[${role}] ${String(row.content).trim()}`);
  }
  const content = lines.join("\n");
  const firstLine = lines[0] ?? "";
  const preview = firstLine.length > 80 ? `${firstLine.slice(0, 77)}...` : firstLine;
  const summary = `Episode (${rows.length} turns): ${preview}`;
  return { content, summary };
}

function buildEpisodeMetadata(rows: MemoryItemRow[], sessionKey: string): Record<string, unknown> {
  let userTurns = 0;
  let assistantTurns = 0;
  let totalContentLength = 0;
  let hasSmallTalk = false;
  for (const row of rows) {
    const meta = parseMetadataJson(row.metadata_json);
    const role = typeof meta?.role === "string" ? meta.role : "";
    if (role === "user") {
      userTurns += 1;
    }
    if (role === "assistant") {
      assistantTurns += 1;
    }
    const contentLen = String(row.content).trim().length;
    totalContentLength += contentLen;
    if (String(row.kind).includes("relationship")) {
      hasSmallTalk = true;
    }
  }
  return {
    sessionKey,
    turnCount: rows.length,
    userTurns,
    assistantTurns,
    totalContentLength,
    avgContentLength: rows.length > 0 ? Math.round(totalContentLength / rows.length) : 0,
    hasSmallTalk,
    episodeStart: Number(rows[0]?.created_at ?? 0),
    episodeEnd: Number(rows[rows.length - 1]?.created_at ?? 0),
  };
}

function deriveSourceDetail(source: SoulMemorySource): SoulMemorySourceDetail {
  if (source === SOURCE_MANUAL_LOG || source === SOURCE_CORE_PREFERENCE) {
    return "explicit";
  }
  if (source === SOURCE_MIGRATION) {
    return "system";
  }
  return "inferred";
}

function normalizeMemoryType(value: string | null | undefined): SoulMemoryType {
  if (value === "semantic") {
    return "semantic";
  }
  return "episodic";
}

function normalizeSourceDetail(value: string | null | undefined): SoulMemorySourceDetail {
  if (value === "explicit") {
    return "explicit";
  }
  if (value === "system") {
    return "system";
  }
  return "inferred";
}

function rowToMemoryItem(row: MemoryItemRow): SoulMemoryItem {
  const reinforcementCount = Math.max(1, Math.floor(Number(row.reinforcement_count ?? 1)));
  return {
    id: String(row.id),
    scopeType: String(row.scope_type),
    scopeId: String(row.scope_id),
    kind: String(row.kind),
    content: String(row.content),
    summary: row.summary == null ? undefined : String(row.summary),
    confidence: Number(row.confidence ?? 0),
    tier: normalizeTier(String(row.tier)),
    source: normalizeSource(String(row.source)),
    recordKind: normalizeRecordKind(String(row.record_kind ?? RECORD_KIND_EXPERIENCE)),
    metadata: parseMetadataJson(row.metadata_json),
    createdAt: Number(row.created_at ?? 0),
    lastAccessedAt: row.last_accessed_at == null ? null : Number(row.last_accessed_at),
    reinforcementCount,
    lastReinforcedAt: row.last_reinforced_at == null ? null : Number(row.last_reinforced_at),
    memoryType: normalizeMemoryType(row.memory_type),
    validFrom: row.valid_from == null ? null : Number(row.valid_from),
    validUntil: row.valid_until == null ? null : Number(row.valid_until),
    sourceDetail: normalizeSourceDetail(row.source_detail),
    isCompacted: (row.is_compacted ?? 0) === 1,
    semanticKey: row.semantic_key == null ? null : String(row.semantic_key),
  };
}

function rowToArchiveEvent(row: ArchiveRow): SoulArchiveEvent {
  return {
    id: String(row.id),
    scopeType: String(row.scope_type),
    scopeId: String(row.scope_id),
    kind: String(row.kind),
    recordKind: normalizeRecordKind(String(row.record_kind ?? RECORD_KIND_FACT)),
    content: String(row.content),
    summary: String(row.summary ?? ""),
    source: normalizeSource(String(row.source)),
    createdAt: Number(row.created_at ?? 0),
    activeMemoryId: row.active_memory_id == null ? undefined : String(row.active_memory_id),
    metadata: parseMetadataJson(row.metadata_json),
  };
}

function normalizeTier(value: string): SoulMemoryTier {
  const normalized = value.trim().toUpperCase();
  if (
    normalized === TIER_P0 ||
    normalized === TIER_P1 ||
    normalized === TIER_P2 ||
    normalized === TIER_P3
  ) {
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
    normalized === SOURCE_AUTO_EXTRACTION ||
    normalized === SOURCE_RUNTIME_EVENT
  ) {
    return normalized;
  }
  return SOURCE_MANUAL_LOG;
}

function normalizeRecordKind(value: string): SoulMemoryRecordKind {
  const normalized = value.trim().toLowerCase();
  if (
    normalized === RECORD_KIND_FACT ||
    normalized === RECORD_KIND_RELATIONSHIP ||
    normalized === RECORD_KIND_EXPERIENCE ||
    normalized === RECORD_KIND_SOUL
  ) {
    return normalized;
  }
  return RECORD_KIND_EXPERIENCE;
}

function resolveSoulMemoryConfig(raw?: SoulMemoryConfig): ResolvedSoulMemoryConfig {
  const config = raw ?? {};
  return {
    forgetConfidenceThreshold: resolveBoundedNumber(
      config.forgetConfidenceThreshold,
      FORGET_CONFIDENCE_THRESHOLD,
      0,
      1,
    ),
    forgetStreakHalfLives: resolveBoundedNumber(
      config.forgetStreakHalfLives,
      FORGET_STREAK_HALF_LIVES,
      0.1,
    ),
    p0ScopePenalty: P0_SCOPE_PENALTY,
    crossScopePenalty: resolveBoundedNumber(config.crossScopePenalty, CROSS_SCOPE_PENALTY, 0),
    matchScopePenalty: resolveBoundedNumber(config.matchScopePenalty, MATCH_SCOPE_PENALTY, 0),
    // Clarity half-lives are all Infinity (no decay). Kept for ClarityDecayConfig compat.
    p0ClarityHalfLifeDays: Infinity,
    p1ClarityHalfLifeDays: Infinity,
    p2ClarityHalfLifeDays: Infinity,
    p3ClarityHalfLifeDays: Infinity,
    scoreSimilarityWeight: resolveBoundedNumber(
      config.scoreSimilarityWeight,
      SCORE_SIMILARITY_WEIGHT,
      0,
    ),
    scoreDecayWeight: resolveBoundedNumber(config.scoreDecayWeight, SCORE_DECAY_WEIGHT, 0),
    reinforcementLogWeight: resolveBoundedNumber(
      config.reinforcementLogWeight,
      REINFORCEMENT_LOG_WEIGHT,
      0,
    ),
    referenceExpansionEnabled:
      config.referenceExpansionEnabled === undefined
        ? REFERENCE_EXPANSION_ENABLED
        : Boolean(config.referenceExpansionEnabled),
    referenceMaxHops: resolveBoundedInteger(config.referenceMaxHops, REFERENCE_MAX_HOPS, 0, 5),
    referenceEdgeDecay: resolveBoundedNumber(config.referenceEdgeDecay, REFERENCE_EDGE_DECAY, 0, 1),
    referenceBoostWeight: resolveBoundedNumber(
      config.referenceBoostWeight,
      REFERENCE_BOOST_WEIGHT,
      0,
    ),
    referenceMaxBoost: resolveBoundedNumber(config.referenceMaxBoost, REFERENCE_MAX_BOOST, 0),
    referenceSeedTopKMultiplier: resolveBoundedInteger(
      config.referenceSeedTopKMultiplier,
      REFERENCE_SEED_TOPK_MULTIPLIER,
      1,
      10,
    ),
    p3Compaction: resolveP3CompactionConfig(config.p3Compaction),
  };
}

function resolveP3CompactionConfig(
  raw?: SoulMemoryConfig["p3Compaction"],
): ResolvedP3CompactionConfig {
  const cfg = raw ?? {};
  return {
    enabled: cfg.enabled ?? false,
    minClusterSize: resolveBoundedInteger(cfg.minClusterSize, 3, 2, 20),
    similarityMin: resolveBoundedNumber(cfg.similarityMin, 0.45, 0, 1),
    similarityMax: resolveBoundedNumber(cfg.similarityMax, 0.85, 0, 1),
    archiveAgeDays: resolveBoundedInteger(cfg.archiveAgeDays, 30, 1),
    orphanAgeDays: resolveBoundedInteger(cfg.orphanAgeDays, 60, 1),
    compactedDiscount: resolveBoundedNumber(cfg.compactedDiscount, 0.5, 0, 1),
    batchLimit: resolveBoundedInteger(cfg.batchLimit, 1000, 50, 10000),
  };
}

function resolveBoundedNumber(value: unknown, fallback: number, min: number, max?: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }
  const numeric = value;
  if (numeric < min) {
    return fallback;
  }
  if (max != null && numeric > max) {
    return fallback;
  }
  return numeric;
}

function resolveBoundedInteger(
  value: unknown,
  fallback: number,
  min: number,
  max?: number,
): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }
  const numeric = Math.floor(value);
  if (numeric < min) {
    return fallback;
  }
  if (max != null && numeric > max) {
    return fallback;
  }
  return numeric;
}

function normalizeScopeValue(value: string): string {
  return value.trim().toLowerCase();
}

function normalizeText(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

/**
 * Batch-annotate query results with unresolved conflict IDs from memory_conflicts.
 * Lightweight: single query, no scoring impact.
 */
function annotateConflictIds(db: DatabaseSync, results: SoulMemoryQueryResult[]): void {
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
  if (normalized === SOURCE_RUNTIME_EVENT || normalized === "runtime" || normalized === "event") {
    return SOURCE_RUNTIME_EVENT;
  }
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

function resolveRecordKind(
  recordKind: SoulMemoryRecordKind | undefined,
  kind: string,
  source: SoulMemorySource,
): SoulMemoryRecordKind {
  if (recordKind) {
    return normalizeRecordKind(recordKind);
  }
  if (source === SOURCE_CORE_PREFERENCE) {
    return RECORD_KIND_SOUL;
  }
  if (
    kind.includes("chat") ||
    kind.includes("relationship") ||
    kind.includes("greeting") ||
    kind.includes("small_talk")
  ) {
    return RECORD_KIND_RELATIONSHIP;
  }
  if (
    source === SOURCE_RUNTIME_EVENT ||
    kind.includes("event") ||
    kind.includes("message") ||
    kind.includes("task") ||
    kind.includes("result") ||
    kind.includes("decision") ||
    kind.includes("session")
  ) {
    return RECORD_KIND_FACT;
  }
  return RECORD_KIND_EXPERIENCE;
}

function normalizeOptionalSummary(summary: string | undefined, content: string): string | null {
  const normalized = summary?.trim();
  if (normalized) {
    return truncateUtf16Safe(normalized, 280);
  }
  const compact = summarizeArchiveContent(content);
  return compact || null;
}

function summarizeArchiveContent(content: string): string {
  const normalized = content.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return "";
  }
  return truncateUtf16Safe(normalized, 280);
}

function stringifyMetadata(metadata: Record<string, unknown> | undefined): string | null {
  if (!metadata || Object.keys(metadata).length === 0) {
    return null;
  }
  try {
    return JSON.stringify(metadata);
  } catch {
    return null;
  }
}

function parseMetadataJson(raw: string | null | undefined): Record<string, unknown> | undefined {
  if (!raw?.trim()) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return undefined;
    }
    return parsed as Record<string, unknown>;
  } catch {
    return undefined;
  }
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

function searchByVector(params: {
  db: DatabaseSync;
  queryVec: number[];
  limit: number;
}): Map<string, { rank: number; distance: number; score: number }> {
  if (params.queryVec.length === 0 || params.limit <= 0 || !hasSoulMemoryVecTable(params.db)) {
    return new Map();
  }
  type VectorRow = { id?: string; dist?: number | null };
  try {
    const rows = params.db
      .prepare(
        `SELECT m.id AS id, vec_distance_cosine(v.embedding, ?) AS dist ` +
          `FROM ${SOUL_MEMORY_VEC_TABLE} v ` +
          "JOIN memory_items m ON m.id = v.id " +
          "ORDER BY dist ASC LIMIT ?",
      )
      .all(vectorToBlob(params.queryVec), params.limit) as VectorRow[];
    const out = new Map<string, { rank: number; distance: number; score: number }>();
    for (const row of rows) {
      const id = String(row.id ?? "").trim();
      if (!id || out.has(id)) {
        continue;
      }
      const distance = Number(row.dist ?? 1);
      out.set(id, {
        rank: out.size + 1,
        distance,
        score: clamp(1 - distance, 0, 1),
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
