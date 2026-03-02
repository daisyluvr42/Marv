import crypto from "node:crypto";
import fsSync from "node:fs";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { resolveStateDir } from "../../core/config/paths.js";
import { requireNodeSqlite } from "./sqlite.js";

const MILLIS_PER_DAY = 24 * 60 * 60 * 1000;
const SOURCE_CORE_PREFERENCE = "core_preference";
const SOURCE_MANUAL_LOG = "manual_log";
const SOURCE_MIGRATION = "migration";
const SOURCE_AUTO_EXTRACTION = "auto_extraction";

const TIER_P0 = "P0";
const TIER_P1 = "P1";
const TIER_P2 = "P2";

const FORGET_CONFIDENCE_THRESHOLD = 0.1;
const FORGET_STREAK_HALF_LIVES = 3;
const DEFAULT_INJECT_THRESHOLD = 0.65;
const P0_CLARITY_HALF_LIFE_DAYS = 365;
const P1_CLARITY_HALF_LIFE_DAYS = 45;
const P2_CLARITY_HALF_LIFE_DAYS = 10;
const P0_RECALL_RELEVANCE_THRESHOLD = 0.7;
const P2_TO_P1_MIN_CLARITY = 0.75;
const P2_TO_P1_MIN_AGE_DAYS = 7;
const P2_TO_P1_MIN_SCOPE_COUNT = 2;
const P1_TO_P0_MIN_CLARITY = 0.75;
const P1_TO_P0_MIN_AGE_DAYS = 150;
const P0_SCOPE_PENALTY = 0.8;
const CROSS_SCOPE_PENALTY = 0.2;
const MATCH_SCOPE_PENALTY = 1;
const P0_TIER_MULTIPLIER = 1.2;
const P1_TIER_MULTIPLIER = 1;
const P2_TIER_MULTIPLIER = 0.75;
const SCORE_SIMILARITY_WEIGHT = 1;
const SCORE_DECAY_WEIGHT = 1;
const REINFORCEMENT_LOG_WEIGHT = 0.2;
const REFERENCE_EXPANSION_ENABLED = true;
const REFERENCE_MAX_HOPS = 2;
const REFERENCE_EDGE_DECAY = 0.8;
const REFERENCE_BOOST_WEIGHT = 0.3;
const REFERENCE_MAX_BOOST = 0.6;
const REFERENCE_SEED_TOPK_MULTIPLIER = 2;
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
const SOUL_MEMORY_VEC_TABLE = "memory_items_vec";
const SOUL_MEMORY_ENTITY_TABLE = "memory_item_entities";
const SOUL_MEMORY_SCOPE_HITS_TABLE = "memory_scope_hits";
const SOUL_MEMORY_REF_TABLE = "memory_item_refs";
const MEMORY_ITEM_SELECT_COLUMNS =
  "id, scope_type, scope_id, kind, content, embedding_json, confidence, tier, source, " +
  "created_at, last_accessed_at, reinforcement_count, last_reinforced_at";
const MEMORY_REF_TOKEN_RE = /\[ref:(mem_[a-z0-9]+)\]/gi;
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

const SOURCE_PROFILE: Record<SoulMemorySource, SourceProfile> = {
  [SOURCE_CORE_PREFERENCE]: { confidence: 0.95, tier: TIER_P0 },
  [SOURCE_MANUAL_LOG]: { confidence: 0.85, tier: TIER_P1 },
  [SOURCE_MIGRATION]: { confidence: 0.85, tier: TIER_P1 },
  [SOURCE_AUTO_EXTRACTION]: { confidence: 0.5, tier: TIER_P2 },
};

// P0 is reserved for durable soul-level facts/policies only.
const DEFAULT_P0_ALLOWED_KINDS = new Set<string>([
  "preference",
  "principle",
  "identity",
  "policy",
  "guardrail",
]);

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
  reinforcement_count: number;
  last_reinforced_at: number | null;
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
  reinforcementCount: number;
  lastReinforcedAt: number | null;
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

export type SoulMemoryConfig = {
  p0AllowedKinds?: string[];
  forgetConfidenceThreshold?: number;
  forgetStreakHalfLives?: number;
  p0ClarityHalfLifeDays?: number;
  p1ClarityHalfLifeDays?: number;
  p2ClarityHalfLifeDays?: number;
  p0RecallRelevanceThreshold?: number;
  p2ToP1MinClarity?: number;
  p2ToP1MinAgeDays?: number;
  p2ToP1MinScopeCount?: number;
  p1ToP0MinClarity?: number;
  p1ToP0MinAgeDays?: number;
  p0ScopePenalty?: number;
  crossScopePenalty?: number;
  matchScopePenalty?: number;
  p0TierMultiplier?: number;
  p1TierMultiplier?: number;
  p2TierMultiplier?: number;
  scoreSimilarityWeight?: number;
  scoreDecayWeight?: number;
  reinforcementLogWeight?: number;
  referenceExpansionEnabled?: boolean;
  referenceMaxHops?: number;
  referenceEdgeDecay?: number;
  referenceBoostWeight?: number;
  referenceMaxBoost?: number;
  referenceSeedTopKMultiplier?: number;
};

type ResolvedSoulMemoryConfig = {
  p0AllowedKinds: Set<string>;
  forgetConfidenceThreshold: number;
  forgetStreakHalfLives: number;
  p0ClarityHalfLifeDays: number;
  p1ClarityHalfLifeDays: number;
  p2ClarityHalfLifeDays: number;
  p0RecallRelevanceThreshold: number;
  p2ToP1MinClarity: number;
  p2ToP1MinAgeDays: number;
  p2ToP1MinScopeCount: number;
  p1ToP0MinClarity: number;
  p1ToP0MinAgeDays: number;
  p0ScopePenalty: number;
  crossScopePenalty: number;
  matchScopePenalty: number;
  p0TierMultiplier: number;
  p1TierMultiplier: number;
  p2TierMultiplier: number;
  scoreSimilarityWeight: number;
  scoreDecayWeight: number;
  reinforcementLogWeight: number;
  referenceExpansionEnabled: boolean;
  referenceMaxHops: number;
  referenceEdgeDecay: number;
  referenceBoostWeight: number;
  referenceMaxBoost: number;
  referenceSeedTopKMultiplier: number;
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
  soulConfig?: SoulMemoryConfig;
  // Deprecated alias; prefer soulConfig.p0AllowedKinds.
  p0AllowedKinds?: string[];
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
  const soulConfig = resolveSoulMemoryConfig({
    ...params.soulConfig,
    p0AllowedKinds: params.soulConfig?.p0AllowedKinds ?? params.p0AllowedKinds,
  });
  const normalizedSource = normalizeSourceForKind(source, kind, soulConfig.p0AllowedKinds);
  const sourceProfile = SOURCE_PROFILE[normalizedSource];
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
      const nextSource = normalizedSource;
      db.prepare(
        "UPDATE memory_items SET confidence = ?, tier = ?, source = ?, " +
          "reinforcement_count = COALESCE(reinforcement_count, 1) + 1, last_reinforced_at = ? " +
          "WHERE id = ?",
      ).run(nextConfidence, nextTier, nextSource, nowMs, existing.id);
      return getSoulMemoryItemInternal(db, existing.id);
    }

    const embeddingVec = embedText(content);
    const embedding = JSON.stringify(embeddingVec);
    const id = `mem_${crypto.randomUUID().replace(/-/g, "")}`;
    db.prepare(
      "INSERT INTO memory_items (" +
        "id, scope_type, scope_id, kind, content, embedding_json, confidence, tier, source, " +
        "created_at, last_accessed_at, reinforcement_count, last_reinforced_at" +
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 1, ?)",
    ).run(
      id,
      scopeType,
      scopeId,
      kind,
      content,
      embedding,
      sourceProfile.confidence,
      sourceProfile.tier,
      normalizedSource,
      nowMs,
      nowMs,
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
        `SELECT ${MEMORY_ITEM_SELECT_COLUMNS} FROM memory_items ` +
          `${where} ORDER BY created_at DESC LIMIT ?`,
      )
      .all(...values, limit) as MemoryItemRow[];
    return rows.map((row) => rowToMemoryItem(row));
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

export function querySoulMemoryMulti(params: {
  agentId: string;
  scopes: SoulMemoryScope[];
  query: string;
  topK: number;
  minScore?: number;
  ttlDays?: number;
  nowMs?: number;
  soulConfig?: SoulMemoryConfig;
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
      if (ttlDays > 0 && ageDays > ttlDays && item.tier !== TIER_P0) {
        continue;
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
      let clarityScore = candidate.clarityScore;
      let wasRecallBoosted = false;
      let decayFactor = candidate.decayFactor;
      if (
        candidate.item.tier === TIER_P0 &&
        relevanceScore >= soulConfig.p0RecallRelevanceThreshold
      ) {
        decayFactor = 1;
        clarityScore = 1;
        wasRecallBoosted = true;
      }
      const similarityScore = computeWeightedScore(
        relevanceScore,
        soulConfig.scoreSimilarityWeight,
      );
      const decayScore = computeWeightedScore(decayFactor, soulConfig.scoreDecayWeight);
      const reinforcementFactor = computeReinforcementFactor(
        candidate.item.reinforcementCount,
        soulConfig,
      );
      const salienceDecay = decayScore;
      const salienceReinforcement = reinforcementFactor;
      const salienceScore = salienceDecay * salienceReinforcement;
      const tierMultiplier = tierPriorityFactor(candidate.item.tier, soulConfig);
      const score = clamp(
        tierMultiplier * candidate.item.confidence * similarityScore * salienceScore,
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
    return ranked;
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
      "tier TEXT NOT NULL DEFAULT 'P1', " +
      "source TEXT NOT NULL DEFAULT 'manual_log', " +
      "created_at INTEGER NOT NULL, " +
      "last_accessed_at INTEGER, " +
      "reinforcement_count INTEGER NOT NULL DEFAULT 1, " +
      "last_reinforced_at INTEGER" +
      ");",
  );
  ensureMemoryItemsColumn(db, "reinforcement_count", "INTEGER NOT NULL DEFAULT 1");
  ensureMemoryItemsColumn(db, "last_reinforced_at", "INTEGER");
  db.exec(
    "UPDATE memory_items SET reinforcement_count = 1 " +
      "WHERE reinforcement_count IS NULL OR reinforcement_count < 1",
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
  ensureSoulMemoryFts(db);
  ensureSoulMemoryVec(db, dbPath);
}

function ensureMemoryItemsColumn(db: DatabaseSync, column: string, definition: string): void {
  const rows = db.prepare("PRAGMA table_info(memory_items)").all() as Array<{ name?: string }>;
  if (rows.some((row) => row.name === column)) {
    return;
  }
  db.exec(`ALTER TABLE memory_items ADD COLUMN ${column} ${definition}`);
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
  const row = db
    .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?")
    .get(SOUL_MEMORY_FTS_TABLE) as { name?: string } | undefined;
  return row?.name === SOUL_MEMORY_FTS_TABLE;
}

function hasSoulMemoryVecTable(db: DatabaseSync): boolean {
  const row = db
    .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?")
    .get(SOUL_MEMORY_VEC_TABLE) as { name?: string } | undefined;
  return row?.name === SOUL_MEMORY_VEC_TABLE;
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

function upsertItemReferences(
  db: DatabaseSync,
  sourceMemoryId: string,
  content: string,
  nowMs: number,
): void {
  db.prepare(`DELETE FROM ${SOUL_MEMORY_REF_TABLE} WHERE source_memory_id = ?`).run(sourceMemoryId);
  const refs = extractMemoryReferenceIds(content).filter((targetId) => targetId !== sourceMemoryId);
  if (refs.length === 0) {
    return;
  }
  const stmt = db.prepare(
    `INSERT OR IGNORE INTO ${SOUL_MEMORY_REF_TABLE} (source_memory_id, target_memory_id, created_at) ` +
      "VALUES (?, ?, ?)",
  );
  for (const targetId of refs) {
    stmt.run(sourceMemoryId, targetId, nowMs);
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

function reinforceRetrievedItems(
  db: DatabaseSync,
  results: Array<{ id: string; wasRecallBoosted: boolean }>,
  nowMs: number,
): void {
  if (results.length === 0) {
    return;
  }
  const boosted = db.prepare(
    "UPDATE memory_items SET " +
      "last_accessed_at = ?, last_reinforced_at = ?, reinforcement_count = reinforcement_count + 1, " +
      "confidence = 1.0 WHERE id = ?",
  );
  const reinforce = db.prepare(
    "UPDATE memory_items SET " +
      "last_accessed_at = ?, last_reinforced_at = ?, reinforcement_count = reinforcement_count + 1, " +
      "confidence = MIN(1.0, confidence + 0.05) WHERE id = ?",
  );
  const handled = new Set<string>();
  for (const result of results) {
    const itemId = result.id;
    if (!itemId || handled.has(itemId)) {
      continue;
    }
    handled.add(itemId);
    if (result.wasRecallBoosted) {
      boosted.run(nowMs, nowMs, itemId);
      continue;
    }
    reinforce.run(nowMs, nowMs, itemId);
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
  deleteSoulMemoryVectorRows(db, unique);
  const deleteStmt = db.prepare("DELETE FROM memory_items WHERE id = ?");
  let deleted = 0;
  for (const memoryId of unique) {
    const result = deleteStmt.run(memoryId) as { changes?: number };
    deleted += Number(result.changes ?? 0);
  }
  return deleted;
}

function rowToMemoryItem(row: MemoryItemRow): SoulMemoryItem {
  const reinforcementCount = Math.max(1, Math.floor(Number(row.reinforcement_count ?? 1)));
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
    reinforcementCount,
    lastReinforcedAt: row.last_reinforced_at == null ? null : Number(row.last_reinforced_at),
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

function normalizeSourceForKind(
  source: SoulMemorySource,
  kind: string,
  p0AllowedKinds: Set<string>,
): SoulMemorySource {
  if (source !== SOURCE_CORE_PREFERENCE) {
    return source;
  }
  if (p0AllowedKinds.has(kind)) {
    return source;
  }
  return SOURCE_MANUAL_LOG;
}

function resolveP0AllowedKinds(raw?: string[]): Set<string> {
  if (!raw || raw.length === 0) {
    return DEFAULT_P0_ALLOWED_KINDS;
  }
  const normalized = new Set(raw.map((entry) => normalizeScopeValue(entry)).filter(Boolean));
  return normalized.size > 0 ? normalized : DEFAULT_P0_ALLOWED_KINDS;
}

function resolveSoulMemoryConfig(raw?: SoulMemoryConfig): ResolvedSoulMemoryConfig {
  const config = raw ?? {};
  return {
    p0AllowedKinds: resolveP0AllowedKinds(config.p0AllowedKinds),
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
    p0ClarityHalfLifeDays: resolveBoundedNumber(
      config.p0ClarityHalfLifeDays,
      P0_CLARITY_HALF_LIFE_DAYS,
      1,
    ),
    p1ClarityHalfLifeDays: resolveBoundedNumber(
      config.p1ClarityHalfLifeDays,
      P1_CLARITY_HALF_LIFE_DAYS,
      1,
    ),
    p2ClarityHalfLifeDays: resolveBoundedNumber(
      config.p2ClarityHalfLifeDays,
      P2_CLARITY_HALF_LIFE_DAYS,
      1,
    ),
    p0RecallRelevanceThreshold: resolveBoundedNumber(
      config.p0RecallRelevanceThreshold,
      P0_RECALL_RELEVANCE_THRESHOLD,
      0,
      1,
    ),
    p2ToP1MinClarity: resolveBoundedNumber(config.p2ToP1MinClarity, P2_TO_P1_MIN_CLARITY, 0, 1),
    p2ToP1MinAgeDays: resolveBoundedInteger(config.p2ToP1MinAgeDays, P2_TO_P1_MIN_AGE_DAYS, 0),
    p2ToP1MinScopeCount: resolveBoundedInteger(
      config.p2ToP1MinScopeCount,
      P2_TO_P1_MIN_SCOPE_COUNT,
      1,
    ),
    p1ToP0MinClarity: resolveBoundedNumber(config.p1ToP0MinClarity, P1_TO_P0_MIN_CLARITY, 0, 1),
    p1ToP0MinAgeDays: resolveBoundedInteger(config.p1ToP0MinAgeDays, P1_TO_P0_MIN_AGE_DAYS, 0),
    p0ScopePenalty: resolveBoundedNumber(config.p0ScopePenalty, P0_SCOPE_PENALTY, 0),
    crossScopePenalty: resolveBoundedNumber(config.crossScopePenalty, CROSS_SCOPE_PENALTY, 0),
    matchScopePenalty: resolveBoundedNumber(config.matchScopePenalty, MATCH_SCOPE_PENALTY, 0),
    p0TierMultiplier: resolveBoundedNumber(config.p0TierMultiplier, P0_TIER_MULTIPLIER, 0),
    p1TierMultiplier: resolveBoundedNumber(config.p1TierMultiplier, P1_TIER_MULTIPLIER, 0),
    p2TierMultiplier: resolveBoundedNumber(config.p2TierMultiplier, P2_TIER_MULTIPLIER, 0),
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

function clarityDecayFactor(
  tier: SoulMemoryTier,
  ageDays: number,
  soulConfig: ResolvedSoulMemoryConfig,
): number {
  const normalizedAgeDays = Math.max(0, ageDays);
  const halfLifeDays =
    tier === TIER_P0
      ? soulConfig.p0ClarityHalfLifeDays
      : tier === TIER_P2
        ? soulConfig.p2ClarityHalfLifeDays
        : soulConfig.p1ClarityHalfLifeDays;
  const factor = 0.5 ** (normalizedAgeDays / halfLifeDays);
  return clamp(factor, 0, 1);
}

function computeCurrentClarity(
  item: SoulMemoryItem,
  ageDays: number,
  soulConfig: ResolvedSoulMemoryConfig,
): number {
  return clamp(item.confidence * clarityDecayFactor(item.tier, ageDays, soulConfig), 0, 1);
}

function resolveTierHalfLifeDays(
  tier: SoulMemoryTier,
  soulConfig: ResolvedSoulMemoryConfig,
): number | null {
  if (tier === TIER_P1) {
    return soulConfig.p1ClarityHalfLifeDays;
  }
  if (tier === TIER_P2) {
    return soulConfig.p2ClarityHalfLifeDays;
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

function shouldPruneMemoryItem(
  item: SoulMemoryItem,
  ageDays: number,
  soulConfig: ResolvedSoulMemoryConfig,
): boolean {
  const halfLifeDays = resolveTierHalfLifeDays(item.tier, soulConfig);
  if (!halfLifeDays || !Number.isFinite(halfLifeDays) || halfLifeDays <= 0) {
    return false;
  }
  const clarity = computeCurrentClarity(item, ageDays, soulConfig);
  if (clarity >= soulConfig.forgetConfidenceThreshold) {
    return false;
  }
  const belowThresholdDays = computeBelowThresholdDurationDays({
    item,
    ageDays,
    threshold: soulConfig.forgetConfidenceThreshold,
    halfLifeDays,
  });
  return belowThresholdDays >= halfLifeDays * soulConfig.forgetStreakHalfLives;
}

function tierPriorityFactor(tier: SoulMemoryTier, soulConfig: ResolvedSoulMemoryConfig): number {
  if (tier === TIER_P0) {
    return soulConfig.p0TierMultiplier;
  }
  if (tier === TIER_P2) {
    return soulConfig.p2TierMultiplier;
  }
  return soulConfig.p1TierMultiplier;
}

function resolveScopePenalty(
  params: {
    item: SoulMemoryItem;
    activeScopeKeySet: Set<string>;
  },
  soulConfig: ResolvedSoulMemoryConfig,
): number {
  if (params.activeScopeKeySet.has(scopeKey(params.item.scopeType, params.item.scopeId))) {
    return soulConfig.matchScopePenalty;
  }
  if (params.item.scopeType === "global" || params.item.scopeType === "user") {
    return soulConfig.p0ScopePenalty;
  }
  return soulConfig.crossScopePenalty;
}

function computeWeightedScore(value: number, weight: number): number {
  const normalizedValue = clamp(value, 0, 1);
  if (!Number.isFinite(weight) || weight === 1) {
    return normalizedValue;
  }
  return clamp(normalizedValue ** weight, 0, 1);
}

function computeReinforcementFactor(
  reinforcementCount: number,
  soulConfig: ResolvedSoulMemoryConfig,
): number {
  const normalizedCount = Math.max(1, Math.floor(reinforcementCount));
  if (soulConfig.reinforcementLogWeight <= 0) {
    return 1;
  }
  const logBoost = Math.log1p(Math.max(0, normalizedCount - 1));
  const factor = 1 + logBoost * soulConfig.reinforcementLogWeight;
  return Math.max(1, factor);
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

function loadReferencesBySourceIds(db: DatabaseSync, sourceIds: string[]): Map<string, string[]> {
  const out = new Map<string, string[]>();
  if (sourceIds.length === 0) {
    return out;
  }
  const placeholders = sourceIds.map(() => "?").join(", ");
  type Row = { source_memory_id?: string; target_memory_id?: string };
  const rows = db
    .prepare(
      `SELECT source_memory_id, target_memory_id FROM ${SOUL_MEMORY_REF_TABLE} ` +
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

function applyReferenceExpansion(params: {
  db: DatabaseSync;
  scoredById: Map<string, SoulMemoryQueryResult>;
  topK: number;
  soulConfig: ResolvedSoulMemoryConfig;
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
    const refsBySource = loadReferencesBySourceIds(params.db, [...frontier.keys()]);
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
  soulConfig?: SoulMemoryConfig;
  // Deprecated alias; prefer soulConfig.p0AllowedKinds.
  p0AllowedKinds?: string[];
}): SoulMemoryPromotionSummary {
  const db = openSoulMemoryDb(params.agentId);
  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  const autoApproveP0 = params.autoApproveP0 === true;
  const approvedP0Ids = new Set((params.approvedP0Ids ?? []).map((entry) => entry.trim()));
  const soulConfig = resolveSoulMemoryConfig({
    ...params.soulConfig,
    p0AllowedKinds: params.soulConfig?.p0AllowedKinds ?? params.p0AllowedKinds,
  });
  try {
    const rows = db
      .prepare(`SELECT ${MEMORY_ITEM_SELECT_COLUMNS} FROM memory_items`)
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
      if (distinctScopeHits < soulConfig.p2ToP1MinScopeCount) {
        continue;
      }
      const survivalAgeDays = Math.max(0, (nowMs - item.createdAt) / MILLIS_PER_DAY);
      const effectiveTs = item.lastAccessedAt ?? item.createdAt;
      const clarityAgeDays = Math.max(0, (nowMs - effectiveTs) / MILLIS_PER_DAY);
      const clarityScore = computeCurrentClarity(item, clarityAgeDays, soulConfig);
      if (
        survivalAgeDays < soulConfig.p2ToP1MinAgeDays ||
        clarityScore < soulConfig.p2ToP1MinClarity
      ) {
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
      if (!soulConfig.p0AllowedKinds.has(item.kind)) {
        continue;
      }
      const survivalAgeDays = Math.max(0, (nowMs - item.createdAt) / MILLIS_PER_DAY);
      const effectiveTs = item.lastAccessedAt ?? item.createdAt;
      const clarityAgeDays = Math.max(0, (nowMs - effectiveTs) / MILLIS_PER_DAY);
      const clarityScore = computeCurrentClarity(item, clarityAgeDays, soulConfig);
      if (
        survivalAgeDays < soulConfig.p1ToP0MinAgeDays ||
        clarityScore < soulConfig.p1ToP0MinClarity
      ) {
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
  soulConfig?: SoulMemoryConfig;
}): { updated: number; deleted: number } {
  const db = openSoulMemoryDb(params.agentId);
  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  const soulConfig = resolveSoulMemoryConfig(params.soulConfig);
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
      .prepare(`SELECT ${MEMORY_ITEM_SELECT_COLUMNS} FROM memory_items${where}`)
      .all(...values) as MemoryItemRow[];

    let updated = 0;
    const staleIds: string[] = [];
    for (const row of rows) {
      const item = rowToMemoryItem(row);
      const effectiveTs = item.lastAccessedAt ?? item.createdAt;
      const ageDays = Math.max(0, (nowMs - effectiveTs) / MILLIS_PER_DAY);
      const decayed = computeCurrentClarity(item, ageDays, soulConfig);
      if (shouldPruneMemoryItem(item, ageDays, soulConfig)) {
        staleIds.push(item.id);
        continue;
      }
      if (Math.abs(decayed - item.confidence) <= 1e-8) {
        continue;
      }
      db.prepare("UPDATE memory_items SET confidence = ? WHERE id = ?").run(decayed, item.id);
      updated += 1;
    }
    const deleted = pruneSoulMemoryItems(db, staleIds);
    return { updated, deleted };
  } finally {
    db.close();
  }
}
