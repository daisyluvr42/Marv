import { truncateUtf16Safe } from "../../utils.js";

// ── Time constants ──────────────────────────────────────────────────────────
export const MILLIS_PER_DAY = 24 * 60 * 60 * 1000;

// ── Source constants ────────────────────────────────────────────────────────
export const SOURCE_CORE_PREFERENCE = "core_preference";
export const SOURCE_MANUAL_LOG = "manual_log";
export const SOURCE_MIGRATION = "migration";
export const SOURCE_AUTO_EXTRACTION = "auto_extraction";
export const SOURCE_RUNTIME_EVENT = "runtime_event";

// ── Tier constant ───────────────────────────────────────────────────────────
// All items reside in the "Memory Palace" — a full episodic memory store with
// structured indexing. The legacy P0–P3 tier hierarchy has been collapsed; the
// operational memory layers are Soul.md + EXPERIENCE + CONTEXT.
export const TIER_PALACE = "palace";

// ── Record kind constants ───────────────────────────────────────────────────
export const RECORD_KIND_FACT = "fact";
export const RECORD_KIND_RELATIONSHIP = "relationship";
export const RECORD_KIND_EXPERIENCE = "experience";
export const RECORD_KIND_SOUL = "soul";

// ── Retrieval constants ─────────────────────────────────────────────────────
export const DEFAULT_INJECT_THRESHOLD = 0.65;
/** Default embedding dimensions (legacy hash-based vectors). */
export const EMBEDDING_DIMS = 128;
/** Alias for migration detection — matches the legacy hash vector size. */
export const LEGACY_EMBEDDING_DIMS = 128;
export const SOUL_MEMORY_PATH_PREFIX = "soul-memory/";
export const SOUL_ARCHIVE_PATH_PREFIX = "soul-archive/";

// ── Table name constants ────────────────────────────────────────────────────
export const SOUL_MEMORY_FTS_TABLE = "memory_items_fts";
export const SOUL_MEMORY_VEC_TABLE = "memory_items_vec";
export const SOUL_MEMORY_ENTITY_TABLE = "memory_item_entities";
export const SOUL_MEMORY_META_TABLE = "soul_memory_meta";
export const MEMORY_ITEM_SELECT_COLUMNS =
  "id, scope_type, scope_id, kind, content, embedding_json, confidence, tier, source, " +
  "record_kind, summary, metadata_json, created_at, last_accessed_at, reinforcement_count, last_reinforced_at, " +
  "memory_type, valid_from, valid_until, source_detail, is_compacted, semantic_key, embedding_version";
export const SOUL_ARCHIVE_TABLE = "memory_archive";
export const SOUL_ARCHIVE_FTS_TABLE = "memory_archive_fts";

// ── RRF / candidate constants ───────────────────────────────────────────────
export const RRF_RANK_CONSTANT = 40;
export const VECTOR_RRF_WEIGHT = 0.6;
export const BM25_RRF_WEIGHT = 0.4;
export const CANDIDATE_FULL_SCAN_MAX_ROWS = 2000;
export const CANDIDATE_MIN_LIMIT = 160;
export const CANDIDATE_MAX_LIMIT = 960;
export const CANDIDATE_PER_TOPK_MULTIPLIER = 24;
export const RECENT_CANDIDATE_SHARE = 0.35;

// ── Internal row types ──────────────────────────────────────────────────────

export type SoulVectorState = {
  available: boolean;
  attempted: boolean;
  dims?: number;
  extensionPath?: string;
  loadError?: string;
};

export const soulVectorStateByDbPath = new Map<string, SoulVectorState>();

export type SourceProfile = {
  confidence: number;
};

// All sources reside in the Memory Palace. Tier promotion is replaced by
// EXPERIENCE.md distillation + reinforcement.
export const SOURCE_PROFILE: Record<SoulMemorySource, SourceProfile> = {
  [SOURCE_CORE_PREFERENCE]: { confidence: 0.95 },
  [SOURCE_MANUAL_LOG]: { confidence: 0.85 },
  [SOURCE_MIGRATION]: { confidence: 0.85 },
  [SOURCE_AUTO_EXTRACTION]: { confidence: 0.5 },
  [SOURCE_RUNTIME_EVENT]: { confidence: 0.5 },
};

// ── Stopwords ───────────────────────────────────────────────────────────────

export const ENTITY_EN_STOPWORDS = new Set<string>([
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

export const ENTITY_ZH_STOPWORDS = new Set<string>([
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

// ── DB row types ────────────────────────────────────────────────────────────

export type MemoryItemRow = {
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
  // Compaction fields
  memory_type: string | null;
  valid_from: number | null;
  valid_until: number | null;
  source_detail: string | null;
  is_compacted: number | null;
  semantic_key: string | null;
  /** 0 = legacy hash (128-dim), 1 = ML embedding */
  embedding_version: number | null;
};

export type ArchiveRow = {
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

// ── Public types ────────────────────────────────────────────────────────────

export type SoulMemorySource =
  | typeof SOURCE_CORE_PREFERENCE
  | typeof SOURCE_MANUAL_LOG
  | typeof SOURCE_MIGRATION
  | typeof SOURCE_AUTO_EXTRACTION
  | typeof SOURCE_RUNTIME_EVENT;

/** All items reside in the Memory Palace. Kept as a string field for DB backward compat. */
export type SoulMemoryTier = string;

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
  // Compaction fields
  memoryType: SoulMemoryType;
  validFrom: number | null;
  validUntil: number | null;
  sourceDetail: SoulMemorySourceDetail;
  isCompacted: boolean;
  semanticKey: string | null;
  /** 0 = legacy hash (128-dim), 1 = ML embedding. */
  embeddingVersion: number;
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
  globalScopePenalty?: number;
  crossScopePenalty?: number;
  matchScopePenalty?: number;
  scoreSimilarityWeight?: number;
  reinforcementLogWeight?: number;
  referenceExpansionEnabled?: boolean;
  referenceMaxHops?: number;
  referenceEdgeDecay?: number;
  referenceBoostWeight?: number;
  referenceMaxBoost?: number;
  referenceSeedTopKMultiplier?: number;
  /** @deprecated Use `compaction` instead. */
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
  compaction?: {
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

export type ResolvedCompactionConfig = {
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

export type ResolvedSoulMemoryConfig = {
  globalScopePenalty: number;
  crossScopePenalty: number;
  matchScopePenalty: number;
  scoreSimilarityWeight: number;
  reinforcementLogWeight: number;
  referenceExpansionEnabled: boolean;
  referenceMaxHops: number;
  referenceEdgeDecay: number;
  referenceBoostWeight: number;
  referenceMaxBoost: number;
  referenceSeedTopKMultiplier: number;
  compaction: ResolvedCompactionConfig;
};

// ── Utility functions ───────────────────────────────────────────────────────

export function normalizeScopeValue(value: string): string {
  return value.trim().toLowerCase();
}

export function normalizeText(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

/** All items belong to the Memory Palace. Legacy tiers (P0–P3) are normalized. */
export function normalizeTier(_value: string): SoulMemoryTier {
  return TIER_PALACE;
}

export function normalizeSource(value: string): SoulMemorySource {
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

export function normalizeRecordKind(value: string): SoulMemoryRecordKind {
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

export function normalizeMemoryType(value: string | null | undefined): SoulMemoryType {
  if (value === "semantic") {
    return "semantic";
  }
  return "episodic";
}

export function normalizeSourceDetail(value: string | null | undefined): SoulMemorySourceDetail {
  if (value === "explicit") {
    return "explicit";
  }
  if (value === "system") {
    return "system";
  }
  return "inferred";
}

export function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, value));
}

export function resolveMemorySource(params: {
  source?: string;
  inputConfidence: number;
}): SoulMemorySource {
  const normalized = (params.source ?? "").trim().toLowerCase();
  if (normalized === SOURCE_RUNTIME_EVENT || normalized === "runtime" || normalized === "event") {
    return SOURCE_RUNTIME_EVENT;
  }
  if (normalized === SOURCE_CORE_PREFERENCE || normalized === "explicit") {
    return SOURCE_CORE_PREFERENCE;
  }
  if (normalized === SOURCE_AUTO_EXTRACTION || normalized === "auto") {
    return SOURCE_AUTO_EXTRACTION;
  }
  if (normalized === SOURCE_MIGRATION || normalized === "migration") {
    return SOURCE_MIGRATION;
  }
  if (normalized === SOURCE_MANUAL_LOG || normalized === "manual") {
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

export function resolveRecordKind(
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

// ── Config resolution ───────────────────────────────────────────────────────

import {
  REFERENCE_BOOST_WEIGHT,
  REFERENCE_EDGE_DECAY,
  REFERENCE_EXPANSION_ENABLED,
  REFERENCE_MAX_BOOST,
  REFERENCE_MAX_HOPS,
  REFERENCE_SEED_TOPK_MULTIPLIER,
} from "../salience/reference-expansion.js";
import { REINFORCEMENT_LOG_WEIGHT } from "../salience/reinforcement.js";
import {
  CROSS_SCOPE_PENALTY,
  GLOBAL_SCOPE_PENALTY,
  MATCH_SCOPE_PENALTY,
  SCORE_SIMILARITY_WEIGHT,
} from "../salience/salience-compute.js";

export function resolveSoulMemoryConfig(raw?: SoulMemoryConfig): ResolvedSoulMemoryConfig {
  const config = raw ?? {};
  return {
    globalScopePenalty: resolveBoundedNumber(config.globalScopePenalty, GLOBAL_SCOPE_PENALTY, 0),
    crossScopePenalty: resolveBoundedNumber(config.crossScopePenalty, CROSS_SCOPE_PENALTY, 0),
    matchScopePenalty: resolveBoundedNumber(config.matchScopePenalty, MATCH_SCOPE_PENALTY, 0),
    scoreSimilarityWeight: resolveBoundedNumber(
      config.scoreSimilarityWeight,
      SCORE_SIMILARITY_WEIGHT,
      0,
    ),
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
    compaction: resolveCompactionConfig(config.compaction ?? config.p3Compaction),
  };
}

export function resolveCompactionConfig(
  raw?: SoulMemoryConfig["compaction"],
): ResolvedCompactionConfig {
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

export function resolveBoundedNumber(
  value: unknown,
  fallback: number,
  min: number,
  max?: number,
): number {
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

export function resolveBoundedInteger(
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

// ── Metadata / summary helpers ──────────────────────────────────────────────

export function normalizeOptionalSummary(
  summary: string | undefined,
  content: string,
): string | null {
  const normalized = summary?.trim();
  if (normalized) {
    return truncateUtf16Safe(normalized, 280);
  }
  const compact = summarizeArchiveContent(content);
  return compact || null;
}

export function summarizeArchiveContent(content: string): string {
  const normalized = content.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return "";
  }
  return truncateUtf16Safe(normalized, 280);
}

export function stringifyMetadata(metadata: Record<string, unknown> | undefined): string | null {
  if (!metadata || Object.keys(metadata).length === 0) {
    return null;
  }
  try {
    return JSON.stringify(metadata);
  } catch {
    return null;
  }
}

export function parseMetadataJson(
  raw: string | null | undefined,
): Record<string, unknown> | undefined {
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

// ── Scope helpers ───────────────────────────────────────────────────────────

export function scopeKey(scopeType: string, scopeId: string): string {
  return `${scopeType}:${scopeId}`;
}

export function dedupeScopes(scopes: SoulMemoryScope[]): SoulMemoryScope[] {
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

export function buildScopedAliasClause(params: { scopes: SoulMemoryScope[]; alias: string }): {
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

// ── Row conversion helpers ──────────────────────────────────────────────────

export function deriveSourceDetail(source: SoulMemorySource): SoulMemorySourceDetail {
  if (source === SOURCE_MANUAL_LOG || source === SOURCE_CORE_PREFERENCE) {
    return "explicit";
  }
  if (source === SOURCE_MIGRATION) {
    return "system";
  }
  return "inferred";
}

export function rowToMemoryItem(row: MemoryItemRow): SoulMemoryItem {
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
    embeddingVersion: Number(row.embedding_version ?? 0),
  };
}

export function rowToArchiveEvent(row: ArchiveRow): SoulArchiveEvent {
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
