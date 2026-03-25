import type { DatabaseSync } from "node:sqlite";
import { buildFtsMatchQuery, extractEntities, vectorToBlob } from "./soul-memory-embedding.js";
import { hasSoulMemoryFtsTable, hasSoulMemoryVecTable } from "./soul-memory-schema.js";
import {
  BM25_RRF_WEIGHT,
  RRF_RANK_CONSTANT,
  SOUL_MEMORY_ENTITY_TABLE,
  SOUL_MEMORY_VEC_TABLE,
  VECTOR_RRF_WEIGHT,
  type SoulMemoryScope,
  buildScopedAliasClause,
  clamp,
} from "./soul-memory-types.js";

export function computeGraphScores(params: {
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

export function computeClusterScores(
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

export function loadEntitiesForMemoryIds(
  db: DatabaseSync,
  memoryIds: string[],
): Map<string, Set<string>> {
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

export function loadRelatedEntityWeights(params: {
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

export function rankByScore(entries: Array<{ id: string; score: number }>): Map<string, number> {
  const ranked = entries
    .slice()
    .toSorted((a, b) => b.score - a.score)
    .map((entry, index) => [entry.id, index + 1] as const);
  return new Map<string, number>(ranked);
}

export function combineRrf(vectorRank: number, bm25Rank: number): number {
  const raw =
    VECTOR_RRF_WEIGHT * reciprocalRank(vectorRank) + BM25_RRF_WEIGHT * reciprocalRank(bm25Rank);
  const topRaw = VECTOR_RRF_WEIGHT * reciprocalRank(1) + BM25_RRF_WEIGHT * reciprocalRank(1);
  if (topRaw <= 0) {
    return 0;
  }
  return clamp(raw / topRaw, 0, 1);
}

export function reciprocalRank(rank: number): number {
  if (!Number.isFinite(rank) || rank <= 0) {
    return 0;
  }
  return 1 / (RRF_RANK_CONSTANT + rank);
}

export function normalizeBm25Score(rawScore: number): number {
  if (!Number.isFinite(rawScore)) {
    return 0;
  }
  // SQLite bm25() lower is better; clamp negatives to strongest match.
  const nonNegative = Math.max(0, rawScore);
  return clamp(1 / (1 + nonNegative), 0, 1);
}

export function searchByBm25(params: {
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

export function searchByVector(params: {
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
