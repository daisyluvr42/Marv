import crypto from "node:crypto";
import type { DatabaseSync } from "node:sqlite";
import { SOUL_MEMORY_REF_TABLE, upsertItemReferences } from "../salience/reference-expansion.js";
import { embedText, extractEntities, vectorToBlob } from "./soul-memory-embedding.js";
import { openSoulMemoryDb } from "./soul-memory-schema.js";
import { hasSoulMemoryVecTable } from "./soul-memory-schema.js";
import {
  MEMORY_ITEM_SELECT_COLUMNS,
  RECORD_KIND_EXPERIENCE,
  SOUL_ARCHIVE_TABLE,
  SOUL_MEMORY_ENTITY_TABLE,
  SOUL_MEMORY_VEC_TABLE,
  SOURCE_MANUAL_LOG,
  SOURCE_PROFILE,
  SOURCE_RUNTIME_EVENT,
  TIER_P3,
  type ArchiveRow,
  type MemoryItemRow,
  type SoulArchiveEvent,
  type SoulMemoryConfig,
  type SoulMemoryItem,
  type SoulMemoryRecordKind,
  type SoulMemorySource,
  type SoulMemoryTier,
  deriveSourceDetail,
  normalizeOptionalSummary,
  normalizeRecordKind,
  normalizeScopeValue,
  normalizeSource,
  normalizeText,
  normalizeTier,
  parseMetadataJson,
  resolveMemorySource,
  resolveRecordKind,
  rowToArchiveEvent,
  rowToMemoryItem,
  stringifyMetadata,
  clamp,
} from "./soul-memory-types.js";

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

export function countSoulMemoryItemsByTier(params: { agentId: string }): Record<string, number> {
  const db = openSoulMemoryDb(params.agentId);
  try {
    type Row = { tier?: string; count?: number };
    const rows = db
      .prepare("SELECT tier, COUNT(*) as count FROM memory_items GROUP BY tier")
      .all() as Row[];
    const counts: Record<string, number> = {};
    for (const row of rows) {
      counts[normalizeTier(String(row.tier ?? "P3"))] = Number(row.count ?? 0);
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

// ── Internal write helpers ──────────────────────────────────────────────────

export function writeSoulArchiveEvent(params: {
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

export function upsertItemEntities(db: DatabaseSync, memoryId: string, content: string): void {
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

export function upsertSoulMemoryVector(
  db: DatabaseSync,
  memoryId: string,
  embedding: number[],
): void {
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

export function deleteSoulMemoryVectorRows(db: DatabaseSync, memoryIds: string[]): void {
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

export function getSoulMemoryItemInternal(db: DatabaseSync, itemId: string): SoulMemoryItem | null {
  const row = db
    .prepare(`SELECT ${MEMORY_ITEM_SELECT_COLUMNS} FROM memory_items WHERE id = ?`)
    .get(itemId) as MemoryItemRow | undefined;
  return row ? rowToMemoryItem(row) : null;
}

export function findExistingMemoryItem(params: {
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

export function pruneSoulMemoryItems(db: DatabaseSync, memoryIds: string[]): number {
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
export function archiveP3BeforeDelete(db: DatabaseSync, memoryIds: string[]): void {
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

export function buildEpisodeContent(rows: MemoryItemRow[]): { content: string; summary: string } {
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

export function buildEpisodeMetadata(
  rows: MemoryItemRow[],
  sessionKey: string,
): Record<string, unknown> {
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
