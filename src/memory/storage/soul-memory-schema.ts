import fsSync from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { SOUL_MEMORY_REF_TABLE } from "../salience/reference-expansion.js";
import { SOUL_MEMORY_SCOPE_HITS_TABLE } from "../salience/reinforcement.js";
import { parseEmbedding, vectorToBlob } from "./soul-memory-embedding.js";
import { resolveSoulMemoryDbPath } from "./soul-memory-path.js";
import {
  EMBEDDING_DIMS,
  SOUL_ARCHIVE_FTS_TABLE,
  SOUL_ARCHIVE_TABLE,
  SOUL_MEMORY_ENTITY_TABLE,
  SOUL_MEMORY_FTS_TABLE,
  SOUL_MEMORY_VEC_TABLE,
  soulVectorStateByDbPath,
} from "./soul-memory-types.js";
import { requireNodeSqlite } from "./sqlite.js";

const require = createRequire(import.meta.url);

export function openSoulMemoryDb(agentId: string): DatabaseSync {
  const dbPath = resolveSoulMemoryDbPath(agentId);
  fsSync.mkdirSync(path.dirname(dbPath), { recursive: true });
  const { DatabaseSync } = requireNodeSqlite();
  const db = new DatabaseSync(dbPath, { allowExtension: true });
  db.exec("PRAGMA foreign_keys = ON;");
  ensureSoulMemorySchema(db, dbPath);
  return db;
}

export function ensureSoulMemorySchema(db: DatabaseSync, dbPath: string): void {
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

export function ensureMemoryItemsColumn(
  db: DatabaseSync,
  column: string,
  definition: string,
): void {
  const rows = db.prepare("PRAGMA table_info(memory_items)").all() as Array<{ name?: string }>;
  if (rows.some((row) => row.name === column)) {
    return;
  }
  db.exec(`ALTER TABLE memory_items ADD COLUMN ${column} ${definition}`);
}

/** One-time backfill: derive source_detail from existing source column. */
export function backfillSourceDetail(db: DatabaseSync): void {
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

export function ensureSoulMemoryFts(db: DatabaseSync): void {
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

export function ensureSoulArchiveFts(db: DatabaseSync): void {
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

export function ensureSoulMemoryVec(db: DatabaseSync, dbPath: string): void {
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

export function ensureSoulVectorReady(
  db: DatabaseSync,
  dbPath: string,
  dimensions: number,
): boolean {
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

export function rebuildSoulMemoryVectors(db: DatabaseSync): void {
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

export function hasSoulMemoryFtsTable(db: DatabaseSync): boolean {
  return hasTable(db, SOUL_MEMORY_FTS_TABLE);
}

export function hasSoulMemoryVecTable(db: DatabaseSync): boolean {
  return hasTable(db, SOUL_MEMORY_VEC_TABLE);
}

export function hasTable(db: DatabaseSync, tableName: string): boolean {
  const row = db
    .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?")
    .get(tableName) as { name?: string } | undefined;
  return row?.name === tableName;
}
