import fsSync from "node:fs";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { requireNodeSqlite } from "../storage/sqlite.js";
import { normalizeTaskId, resolveTaskContextDbPath } from "./store.js";
import type { TaskContextEntry } from "./types.js";

type TaskContextStateRow = {
  task_id: string;
  rolling_summary: string | null;
  updated_at: number;
};

type TaskContextEntryRow = {
  id: string;
  task_id: string;
  sequence: number;
  role: string;
  content: string;
  content_hash: string;
  summary: string | null;
  token_count: number;
  created_at: number;
  metadata_json: string | null;
  summarized: number;
};

export type TaskContextState = {
  taskId: string;
  rollingSummary?: string;
  updatedAt: number;
};

export function getTaskContextState(params: {
  agentId: string;
  taskId: string;
  env?: NodeJS.ProcessEnv;
}): TaskContextState | null {
  const opened = openTaskContextDb({
    agentId: params.agentId,
    taskId: params.taskId,
    env: params.env,
    createIfMissing: false,
  });
  if (!opened) {
    return null;
  }
  try {
    const row = opened.db
      .prepare(
        "SELECT task_id, rolling_summary, updated_at FROM task_context_state WHERE task_id = ?",
      )
      .get(opened.taskId) as TaskContextStateRow | undefined;
    return row ? rowToState(row) : null;
  } finally {
    opened.db.close();
  }
}

export function getTaskContextRollingSummary(params: {
  agentId: string;
  taskId: string;
  env?: NodeJS.ProcessEnv;
}): string | null {
  const state = getTaskContextState(params);
  return state?.rollingSummary?.trim() || null;
}

export function setTaskContextRollingSummary(params: {
  agentId: string;
  taskId: string;
  summary: string;
  updatedAt?: number;
  env?: NodeJS.ProcessEnv;
}): TaskContextState | null {
  const summary = params.summary.trim();
  if (!summary) {
    return null;
  }
  const opened = openTaskContextDb({
    agentId: params.agentId,
    taskId: params.taskId,
    env: params.env,
    createIfMissing: false,
  });
  if (!opened) {
    return null;
  }
  try {
    const updatedAt = normalizeTimestamp(params.updatedAt);
    opened.db
      .prepare(
        "INSERT INTO task_context_state (task_id, rolling_summary, updated_at) " +
          "VALUES (?, ?, ?) " +
          "ON CONFLICT(task_id) DO UPDATE SET rolling_summary=excluded.rolling_summary, updated_at=excluded.updated_at",
      )
      .run(opened.taskId, summary, updatedAt);
    const row = opened.db
      .prepare(
        "SELECT task_id, rolling_summary, updated_at FROM task_context_state WHERE task_id = ?",
      )
      .get(opened.taskId) as TaskContextStateRow | undefined;
    return row ? rowToState(row) : null;
  } finally {
    opened.db.close();
  }
}

export function markTaskContextEntriesSummarized(params: {
  agentId: string;
  taskId: string;
  entryIds: string[];
  summary: string;
  env?: NodeJS.ProcessEnv;
}): number {
  const entryIds = dedupeIds(params.entryIds);
  if (entryIds.length === 0) {
    return 0;
  }
  const opened = openTaskContextDb({
    agentId: params.agentId,
    taskId: params.taskId,
    env: params.env,
    createIfMissing: false,
  });
  if (!opened) {
    return 0;
  }
  try {
    const stmt = opened.db.prepare(
      "UPDATE task_context_entries SET summarized = 1, summary = ? WHERE task_id = ? AND id = ?",
    );
    let updated = 0;
    for (const id of entryIds) {
      stmt.run(params.summary, opened.taskId, id);
      const changeRow = opened.db.prepare("SELECT changes() AS changes").get() as
        | { changes?: number }
        | undefined;
      updated += Number(changeRow?.changes ?? 0);
    }
    return updated;
  } finally {
    opened.db.close();
  }
}

export function listUnsummarizedTaskContextEntries(params: {
  agentId: string;
  taskId: string;
  limit?: number;
  env?: NodeJS.ProcessEnv;
}): TaskContextEntry[] {
  const opened = openTaskContextDb({
    agentId: params.agentId,
    taskId: params.taskId,
    env: params.env,
    createIfMissing: false,
  });
  if (!opened) {
    return [];
  }
  try {
    const limit = Number.isFinite(params.limit)
      ? Math.max(1, Math.floor(params.limit as number))
      : 2000;
    const rows = opened.db
      .prepare(
        "SELECT id, task_id, sequence, role, content, content_hash, summary, token_count, created_at, metadata_json, summarized " +
          "FROM task_context_entries WHERE task_id = ? AND COALESCE(summarized, 0) = 0 " +
          "ORDER BY sequence ASC LIMIT ?",
      )
      .all(opened.taskId, limit) as TaskContextEntryRow[];
    return rows.map((row) => ({
      id: row.id,
      taskId: row.task_id,
      sequence: Math.max(1, Math.floor(Number(row.sequence ?? 1))),
      role: normalizeRole(row.role),
      content: row.content,
      contentHash: row.content_hash,
      summary: row.summary ?? undefined,
      tokenCount: Math.max(1, Math.floor(Number(row.token_count ?? 1))),
      createdAt: Number(row.created_at ?? 0),
      metadata: row.metadata_json ?? undefined,
      summarized: Number(row.summarized ?? 0) > 0,
    }));
  } finally {
    opened.db.close();
  }
}

function openTaskContextDb(params: {
  agentId: string;
  taskId: string;
  env?: NodeJS.ProcessEnv;
  createIfMissing: boolean;
}): { db: DatabaseSync; taskId: string } | null {
  const taskId = normalizeTaskId(params.taskId);
  const dbPath = resolveTaskContextDbPath({
    agentId: params.agentId,
    taskId,
    env: params.env,
  });
  if (!params.createIfMissing && !fsSync.existsSync(dbPath)) {
    return null;
  }
  fsSync.mkdirSync(path.dirname(dbPath), { recursive: true });
  const { DatabaseSync } = requireNodeSqlite();
  const db = new DatabaseSync(dbPath);
  db.exec("PRAGMA foreign_keys = ON;");
  ensureTaskContextStateSchema(db);
  return { db, taskId };
}

function ensureTaskContextStateSchema(db: DatabaseSync): void {
  db.exec(
    "CREATE TABLE IF NOT EXISTS task_context_state (" +
      "task_id TEXT PRIMARY KEY, " +
      "rolling_summary TEXT, " +
      "updated_at INTEGER NOT NULL, " +
      "FOREIGN KEY (task_id) REFERENCES task_context(task_id) ON DELETE CASCADE" +
      ");",
  );
}

function rowToState(row: TaskContextStateRow): TaskContextState {
  return {
    taskId: row.task_id,
    rollingSummary: row.rolling_summary ?? undefined,
    updatedAt: Number(row.updated_at ?? 0),
  };
}

function dedupeIds(ids: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of ids) {
    const id = raw.trim();
    if (!id || seen.has(id)) {
      continue;
    }
    seen.add(id);
    out.push(id);
  }
  return out;
}

function normalizeRole(value: string): "user" | "assistant" | "system" | "tool" {
  const normalized = value.trim().toLowerCase();
  if (
    normalized === "user" ||
    normalized === "assistant" ||
    normalized === "system" ||
    normalized === "tool"
  ) {
    return normalized;
  }
  return "assistant";
}

function normalizeTimestamp(value: number | undefined): number {
  if (Number.isFinite(value)) {
    return Math.floor(value as number);
  }
  return Date.now();
}
