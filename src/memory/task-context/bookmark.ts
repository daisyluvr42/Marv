import crypto from "node:crypto";
import fsSync from "node:fs";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { requireNodeSqlite } from "../storage/sqlite.js";
import { normalizeTaskId, resolveTaskContextDbPath } from "./store.js";

export type TaskDecisionBookmark = {
  id: string;
  taskId: string;
  sequence: number | null;
  content: string;
  sourceEntryId?: string;
  tags?: string[];
  createdAt: number;
};

type BookmarkRow = {
  id: string;
  task_id: string;
  sequence: number | null;
  content: string;
  source_entry_id: string | null;
  tags_json: string | null;
  created_at: number;
};

export function addTaskDecisionBookmark(params: {
  agentId: string;
  taskId: string;
  content: string;
  sequence?: number | null;
  sourceEntryId?: string;
  tags?: string[];
  createdAt?: number;
  env?: NodeJS.ProcessEnv;
}): TaskDecisionBookmark | null {
  const content = params.content.trim();
  if (!content) {
    return null;
  }
  const opened = openTaskContextDb({
    agentId: params.agentId,
    taskId: params.taskId,
    createIfMissing: false,
    env: params.env,
  });
  if (!opened) {
    return null;
  }
  try {
    const id = `tcb_${crypto.randomUUID().replace(/-/g, "")}`;
    const createdAt = Number.isFinite(params.createdAt)
      ? Math.floor(params.createdAt as number)
      : Date.now();
    const sequence =
      params.sequence != null && Number.isFinite(params.sequence)
        ? Math.max(1, Math.floor(params.sequence))
        : null;
    const sourceEntryId = params.sourceEntryId?.trim() || null;
    const tagsJson =
      params.tags && params.tags.length > 0 ? JSON.stringify(dedupeTags(params.tags)) : null;
    opened.db
      .prepare(
        "INSERT INTO task_context_bookmarks (" +
          "id, task_id, sequence, content, source_entry_id, tags_json, created_at" +
          ") VALUES (?, ?, ?, ?, ?, ?, ?)",
      )
      .run(id, opened.taskId, sequence, content, sourceEntryId, tagsJson, createdAt);
    const row = opened.db
      .prepare(
        "SELECT id, task_id, sequence, content, source_entry_id, tags_json, created_at " +
          "FROM task_context_bookmarks WHERE id = ?",
      )
      .get(id) as BookmarkRow | undefined;
    return row ? rowToBookmark(row) : null;
  } finally {
    opened.db.close();
  }
}

export function listTaskDecisionBookmarks(params: {
  agentId: string;
  taskId: string;
  limit?: number;
  env?: NodeJS.ProcessEnv;
}): TaskDecisionBookmark[] {
  const opened = openTaskContextDb({
    agentId: params.agentId,
    taskId: params.taskId,
    createIfMissing: false,
    env: params.env,
  });
  if (!opened) {
    return [];
  }
  try {
    const limit = Number.isFinite(params.limit)
      ? Math.max(1, Math.floor(params.limit as number))
      : 200;
    const rows = opened.db
      .prepare(
        "SELECT id, task_id, sequence, content, source_entry_id, tags_json, created_at " +
          "FROM task_context_bookmarks WHERE task_id = ? ORDER BY created_at ASC, id ASC LIMIT ?",
      )
      .all(opened.taskId, limit) as BookmarkRow[];
    return rows.map((row) => rowToBookmark(row));
  } finally {
    opened.db.close();
  }
}

export function extractDecisionCandidates(input: string): string[] {
  const lines = input
    .split(/\r?\n+/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) {
    return [];
  }
  const decisionHints = [
    "决定",
    "约束",
    "必须",
    "禁止",
    "should",
    "must",
    "never",
    "always",
    "decision",
    "constraint",
    "agreed",
  ];
  const out: string[] = [];
  for (const line of lines) {
    const lower = line.toLowerCase();
    if (decisionHints.some((hint) => lower.includes(hint.toLowerCase()))) {
      out.push(line);
    }
    if (out.length >= 12) {
      break;
    }
  }
  return dedupeText(out);
}

function rowToBookmark(row: BookmarkRow): TaskDecisionBookmark {
  return {
    id: row.id,
    taskId: row.task_id,
    sequence: row.sequence == null ? null : Math.max(1, Math.floor(Number(row.sequence))),
    content: row.content,
    sourceEntryId: row.source_entry_id ?? undefined,
    tags: parseTags(row.tags_json),
    createdAt: Number(row.created_at ?? 0),
  };
}

function parseTags(raw: string | null): string[] | undefined {
  if (!raw) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) {
      return undefined;
    }
    const tags = parsed
      .map((tag) => (typeof tag === "string" ? tag.trim().toLowerCase() : ""))
      .filter(Boolean);
    return tags.length > 0 ? tags : undefined;
  } catch {
    return undefined;
  }
}

function dedupeTags(input: string[]): string[] {
  const seen = new Set<string>();
  const tags: string[] = [];
  for (const raw of input) {
    const normalized = raw.trim().toLowerCase();
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    tags.push(normalized);
  }
  return tags;
}

function dedupeText(lines: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const line of lines) {
    const normalized = line.trim().toLowerCase();
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    out.push(line.trim());
  }
  return out;
}

function openTaskContextDb(params: {
  agentId: string;
  taskId: string;
  createIfMissing: boolean;
  env?: NodeJS.ProcessEnv;
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
  ensureBookmarkSchema(db);
  return { db, taskId };
}

function ensureBookmarkSchema(db: DatabaseSync): void {
  db.exec(
    "CREATE TABLE IF NOT EXISTS task_context_bookmarks (" +
      "id TEXT PRIMARY KEY, " +
      "task_id TEXT NOT NULL, " +
      "sequence INTEGER, " +
      "content TEXT NOT NULL, " +
      "source_entry_id TEXT, " +
      "tags_json TEXT, " +
      "created_at INTEGER NOT NULL, " +
      "FOREIGN KEY (task_id) REFERENCES task_context(task_id) ON DELETE CASCADE" +
      ");",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_task_context_bookmarks_task_created " +
      "ON task_context_bookmarks (task_id, created_at);",
  );
}
