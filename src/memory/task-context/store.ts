import crypto from "node:crypto";
import fsSync from "node:fs";
import os from "node:os";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { resolveStateDir } from "../../core/config/paths.js";
import { normalizeAgentId } from "../../routing/session-key.js";
import { hashText } from "../internal.js";
import { requireNodeSqlite } from "../storage/sqlite.js";
import type { TaskContext, TaskContextEntry, TaskContextRole, TaskStatus } from "./types.js";

const TASK_CONTEXT_SELECT_COLUMNS =
  "task_id, agent_id, title, status, parent_task_id, scope_id, created_at, updated_at, completed_at, total_entries, total_tokens";
const TASK_CONTEXT_ENTRY_SELECT_COLUMNS =
  "id, task_id, sequence, role, content, content_hash, summary, token_count, created_at, metadata_json, summarized";
const DEFAULT_SCOPE_PREFIX = "task";
const MAX_LIST_LIMIT = 2000;
const DEFAULT_LIST_LIMIT = 200;

const TASK_ROLES = new Set<TaskContextRole>(["user", "assistant", "system", "tool"]);
const TASK_STATUSES = new Set<TaskStatus>(["active", "paused", "completed", "archived"]);

export const SAFE_TASK_ID_RE = /^[a-z0-9][a-z0-9._-]{0,127}$/i;

type TaskContextRow = {
  task_id: string;
  agent_id: string;
  title: string;
  status: string;
  parent_task_id: string | null;
  scope_id: string;
  created_at: number;
  updated_at: number;
  completed_at: number | null;
  total_entries: number;
  total_tokens: number;
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

export type TaskEntryMetadataInput = string | Record<string, unknown>;

export type CreateTaskContextParams = {
  agentId: string;
  taskId: string;
  title: string;
  parentTaskId?: string;
  scopeId?: string;
  status?: TaskStatus;
  nowMs?: number;
  env?: NodeJS.ProcessEnv;
};

export type AppendTaskContextEntryParams = {
  agentId: string;
  taskId: string;
  role: TaskContextRole;
  content: string;
  contentHash?: string;
  summary?: string;
  tokenCount?: number;
  createdAt?: number;
  metadata?: TaskEntryMetadataInput;
  summarized?: boolean;
  env?: NodeJS.ProcessEnv;
};

export type ListTaskContextEntriesParams = {
  agentId: string;
  taskId: string;
  limit?: number;
  afterSequence?: number;
  beforeSequence?: number;
  env?: NodeJS.ProcessEnv;
};

export type UpdateTaskContextStatusParams = {
  agentId: string;
  taskId: string;
  status: TaskStatus;
  updatedAt?: number;
  completedAt?: number | null;
  env?: NodeJS.ProcessEnv;
};

type OpenTaskContextDbParams = {
  agentId: string;
  taskId: string;
  createIfMissing: boolean;
  env?: NodeJS.ProcessEnv;
};

type OpenTaskContextDbResult = {
  db: DatabaseSync;
  dbPath: string;
  agentId: string;
  taskId: string;
};

export function normalizeTaskId(value: string | undefined | null): string {
  const trimmed = (value ?? "").trim();
  if (!trimmed) {
    return "task";
  }
  if (SAFE_TASK_ID_RE.test(trimmed)) {
    return trimmed.toLowerCase();
  }
  return (
    trimmed
      .toLowerCase()
      .replace(/[^a-z0-9._-]+/g, "-")
      .replace(/^[._-]+/, "")
      .replace(/[._-]+$/, "")
      .slice(0, 128) || "task"
  );
}

export function resolveTaskContextRootDir(env: NodeJS.ProcessEnv = process.env): string {
  const stateDir = resolveStateDir(env, os.homedir);
  return path.join(stateDir, "tasks");
}

export function resolveTaskContextAgentDir(params: {
  agentId: string;
  env?: NodeJS.ProcessEnv;
}): string {
  const agentId = normalizeAgentId(params.agentId);
  return path.join(resolveTaskContextRootDir(params.env), agentId);
}

export function resolveTaskContextDbPath(params: {
  agentId: string;
  taskId: string;
  env?: NodeJS.ProcessEnv;
}): string {
  const agentId = normalizeAgentId(params.agentId);
  const taskId = normalizeTaskId(params.taskId);
  return path.join(resolveTaskContextAgentDir({ agentId, env: params.env }), `${taskId}.sqlite`);
}

export function createTaskContext(params: CreateTaskContextParams): TaskContext {
  const opened = openTaskContextDb({
    agentId: params.agentId,
    taskId: params.taskId,
    createIfMissing: true,
    env: params.env,
  });
  if (!opened) {
    throw new Error("failed to open task context store");
  }
  const { db, agentId, taskId } = opened;
  try {
    const existing = readTaskContextRow(db, taskId);
    if (existing) {
      throw new Error(`Task context already exists: ${taskId}`);
    }
    const nowMs = normalizeTimestamp(params.nowMs);
    const title = params.title.trim() || taskId;
    const status = normalizeTaskStatus(params.status ?? "active");
    const parentTaskId = params.parentTaskId?.trim() ? normalizeTaskId(params.parentTaskId) : null;
    const scopeId = normalizeScopeId(params.scopeId, agentId, taskId);
    const completedAt = status === "completed" ? nowMs : null;

    db.prepare(
      "INSERT INTO task_context (" +
        "task_id, agent_id, title, status, parent_task_id, scope_id, created_at, updated_at, completed_at, total_entries, total_tokens" +
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)",
    ).run(taskId, agentId, title, status, parentTaskId, scopeId, nowMs, nowMs, completedAt);

    const row = readTaskContextRow(db, taskId);
    if (!row) {
      throw new Error(`failed to persist task context: ${taskId}`);
    }
    return rowToTaskContext(row);
  } finally {
    db.close();
  }
}

export function getTaskContext(params: {
  agentId: string;
  taskId: string;
  env?: NodeJS.ProcessEnv;
}): TaskContext | null {
  const opened = openTaskContextDb({
    agentId: params.agentId,
    taskId: params.taskId,
    createIfMissing: false,
    env: params.env,
  });
  if (!opened) {
    return null;
  }
  const { db, taskId } = opened;
  try {
    const row = readTaskContextRow(db, taskId);
    return row ? rowToTaskContext(row) : null;
  } finally {
    db.close();
  }
}

export function listTaskContextsForAgent(params: {
  agentId: string;
  status?: TaskStatus;
  limit?: number;
  env?: NodeJS.ProcessEnv;
}): TaskContext[] {
  const agentId = normalizeAgentId(params.agentId);
  const status = params.status ? normalizeTaskStatus(params.status) : undefined;
  const limit = normalizeListLimit(params.limit, DEFAULT_LIST_LIMIT);
  const taskDir = resolveTaskContextAgentDir({ agentId, env: params.env });

  let files: fsSync.Dirent[];
  try {
    files = fsSync.readdirSync(taskDir, { withFileTypes: true });
  } catch {
    return [];
  }

  const tasks: TaskContext[] = [];
  for (const file of files) {
    if (!file.isFile() || !file.name.endsWith(".sqlite")) {
      continue;
    }
    const taskId = normalizeTaskId(file.name.slice(0, -".sqlite".length));
    const opened = openTaskContextDb({
      agentId,
      taskId,
      createIfMissing: false,
      env: params.env,
    });
    if (!opened) {
      continue;
    }
    try {
      const row = readTaskContextRow(opened.db, taskId);
      if (!row) {
        continue;
      }
      const context = rowToTaskContext(row);
      if (status && context.status !== status) {
        continue;
      }
      tasks.push(context);
    } finally {
      opened.db.close();
    }
  }

  tasks.sort((a, b) => {
    if (a.updatedAt !== b.updatedAt) {
      return b.updatedAt - a.updatedAt;
    }
    if (a.createdAt !== b.createdAt) {
      return b.createdAt - a.createdAt;
    }
    return a.taskId.localeCompare(b.taskId);
  });
  return tasks.slice(0, limit);
}

export function appendTaskContextEntry(
  params: AppendTaskContextEntryParams,
): TaskContextEntry | null {
  const opened = openTaskContextDb({
    agentId: params.agentId,
    taskId: params.taskId,
    createIfMissing: false,
    env: params.env,
  });
  if (!opened) {
    throw new Error(`Task context not found: ${normalizeTaskId(params.taskId)}`);
  }
  const { db, taskId } = opened;
  try {
    const task = readTaskContextRow(db, taskId);
    if (!task) {
      throw new Error(`Task context not found: ${taskId}`);
    }

    const content = params.content.trim();
    if (!content) {
      return null;
    }
    const role = normalizeTaskRole(params.role);
    const createdAt = normalizeTimestamp(params.createdAt);
    const summary = params.summary?.trim() || null;
    const metadataJson = normalizeMetadata(params.metadata);
    const tokenCount = normalizeTokenCount(params.tokenCount, content);
    const contentHash = params.contentHash?.trim() || hashText(content);
    const summarized = params.summarized ? 1 : 0;
    const latest = db
      .prepare(
        `SELECT ${TASK_CONTEXT_ENTRY_SELECT_COLUMNS} FROM task_context_entries ` +
          "WHERE task_id = ? ORDER BY sequence DESC LIMIT 1",
      )
      .get(taskId) as TaskContextEntryRow | undefined;
    if (latest) {
      const latestRole = coerceTaskRole(latest.role);
      const latestHash = String(latest.content_hash ?? "").trim();
      if (latestRole === role && latestHash && latestHash === contentHash) {
        return rowToTaskContextEntry(latest);
      }
    }
    const sequence = nextEntrySequence(db, taskId);
    const id = `tce_${crypto.randomUUID().replace(/-/g, "")}`;

    db.prepare(
      "INSERT INTO task_context_entries (" +
        "id, task_id, sequence, role, content, content_hash, summary, token_count, created_at, metadata_json, summarized" +
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    ).run(
      id,
      taskId,
      sequence,
      role,
      content,
      contentHash,
      summary,
      tokenCount,
      createdAt,
      metadataJson,
      summarized,
    );

    db.prepare(
      "UPDATE task_context " +
        "SET total_entries = total_entries + 1, total_tokens = total_tokens + ?, updated_at = ? " +
        "WHERE task_id = ?",
    ).run(tokenCount, createdAt, taskId);

    const row = db
      .prepare(`SELECT ${TASK_CONTEXT_ENTRY_SELECT_COLUMNS} FROM task_context_entries WHERE id = ?`)
      .get(id) as TaskContextEntryRow | undefined;
    if (!row) {
      throw new Error(`failed to persist task context entry: ${id}`);
    }
    return rowToTaskContextEntry(row);
  } finally {
    db.close();
  }
}

export function listTaskContextEntries(params: ListTaskContextEntriesParams): TaskContextEntry[] {
  const opened = openTaskContextDb({
    agentId: params.agentId,
    taskId: params.taskId,
    createIfMissing: false,
    env: params.env,
  });
  if (!opened) {
    return [];
  }
  const { db, taskId } = opened;
  try {
    const clauses = ["task_id = ?"];
    const values: Array<string | number> = [taskId];
    if (Number.isFinite(params.afterSequence)) {
      clauses.push("sequence > ?");
      values.push(Math.floor(params.afterSequence as number));
    }
    if (Number.isFinite(params.beforeSequence)) {
      clauses.push("sequence < ?");
      values.push(Math.floor(params.beforeSequence as number));
    }
    const limit = normalizeListLimit(params.limit, DEFAULT_LIST_LIMIT);
    const rows = db
      .prepare(
        `SELECT ${TASK_CONTEXT_ENTRY_SELECT_COLUMNS} FROM task_context_entries ` +
          `WHERE ${clauses.join(" AND ")} ORDER BY sequence ASC LIMIT ?`,
      )
      .all(...values, limit) as TaskContextEntryRow[];
    return rows.map((row) => rowToTaskContextEntry(row));
  } finally {
    db.close();
  }
}

export function updateTaskContextStatus(params: UpdateTaskContextStatusParams): TaskContext | null {
  const opened = openTaskContextDb({
    agentId: params.agentId,
    taskId: params.taskId,
    createIfMissing: false,
    env: params.env,
  });
  if (!opened) {
    return null;
  }
  const { db, taskId } = opened;
  try {
    const existing = readTaskContextRow(db, taskId);
    if (!existing) {
      return null;
    }
    const status = normalizeTaskStatus(params.status);
    const updatedAt = normalizeTimestamp(params.updatedAt);
    const completedAt = resolveCompletedAt({
      status,
      updatedAt,
      completedAt: params.completedAt,
      existingCompletedAt: existing.completed_at,
    });

    db.prepare(
      "UPDATE task_context SET status = ?, updated_at = ?, completed_at = ? WHERE task_id = ?",
    ).run(status, updatedAt, completedAt, taskId);
    const row = readTaskContextRow(db, taskId);
    return row ? rowToTaskContext(row) : null;
  } finally {
    db.close();
  }
}

function openTaskContextDb(params: OpenTaskContextDbParams): OpenTaskContextDbResult | null {
  const agentId = normalizeAgentId(params.agentId);
  const taskId = normalizeTaskId(params.taskId);
  const dbPath = resolveTaskContextDbPath({ agentId, taskId, env: params.env });
  if (!params.createIfMissing && !fsSync.existsSync(dbPath)) {
    return null;
  }
  fsSync.mkdirSync(path.dirname(dbPath), { recursive: true });
  const { DatabaseSync } = requireNodeSqlite();
  const db = new DatabaseSync(dbPath);
  db.exec("PRAGMA foreign_keys = ON;");
  ensureTaskContextSchema(db);
  return { db, dbPath, agentId, taskId };
}

function ensureTaskContextSchema(db: DatabaseSync): void {
  db.exec(
    "CREATE TABLE IF NOT EXISTS task_context (" +
      "task_id TEXT PRIMARY KEY, " +
      "agent_id TEXT NOT NULL, " +
      "title TEXT NOT NULL, " +
      "status TEXT NOT NULL, " +
      "parent_task_id TEXT, " +
      "scope_id TEXT NOT NULL, " +
      "created_at INTEGER NOT NULL, " +
      "updated_at INTEGER NOT NULL, " +
      "completed_at INTEGER, " +
      "total_entries INTEGER NOT NULL DEFAULT 0, " +
      "total_tokens INTEGER NOT NULL DEFAULT 0" +
      ");",
  );
  db.exec(
    "CREATE TABLE IF NOT EXISTS task_context_entries (" +
      "id TEXT PRIMARY KEY, " +
      "task_id TEXT NOT NULL, " +
      "sequence INTEGER NOT NULL, " +
      "role TEXT NOT NULL, " +
      "content TEXT NOT NULL, " +
      "content_hash TEXT NOT NULL, " +
      "summary TEXT, " +
      "token_count INTEGER NOT NULL, " +
      "created_at INTEGER NOT NULL, " +
      "metadata_json TEXT, " +
      "summarized INTEGER NOT NULL DEFAULT 0, " +
      "FOREIGN KEY (task_id) REFERENCES task_context(task_id) ON DELETE CASCADE" +
      ");",
  );
  db.exec(
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_task_context_entries_sequence " +
      "ON task_context_entries (task_id, sequence);",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_task_context_entries_created_at " +
      "ON task_context_entries (task_id, created_at, sequence);",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_task_context_entries_content_hash " +
      "ON task_context_entries (task_id, content_hash);",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_task_context_status " +
      "ON task_context (agent_id, status, updated_at);",
  );
  ensureColumn(db, "task_context_entries", "metadata_json", "TEXT");
  ensureColumn(db, "task_context_entries", "summarized", "INTEGER NOT NULL DEFAULT 0");
}

function ensureColumn(
  db: DatabaseSync,
  table: "task_context_entries",
  column: string,
  definition: string,
): void {
  const rows = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name?: string }>;
  if (rows.some((row) => row.name === column)) {
    return;
  }
  db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${definition}`);
}

function readTaskContextRow(db: DatabaseSync, taskId: string): TaskContextRow | undefined {
  return db
    .prepare(`SELECT ${TASK_CONTEXT_SELECT_COLUMNS} FROM task_context WHERE task_id = ?`)
    .get(taskId) as TaskContextRow | undefined;
}

function rowToTaskContext(row: TaskContextRow): TaskContext {
  const status = coerceTaskStatus(row.status);
  return {
    taskId: row.task_id,
    agentId: row.agent_id,
    title: row.title,
    status,
    parentTaskId: row.parent_task_id ?? undefined,
    scopeId: row.scope_id,
    createdAt: Number(row.created_at ?? 0),
    updatedAt: Number(row.updated_at ?? 0),
    completedAt: row.completed_at == null ? undefined : Number(row.completed_at),
    totalEntries: Math.max(0, Math.floor(Number(row.total_entries ?? 0))),
    totalTokens: Math.max(0, Math.floor(Number(row.total_tokens ?? 0))),
  };
}

function rowToTaskContextEntry(row: TaskContextEntryRow): TaskContextEntry {
  return {
    id: row.id,
    taskId: row.task_id,
    sequence: Math.max(1, Math.floor(Number(row.sequence ?? 1))),
    role: coerceTaskRole(row.role),
    content: row.content,
    contentHash: row.content_hash,
    summary: row.summary ?? undefined,
    tokenCount: Math.max(1, Math.floor(Number(row.token_count ?? 1))),
    createdAt: Number(row.created_at ?? 0),
    metadata: row.metadata_json ?? undefined,
    summarized: Number(row.summarized ?? 0) > 0,
  };
}

function normalizeScopeId(scopeId: string | undefined, agentId: string, taskId: string): string {
  const normalized = scopeId?.trim();
  if (normalized) {
    return normalized.toLowerCase();
  }
  return `${DEFAULT_SCOPE_PREFIX}:${agentId}:${taskId}`;
}

function normalizeTaskStatus(value: string): TaskStatus {
  const normalized = value.trim().toLowerCase();
  if (TASK_STATUSES.has(normalized as TaskStatus)) {
    return normalized as TaskStatus;
  }
  throw new Error(`Invalid task status: ${value}`);
}

function normalizeTaskRole(value: string): TaskContextRole {
  const normalized = value.trim().toLowerCase();
  if (TASK_ROLES.has(normalized as TaskContextRole)) {
    return normalized as TaskContextRole;
  }
  throw new Error(`Invalid task context role: ${value}`);
}

function coerceTaskStatus(value: string): TaskStatus {
  const normalized = value.trim().toLowerCase();
  if (TASK_STATUSES.has(normalized as TaskStatus)) {
    return normalized as TaskStatus;
  }
  return "active";
}

function coerceTaskRole(value: string): TaskContextRole {
  const normalized = value.trim().toLowerCase();
  if (TASK_ROLES.has(normalized as TaskContextRole)) {
    return normalized as TaskContextRole;
  }
  return "assistant";
}

function normalizeMetadata(input: TaskEntryMetadataInput | undefined): string | null {
  if (input === undefined) {
    return null;
  }
  if (typeof input === "string") {
    const normalized = input.trim();
    return normalized || null;
  }
  try {
    return JSON.stringify(input);
  } catch {
    return null;
  }
}

function normalizeTokenCount(value: number | undefined, content: string): number {
  if (Number.isFinite(value)) {
    return Math.max(1, Math.floor(value as number));
  }
  return Math.max(1, Math.ceil(content.length / 4));
}

function nextEntrySequence(db: DatabaseSync, taskId: string): number {
  const row = db
    .prepare(
      "SELECT COALESCE(MAX(sequence), 0) AS max_sequence FROM task_context_entries WHERE task_id = ?",
    )
    .get(taskId) as { max_sequence?: number } | undefined;
  const maxSequence = Math.max(0, Math.floor(Number(row?.max_sequence ?? 0)));
  return maxSequence + 1;
}

function normalizeTimestamp(value: number | undefined): number {
  if (Number.isFinite(value)) {
    return Math.floor(value as number);
  }
  return Date.now();
}

function normalizeListLimit(value: number | undefined, fallback: number): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(1, Math.min(MAX_LIST_LIMIT, Math.floor(value as number)));
}

function resolveCompletedAt(params: {
  status: TaskStatus;
  updatedAt: number;
  completedAt: number | null | undefined;
  existingCompletedAt: number | null;
}): number | null {
  if (params.status === "completed") {
    if (Number.isFinite(params.completedAt)) {
      return Math.floor(params.completedAt as number);
    }
    return params.updatedAt;
  }
  if (params.status === "active" || params.status === "paused") {
    return null;
  }
  if (Number.isFinite(params.completedAt)) {
    return Math.floor(params.completedAt as number);
  }
  return params.existingCompletedAt == null ? null : Math.floor(params.existingCompletedAt);
}

export function buildTaskContextEntryHash(content: string): string {
  return hashText(content.trim());
}
