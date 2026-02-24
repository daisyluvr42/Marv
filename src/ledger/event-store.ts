import crypto from "node:crypto";
import fsSync from "node:fs";
import os from "node:os";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { resolveStateDir } from "../config/paths.js";
import { requireNodeSqlite } from "../memory/sqlite.js";

export type LedgerEventRecord = {
  id: number;
  eventId: string;
  taskId: string | null;
  conversationId: string;
  type: string;
  ts: number;
  actorId: string | null;
  payload: Record<string, unknown>;
};

export type LedgerAppendEventParams = {
  eventId?: string;
  taskId?: string;
  conversationId: string;
  type: string;
  ts?: number;
  actorId?: string;
  payload?: Record<string, unknown>;
};

export type LedgerQueryParams = {
  conversationId: string;
  taskId?: string;
  type?: string;
  fromTs?: number;
  toTs?: number;
  limit?: number;
};

type LedgerEventRow = {
  id: number;
  event_id: string;
  task_id: string | null;
  conversation_id: string;
  type: string;
  ts: number;
  actor_id: string | null;
  payload_json: string;
};

export function resolveLedgerDbPath(): string {
  const stateDir = resolveStateDir(process.env, os.homedir);
  return path.join(stateDir, "ledger", "events.sqlite");
}

export function appendLedgerEvent(params: LedgerAppendEventParams): LedgerEventRecord {
  const conversationId = params.conversationId.trim();
  const eventType = params.type.trim();
  if (!conversationId) {
    throw new Error("conversationId required");
  }
  if (!eventType) {
    throw new Error("type required");
  }

  const db = openLedgerDb();
  try {
    const eventId = params.eventId?.trim() || `evt_${crypto.randomUUID().replace(/-/g, "")}`;
    const ts = Number.isFinite(params.ts) ? Math.floor(params.ts as number) : Date.now();
    const payloadJson = JSON.stringify(params.payload ?? {});
    db.prepare(
      "INSERT INTO ledger_events (" +
        "event_id, task_id, conversation_id, type, ts, actor_id, payload_json" +
        ") VALUES (?, ?, ?, ?, ?, ?, ?)",
    ).run(
      eventId,
      params.taskId?.trim() || null,
      conversationId,
      eventType,
      ts,
      params.actorId?.trim() || null,
      payloadJson,
    );

    const row = db
      .prepare(
        "SELECT id, event_id, task_id, conversation_id, type, ts, actor_id, payload_json " +
          "FROM ledger_events WHERE event_id = ?",
      )
      .get(eventId) as LedgerEventRow | undefined;
    if (!row) {
      throw new Error("failed to read appended ledger event");
    }
    return rowToRecord(row);
  } finally {
    db.close();
  }
}

export function queryLedgerEvents(params: LedgerQueryParams): LedgerEventRecord[] {
  const conversationId = params.conversationId.trim();
  if (!conversationId) {
    return [];
  }

  const clauses = ["conversation_id = ?"];
  const values: Array<string | number> = [conversationId];

  if (params.taskId?.trim()) {
    clauses.push("task_id = ?");
    values.push(params.taskId.trim());
  }
  if (params.type?.trim()) {
    clauses.push("type = ?");
    values.push(params.type.trim());
  }
  if (Number.isFinite(params.fromTs)) {
    clauses.push("ts >= ?");
    values.push(Math.floor(params.fromTs as number));
  }
  if (Number.isFinite(params.toTs)) {
    clauses.push("ts <= ?");
    values.push(Math.floor(params.toTs as number));
  }

  const limit = Math.max(1, Math.min(2000, Math.floor(params.limit ?? 200)));
  const db = openLedgerDb();
  try {
    const rows = db
      .prepare(
        "SELECT id, event_id, task_id, conversation_id, type, ts, actor_id, payload_json " +
          "FROM ledger_events WHERE " +
          clauses.join(" AND ") +
          " ORDER BY ts ASC, id ASC LIMIT ?",
      )
      .all(...values, limit) as LedgerEventRow[];
    return rows.map((row) => rowToRecord(row));
  } finally {
    db.close();
  }
}

function rowToRecord(row: LedgerEventRow): LedgerEventRecord {
  return {
    id: row.id,
    eventId: row.event_id,
    taskId: row.task_id,
    conversationId: row.conversation_id,
    type: row.type,
    ts: row.ts,
    actorId: row.actor_id,
    payload: parsePayload(row.payload_json),
  };
}

function parsePayload(raw: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // Fall through to empty payload.
  }
  return {};
}

function openLedgerDb(): DatabaseSync {
  const dbPath = resolveLedgerDbPath();
  fsSync.mkdirSync(path.dirname(dbPath), { recursive: true });
  const { DatabaseSync } = requireNodeSqlite();
  const db = new DatabaseSync(dbPath);
  ensureLedgerSchema(db);
  return db;
}

function ensureLedgerSchema(db: DatabaseSync): void {
  db.exec(
    "CREATE TABLE IF NOT EXISTS ledger_events (" +
      "id INTEGER PRIMARY KEY AUTOINCREMENT, " +
      "event_id TEXT NOT NULL UNIQUE, " +
      "task_id TEXT, " +
      "conversation_id TEXT NOT NULL, " +
      "type TEXT NOT NULL, " +
      "ts INTEGER NOT NULL, " +
      "actor_id TEXT, " +
      "payload_json TEXT NOT NULL" +
      ");",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_ledger_events_conversation_ts " +
      "ON ledger_events (conversation_id, ts, id);",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_ledger_events_task_ts ON ledger_events (task_id, ts, id);",
  );
  db.exec("CREATE INDEX IF NOT EXISTS idx_ledger_events_type_ts ON ledger_events (type, ts, id);");
}
