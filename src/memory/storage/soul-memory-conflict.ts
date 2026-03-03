import crypto from "node:crypto";
import fsSync from "node:fs";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { resolveSoulMemoryDbPath } from "./soul-memory-store.js";
import { requireNodeSqlite } from "./sqlite.js";

type MemoryRow = {
  id: string;
  scope_type: string;
  scope_id: string;
  kind: string;
  content: string;
  confidence: number;
};

type ConflictRow = {
  id: string;
  memory_id_a: string;
  memory_id_b: string;
  content_a: string;
  content_b: string;
  conflict_reason: string;
  detected_at: number;
  resolved_at: number | null;
  resolution: string | null;
  resolved_by: string | null;
};

type MemoryPair = {
  left: MemoryRow;
  right: MemoryRow;
};

export type SoulMemoryConflict = {
  id: string;
  memoryIdA: string;
  memoryIdB: string;
  contentA: string;
  contentB: string;
  conflictReason: string;
  detectedAt: number;
  resolvedAt?: number;
  resolution?: string;
  resolvedBy?: string;
};

export type SoulMemoryConflictDetectionResult = {
  inserted: number;
  conflicts: SoulMemoryConflict[];
};

const NEGATION_HINTS = [
  "not",
  "never",
  "dont",
  "don't",
  "cannot",
  "can't",
  "禁止",
  "不要",
  "不能",
  "不可",
  "永远不要",
];

const POLARITY_HINTS = [
  ["always", "never"],
  ["must", "must not"],
  ["allow", "forbid"],
  ["enable", "disable"],
  ["should", "should not"],
  ["总是", "不要"],
  ["必须", "禁止"],
  ["允许", "禁止"],
  ["可以", "不可以"],
];

const SOFT_STOPWORDS = new Set([
  "the",
  "a",
  "an",
  "to",
  "for",
  "of",
  "in",
  "on",
  "and",
  "or",
  "is",
  "are",
  "be",
  "this",
  "that",
  "with",
  "we",
  "you",
  "should",
  "must",
  "not",
  "never",
  "do",
  "does",
  "did",
]);

export function detectSoulMemoryConflicts(params: {
  agentId: string;
  minConfidence?: number;
  overlapThreshold?: number;
  nowMs?: number;
  judgeConflict?: (input: { left: string; right: string; kind: string }) => string | null;
}): SoulMemoryConflictDetectionResult {
  const minConfidence = clamp(params.minConfidence ?? 0.7, 0, 1);
  const overlapThreshold = clamp(params.overlapThreshold ?? 0.2, 0, 1);
  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  const db = openSoulMemoryDb(params.agentId);
  try {
    ensureConflictSchema(db);
    const memories = loadMemoryRows(db);
    if (memories.length < 2) {
      return { inserted: 0, conflicts: [] };
    }

    const pairs = collectCandidatePairs(memories, minConfidence);
    const inserted: SoulMemoryConflict[] = [];

    const insertStmt = db.prepare(
      "INSERT INTO memory_conflicts (" +
        "id, memory_id_a, memory_id_b, content_a, content_b, conflict_reason, detected_at, resolved_at, resolution, resolved_by" +
        ") VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
    );
    const existsStmt = db.prepare(
      "SELECT id FROM memory_conflicts WHERE memory_id_a = ? AND memory_id_b = ? AND resolved_at IS NULL LIMIT 1",
    );

    for (const pair of pairs) {
      const overlap = tokenOverlap(pair.left.content, pair.right.content);
      if (overlap < overlapThreshold) {
        continue;
      }
      const conflictReason =
        detectRuleBasedConflict(pair.left.content, pair.right.content) ??
        params.judgeConflict?.({
          left: pair.left.content,
          right: pair.right.content,
          kind: pair.left.kind,
        }) ??
        null;
      if (!conflictReason) {
        continue;
      }
      const ordered = orderPair(pair.left.id, pair.right.id);
      const exists = existsStmt.get(ordered.first, ordered.second) as { id?: string } | undefined;
      if (exists?.id) {
        continue;
      }
      const id = `mcf_${crypto.randomUUID().replace(/-/g, "")}`;
      insertStmt.run(
        id,
        ordered.first,
        ordered.second,
        ordered.first === pair.left.id ? pair.left.content : pair.right.content,
        ordered.second === pair.right.id ? pair.right.content : pair.left.content,
        conflictReason,
        nowMs,
      );
      inserted.push({
        id,
        memoryIdA: ordered.first,
        memoryIdB: ordered.second,
        contentA: ordered.first === pair.left.id ? pair.left.content : pair.right.content,
        contentB: ordered.second === pair.right.id ? pair.right.content : pair.left.content,
        conflictReason,
        detectedAt: nowMs,
      });
    }

    return {
      inserted: inserted.length,
      conflicts: inserted,
    };
  } finally {
    db.close();
  }
}

export function listSoulMemoryConflicts(params: {
  agentId: string;
  unresolvedOnly?: boolean;
  limit?: number;
}): SoulMemoryConflict[] {
  const db = openSoulMemoryDb(params.agentId);
  try {
    ensureConflictSchema(db);
    const unresolvedOnly = params.unresolvedOnly !== false;
    const limit = Number.isFinite(params.limit)
      ? Math.max(1, Math.floor(params.limit as number))
      : 200;
    const where = unresolvedOnly ? "WHERE resolved_at IS NULL" : "";
    const rows = db
      .prepare(
        "SELECT id, memory_id_a, memory_id_b, content_a, content_b, conflict_reason, detected_at, " +
          "resolved_at, resolution, resolved_by FROM memory_conflicts " +
          `${where} ORDER BY detected_at DESC LIMIT ?`,
      )
      .all(limit) as ConflictRow[];
    return rows.map((row) => ({
      id: row.id,
      memoryIdA: row.memory_id_a,
      memoryIdB: row.memory_id_b,
      contentA: row.content_a,
      contentB: row.content_b,
      conflictReason: row.conflict_reason,
      detectedAt: Number(row.detected_at ?? 0),
      resolvedAt: row.resolved_at == null ? undefined : Number(row.resolved_at),
      resolution: row.resolution ?? undefined,
      resolvedBy: row.resolved_by ?? undefined,
    }));
  } finally {
    db.close();
  }
}

function collectCandidatePairs(memories: MemoryRow[], minConfidence: number): MemoryPair[] {
  const pairs: MemoryPair[] = [];
  const grouped = new Map<string, MemoryRow[]>();
  for (const item of memories) {
    const key = `${item.scope_type}::${item.scope_id}::${item.kind}`;
    const bucket = grouped.get(key) ?? [];
    bucket.push(item);
    grouped.set(key, bucket);
  }
  for (const group of grouped.values()) {
    if (group.length < 2) {
      continue;
    }
    for (let i = 0; i < group.length; i += 1) {
      const left = group[i];
      if (!left) {
        continue;
      }
      for (let j = i + 1; j < group.length; j += 1) {
        const right = group[j];
        if (!right) {
          continue;
        }
        const leftConfidence = Number(left.confidence ?? 0);
        const rightConfidence = Number(right.confidence ?? 0);
        if (leftConfidence < minConfidence && rightConfidence < minConfidence) {
          continue;
        }
        pairs.push({ left, right });
      }
    }
  }
  return pairs;
}

function detectRuleBasedConflict(leftRaw: string, rightRaw: string): string | null {
  const left = normalizeText(leftRaw);
  const right = normalizeText(rightRaw);
  if (!left || !right || left === right) {
    return null;
  }
  for (const [positive, negative] of POLARITY_HINTS) {
    const leftHasPositive = left.includes(positive);
    const leftHasNegative = left.includes(negative);
    const rightHasPositive = right.includes(positive);
    const rightHasNegative = right.includes(negative);
    if ((leftHasPositive && rightHasNegative) || (leftHasNegative && rightHasPositive)) {
      return `opposite policy markers: "${positive}" vs "${negative}"`;
    }
  }

  const leftNegated = hasNegation(left);
  const rightNegated = hasNegation(right);
  if (leftNegated === rightNegated) {
    return null;
  }
  const overlap = subjectTokenOverlap(left, right);
  if (overlap < 0.34) {
    return null;
  }
  return "negation conflict on overlapping subject";
}

function subjectTokenOverlap(a: string, b: string): number {
  const left = tokenize(a, { dropNegation: true });
  const right = tokenize(b, { dropNegation: true });
  if (left.size === 0 || right.size === 0) {
    return 0;
  }
  let intersection = 0;
  for (const token of left) {
    if (right.has(token)) {
      intersection += 1;
    }
  }
  const union = left.size + right.size - intersection;
  return union > 0 ? intersection / union : 0;
}

function tokenOverlap(a: string, b: string): number {
  const left = tokenize(normalizeText(a), { dropNegation: false });
  const right = tokenize(normalizeText(b), { dropNegation: false });
  if (left.size === 0 || right.size === 0) {
    return 0;
  }
  let intersection = 0;
  for (const token of left) {
    if (right.has(token)) {
      intersection += 1;
    }
  }
  const union = left.size + right.size - intersection;
  return union > 0 ? intersection / union : 0;
}

function hasNegation(input: string): boolean {
  return NEGATION_HINTS.some((hint) => input.includes(hint));
}

function tokenize(
  value: string,
  opts: {
    dropNegation: boolean;
  },
): Set<string> {
  const out = new Set<string>();
  const matches = value.match(/[a-z0-9_]+|[\u4e00-\u9fff]+/g) ?? [];
  for (const token of matches) {
    if (!token) {
      continue;
    }
    if (SOFT_STOPWORDS.has(token)) {
      continue;
    }
    if (opts.dropNegation && NEGATION_HINTS.includes(token)) {
      continue;
    }
    out.add(token);
  }
  return out;
}

function loadMemoryRows(db: DatabaseSync): MemoryRow[] {
  return db
    .prepare(
      "SELECT id, scope_type, scope_id, kind, content, confidence " +
        "FROM memory_items ORDER BY confidence DESC, created_at ASC",
    )
    .all() as MemoryRow[];
}

function orderPair(a: string, b: string): { first: string; second: string } {
  return a.localeCompare(b) <= 0 ? { first: a, second: b } : { first: b, second: a };
}

function ensureConflictSchema(db: DatabaseSync): void {
  db.exec(
    "CREATE TABLE IF NOT EXISTS memory_conflicts (" +
      "id TEXT PRIMARY KEY, " +
      "memory_id_a TEXT NOT NULL, " +
      "memory_id_b TEXT NOT NULL, " +
      "content_a TEXT NOT NULL, " +
      "content_b TEXT NOT NULL, " +
      "conflict_reason TEXT NOT NULL, " +
      "detected_at INTEGER NOT NULL, " +
      "resolved_at INTEGER, " +
      "resolution TEXT, " +
      "resolved_by TEXT, " +
      "FOREIGN KEY (memory_id_a) REFERENCES memory_items(id) ON DELETE CASCADE, " +
      "FOREIGN KEY (memory_id_b) REFERENCES memory_items(id) ON DELETE CASCADE" +
      ");",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_memory_conflicts_detected_at ON memory_conflicts (detected_at DESC);",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_memory_conflicts_pair ON memory_conflicts (memory_id_a, memory_id_b, resolved_at);",
  );
}

function openSoulMemoryDb(agentId: string): DatabaseSync {
  const dbPath = resolveSoulMemoryDbPath(agentId);
  fsSync.mkdirSync(path.dirname(dbPath), { recursive: true });
  const { DatabaseSync } = requireNodeSqlite();
  const db = new DatabaseSync(dbPath);
  db.exec("PRAGMA foreign_keys = ON;");
  return db;
}

function normalizeText(value: string): string {
  return value.toLowerCase().replace(/\s+/g, " ").trim();
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, value));
}
