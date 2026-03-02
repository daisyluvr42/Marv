import crypto from "node:crypto";
import fsSync from "node:fs";
import os from "node:os";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { requireNodeSqlite } from "../../memory/storage/sqlite.js";
import { resolveStateDir } from "./paths.js";
import type { MarvConfig } from "./types.marv.js";

export type SemanticPatchRiskLevel = "L1" | "L2" | "L3";
export type SemanticPatchProposalStatus = "open" | "committed" | "rejected";
export type SemanticConfigRevisionStatus = "committed" | "rolled_back";

export type SemanticPatchCompilation = {
  patch: Record<string, unknown>;
  riskLevel: SemanticPatchRiskLevel;
  explanation: string;
  needsApproval: boolean;
};

export type SemanticPatchProposal = {
  proposalId: string;
  scopeType: string;
  scopeId: string;
  naturalLanguage: string;
  patch: Record<string, unknown>;
  riskLevel: SemanticPatchRiskLevel;
  explanation: string;
  needsApproval: boolean;
  createdAt: number;
  actorId: string;
  status: SemanticPatchProposalStatus;
};

export type SemanticConfigRevision = {
  revision: string;
  proposalId: string | null;
  scopeType: string;
  scopeId: string;
  createdAt: number;
  actorId: string;
  patch: Record<string, unknown>;
  explanation: string;
  riskLevel: SemanticPatchRiskLevel;
  status: SemanticConfigRevisionStatus;
  beforeConfig: MarvConfig | null;
  afterConfig: MarvConfig | null;
};

export function resolveSemanticPatchDbPath(): string {
  const stateDir = resolveStateDir(process.env, os.homedir);
  return path.join(stateDir, "config", "semantic-patches.sqlite");
}

export function compileSemanticPatch(naturalLanguage: string): SemanticPatchCompilation {
  const text = naturalLanguage.trim();
  const lowered = text.toLowerCase();

  const explicitPatch = tryExtractPatchObject(text);
  if (explicitPatch) {
    return {
      patch: explicitPatch,
      riskLevel: "L2",
      explanation: "Detected explicit JSON patch payload and routed it as semantic patch.",
      needsApproval: true,
    };
  }

  if (hasAny(lowered, ["简洁", "简短", "concise", "shorter", "less verbose"])) {
    return {
      patch: {
        agents: {
          defaults: {
            thinkingDefault: "low",
          },
        },
      },
      riskLevel: "L1",
      explanation: "Lowered default thinking depth for more concise responses.",
      needsApproval: false,
    };
  }

  if (hasAny(lowered, ["详细", "更详细", "深入", "detailed", "deep dive"])) {
    return {
      patch: {
        agents: {
          defaults: {
            thinkingDefault: "high",
          },
        },
      },
      riskLevel: "L1",
      explanation: "Raised default thinking depth for more detailed responses.",
      needsApproval: false,
    };
  }

  if (
    hasAny(lowered, [
      "外部写",
      "workspace 外",
      "workspace外",
      "unrestricted file",
      "external write",
      "allow outside workspace",
    ])
  ) {
    return {
      patch: {
        tools: {
          fs: {
            workspaceOnly: false,
          },
        },
      },
      riskLevel: "L3",
      explanation: "This change relaxes filesystem write boundaries and is high risk.",
      needsApproval: true,
    };
  }

  if (hasAny(lowered, ["只允许工作区", "workspace only", "restrict filesystem"])) {
    return {
      patch: {
        tools: {
          fs: {
            workspaceOnly: true,
          },
        },
      },
      riskLevel: "L1",
      explanation: "Enforced workspace-only filesystem access for safer operations.",
      needsApproval: false,
    };
  }

  if (hasAny(lowered, ["关闭记忆刷新", "关闭 memory flush", "disable memory flush"])) {
    return {
      patch: {
        agents: {
          defaults: {
            compaction: {
              memoryFlush: {
                enabled: false,
              },
            },
          },
        },
      },
      riskLevel: "L2",
      explanation:
        "Disabled pre-compaction memory flush; this can reduce memory persistence reliability.",
      needsApproval: true,
    };
  }

  if (hasAny(lowered, ["启用记忆刷新", "enable memory flush", "memory flush on"])) {
    return {
      patch: {
        agents: {
          defaults: {
            compaction: {
              memoryFlush: {
                enabled: true,
              },
            },
          },
        },
      },
      riskLevel: "L1",
      explanation: "Enabled pre-compaction memory flush for stronger durable memory capture.",
      needsApproval: false,
    };
  }

  return {
    patch: {
      agents: {
        defaults: {
          thinkingDefault: "medium",
        },
      },
    },
    riskLevel: "L2",
    explanation: "No explicit semantic rule matched; applying balanced-thinking default patch.",
    needsApproval: true,
  };
}

export function createSemanticPatchProposal(params: {
  scopeType: string;
  scopeId: string;
  naturalLanguage: string;
  actorId: string;
  nowMs?: number;
}): SemanticPatchProposal {
  const scopeType = normalizeScope(params.scopeType, "global");
  const scopeId = normalizeScope(params.scopeId, "gateway");
  const naturalLanguage = params.naturalLanguage.trim();
  const actorId = params.actorId.trim() || "gateway";
  if (!naturalLanguage) {
    throw new Error("naturalLanguage required");
  }

  const compiled = compileSemanticPatch(naturalLanguage);
  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  const proposalId = `pp_${crypto.randomUUID().replace(/-/g, "")}`;

  const db = openSemanticPatchDb();
  try {
    db.prepare(
      "INSERT INTO semantic_patch_proposals (" +
        "proposal_id, scope_type, scope_id, natural_language, patch_json, risk_level, explanation, " +
        "needs_approval, created_at, actor_id, status" +
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    ).run(
      proposalId,
      scopeType,
      scopeId,
      naturalLanguage,
      JSON.stringify(compiled.patch),
      compiled.riskLevel,
      compiled.explanation,
      compiled.needsApproval ? 1 : 0,
      nowMs,
      actorId,
      "open",
    );
    const row = db
      .prepare(
        "SELECT proposal_id, scope_type, scope_id, natural_language, patch_json, risk_level, explanation, " +
          "needs_approval, created_at, actor_id, status " +
          "FROM semantic_patch_proposals WHERE proposal_id = ?",
      )
      .get(proposalId) as SemanticPatchProposalRow | undefined;
    if (!row) {
      throw new Error("failed to read created proposal");
    }
    return rowToProposal(row);
  } finally {
    db.close();
  }
}

export function getSemanticPatchProposal(proposalId: string): SemanticPatchProposal | null {
  const id = proposalId.trim();
  if (!id) {
    return null;
  }
  const db = openSemanticPatchDb();
  try {
    const row = db
      .prepare(
        "SELECT proposal_id, scope_type, scope_id, natural_language, patch_json, risk_level, explanation, " +
          "needs_approval, created_at, actor_id, status " +
          "FROM semantic_patch_proposals WHERE proposal_id = ?",
      )
      .get(id) as SemanticPatchProposalRow | undefined;
    return row ? rowToProposal(row) : null;
  } finally {
    db.close();
  }
}

export function updateSemanticPatchProposalStatus(params: {
  proposalId: string;
  status: SemanticPatchProposalStatus;
}): SemanticPatchProposal | null {
  const proposalId = params.proposalId.trim();
  if (!proposalId) {
    return null;
  }
  const db = openSemanticPatchDb();
  try {
    const result = db
      .prepare("UPDATE semantic_patch_proposals SET status = ? WHERE proposal_id = ?")
      .run(params.status, proposalId);
    if ((result.changes ?? 0) <= 0) {
      return null;
    }
    const row = db
      .prepare(
        "SELECT proposal_id, scope_type, scope_id, natural_language, patch_json, risk_level, explanation, " +
          "needs_approval, created_at, actor_id, status " +
          "FROM semantic_patch_proposals WHERE proposal_id = ?",
      )
      .get(proposalId) as SemanticPatchProposalRow | undefined;
    return row ? rowToProposal(row) : null;
  } finally {
    db.close();
  }
}

export function createSemanticConfigRevision(params: {
  proposalId?: string;
  scopeType: string;
  scopeId: string;
  actorId: string;
  patch: Record<string, unknown>;
  explanation: string;
  riskLevel: SemanticPatchRiskLevel;
  status?: SemanticConfigRevisionStatus;
  beforeConfig?: MarvConfig | null;
  afterConfig?: MarvConfig | null;
  nowMs?: number;
}): SemanticConfigRevision {
  const revision = `rev_${crypto.randomUUID().replace(/-/g, "")}`;
  const scopeType = normalizeScope(params.scopeType, "global");
  const scopeId = normalizeScope(params.scopeId, "gateway");
  const actorId = params.actorId.trim() || "gateway";
  const status = params.status ?? "committed";
  const createdAt = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  const patch = normalizePatch(params.patch);

  const db = openSemanticPatchDb();
  try {
    db.prepare(
      "INSERT INTO semantic_config_revisions (" +
        "revision, proposal_id, scope_type, scope_id, created_at, actor_id, patch_json, explanation, " +
        "risk_level, status, before_config_json, after_config_json" +
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    ).run(
      revision,
      params.proposalId?.trim() || null,
      scopeType,
      scopeId,
      createdAt,
      actorId,
      JSON.stringify(patch),
      params.explanation,
      params.riskLevel,
      status,
      params.beforeConfig ? JSON.stringify(params.beforeConfig) : null,
      params.afterConfig ? JSON.stringify(params.afterConfig) : null,
    );

    const row = db
      .prepare(
        "SELECT revision, proposal_id, scope_type, scope_id, created_at, actor_id, patch_json, explanation, " +
          "risk_level, status, before_config_json, after_config_json " +
          "FROM semantic_config_revisions WHERE revision = ?",
      )
      .get(revision) as SemanticConfigRevisionRow | undefined;
    if (!row) {
      throw new Error("failed to read created semantic config revision");
    }
    return rowToRevision(row);
  } finally {
    db.close();
  }
}

export function getSemanticConfigRevision(revision: string): SemanticConfigRevision | null {
  const id = revision.trim();
  if (!id) {
    return null;
  }
  const db = openSemanticPatchDb();
  try {
    const row = db
      .prepare(
        "SELECT revision, proposal_id, scope_type, scope_id, created_at, actor_id, patch_json, explanation, " +
          "risk_level, status, before_config_json, after_config_json " +
          "FROM semantic_config_revisions WHERE revision = ?",
      )
      .get(id) as SemanticConfigRevisionRow | undefined;
    return row ? rowToRevision(row) : null;
  } finally {
    db.close();
  }
}

export function listSemanticConfigRevisions(params?: {
  scopeType?: string;
  scopeId?: string;
  limit?: number;
}): SemanticConfigRevision[] {
  const clauses: string[] = [];
  const values: string[] = [];
  if (params?.scopeType?.trim()) {
    clauses.push("scope_type = ?");
    values.push(normalizeScope(params.scopeType, "global"));
  }
  if (params?.scopeId?.trim()) {
    clauses.push("scope_id = ?");
    values.push(normalizeScope(params.scopeId, "gateway"));
  }
  const where = clauses.length > 0 ? ` WHERE ${clauses.join(" AND ")}` : "";
  const limit = Math.max(1, Math.min(500, Math.floor(params?.limit ?? 100)));

  const db = openSemanticPatchDb();
  try {
    const rows = db
      .prepare(
        "SELECT revision, proposal_id, scope_type, scope_id, created_at, actor_id, patch_json, explanation, " +
          "risk_level, status, before_config_json, after_config_json " +
          "FROM semantic_config_revisions" +
          where +
          " ORDER BY created_at DESC LIMIT ?",
      )
      .all(...values, limit) as SemanticConfigRevisionRow[];
    return rows.map((row) => rowToRevision(row));
  } finally {
    db.close();
  }
}

export function updateSemanticConfigRevisionStatus(params: {
  revision: string;
  status: SemanticConfigRevisionStatus;
}): SemanticConfigRevision | null {
  const revision = params.revision.trim();
  if (!revision) {
    return null;
  }
  const db = openSemanticPatchDb();
  try {
    const result = db
      .prepare("UPDATE semantic_config_revisions SET status = ? WHERE revision = ?")
      .run(params.status, revision);
    if ((result.changes ?? 0) <= 0) {
      return null;
    }
    const row = db
      .prepare(
        "SELECT revision, proposal_id, scope_type, scope_id, created_at, actor_id, patch_json, explanation, " +
          "risk_level, status, before_config_json, after_config_json " +
          "FROM semantic_config_revisions WHERE revision = ?",
      )
      .get(revision) as SemanticConfigRevisionRow | undefined;
    return row ? rowToRevision(row) : null;
  } finally {
    db.close();
  }
}

export function findCommittedRevisionByProposalId(
  proposalId: string,
): SemanticConfigRevision | null {
  const id = proposalId.trim();
  if (!id) {
    return null;
  }
  const db = openSemanticPatchDb();
  try {
    const row = db
      .prepare(
        "SELECT revision, proposal_id, scope_type, scope_id, created_at, actor_id, patch_json, explanation, " +
          "risk_level, status, before_config_json, after_config_json " +
          "FROM semantic_config_revisions WHERE proposal_id = ? AND status = 'committed' " +
          "ORDER BY created_at DESC LIMIT 1",
      )
      .get(id) as SemanticConfigRevisionRow | undefined;
    return row ? rowToRevision(row) : null;
  } finally {
    db.close();
  }
}

export function buildSemanticConfigConversationId(scopeType: string, scopeId: string): string {
  return `config:${normalizeScope(scopeType, "global")}:${normalizeScope(scopeId, "gateway")}`;
}

type SemanticPatchProposalRow = {
  proposal_id: string;
  scope_type: string;
  scope_id: string;
  natural_language: string;
  patch_json: string;
  risk_level: SemanticPatchRiskLevel;
  explanation: string;
  needs_approval: number;
  created_at: number;
  actor_id: string;
  status: SemanticPatchProposalStatus;
};

type SemanticConfigRevisionRow = {
  revision: string;
  proposal_id: string | null;
  scope_type: string;
  scope_id: string;
  created_at: number;
  actor_id: string;
  patch_json: string;
  explanation: string;
  risk_level: SemanticPatchRiskLevel;
  status: SemanticConfigRevisionStatus;
  before_config_json: string | null;
  after_config_json: string | null;
};

function rowToProposal(row: SemanticPatchProposalRow): SemanticPatchProposal {
  return {
    proposalId: row.proposal_id,
    scopeType: row.scope_type,
    scopeId: row.scope_id,
    naturalLanguage: row.natural_language,
    patch: parsePatchJson(row.patch_json),
    riskLevel: row.risk_level,
    explanation: row.explanation,
    needsApproval: row.needs_approval === 1,
    createdAt: row.created_at,
    actorId: row.actor_id,
    status: row.status,
  };
}

function rowToRevision(row: SemanticConfigRevisionRow): SemanticConfigRevision {
  return {
    revision: row.revision,
    proposalId: row.proposal_id,
    scopeType: row.scope_type,
    scopeId: row.scope_id,
    createdAt: row.created_at,
    actorId: row.actor_id,
    patch: parsePatchJson(row.patch_json),
    explanation: row.explanation,
    riskLevel: row.risk_level,
    status: row.status,
    beforeConfig: parseConfigJson(row.before_config_json),
    afterConfig: parseConfigJson(row.after_config_json),
  };
}

function parsePatchJson(raw: string): Record<string, unknown> {
  try {
    return normalizePatch(JSON.parse(raw));
  } catch {
    return {};
  }
}

function parseConfigJson(raw: string | null): MarvConfig | null {
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as MarvConfig;
    }
  } catch {
    // ignore
  }
  return null;
}

function normalizePatch(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return value as Record<string, unknown>;
}

function normalizeScope(value: string, fallback: string): string {
  const normalized = value.trim().toLowerCase();
  return normalized || fallback;
}

function hasAny(loweredText: string, needles: string[]): boolean {
  return needles.some((needle) => loweredText.includes(needle));
}

function tryExtractPatchObject(text: string): Record<string, unknown> | null {
  if (!text.trim()) {
    return null;
  }
  const fenced = text.match(/```(?:json|json5)?\s*([\s\S]+?)\s*```/i);
  if (fenced?.[1]) {
    const parsed = tryParseObject(fenced[1]);
    if (parsed) {
      return parsed;
    }
  }
  if (text.trim().startsWith("{") && text.trim().endsWith("}")) {
    const parsed = tryParseObject(text);
    if (parsed) {
      return parsed;
    }
  }
  return null;
}

function tryParseObject(raw: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // ignore parse error
  }
  return null;
}

function openSemanticPatchDb(): DatabaseSync {
  const dbPath = resolveSemanticPatchDbPath();
  fsSync.mkdirSync(path.dirname(dbPath), { recursive: true });
  const { DatabaseSync } = requireNodeSqlite();
  const db = new DatabaseSync(dbPath);
  ensureSemanticPatchSchema(db);
  return db;
}

function ensureSemanticPatchSchema(db: DatabaseSync): void {
  db.exec(
    "CREATE TABLE IF NOT EXISTS semantic_patch_proposals (" +
      "proposal_id TEXT PRIMARY KEY, " +
      "scope_type TEXT NOT NULL, " +
      "scope_id TEXT NOT NULL, " +
      "natural_language TEXT NOT NULL, " +
      "patch_json TEXT NOT NULL, " +
      "risk_level TEXT NOT NULL, " +
      "explanation TEXT NOT NULL, " +
      "needs_approval INTEGER NOT NULL, " +
      "created_at INTEGER NOT NULL, " +
      "actor_id TEXT NOT NULL, " +
      "status TEXT NOT NULL" +
      ");",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_semantic_patch_scope ON semantic_patch_proposals (scope_type, scope_id, created_at);",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_semantic_patch_status ON semantic_patch_proposals (status, created_at);",
  );

  db.exec(
    "CREATE TABLE IF NOT EXISTS semantic_config_revisions (" +
      "revision TEXT PRIMARY KEY, " +
      "proposal_id TEXT, " +
      "scope_type TEXT NOT NULL, " +
      "scope_id TEXT NOT NULL, " +
      "created_at INTEGER NOT NULL, " +
      "actor_id TEXT NOT NULL, " +
      "patch_json TEXT NOT NULL, " +
      "explanation TEXT NOT NULL, " +
      "risk_level TEXT NOT NULL, " +
      "status TEXT NOT NULL, " +
      "before_config_json TEXT, " +
      "after_config_json TEXT" +
      ");",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_semantic_revision_scope ON semantic_config_revisions (scope_type, scope_id, created_at);",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_semantic_revision_status ON semantic_config_revisions (status, created_at);",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_semantic_revision_proposal ON semantic_config_revisions (proposal_id, created_at);",
  );
}
