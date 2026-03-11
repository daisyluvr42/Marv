import { LEGACY_CONFIG_MIGRATIONS } from "./legacy.migrations.js";
import { LEGACY_CONFIG_RULES } from "./legacy.rules.js";
import type { LegacyConfigIssue } from "./types.js";

export function findLegacyConfigIssues(raw: unknown): LegacyConfigIssue[] {
  if (!raw || typeof raw !== "object") {
    return [];
  }
  const root = raw as Record<string, unknown>;
  const issues: LegacyConfigIssue[] = [];
  for (const rule of LEGACY_CONFIG_RULES) {
    let cursor: unknown = root;
    for (const key of rule.path) {
      if (!cursor || typeof cursor !== "object") {
        cursor = undefined;
        break;
      }
      cursor = (cursor as Record<string, unknown>)[key];
    }
    if (cursor !== undefined && (!rule.match || rule.match(cursor, root))) {
      issues.push({ path: rule.path.join("."), message: rule.message });
    }
  }
  return issues;
}

export function findRemovedTopLevelMultiAgentIssues(raw: unknown): LegacyConfigIssue[] {
  if (!raw || typeof raw !== "object") {
    return [];
  }
  const root = raw as Record<string, unknown>;
  const issues: LegacyConfigIssue[] = [];

  const agents = root.agents;
  const list =
    agents && typeof agents === "object" ? (agents as { list?: unknown }).list : undefined;
  if (list !== undefined) {
    issues.push({
      path: "agents.list",
      message:
        'Top-level agents.list was removed. Keep only agents.defaults for "main" and use enhanced subagents for delegation.',
    });
  }

  if (root.bindings !== undefined) {
    issues.push({
      path: "bindings",
      message: 'Top-level bindings were removed. Routing now targets only "main".',
    });
  }

  return issues;
}

export function applyLegacyMigrations(raw: unknown): {
  next: Record<string, unknown> | null;
  changes: string[];
} {
  if (!raw || typeof raw !== "object") {
    return { next: null, changes: [] };
  }
  const next = structuredClone(raw) as Record<string, unknown>;
  const changes: string[] = [];
  for (const migration of LEGACY_CONFIG_MIGRATIONS) {
    migration.apply(next, changes);
  }
  if (changes.length === 0) {
    return { next: null, changes: [] };
  }
  return { next, changes };
}
