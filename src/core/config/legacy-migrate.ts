import { applyLegacyMigrations } from "./legacy.js";
import type { MarvConfig } from "./types.js";
import { validateConfigObjectWithPlugins } from "./validation.js";

function normalizeLegacyAgentId(value: unknown): string {
  return typeof value === "string" ? value.trim().toLowerCase() : "";
}

function listLegacyTopLevelAgentIds(raw: unknown): string[] {
  if (!raw || typeof raw !== "object") {
    return [];
  }
  const root = raw as Record<string, unknown>;
  const ids = new Set<string>();
  const agents = root.agents;
  if (agents && typeof agents === "object") {
    const list = (agents as { list?: unknown }).list;
    if (Array.isArray(list)) {
      for (const entry of list) {
        if (!entry || typeof entry !== "object") {
          continue;
        }
        const id = normalizeLegacyAgentId((entry as { id?: unknown }).id);
        if (id) {
          ids.add(id);
        }
      }
    }
  }
  if (Array.isArray(root.bindings)) {
    for (const binding of root.bindings) {
      if (!binding || typeof binding !== "object") {
        continue;
      }
      const id = normalizeLegacyAgentId((binding as { agentId?: unknown }).agentId);
      if (id) {
        ids.add(id);
      }
    }
  }
  return [...ids];
}

function hasLegacyTopLevelMultiAgentConfig(raw: unknown): boolean {
  if (!raw || typeof raw !== "object") {
    return false;
  }
  const root = raw as Record<string, unknown>;
  const agents = root.agents;
  if (agents && typeof agents === "object" && (agents as { list?: unknown }).list !== undefined) {
    return true;
  }
  return root.bindings !== undefined;
}

export function migrateLegacyConfig(raw: unknown): {
  config: MarvConfig | null;
  changes: string[];
} {
  const legacyAgentIds = listLegacyTopLevelAgentIds(raw);
  if (hasLegacyTopLevelMultiAgentConfig(raw)) {
    return {
      config: null,
      changes: [
        "Top-level multi-agent config is no longer supported.",
        ...(legacyAgentIds.length > 0
          ? [`Configured legacy agent ids: ${legacyAgentIds.join(", ")}`]
          : []),
        'Remove agents.list and bindings manually, keep only agents.defaults for "main", then retry.',
      ],
    };
  }
  const { next, changes } = applyLegacyMigrations(raw);
  if (!next) {
    return { config: null, changes: [] };
  }
  if (hasLegacyTopLevelMultiAgentConfig(next)) {
    return {
      config: null,
      changes: [
        ...changes,
        "Legacy routing/identity migration would produce removed top-level multi-agent fields.",
        'Rewrite the config directly to agents.defaults for "main" and remove legacy routing/bindings fields.',
      ],
    };
  }
  const validated = validateConfigObjectWithPlugins(next);
  if (!validated.ok) {
    changes.push("Migration applied, but config still invalid; fix remaining issues manually.");
    return { config: null, changes };
  }
  return { config: validated.config, changes };
}
