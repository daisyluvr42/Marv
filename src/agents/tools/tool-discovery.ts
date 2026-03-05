import type { MarvConfig } from "../../core/config/config.js";
import { loadWorkspaceSkillEntries, type SkillEntry } from "../skills.js";
import type { ParsedSkillFrontmatter } from "../skills/types.js";

export type MissingCapability = {
  description: string;
  suggestedTools?: string[];
  contextTaskId?: string;
};

export type DiscoveredSkill = {
  skillId: string;
  source: "bundled" | "managed";
  metadata: ParsedSkillFrontmatter;
  confidenceScore: number;
};

type DiscoveryScope = "bundled" | "managed" | "all";

function normalizeScope(config?: MarvConfig): DiscoveryScope {
  const raw = config?.autonomy?.discovery?.scope;
  if (raw === "bundled" || raw === "managed" || raw === "all") {
    return raw;
  }
  return "all";
}

function isDiscoveryEnabled(config?: MarvConfig): boolean {
  const enabled = config?.autonomy?.discovery?.enabled;
  return enabled !== false;
}

function normalizeSource(raw: string): "bundled" | "managed" | undefined {
  const source = raw.trim().toLowerCase();
  if (source.includes("bundled")) {
    return "bundled";
  }
  if (source.includes("managed")) {
    return "managed";
  }
  return undefined;
}

function tokenize(text: string): string[] {
  const normalized = text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
  if (!normalized) {
    return [];
  }
  return normalized.split(/\s+/).filter((token) => token.length > 1);
}

function clamp01(value: number): number {
  if (value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
}

function computeScore(entry: SkillEntry, capability: MissingCapability): number {
  const queryTokens = tokenize(
    [capability.description, ...(capability.suggestedTools ?? [])].join(" "),
  );
  if (queryTokens.length === 0) {
    return 0;
  }

  const name = entry.skill.name.toLowerCase();
  const description = entry.skill.description.toLowerCase();
  const key = (entry.metadata?.skillKey ?? "").toLowerCase();
  const frontmatterDescription = (entry.frontmatter.description ?? "").toLowerCase();
  const requiresBins = (entry.metadata?.requires?.bins ?? []).map((bin) => bin.toLowerCase());
  const installBins = (entry.metadata?.install ?? [])
    .flatMap((spec) => spec.bins ?? [])
    .map((bin) => bin.toLowerCase());

  let score = 0;
  for (const token of queryTokens) {
    if (name === token || key === token) {
      score += 4;
      continue;
    }
    if (name.includes(token) || key.includes(token)) {
      score += 2.4;
    }
    if (description.includes(token)) {
      score += 1.8;
    }
    if (frontmatterDescription.includes(token)) {
      score += 1.2;
    }
    if (requiresBins.some((bin) => bin === token || bin.includes(token))) {
      score += 2.6;
    }
    if (installBins.some((bin) => bin === token || bin.includes(token))) {
      score += 2.2;
    }
  }

  return score;
}

export class ToolDiscoveryService {
  discover(params: {
    workspaceDir: string;
    capability: MissingCapability;
    config?: MarvConfig;
    limit?: number;
    entries?: SkillEntry[];
  }): DiscoveredSkill[] {
    if (!isDiscoveryEnabled(params.config)) {
      return [];
    }
    const scope = normalizeScope(params.config);
    const limit = Math.max(1, Math.min(params.limit ?? 8, 50));
    const entries =
      params.entries ??
      loadWorkspaceSkillEntries(params.workspaceDir, {
        config: params.config,
      });

    const discovered: DiscoveredSkill[] = [];
    for (const entry of entries) {
      const source = normalizeSource(entry.skill.source);
      if (!source) {
        continue;
      }
      if (scope !== "all" && source !== scope) {
        continue;
      }

      const score = computeScore(entry, params.capability);
      if (score <= 0) {
        continue;
      }

      discovered.push({
        skillId: entry.skill.name,
        source,
        metadata: entry.frontmatter,
        confidenceScore: clamp01(score / 12),
      });
    }

    return discovered
      .toSorted(
        (a, b) => b.confidenceScore - a.confidenceScore || a.skillId.localeCompare(b.skillId),
      )
      .slice(0, limit);
  }
}
