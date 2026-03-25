import type { MarvConfig } from "../../../core/config/config.js";
import { loadConfig } from "../../../core/config/config.js";
import { isSkillQuarantined, type SkillUsageRecord } from "../../skill-usage-records.js";
import { loadWorkspaceSkillEntries, type SkillEntry } from "../../skills.js";
import type { ParsedSkillFrontmatter } from "../../skills/types.js";
import { listManagedCliProfiles } from "../cli/cli-profile-registry.js";

export type MissingCapability = {
  description: string;
  suggestedTools?: string[];
  contextTaskId?: string;
};

export type DiscoveredSkill = {
  skillId: string;
  source: "bundled" | "managed" | "workspace" | "cli-profile" | "registry";
  metadata: ParsedSkillFrontmatter;
  confidenceScore: number;
  /** True when the skill is already locally present (workspace source) and needs no installation. */
  alreadyInstalled?: boolean;
  /** For registry-sourced skills: how to install them. */
  registryInstall?: { repo?: string; npm?: string };
};

export type RegistrySkillEntry = {
  id: string;
  name?: string;
  description?: string;
  repo?: string;
  npm?: string;
  capabilities?: string[];
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

function normalizeSource(raw: string): "bundled" | "managed" | "workspace" | undefined {
  const source = raw.trim().toLowerCase();
  if (source.includes("bundled")) {
    return "bundled";
  }
  if (source.includes("managed")) {
    return "managed";
  }
  // Workspace and agents-skills sources are locally present; surfaced as already installed.
  if (source.includes("workspace") || source.includes("agents-skills")) {
    return "workspace";
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

/** Score a registry skill entry against a capability query. */
function computeRegistryScore(entry: RegistrySkillEntry, capability: MissingCapability): number {
  const queryTokens = tokenize(
    [capability.description, ...(capability.suggestedTools ?? [])].join(" "),
  );
  if (queryTokens.length === 0) {
    return 0;
  }

  const id = (entry.id ?? "").toLowerCase();
  const name = (entry.name ?? "").toLowerCase();
  const description = (entry.description ?? "").toLowerCase();
  const caps = (entry.capabilities ?? []).map((c) => c.toLowerCase());

  let score = 0;
  for (const token of queryTokens) {
    if (id === token || name === token) {
      score += 4;
      continue;
    }
    if (id.includes(token) || name.includes(token)) {
      score += 2.4;
    }
    if (description.includes(token)) {
      score += 1.8;
    }
    if (caps.some((c) => c === token || c.includes(token))) {
      score += 2.2;
    }
  }
  return score;
}

/** Fetch and search configured registry sources for matching skills. */
async function searchRegistrySources(
  capability: MissingCapability,
  config?: MarvConfig,
): Promise<DiscoveredSkill[]> {
  const cfg = config ?? loadConfig();
  const sources = (cfg as Record<string, unknown>).skills as Record<string, unknown> | undefined;
  const registrySources = (sources?.sources ?? {}) as Record<string, string>;
  const urls = Object.values(registrySources).filter(
    (u) => typeof u === "string" && u.startsWith("http"),
  );
  if (urls.length === 0) {
    return [];
  }

  const results: DiscoveredSkill[] = [];
  for (const url of urls) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(10_000) });
      if (!res.ok) {
        continue;
      }
      const data = (await res.json()) as { skills?: RegistrySkillEntry[] };
      const skills = data.skills ?? [];
      for (const entry of skills) {
        const score = computeRegistryScore(entry, capability);
        if (score <= 0) {
          continue;
        }
        results.push({
          skillId: entry.id ?? entry.name ?? "unknown",
          source: "registry",
          metadata: {
            description: entry.description ?? "",
          } as ParsedSkillFrontmatter,
          confidenceScore: clamp01(score / 12),
          registryInstall: {
            ...(entry.repo ? { repo: entry.repo } : {}),
            ...(entry.npm ? { npm: entry.npm } : {}),
          },
        });
      }
    } catch {
      // Registry fetch failed — skip silently.
    }
  }
  return results;
}

export class ToolDiscoveryService {
  /** Discover skills from local workspace, managed CLI profiles, and optionally registry sources. */
  async discoverAsync(params: {
    workspaceDir: string;
    capability: MissingCapability;
    config?: MarvConfig;
    limit?: number;
    entries?: SkillEntry[];
    usageRecords?: Record<string, SkillUsageRecord>;
    searchRegistries?: boolean;
  }): Promise<DiscoveredSkill[]> {
    const localResults = this.discover(params);

    // If local results are strong enough, skip remote search.
    const hasStrongLocal = localResults.some((r) => r.confidenceScore >= 0.5);
    if (hasStrongLocal || params.searchRegistries === false) {
      return localResults;
    }

    // Search managed CLI profiles.
    const cliResults = await this.discoverCliProfiles(params.capability);

    // Search configured registry sources.
    const registryResults = await searchRegistrySources(params.capability, params.config);

    const limit = Math.max(1, Math.min(params.limit ?? 8, 50));
    return [...localResults, ...cliResults, ...registryResults]
      .toSorted(
        (a, b) => b.confidenceScore - a.confidenceScore || a.skillId.localeCompare(b.skillId),
      )
      .slice(0, limit);
  }

  /** Synchronous local-only discovery (original behavior). */
  discover(params: {
    workspaceDir: string;
    capability: MissingCapability;
    config?: MarvConfig;
    limit?: number;
    entries?: SkillEntry[];
    usageRecords?: Record<string, SkillUsageRecord>;
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
    const usageRecords = params.usageRecords;

    const discovered: DiscoveredSkill[] = [];
    for (const entry of entries) {
      if (isSkillQuarantined(entry.skill.name, usageRecords)) {
        continue;
      }
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
        ...(source === "workspace" ? { alreadyInstalled: true } : {}),
      });
    }

    return discovered
      .toSorted(
        (a, b) => b.confidenceScore - a.confidenceScore || a.skillId.localeCompare(b.skillId),
      )
      .slice(0, limit);
  }

  /** Search managed CLI profiles for matching capabilities. */
  private async discoverCliProfiles(capability: MissingCapability): Promise<DiscoveredSkill[]> {
    try {
      const profiles = await listManagedCliProfiles();
      const queryTokens = tokenize(
        [capability.description, ...(capability.suggestedTools ?? [])].join(" "),
      );
      if (queryTokens.length === 0) {
        return [];
      }

      const results: DiscoveredSkill[] = [];
      for (const record of profiles) {
        const entry = record.entry;
        // Only show active/verified profiles.
        if (entry.state !== "active" && entry.state !== "verified") {
          continue;
        }
        const id = entry.id.toLowerCase();
        const name = (entry.name ?? "").toLowerCase();
        const desc = (entry.description ?? "").toLowerCase();
        const caps = (entry.capabilities ?? []).map((c: string) => c.toLowerCase());

        let score = 0;
        for (const token of queryTokens) {
          if (id === token || name === token) {
            score += 4;
            continue;
          }
          if (id.includes(token) || name.includes(token)) {
            score += 2.4;
          }
          if (desc.includes(token)) {
            score += 1.8;
          }
          if (caps.some((c: string) => c === token || c.includes(token))) {
            score += 2.6;
          }
        }
        if (score <= 0) {
          continue;
        }

        results.push({
          skillId: entry.id,
          source: "cli-profile",
          metadata: { description: entry.description ?? "" } as ParsedSkillFrontmatter,
          confidenceScore: clamp01(score / 12),
          alreadyInstalled: true,
        });
      }
      return results;
    } catch {
      return [];
    }
  }
}
