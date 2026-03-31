import type { MarvConfig, SkillConfig } from "../../core/config/config.js";
import {
  evaluateRuntimeRequires,
  hasBinary,
  isConfigPathTruthyWithDefaults,
  resolveConfigPath,
  resolveRuntimePlatform,
} from "../../shared/config-eval.js";
import { resolveSkillKey } from "./frontmatter.js";
import type { SkillEligibilityContext, SkillEntry } from "./types.js";

const DEFAULT_CONFIG_VALUES: Record<string, boolean> = {
  "browser.enabled": true,
  "browser.evaluateEnabled": true,
};

export { hasBinary, resolveConfigPath, resolveRuntimePlatform };

export function isConfigPathTruthy(config: MarvConfig | undefined, pathStr: string): boolean {
  return isConfigPathTruthyWithDefaults(config, pathStr, DEFAULT_CONFIG_VALUES);
}

export function resolveSkillConfig(
  config: MarvConfig | undefined,
  skillKey: string,
): SkillConfig | undefined {
  const skills = config?.skills?.entries;
  if (!skills || typeof skills !== "object") {
    return undefined;
  }
  const entry = (skills as Record<string, SkillConfig | undefined>)[skillKey];
  if (!entry || typeof entry !== "object") {
    return undefined;
  }
  return entry;
}

function normalizeAllowlist(input: unknown): string[] | undefined {
  if (!input) {
    return undefined;
  }
  if (!Array.isArray(input)) {
    return undefined;
  }
  const normalized = input.map((entry) => String(entry).trim()).filter(Boolean);
  return normalized.length > 0 ? normalized : undefined;
}

const BUNDLED_SOURCES = new Set(["marv-bundled", "marv-bundled"]);

function isBundledSkill(entry: SkillEntry): boolean {
  return BUNDLED_SOURCES.has(entry.skill.source);
}

export function resolveBundledAllowlist(config?: MarvConfig): string[] | undefined {
  return normalizeAllowlist(config?.skills?.allowBundled);
}

export function isBundledSkillAllowed(entry: SkillEntry, allowlist?: string[]): boolean {
  if (!allowlist || allowlist.length === 0) {
    return true;
  }
  if (!isBundledSkill(entry)) {
    return true;
  }
  const key = resolveSkillKey(entry.skill, entry);
  return allowlist.includes(key) || allowlist.includes(entry.skill.name);
}

function resolveAvailableToolsSet(eligibility?: SkillEligibilityContext): Set<string> {
  return new Set(
    (eligibility?.availableTools ?? []).map((tool) => tool.trim().toLowerCase()).filter(Boolean),
  );
}

function matchesActivation(entry: SkillEntry, eligibility?: SkillEligibilityContext): boolean {
  const activation = entry.metadata?.activation;
  if (!activation) {
    return true;
  }
  const availableTools = resolveAvailableToolsSet(eligibility);
  if (availableTools.size === 0) {
    return true;
  }
  const requiresTools = (activation.requiresTools ?? []).map((tool) => tool.toLowerCase());
  if (requiresTools.length > 0 && !requiresTools.every((tool) => availableTools.has(tool))) {
    return false;
  }
  const requiresAnyTool = (activation.requiresAnyTool ?? []).map((tool) => tool.toLowerCase());
  if (requiresAnyTool.length > 0 && !requiresAnyTool.some((tool) => availableTools.has(tool))) {
    return false;
  }
  const fallbackForTools = (activation.fallbackForTools ?? []).map((tool) => tool.toLowerCase());
  if (fallbackForTools.some((tool) => availableTools.has(tool))) {
    return false;
  }
  return true;
}

export function shouldIncludeSkill(params: {
  entry: SkillEntry;
  config?: MarvConfig;
  eligibility?: SkillEligibilityContext;
}): boolean {
  const { entry, config, eligibility } = params;
  const skillKey = resolveSkillKey(entry.skill, entry);
  const skillConfig = resolveSkillConfig(config, skillKey);

  // User-disabled skills are always excluded, regardless of autonomy mode.
  if (skillConfig?.enabled === false) {
    return false;
  }

  // OS filtering is always applied (a macOS skill cannot run on Linux).
  const osList = entry.metadata?.os ?? [];
  const remotePlatforms = eligibility?.remote?.platforms ?? [];
  if (
    osList.length > 0 &&
    !osList.includes(resolveRuntimePlatform()) &&
    !remotePlatforms.some((platform) => osList.includes(platform))
  ) {
    return false;
  }

  if (!matchesActivation(entry, eligibility)) {
    return false;
  }

  // Autonomy "all" mode: skip bundled allowlist and runtime requirement checks.
  const autonomySkills =
    config?.autonomy?.skills ?? (config?.autonomy?.mode === "full" ? "all" : undefined);
  if (autonomySkills === "all") {
    return true;
  }

  // Standard filtering: bundled allowlist + runtime requirements.
  const allowBundled = normalizeAllowlist(config?.skills?.allowBundled);
  if (!isBundledSkillAllowed(entry, allowBundled)) {
    return false;
  }
  if (entry.metadata?.always === true) {
    return true;
  }

  return evaluateRuntimeRequires({
    requires: entry.metadata?.requires,
    hasBin: hasBinary,
    hasRemoteBin: eligibility?.remote?.hasBin,
    hasAnyRemoteBin: eligibility?.remote?.hasAnyBin,
    hasEnv: (envName) =>
      Boolean(
        process.env[envName] ||
        skillConfig?.env?.[envName] ||
        (skillConfig?.apiKey && entry.metadata?.primaryEnv === envName),
      ),
    isConfigPathTruthy: (configPath) => isConfigPathTruthy(config, configPath),
  });
}
