import fs from "node:fs/promises";
import path from "node:path";
import { loadConfig, writeConfigFile, type MarvConfig } from "../../core/config/config.js";
import { listManagedCliProfiles } from "./cli-profile-registry.js";

// --- Config get/set/unset via dot-path ---

/** Navigate into a nested object by dot-path, returning the value. */
function getByPath(obj: Record<string, unknown>, dotPath: string): unknown {
  const keys = dotPath.split(".");
  let current: unknown = obj;
  for (const key of keys) {
    if (current == null || typeof current !== "object") {
      return undefined;
    }
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}

/** Set a value in a nested object by dot-path, creating intermediate objects. */
function setByPath(obj: Record<string, unknown>, dotPath: string, value: unknown): void {
  const keys = dotPath.split(".");
  let current: Record<string, unknown> = obj;
  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];
    if (current[key] == null || typeof current[key] !== "object") {
      current[key] = {};
    }
    current = current[key] as Record<string, unknown>;
  }
  current[keys[keys.length - 1]] = value;
}

/** Remove a key from a nested object by dot-path. */
function unsetByPath(obj: Record<string, unknown>, dotPath: string): boolean {
  const keys = dotPath.split(".");
  let current: Record<string, unknown> = obj;
  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];
    if (current[key] == null || typeof current[key] !== "object") {
      return false;
    }
    current = current[key] as Record<string, unknown>;
  }
  const lastKey = keys[keys.length - 1];
  if (lastKey in current) {
    delete current[lastKey];
    return true;
  }
  return false;
}

/** Parse a value string as JSON if possible, otherwise keep as string. */
function parseValue(raw: string): unknown {
  const trimmed = raw.trim();
  if (trimmed === "true") {
    return true;
  }
  if (trimmed === "false") {
    return false;
  }
  if (trimmed === "null") {
    return null;
  }
  if (/^\d+$/.test(trimmed)) {
    return Number.parseInt(trimmed, 10);
  }
  if (/^\d+\.\d+$/.test(trimmed)) {
    return Number.parseFloat(trimmed);
  }
  try {
    return JSON.parse(trimmed);
  } catch {
    return trimmed;
  }
}

// Paths that should never be read/written by the agent for security.
const BLOCKED_PATHS = new Set(["auth", "auth.profiles"]);

function isBlockedPath(dotPath: string): boolean {
  const lower = dotPath.toLowerCase();
  for (const blocked of BLOCKED_PATHS) {
    if (lower === blocked || lower.startsWith(`${blocked}.`)) {
      return true;
    }
  }
  return false;
}

function isSecretPath(dotPath: string): boolean {
  const lower = dotPath.toLowerCase();
  return (
    lower.includes("secret") ||
    lower.includes("password") ||
    lower.includes("token") ||
    lower.includes("apikey") ||
    lower.includes("api_key")
  );
}

export async function handleConfigGet(dotPath: string): Promise<{
  ok: boolean;
  path: string;
  value?: unknown;
  error?: string;
}> {
  if (isBlockedPath(dotPath)) {
    return { ok: false, path: dotPath, error: "Access denied for this config path." };
  }
  const cfg = loadConfig() as Record<string, unknown>;
  const value = getByPath(cfg, dotPath);
  // Redact secrets
  if (isSecretPath(dotPath) && typeof value === "string" && value.length > 0) {
    return { ok: true, path: dotPath, value: `${value.slice(0, 4)}...${value.slice(-4)}` };
  }
  return { ok: true, path: dotPath, value: value ?? null };
}

export async function handleConfigSet(expr: string): Promise<{
  ok: boolean;
  path?: string;
  value?: unknown;
  error?: string;
}> {
  const eqIndex = expr.indexOf("=");
  if (eqIndex < 1) {
    return { ok: false, error: "Format: 'path=value' (e.g. 'tools.web.search.provider=tavily')." };
  }
  const dotPath = expr.slice(0, eqIndex).trim();
  const rawValue = expr.slice(eqIndex + 1).trim();

  if (isBlockedPath(dotPath)) {
    return { ok: false, path: dotPath, error: "Access denied for this config path." };
  }

  const cfg = loadConfig() as Record<string, unknown>;
  const value = parseValue(rawValue);
  setByPath(cfg, dotPath, value);
  await writeConfigFile(cfg as MarvConfig);

  // Redact secret values in response
  const displayValue = isSecretPath(dotPath) && typeof value === "string" ? "[configured]" : value;
  return { ok: true, path: dotPath, value: displayValue };
}

export async function handleConfigUnset(dotPath: string): Promise<{
  ok: boolean;
  path: string;
  removed: boolean;
  error?: string;
}> {
  if (isBlockedPath(dotPath)) {
    return {
      ok: false,
      path: dotPath,
      removed: false,
      error: "Access denied for this config path.",
    };
  }
  const cfg = loadConfig() as Record<string, unknown>;
  const removed = unsetByPath(cfg, dotPath);
  if (removed) {
    await writeConfigFile(cfg as MarvConfig);
  }
  return { ok: true, path: dotPath, removed };
}

// --- Skill list and source management ---

const EXTENSIONS_DIR_DEFAULT = path.join(
  process.env.HOME ?? process.env.USERPROFILE ?? ".",
  ".marv",
  "extensions",
);

const SKILLS_DIR_DEFAULT = path.join(
  process.env.HOME ?? process.env.USERPROFILE ?? ".",
  ".marv",
  "skills",
);

type SkillListEntry = {
  id: string;
  path: string;
  type: "extension" | "synthesized" | "cli-profile";
  state?: string;
  hasManifest: boolean;
};

/** List all installed skills across extensions, synthesized skills, and managed CLI profiles. */
export async function handleSkillList(): Promise<{
  ok: boolean;
  skills: SkillListEntry[];
}> {
  const skills: SkillListEntry[] = [];

  // List installed extensions (~/.marv/extensions/)
  try {
    const entries = await fs.readdir(EXTENSIONS_DIR_DEFAULT, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory() && !entry.isSymbolicLink()) {
        continue;
      }
      if (entry.name.startsWith(".")) {
        continue;
      }
      const skillPath = path.join(EXTENSIONS_DIR_DEFAULT, entry.name);
      const hasPkg = await fs
        .access(path.join(skillPath, "package.json"))
        .then(() => true)
        .catch(() => false);
      skills.push({
        id: entry.name,
        path: skillPath,
        type: "extension",
        hasManifest: hasPkg,
      });
    }
  } catch {
    // extensions dir may not exist yet
  }

  // List synthesized skills (~/.marv/skills/)
  try {
    const entries = await fs.readdir(SKILLS_DIR_DEFAULT, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory() && !entry.isSymbolicLink()) {
        continue;
      }
      if (entry.name.startsWith(".")) {
        continue;
      }
      const skillPath = path.join(SKILLS_DIR_DEFAULT, entry.name);
      const hasSkillMd = await fs
        .access(path.join(skillPath, "SKILL.md"))
        .then(() => true)
        .catch(() => false);
      skills.push({
        id: entry.name,
        path: skillPath,
        type: "synthesized",
        hasManifest: hasSkillMd,
      });
    }
  } catch {
    // skills dir may not exist yet
  }

  // List managed CLI profiles (~/.marv/tools/managed-cli/)
  try {
    const profiles = await listManagedCliProfiles();
    for (const record of profiles) {
      skills.push({
        id: record.entry.id,
        path: record.entry.manifestPath ?? "",
        type: "cli-profile",
        state: record.entry.state,
        hasManifest: true,
      });
    }
  } catch {
    // managed CLI dir may not exist yet
  }

  return { ok: true, skills };
}

export async function handleSkillSourceAdd(expr: string): Promise<{
  ok: boolean;
  message: string;
}> {
  const eqIndex = expr.indexOf("=");
  if (eqIndex < 1) {
    return {
      ok: false,
      message: "Format: 'name=url' (e.g. 'community=https://example.com/registry.json').",
    };
  }
  const name = expr.slice(0, eqIndex).trim();
  const url = expr.slice(eqIndex + 1).trim();

  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    return { ok: false, message: "Source URL must start with http:// or https://." };
  }

  const cfg = loadConfig() as Record<string, unknown>;
  if (!cfg.skills || typeof cfg.skills !== "object") {
    cfg.skills = {};
  }
  const skillsSection = cfg.skills as Record<string, unknown>;
  if (!skillsSection.sources || typeof skillsSection.sources !== "object") {
    skillsSection.sources = {};
  }
  const sources = skillsSection.sources as Record<string, unknown>;
  sources[name] = url;
  await writeConfigFile(cfg as MarvConfig);

  return { ok: true, message: `Skill source '${name}' added: ${url}` };
}

export async function handleSkillSourceList(): Promise<{
  ok: boolean;
  sources: Record<string, string>;
}> {
  const cfg = loadConfig() as Record<string, unknown>;
  const skillsSection = (cfg.skills ?? {}) as Record<string, unknown>;
  const sources = (skillsSection.sources ?? {}) as Record<string, string>;
  return { ok: true, sources };
}
