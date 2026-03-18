import fs from "node:fs/promises";
import path from "node:path";
import { loadConfig, writeConfigFile, type MarvConfig } from "../../core/config/config.js";

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
  // Block any path containing "secret", "password", "token" reads
  // (writes are allowed so agent can configure API keys)
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

// --- Skill install/list/source management ---

const EXTENSIONS_DIR_DEFAULT = path.join(
  process.env.HOME ?? process.env.USERPROFILE ?? ".",
  ".marv",
  "extensions",
);

export async function handleSkillInstall(source: string): Promise<{
  ok: boolean;
  skillId?: string;
  source: string;
  message: string;
}> {
  const trimmed = source.trim();

  // Determine install method
  if (trimmed.startsWith("https://github.com/") || trimmed.startsWith("git@")) {
    return installFromGitHub(trimmed);
  }
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return installFromUrl(trimmed);
  }
  if (trimmed.startsWith("/") || trimmed.startsWith("./") || trimmed.startsWith("~")) {
    return installFromLocalPath(trimmed);
  }
  // Assume npm package
  return installFromNpm(trimmed);
}

async function installFromGitHub(repoUrl: string): Promise<{
  ok: boolean;
  skillId?: string;
  source: string;
  message: string;
}> {
  // Extract repo name for skill ID
  const match = repoUrl.match(/github\.com[/:]([^/]+)\/([^/.]+)/);
  const skillId = match ? match[2].replace(/^marv-/, "") : `github-${Date.now()}`;
  const installDir = path.join(EXTENSIONS_DIR_DEFAULT, skillId);

  try {
    await fs.mkdir(installDir, { recursive: true });
    const { execSync } = await import("node:child_process");
    execSync(`git clone --depth 1 ${repoUrl} "${installDir}"`, {
      stdio: "pipe",
      timeout: 60_000,
    });

    // Run npm install if package.json exists
    const pkgPath = path.join(installDir, "package.json");
    try {
      await fs.access(pkgPath);
      execSync("npm install --omit=dev", { cwd: installDir, stdio: "pipe", timeout: 120_000 });
    } catch {
      // No package.json or install failed — skill may not need deps
    }

    return {
      ok: true,
      skillId,
      source: repoUrl,
      message: `Skill '${skillId}' installed from GitHub. Restart gateway to activate.`,
    };
  } catch (err) {
    // Clean up on failure
    await fs.rm(installDir, { recursive: true, force: true }).catch(() => {});
    return {
      ok: false,
      skillId,
      source: repoUrl,
      message: `Failed to install from GitHub: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

async function installFromNpm(packageName: string): Promise<{
  ok: boolean;
  skillId?: string;
  source: string;
  message: string;
}> {
  const skillId = packageName.replace(/^@[^/]+\//, "").replace(/^marv-/, "");
  const installDir = path.join(EXTENSIONS_DIR_DEFAULT, skillId);

  try {
    await fs.mkdir(installDir, { recursive: true });
    const { execSync } = await import("node:child_process");

    // Create a temporary package.json and install the package
    const tmpPkg = {
      name: `marv-skill-${skillId}`,
      version: "1.0.0",
      dependencies: { [packageName]: "latest" },
    };
    await fs.writeFile(path.join(installDir, "package.json"), JSON.stringify(tmpPkg, null, 2));
    execSync("npm install --omit=dev", { cwd: installDir, stdio: "pipe", timeout: 120_000 });

    return {
      ok: true,
      skillId,
      source: packageName,
      message: `Skill '${skillId}' installed from npm (${packageName}). Restart gateway to activate.`,
    };
  } catch (err) {
    await fs.rm(installDir, { recursive: true, force: true }).catch(() => {});
    return {
      ok: false,
      skillId,
      source: packageName,
      message: `Failed to install from npm: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

async function installFromUrl(url: string): Promise<{
  ok: boolean;
  skillId?: string;
  source: string;
  message: string;
}> {
  // Fetch registry JSON or treat as a direct package tarball
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(30_000) });
    if (!res.ok) {
      return { ok: false, source: url, message: `HTTP ${res.status}: ${res.statusText}` };
    }
    const contentType = res.headers.get("content-type") ?? "";

    if (contentType.includes("json") || url.endsWith(".json")) {
      // It's a registry JSON — parse and list available skills
      const registry = (await res.json()) as {
        skills?: Array<{ id: string; name?: string; repo?: string; npm?: string }>;
      };
      const skills = registry.skills ?? [];
      return {
        ok: true,
        source: url,
        message: `Found ${skills.length} skills in registry: ${skills.map((s) => s.id ?? s.name).join(", ")}. Use skillInstall with specific skill name or repo URL to install.`,
      };
    }

    // Tarball or unknown — download and extract
    return {
      ok: false,
      source: url,
      message:
        "Direct URL install for tarballs is not yet supported. Use a GitHub repo URL or npm package name instead.",
    };
  } catch (err) {
    return {
      ok: false,
      source: url,
      message: `Failed to fetch URL: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

async function installFromLocalPath(localPath: string): Promise<{
  ok: boolean;
  skillId?: string;
  source: string;
  message: string;
}> {
  const resolved = localPath.startsWith("~")
    ? path.join(process.env.HOME ?? ".", localPath.slice(1))
    : path.resolve(localPath);

  try {
    const stat = await fs.stat(resolved);
    if (!stat.isDirectory()) {
      return { ok: false, source: localPath, message: "Path is not a directory." };
    }
    const pkgPath = path.join(resolved, "package.json");
    const pkgData = JSON.parse(await fs.readFile(pkgPath, "utf-8"));
    const skillId = (pkgData.name ?? path.basename(resolved))
      .replace(/^@[^/]+\//, "")
      .replace(/^marv-/, "");

    // Symlink into extensions
    const installDir = path.join(EXTENSIONS_DIR_DEFAULT, skillId);
    await fs.rm(installDir, { recursive: true, force: true }).catch(() => {});
    await fs.symlink(resolved, installDir);

    return {
      ok: true,
      skillId,
      source: localPath,
      message: `Skill '${skillId}' linked from local path. Restart gateway to activate.`,
    };
  } catch (err) {
    return {
      ok: false,
      source: localPath,
      message: `Failed to install from local path: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

const SKILLS_DIR_DEFAULT = path.join(
  process.env.HOME ?? process.env.USERPROFILE ?? ".",
  ".marv",
  "skills",
);

export async function handleSkillList(): Promise<{
  ok: boolean;
  skills: Array<{
    id: string;
    path: string;
    type: "extension" | "synthesized";
    hasPackageJson: boolean;
  }>;
}> {
  const skills: Array<{
    id: string;
    path: string;
    type: "extension" | "synthesized";
    hasPackageJson: boolean;
  }> = [];

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
      skills.push({ id: entry.name, path: skillPath, type: "extension", hasPackageJson: hasPkg });
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
        hasPackageJson: hasSkillMd,
      });
    }
  } catch {
    // skills dir may not exist yet
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
  const skills = cfg.skills as Record<string, unknown>;
  if (!skills.sources || typeof skills.sources !== "object") {
    skills.sources = {};
  }
  const sources = skills.sources as Record<string, unknown>;
  sources[name] = url;
  await writeConfigFile(cfg as MarvConfig);

  return { ok: true, message: `Skill source '${name}' added: ${url}` };
}

export async function handleSkillSourceList(): Promise<{
  ok: boolean;
  sources: Record<string, string>;
}> {
  const cfg = loadConfig() as Record<string, unknown>;
  const skills = (cfg.skills ?? {}) as Record<string, unknown>;
  const sources = (skills.sources ?? {}) as Record<string, string>;
  return { ok: true, sources };
}
