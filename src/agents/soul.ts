import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import type { MarvConfig } from "../core/config/config.js";
import { resolveStateDir } from "../core/config/paths.js";
import { resolveAgentWorkspaceDir } from "./agent-scope.js";
import type { EmbeddedContextFile } from "./runner/pi-embedded-helpers.js";

const SOUL_FILENAME = "Soul.md";
const MIGRATION_MARKER = ".soul-migrated";

/**
 * Resolve the on-disk path for an agent's Soul.md.
 * Path: `<stateDir>/soul/<agentId>/Soul.md`
 */
export function resolveSoulFilePath(agentId: string): string {
  const stateDir = resolveStateDir(process.env, () => process.env.HOME ?? "");
  return path.join(stateDir, "soul", normalizeId(agentId), SOUL_FILENAME);
}

/**
 * Load Soul.md content for an agent. Returns empty string if missing.
 * Agent can only read, never write.
 */
export function loadSoulFileSync(agentId: string): string {
  try {
    return fs.readFileSync(resolveSoulFilePath(agentId), "utf-8");
  } catch {
    return "";
  }
}

/**
 * Async load Soul.md content.
 */
export async function loadSoulFile(agentId: string): Promise<string> {
  try {
    return await fsp.readFile(resolveSoulFilePath(agentId), "utf-8");
  } catch {
    return "";
  }
}

/**
 * Build Soul.md as an EmbeddedContextFile for system prompt injection.
 * Returns empty array if no Soul.md exists.
 */
export function buildSoulContextFile(agentId: string): EmbeddedContextFile[] {
  const content = loadSoulFileSync(agentId);
  if (!content.trim()) {
    return [];
  }
  return [{ path: "Soul", content }];
}

/**
 * Migrate legacy workspace files (SOUL.md, IDENTITY.md, USER.md) → Soul.md.
 * Only runs once; idempotent via marker file.
 */
export async function migrateLegacyP0ToSoul(params: {
  agentId: string;
  cfg?: MarvConfig;
  warn?: (msg: string) => void;
}): Promise<boolean> {
  const { agentId, cfg, warn } = params;
  const soulPath = resolveSoulFilePath(agentId);
  const soulDir = path.dirname(soulPath);
  const markerPath = path.join(soulDir, MIGRATION_MARKER);

  // Already migrated or Soul.md exists
  if (fileExists(soulPath) || fileExists(markerPath)) {
    return false;
  }

  // Try to read legacy workspace files
  const workspaceDir = cfg ? resolveAgentWorkspaceDir(cfg, agentId) : undefined;
  const legacyFiles: Record<string, string> = {};

  if (workspaceDir) {
    for (const name of ["SOUL.md", "IDENTITY.md", "USER.md"]) {
      try {
        const content = fs.readFileSync(path.join(workspaceDir, name), "utf-8");
        if (content.trim()) {
          legacyFiles[name] = content.trim();
        }
      } catch {
        // File doesn't exist, skip
      }
    }
  }

  if (Object.keys(legacyFiles).length === 0) {
    // No legacy files found, write marker only
    await fsp.mkdir(soulDir, { recursive: true });
    await fsp.writeFile(markerPath, `migrated=${new Date().toISOString()}\nno_legacy_files=true\n`);
    return false;
  }

  // Merge into Soul.md
  const sections: string[] = ["# Soul\n"];

  if (legacyFiles["SOUL.md"]) {
    sections.push("## Background\n", legacyFiles["SOUL.md"], "\n");
  }

  if (legacyFiles["IDENTITY.md"]) {
    sections.push("## Identity\n", legacyFiles["IDENTITY.md"], "\n");
  }

  if (legacyFiles["USER.md"]) {
    sections.push("## User\n", legacyFiles["USER.md"], "\n");
  }

  const merged = sections.join("\n");

  // Write Soul.md
  await fsp.mkdir(soulDir, { recursive: true });
  await fsp.writeFile(soulPath, merged, "utf-8");

  // Backup legacy files
  if (workspaceDir) {
    for (const name of Object.keys(legacyFiles)) {
      const src = path.join(workspaceDir, name);
      const bak = `${src}.bak`;
      try {
        if (fs.existsSync(src) && !fs.existsSync(bak)) {
          await fsp.rename(src, bak);
        }
      } catch {
        warn?.(`Failed to backup ${name}`);
      }
    }
  }

  // Write marker
  await fsp.writeFile(
    markerPath,
    `migrated=${new Date().toISOString()}\nfiles=${Object.keys(legacyFiles).join(",")}\n`,
  );

  warn?.(`Migrated P0 files → Soul.md: ${Object.keys(legacyFiles).join(", ")}`);
  return true;
}

function fileExists(filePath: string): boolean {
  try {
    return fs.existsSync(filePath);
  } catch {
    return false;
  }
}

function normalizeId(id: string): string {
  return id
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]/g, "_");
}
