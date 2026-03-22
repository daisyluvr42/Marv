import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { resolveStateDir } from "../../core/config/paths.js";

/**
 * Budget defaults (character counts, excluding attribution markers).
 */
export const EXPERIENCE_BUDGET_CHARS = 800;
export const CONTEXT_BUDGET_CHARS = 400;

/** Known experience file names. */
export type ExperienceFileName =
  | "MARV_EXPERIENCE.md"
  | "MARV_EXPERIENCE_LOG.md"
  | "MARV_CONTEXT.md";

/**
 * Resolve the on-disk directory for an agent's experience files.
 * Path: `<stateDir>/experience/<agentId>/`
 */
export function resolveExperienceDir(agentId: string): string {
  const stateDir = resolveStateDir(process.env, () => process.env.HOME ?? "");
  return path.join(stateDir, "experience", normalizeId(agentId));
}

/**
 * Read an experience file. Returns empty string if missing.
 */
export function readExperienceFileSync(agentId: string, fileName: ExperienceFileName): string {
  const filePath = path.join(resolveExperienceDir(agentId), fileName);
  try {
    return fs.readFileSync(filePath, "utf-8");
  } catch {
    return "";
  }
}

/**
 * Async read. Returns empty string if missing.
 */
export async function readExperienceFile(
  agentId: string,
  fileName: ExperienceFileName,
): Promise<string> {
  const filePath = path.join(resolveExperienceDir(agentId), fileName);
  try {
    return await fsp.readFile(filePath, "utf-8");
  } catch {
    return "";
  }
}

/**
 * Write an experience file atomically (write to tmp then rename).
 */
export async function writeExperienceFile(
  agentId: string,
  fileName: ExperienceFileName,
  content: string,
): Promise<void> {
  const dir = resolveExperienceDir(agentId);
  await fsp.mkdir(dir, { recursive: true });
  const filePath = path.join(dir, fileName);
  const tmpPath = `${filePath}.tmp.${Date.now()}`;
  await fsp.writeFile(tmpPath, content, "utf-8");
  await fsp.rename(tmpPath, filePath);
}

/**
 * Append to a log file (MARV_EXPERIENCE_LOG.md). Creates file if missing.
 */
export async function appendExperienceLog(agentId: string, entry: string): Promise<void> {
  const dir = resolveExperienceDir(agentId);
  await fsp.mkdir(dir, { recursive: true });
  const filePath = path.join(dir, "MARV_EXPERIENCE_LOG.md");
  await fsp.appendFile(filePath, entry + "\n", "utf-8");
}

/**
 * Compute content length for budget check.
 * Strips attribution markers (`@YYYY-MM | a:\d+ p:\d+`) before counting.
 */
export function measureExperienceContent(content: string): number {
  const stripped = content.replace(/@\d{4}-\d{2}\s*\|\s*a:\d+\s+p:\d+/g, "");
  return stripped.length;
}

/**
 * Check whether content exceeds the given budget.
 */
export function isOverBudget(content: string, budget: number): boolean {
  return measureExperienceContent(content) > budget;
}

function normalizeId(id: string): string {
  return id
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]/g, "_");
}
