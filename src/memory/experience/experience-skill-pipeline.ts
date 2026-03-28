import fsp from "node:fs/promises";
import path from "node:path";
import { resolveStateDir } from "../../core/config/paths.js";

/**
 * Resolve the on-disk directory for an agent's skill candidates.
 * Path: `<stateDir>/agents/<agentId>/skill-candidates/`
 */
export function resolveSkillCandidatesDir(agentId: string): string {
  const stateDir = resolveStateDir(process.env, () => process.env.HOME ?? "");
  return path.join(stateDir, "agents", agentId, "skill-candidates");
}

/**
 * Write skill candidates from a distillation result to disk.
 * Each candidate is written as a markdown file for human review.
 * Skips candidates that already have a file on disk.
 *
 * @returns Number of new candidate files written.
 */
export async function writeSkillCandidates(agentId: string, candidates: string[]): Promise<number> {
  if (candidates.length === 0) {
    return 0;
  }
  const dir = resolveSkillCandidatesDir(agentId);
  await fsp.mkdir(dir, { recursive: true });

  let written = 0;
  for (const raw of candidates) {
    const name = normalizeSkillName(raw);
    if (!name) {
      continue;
    }
    const filePath = path.join(dir, `${name}.md`);
    // Skip if already exists (avoid overwriting reviewed candidates)
    try {
      await fsp.access(filePath);
      continue;
    } catch {
      // File doesn't exist — write it
    }
    const content = buildCandidateMarkdown(name, raw);
    await fsp.writeFile(filePath, content, "utf-8");
    written += 1;
  }
  return written;
}

/**
 * List existing skill candidate files for an agent.
 */
export async function listSkillCandidates(agentId: string): Promise<string[]> {
  const dir = resolveSkillCandidatesDir(agentId);
  try {
    const entries = await fsp.readdir(dir);
    return entries
      .filter((entry) => entry.endsWith(".md"))
      .map((entry) => entry.replace(/\.md$/, ""))
      .toSorted();
  } catch {
    return [];
  }
}

// ── Internal helpers ────────────────────────────────────────────────────────

/** Normalize a raw candidate name to a safe filename slug. */
function normalizeSkillName(raw: string): string {
  return raw
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff_-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60);
}

/** Build the markdown content for a skill candidate file. */
function buildCandidateMarkdown(name: string, rawLabel: string): string {
  const date = new Date().toISOString().slice(0, 10);
  return `---
name: ${rawLabel.trim()}
status: candidate
discovered: ${date}
---

# ${rawLabel.trim()}

Skill candidate discovered during experience distillation.

## Pattern

_Describe the reusable pattern here._

## When to Use

_Describe trigger conditions._

## Steps

1. _Step 1_
2. _Step 2_
`;
}
