import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { isPathInside } from "../../security/scan-paths.js";

export const SKILL_SYNC_MANIFEST_FILENAME = ".sync-manifest.json";

export type SkillSyncManifest = Record<string, string>;

export function resolveSkillSyncManifestPath(skillsDir: string): string {
  return path.join(skillsDir, SKILL_SYNC_MANIFEST_FILENAME);
}

async function listFilesRecursive(rootDir: string): Promise<string[]> {
  const entries = await fs.readdir(rootDir, { withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    if (entry.name === SKILL_SYNC_MANIFEST_FILENAME) {
      continue;
    }
    const fullPath = path.join(rootDir, entry.name);
    if (!isPathInside(rootDir, fullPath)) {
      continue;
    }
    if (entry.isDirectory()) {
      files.push(...(await listFilesRecursive(fullPath)));
      continue;
    }
    if (entry.isFile()) {
      files.push(fullPath);
    }
  }
  return files;
}

export async function computeSkillDirectoryHash(skillDir: string): Promise<string> {
  const hash = crypto.createHash("sha256");
  const files = (await listFilesRecursive(skillDir)).sort((left, right) =>
    left.localeCompare(right),
  );
  for (const file of files) {
    const relativePath = path.relative(skillDir, file).replace(/\\/g, "/");
    hash.update(`FILE:${relativePath}\n`);
    hash.update(await fs.readFile(file));
    hash.update("\n");
  }
  return `sha256:${hash.digest("hex")}`;
}

export async function readSkillSyncManifest(skillsDir: string): Promise<SkillSyncManifest> {
  try {
    const raw = await fs.readFile(resolveSkillSyncManifestPath(skillsDir), "utf-8");
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    return Object.fromEntries(
      Object.entries(parsed as Record<string, unknown>).filter(
        (entry): entry is [string, string] =>
          typeof entry[0] === "string" && typeof entry[1] === "string",
      ),
    );
  } catch {
    return {};
  }
}

export async function writeSkillSyncManifest(
  skillsDir: string,
  manifest: SkillSyncManifest,
): Promise<void> {
  await fs.mkdir(skillsDir, { recursive: true });
  await fs.writeFile(
    resolveSkillSyncManifestPath(skillsDir),
    `${JSON.stringify(manifest, null, 2)}\n`,
    "utf-8",
  );
}
