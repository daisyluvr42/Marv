import fsSync from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import type { SkillPackageMetadata, SkillTrustLevel } from "./types.js";

export const SKILL_PACKAGE_METADATA_FILENAME = ".marv-skill.json";

export function resolveSkillPackageMetadataPath(skillDir: string): string {
  return path.join(skillDir, SKILL_PACKAGE_METADATA_FILENAME);
}

export function resolveSkillTrustLevelForSource(source?: string): SkillTrustLevel | undefined {
  const normalized = source?.trim().toLowerCase();
  if (!normalized) {
    return undefined;
  }
  if (normalized.includes("bundled")) {
    return "builtin";
  }
  if (normalized.includes("managed")) {
    return "managed";
  }
  if (normalized.includes("workspace") || normalized.includes("agents-skills")) {
    return "workspace";
  }
  return undefined;
}

function normalizeString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function normalizeTrust(value: unknown): SkillTrustLevel | undefined {
  if (
    value === "builtin" ||
    value === "managed" ||
    value === "workspace" ||
    value === "agent-created" ||
    value === "community"
  ) {
    return value;
  }
  return undefined;
}

export async function readSkillPackageMetadata(skillDir: string): Promise<SkillPackageMetadata> {
  return normalizeSkillPackageMetadata(
    await readRawSkillPackageMetadata(resolveSkillPackageMetadataPath(skillDir)),
  );
}

export function readSkillPackageMetadataSync(skillDir: string): SkillPackageMetadata {
  return normalizeSkillPackageMetadata(
    readRawSkillPackageMetadataSync(resolveSkillPackageMetadataPath(skillDir)),
  );
}

function normalizeSkillPackageMetadata(parsed: unknown): SkillPackageMetadata {
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return {};
  }
  const record = parsed as Record<string, unknown>;
  return {
    source: normalizeString(record.source),
    originHash: normalizeString(record.originHash),
    trust: normalizeTrust(record.trust),
    adaptedAt: normalizeString(record.adaptedAt),
    adaptedFrom: normalizeString(record.adaptedFrom),
  };
}

async function readRawSkillPackageMetadata(filePath: string): Promise<unknown> {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    return JSON.parse(raw) as unknown;
  } catch {
    return undefined;
  }
}

function readRawSkillPackageMetadataSync(filePath: string): unknown {
  try {
    const raw = fsSync.readFileSync(filePath, "utf-8");
    return JSON.parse(raw) as unknown;
  } catch {
    return undefined;
  }
}

export async function writeSkillPackageMetadata(
  skillDir: string,
  metadata: SkillPackageMetadata,
): Promise<void> {
  const payload: SkillPackageMetadata = {
    source: normalizeString(metadata.source),
    originHash: normalizeString(metadata.originHash),
    trust: normalizeTrust(metadata.trust),
    adaptedAt: normalizeString(metadata.adaptedAt),
    adaptedFrom: normalizeString(metadata.adaptedFrom),
  };
  await fs.mkdir(skillDir, { recursive: true });
  await fs.writeFile(
    resolveSkillPackageMetadataPath(skillDir),
    `${JSON.stringify(payload, null, 2)}\n`,
    "utf-8",
  );
}
