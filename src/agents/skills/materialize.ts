import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { parseFrontmatter } from "./frontmatter.js";
import { writeSkillPackageMetadata } from "./package-metadata.js";
import type { SkillPackageMetadata } from "./types.js";

export type MaterializedSkillFile = {
  path: string;
  content: string;
};

export function stripSkillFrontmatter(content: string): string {
  const normalized = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  if (!normalized.startsWith("---\n")) {
    return normalized.trim();
  }
  const endIndex = normalized.indexOf("\n---\n", 4);
  if (endIndex === -1) {
    return normalized.trim();
  }
  return normalized.slice(endIndex + 5).trim();
}

function ensureTrailingNewline(input: string): string {
  return input.endsWith("\n") ? input : `${input}\n`;
}

function normalizeSkillBody(input: string): string {
  const stripped = stripSkillFrontmatter(input).trim();
  if (!stripped) {
    return "## Overview\n\nTBD.\n";
  }
  return ensureTrailingNewline(stripped);
}

function escapeFrontmatterValue(value: string): string {
  if (/^[A-Za-z0-9._/-]+$/.test(value)) {
    return value;
  }
  return JSON.stringify(value);
}

export function buildCanonicalSkillMarkdown(params: {
  name: string;
  description: string;
  body: string;
}): string {
  const name = params.name.trim();
  const description = params.description.trim();
  if (!name) {
    throw new Error("skill name required");
  }
  if (!description) {
    throw new Error("skill description required");
  }
  const body = normalizeSkillBody(params.body);
  return ensureTrailingNewline(
    `---\nname: ${escapeFrontmatterValue(name)}\ndescription: ${escapeFrontmatterValue(description)}\n---\n\n${body}`,
  );
}

export function validateCanonicalSkillMarkdown(skillMd: string): {
  name: string;
  description: string;
} {
  const frontmatter = parseFrontmatter(skillMd);
  const name = frontmatter.name?.trim();
  const description = frontmatter.description?.trim();
  if (!name) {
    throw new Error("canonical skill missing frontmatter name");
  }
  if (!description) {
    throw new Error("canonical skill missing frontmatter description");
  }
  return { name, description };
}

function assertRelativeSkillFilePath(filePath: string): string {
  const normalized = filePath.replace(/\\/g, "/").trim();
  if (!normalized) {
    throw new Error("skill file path required");
  }
  if (normalized.startsWith("/") || normalized.startsWith("../") || normalized.includes("/../")) {
    throw new Error(`invalid skill file path: ${filePath}`);
  }
  return normalized;
}

export function computeSkillPackageHash(params: {
  skillMd: string;
  files?: MaterializedSkillFile[];
}): string {
  const hash = crypto.createHash("sha256");
  hash.update("SKILL.md\n");
  hash.update(params.skillMd);
  const files = [...(params.files ?? [])].sort((left, right) =>
    left.path.localeCompare(right.path),
  );
  for (const file of files) {
    hash.update(`\nFILE:${file.path}\n`);
    hash.update(file.content);
  }
  return `sha256:${hash.digest("hex")}`;
}

async function writeSkillFiles(params: {
  dir: string;
  skillMd: string;
  files?: MaterializedSkillFile[];
}) {
  await fs.mkdir(params.dir, { recursive: true });
  await fs.writeFile(path.join(params.dir, "SKILL.md"), params.skillMd, "utf-8");
  for (const file of params.files ?? []) {
    const relativePath = assertRelativeSkillFilePath(file.path);
    const destination = path.join(params.dir, relativePath);
    await fs.mkdir(path.dirname(destination), { recursive: true });
    await fs.writeFile(destination, ensureTrailingNewline(file.content), "utf-8");
  }
}

export async function materializeSkillPackage(params: {
  targetDir: string;
  name: string;
  description: string;
  body: string;
  files?: MaterializedSkillFile[];
  metadata?: SkillPackageMetadata;
}): Promise<{ targetDir: string; skillMd: string; contentHash: string }> {
  const targetDir = path.resolve(params.targetDir);
  const parentDir = path.dirname(targetDir);
  const tempDir = path.join(parentDir, `${path.basename(targetDir)}.${crypto.randomUUID()}.tmp`);
  const skillMd = buildCanonicalSkillMarkdown({
    name: params.name,
    description: params.description,
    body: params.body,
  });
  validateCanonicalSkillMarkdown(skillMd);
  const contentHash = computeSkillPackageHash({ skillMd, files: params.files });

  await fs.rm(tempDir, { recursive: true, force: true });
  await writeSkillFiles({
    dir: tempDir,
    skillMd,
    files: params.files,
  });
  await writeSkillPackageMetadata(tempDir, {
    ...params.metadata,
    adaptedAt: params.metadata?.adaptedAt ?? new Date().toISOString(),
  });

  await fs.mkdir(parentDir, { recursive: true });
  await fs.rm(targetDir, { recursive: true, force: true });
  await fs.rename(tempDir, targetDir);

  return { targetDir, skillMd, contentHash };
}
