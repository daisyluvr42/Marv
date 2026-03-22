import fs from "node:fs/promises";
import path from "node:path";
import type { MarvConfig } from "../core/config/config.js";
import {
  deleteSoulMemoryScopeItems,
  writeSoulMemory,
} from "../memory/storage/soul-memory-store.js";
import { chunkMarkdownByHeadings, extractFrontmatter } from "./chunker.js";
import {
  computeFileHash,
  loadFileHashState,
  saveFileHashState,
  type FileHashEntry,
} from "./file-hash-registry.js";

const DEFAULT_EXCLUDES = [".obsidian", "Templates", ".trash", "node_modules", ".git"];
const DEFAULT_SYNC_INTERVAL_MS = 5 * 60_000;

export type IndexResult = {
  filesScanned: number;
  filesIndexed: number;
  filesSkipped: number;
  filesRemoved: number;
  chunksWritten: number;
  durationMs: number;
};

export async function indexDirectory(params: {
  agentId: string;
  directory: string;
  vaultName: string;
  exclude?: string[];
  dryRun?: boolean;
  registryId?: string;
}): Promise<IndexResult> {
  const startedAt = Date.now();
  const registry = await loadFileHashState(params.agentId, params.registryId);
  const files = await collectMarkdownFiles(params.directory, params.exclude ?? []);
  let filesIndexed = 0;
  let filesSkipped = 0;
  let filesRemoved = 0;
  let chunksWritten = 0;
  const nextEntries = new Map<string, FileHashEntry>();

  for (const filePath of files) {
    const relativePath = path.relative(params.directory, filePath).replace(/\\/g, "/");
    const content = await fs.readFile(filePath, "utf-8");
    const contentHash = computeFileHash(content);
    const existing = registry.entries.get(relativePath);
    if (existing?.contentHash === contentHash) {
      nextEntries.set(relativePath, existing);
      filesSkipped += 1;
      continue;
    }
    if (!params.dryRun) {
      const result = await indexSingleFile({
        agentId: params.agentId,
        filePath,
        relativePath,
        vaultName: params.vaultName,
      });
      chunksWritten += result.chunksWritten;
      nextEntries.set(relativePath, {
        relativePath,
        contentHash,
        lastIndexedAt: Date.now(),
        chunkCount: result.chunksWritten,
      });
    }
    filesIndexed += 1;
  }

  for (const [relativePath] of registry.entries) {
    if (nextEntries.has(relativePath)) {
      continue;
    }
    filesRemoved += 1;
    if (!params.dryRun) {
      await removeFileFromIndex({
        agentId: params.agentId,
        relativePath,
        vaultName: params.vaultName,
      });
    }
  }

  if (!params.dryRun) {
    await saveFileHashState(
      params.agentId,
      {
        lastScanAt: Date.now(),
        entries: nextEntries,
      },
      params.registryId,
    );
  }

  return {
    filesScanned: files.length,
    filesIndexed,
    filesSkipped,
    filesRemoved,
    chunksWritten,
    durationMs: Date.now() - startedAt,
  };
}

export async function indexSingleFile(params: {
  agentId: string;
  filePath: string;
  relativePath: string;
  vaultName: string;
}): Promise<{ chunksWritten: number }> {
  const raw = await fs.readFile(params.filePath, "utf-8");
  const { frontmatter, body } = extractFrontmatter(raw);
  const chunks = chunkMarkdownByHeadings(raw);
  const scopeId = buildDocumentScopeId(params.relativePath);
  deleteSoulMemoryScopeItems({
    agentId: params.agentId,
    scopeType: "document",
    scopeId,
  });

  let chunksWritten = 0;
  const stats = await fs.stat(params.filePath);
  for (const chunk of chunks) {
    const item = writeSoulMemory({
      agentId: params.agentId,
      scopeType: "document",
      scopeId,
      kind: "document_chunk",
      content: chunk.content,
      summary: chunk.heading ?? firstNonEmptyLine(chunk.content),
      confidence: 1,
      tier: "P3",
      source: "manual",
      recordKind: "fact",
      memoryType: "knowledge",
      metadata: {
        filePath: path.resolve(params.filePath),
        relativePath: params.relativePath,
        vault: params.vaultName,
        heading: chunk.heading,
        chunkIndex: chunk.chunkIndex,
        startLine: chunk.startLine,
        endLine: chunk.endLine,
        frontmatter,
        lastModified: stats.mtimeMs,
        contentHash: computeFileHash(body),
      },
    });
    if (item) {
      chunksWritten += 1;
    }
  }
  return { chunksWritten };
}

export async function removeFileFromIndex(params: {
  agentId: string;
  relativePath: string;
  vaultName: string;
}): Promise<void> {
  void params.vaultName;
  deleteSoulMemoryScopeItems({
    agentId: params.agentId,
    scopeType: "document",
    scopeId: buildDocumentScopeId(params.relativePath),
  });
}

export async function syncConfiguredKnowledgeBases(params: {
  agentId: string;
  config?: MarvConfig;
  force?: boolean;
  reason?: string;
}): Promise<void> {
  void params.reason;
  const knowledge = params.config?.memory?.knowledge;
  if (!knowledge?.enabled) {
    return;
  }
  const vaults = knowledge.vaults ?? [];
  if (vaults.length === 0) {
    return;
  }
  const autoSyncOnSearch = knowledge.autoSyncOnSearch !== false;
  if (!params.force && !autoSyncOnSearch) {
    return;
  }
  const syncIntervalMs =
    typeof knowledge.syncIntervalMs === "number" && Number.isFinite(knowledge.syncIntervalMs)
      ? Math.max(0, Math.floor(knowledge.syncIntervalMs))
      : DEFAULT_SYNC_INTERVAL_MS;

  for (const vault of vaults) {
    const registryId = buildRegistryId(vault.name, vault.path);
    const state = await loadFileHashState(params.agentId, registryId);
    if (
      !params.force &&
      typeof state.lastScanAt === "number" &&
      Date.now() - state.lastScanAt < syncIntervalMs
    ) {
      continue;
    }
    await indexDirectory({
      agentId: params.agentId,
      directory: vault.path,
      vaultName: vault.name?.trim() || path.basename(vault.path),
      exclude: vault.exclude,
      registryId,
    });
  }
}

function buildDocumentScopeId(relativePath: string): string {
  return `obsidian:${relativePath}`;
}

async function collectMarkdownFiles(root: string, exclude: string[]): Promise<string[]> {
  const results: string[] = [];
  await walkDirectory(root, root, [...DEFAULT_EXCLUDES, ...exclude], results);
  return results.toSorted((a, b) => a.localeCompare(b));
}

async function walkDirectory(
  root: string,
  current: string,
  exclude: string[],
  results: string[],
): Promise<void> {
  const entries = await fs.readdir(current, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(current, entry.name);
    const relativePath = path.relative(root, fullPath).replace(/\\/g, "/");
    if (shouldExclude(relativePath, exclude)) {
      continue;
    }
    if (entry.isDirectory()) {
      await walkDirectory(root, fullPath, exclude, results);
      continue;
    }
    if (entry.isFile() && entry.name.toLowerCase().endsWith(".md")) {
      results.push(fullPath);
    }
  }
}

function shouldExclude(relativePath: string, exclude: string[]): boolean {
  const segments = relativePath.split("/").filter(Boolean);
  if (segments.some((segment) => DEFAULT_EXCLUDES.includes(segment))) {
    return true;
  }
  return exclude.some((pattern) => {
    const normalized = pattern.trim().replace(/\\/g, "/").replace(/\/+$/g, "");
    if (!normalized) {
      return false;
    }
    return relativePath === normalized || relativePath.startsWith(`${normalized}/`);
  });
}

function buildRegistryId(name: string | undefined, vaultPath: string): string {
  const stem = (name?.trim() || path.basename(vaultPath) || "vault")
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  const hash = computeFileHash(path.resolve(vaultPath)).slice(-12);
  return `${stem || "vault"}-${hash}`;
}

function firstNonEmptyLine(text: string): string {
  return (
    text
      .split("\n")
      .map((line) => line.trim())
      .find(Boolean) ?? "document chunk"
  );
}
