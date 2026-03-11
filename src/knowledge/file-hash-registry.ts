import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { resolveStateDir } from "../core/config/paths.js";

export type FileHashEntry = {
  relativePath: string;
  contentHash: string;
  lastIndexedAt: number;
  chunkCount: number;
};

type FileHashRegistryRecord = {
  version: 1;
  lastScanAt?: number;
  entries: Record<string, FileHashEntry>;
};

export type FileHashRegistryState = {
  lastScanAt?: number;
  entries: Map<string, FileHashEntry>;
};

export function loadFileHashes(
  agentId: string,
  registryId = "default",
): Promise<Map<string, FileHashEntry>> {
  return loadFileHashState(agentId, registryId).then((state) => state.entries);
}

export async function saveFileHashes(
  agentId: string,
  hashes: Map<string, FileHashEntry>,
  registryId = "default",
): Promise<void> {
  await saveFileHashState(
    agentId,
    {
      entries: hashes,
    },
    registryId,
  );
}

export async function loadFileHashState(
  agentId: string,
  registryId = "default",
): Promise<FileHashRegistryState> {
  const filePath = resolveFileHashRegistryPath(agentId, registryId);
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const parsed = JSON.parse(raw) as FileHashRegistryRecord;
    const entries = new Map<string, FileHashEntry>(
      Object.entries(parsed.entries ?? {}).map(([key, value]) => [key, value]),
    );
    return {
      lastScanAt: parsed.lastScanAt,
      entries,
    };
  } catch {
    return {
      entries: new Map<string, FileHashEntry>(),
    };
  }
}

export async function saveFileHashState(
  agentId: string,
  state: FileHashRegistryState,
  registryId = "default",
): Promise<void> {
  const filePath = resolveFileHashRegistryPath(agentId, registryId);
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const payload: FileHashRegistryRecord = {
    version: 1,
    lastScanAt: state.lastScanAt,
    entries: Object.fromEntries(state.entries.entries()),
  };
  await fs.writeFile(filePath, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");
}

export function computeFileHash(content: string): string {
  return `sha256:${crypto.createHash("sha256").update(content).digest("hex")}`;
}

export function resolveFileHashRegistryPath(agentId: string, registryId = "default"): string {
  return path.join(
    resolveStateDir(process.env),
    "knowledge",
    agentId,
    `${registryId}-file-hashes.json`,
  );
}
