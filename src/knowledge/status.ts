import path from "node:path";
import type { MarvConfig } from "../core/config/config.js";
import { computeFileHash, loadFileHashState } from "./file-hash-registry.js";

export type KnowledgeVaultStatus = {
  name: string;
  path: string;
  registryId: string;
  exclude: string[];
  fileCount: number;
  chunkCount: number;
  lastScanAt: number | null;
};

export type KnowledgeStatusSnapshot = {
  agentId: string;
  enabled: boolean;
  autoSyncOnSearch: boolean;
  autoSyncOnBoot: boolean;
  syncIntervalMs: number | null;
  vaultCount: number;
  totalFiles: number;
  totalChunks: number;
  lastScanAt: number | null;
  vaults: KnowledgeVaultStatus[];
};

export async function getKnowledgeStatusSnapshot(params: {
  agentId: string;
  config?: MarvConfig;
}): Promise<KnowledgeStatusSnapshot> {
  const cfg = params.config ?? {};
  const knowledge = cfg.memory?.knowledge;
  const vaults = knowledge?.vaults ?? [];
  const vaultStates = await Promise.all(
    vaults.map(async (vault) => {
      const registryId = buildKnowledgeRegistryId(vault.name, vault.path);
      const state = await loadFileHashState(params.agentId, registryId);
      let chunkCount = 0;
      for (const entry of state.entries.values()) {
        chunkCount += Math.max(0, Math.floor(entry.chunkCount ?? 0));
      }
      return {
        name: vault.name?.trim() || path.basename(vault.path) || "vault",
        path: path.resolve(vault.path),
        registryId,
        exclude: Array.isArray(vault.exclude) ? vault.exclude.filter(Boolean) : [],
        fileCount: state.entries.size,
        chunkCount,
        lastScanAt: typeof state.lastScanAt === "number" ? state.lastScanAt : null,
      } satisfies KnowledgeVaultStatus;
    }),
  );
  const totalFiles = vaultStates.reduce((sum, vault) => sum + vault.fileCount, 0);
  const totalChunks = vaultStates.reduce((sum, vault) => sum + vault.chunkCount, 0);
  const lastScanAt = vaultStates.reduce<number | null>(
    (latest, vault) =>
      latest == null || (vault.lastScanAt != null && vault.lastScanAt > latest)
        ? vault.lastScanAt
        : latest,
    null,
  );
  return {
    agentId: params.agentId,
    enabled: knowledge?.enabled === true,
    autoSyncOnSearch: knowledge?.autoSyncOnSearch !== false,
    autoSyncOnBoot: knowledge?.autoSyncOnBoot === true,
    syncIntervalMs:
      typeof knowledge?.syncIntervalMs === "number" && Number.isFinite(knowledge.syncIntervalMs)
        ? Math.max(0, Math.floor(knowledge.syncIntervalMs))
        : null,
    vaultCount: vaultStates.length,
    totalFiles,
    totalChunks,
    lastScanAt,
    vaults: vaultStates,
  };
}

function buildKnowledgeRegistryId(name: string | undefined, vaultPath: string): string {
  const stem = (name?.trim() || path.basename(vaultPath) || "vault")
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  const hash = computeFileHash(path.resolve(vaultPath)).slice(-12);
  return `${stem || "vault"}-${hash}`;
}
