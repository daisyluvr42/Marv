import { listAgentIds } from "../../agents/agent-scope.js";
import type { MarvConfig } from "../../core/config/config.js";
import { normalizeAgentId } from "../../routing/session-key.js";
import { compactEpisodic } from "./soul-memory-compaction.js";
import { detectSoulMemoryConflicts, type SoulMemoryConflict } from "./soul-memory-conflict.js";
import { consolidateSoulMemories } from "./soul-memory-consolidation.js";
import { dedupeSoulMemories } from "./soul-memory-dedupe.js";
import { type SoulMemoryConfig } from "./soul-memory-store.js";

export type SoulMemoryMaintenancePerAgent = {
  agentId: string;
  deduplicationMergedPairs: number;
  deduplicationRemoved: number;
  consolidationGeneralized: number;
  compactionClusters: number;
  compactionEpisodic: number;
  compactionArchived: number;
  compactionEvolved: number;
  compactionEvolutionConflicts: number;
  conflictsDetected: number;
  error?: string;
};

export type SoulMemoryMaintenanceReport = {
  agents: SoulMemoryMaintenancePerAgent[];
  totals: {
    deduplicationMergedPairs: number;
    deduplicationRemoved: number;
    consolidationGeneralized: number;
    compactionClusters: number;
    compactionEpisodic: number;
    compactionArchived: number;
    compactionEvolved: number;
    compactionEvolutionConflicts: number;
    conflictsDetected: number;
  };
  deduplication: {
    mergedPairs: number;
    removedIds: string[];
  };
  consolidation: {
    generalizedCount: number;
    consolidatedIds: string[];
  };
  conflicts: SoulMemoryConflict[];
  failedAgents: number;
};

function resolveMemorySoulConfig(cfg: MarvConfig): SoulMemoryConfig | undefined {
  return cfg.memory?.soul;
}

function resolveMaintenanceAgentIds(params: { cfg: MarvConfig; agentId?: string }): string[] {
  const single = params.agentId?.trim();
  if (single) {
    return [normalizeAgentId(single)];
  }
  return listAgentIds(params.cfg).map((id) => normalizeAgentId(id));
}

export function runSoulMemoryMaintenance(params: {
  cfg: MarvConfig;
  nowMs?: number;
  agentId?: string;
}): SoulMemoryMaintenanceReport {
  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  const soulConfig = resolveMemorySoulConfig(params.cfg);
  const agentIds = resolveMaintenanceAgentIds(params);
  const agents: SoulMemoryMaintenancePerAgent[] = [];

  let deduplicationMergedPairs = 0;
  let deduplicationRemoved = 0;
  let consolidationGeneralized = 0;
  let compactionClusters = 0;
  let compactionEpisodic = 0;
  let compactionArchived = 0;
  let compactionEvolved = 0;
  let compactionEvolutionConflicts = 0;
  let conflictsDetected = 0;
  const dedupRemovedIds: string[] = [];
  const consolidatedIds: string[] = [];
  const conflicts: SoulMemoryConflict[] = [];
  let failedAgents = 0;

  for (const agentId of agentIds) {
    try {
      const compactionCfg = soulConfig?.compaction ?? soulConfig?.p3Compaction;
      const compactionEnabled = compactionCfg?.enabled === true;
      const dedupe = dedupeSoulMemories({
        agentId,
        compactionEnabled: compactionEnabled,
      });
      const consolidation = consolidateSoulMemories({
        agentId,
        nowMs,
      });
      // Compaction: cluster episodic -> semantic, archive compacted footage
      const compaction = compactionEnabled
        ? compactEpisodic({
            agentId,
            config: {
              enabled: true,
              minClusterSize: compactionCfg?.minClusterSize ?? 3,
              similarityMin: compactionCfg?.similarityMin ?? 0.45,
              similarityMax: compactionCfg?.similarityMax ?? 0.85,
              archiveAgeDays: compactionCfg?.archiveAgeDays ?? 30,
              orphanAgeDays: compactionCfg?.orphanAgeDays ?? 60,
              compactedDiscount: compactionCfg?.compactedDiscount ?? 0.5,
              batchLimit: compactionCfg?.batchLimit ?? 1000,
            },
            nowMs,
          })
        : {
            compactedClusters: 0,
            compactedEpisodic: 0,
            archivedCompacted: 0,
            archivedOrphans: 0,
            semanticIds: [],
            evolvedSemantics: 0,
            evolutionConflicts: 0,
          };
      const conflictResult = detectSoulMemoryConflicts({
        agentId,
        nowMs,
      });
      const entry: SoulMemoryMaintenancePerAgent = {
        agentId,
        deduplicationMergedPairs: dedupe.mergedPairs,
        deduplicationRemoved: dedupe.removedIds.length,
        consolidationGeneralized: consolidation.generalizedCount,
        compactionClusters: compaction.compactedClusters,
        compactionEpisodic: compaction.compactedEpisodic,
        compactionArchived: compaction.archivedCompacted + compaction.archivedOrphans,
        compactionEvolved: compaction.evolvedSemantics,
        compactionEvolutionConflicts: compaction.evolutionConflicts,
        conflictsDetected: conflictResult.inserted,
      };
      agents.push(entry);

      deduplicationMergedPairs += entry.deduplicationMergedPairs;
      deduplicationRemoved += entry.deduplicationRemoved;
      consolidationGeneralized += entry.consolidationGeneralized;
      compactionClusters += entry.compactionClusters;
      compactionEpisodic += entry.compactionEpisodic;
      compactionArchived += entry.compactionArchived;
      compactionEvolved += entry.compactionEvolved;
      compactionEvolutionConflicts += entry.compactionEvolutionConflicts;
      conflictsDetected += entry.conflictsDetected;
      dedupRemovedIds.push(...dedupe.removedIds);
      consolidatedIds.push(...consolidation.consolidatedIds);
      conflicts.push(...conflictResult.conflicts);
    } catch (err) {
      failedAgents += 1;
      agents.push({
        agentId,
        deduplicationMergedPairs: 0,
        deduplicationRemoved: 0,
        consolidationGeneralized: 0,
        compactionClusters: 0,
        compactionEpisodic: 0,
        compactionArchived: 0,
        compactionEvolved: 0,
        compactionEvolutionConflicts: 0,
        conflictsDetected: 0,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  return {
    agents,
    totals: {
      deduplicationMergedPairs,
      deduplicationRemoved,
      consolidationGeneralized,
      compactionClusters,
      compactionEpisodic,
      compactionArchived,
      compactionEvolved,
      compactionEvolutionConflicts,
      conflictsDetected,
    },
    deduplication: {
      mergedPairs: deduplicationMergedPairs,
      removedIds: dedupRemovedIds,
    },
    consolidation: {
      generalizedCount: consolidationGeneralized,
      consolidatedIds,
    },
    conflicts,
    failedAgents,
  };
}

export function formatSoulMemoryMaintenanceSummary(report: SoulMemoryMaintenanceReport): string {
  return (
    "Soul maintenance complete: " +
    `agents=${report.agents.length}, ` +
    `failed=${report.failedAgents}, ` +
    `deduped=${report.totals.deduplicationMergedPairs}, ` +
    `generalized=${report.totals.consolidationGeneralized}, ` +
    `compacted=${report.totals.compactionClusters}/${report.totals.compactionEpisodic}, ` +
    `evolved=${report.totals.compactionEvolved}, ` +
    `evo_conflicts=${report.totals.compactionEvolutionConflicts}, ` +
    `archived=${report.totals.compactionArchived}, ` +
    `conflicts=${report.totals.conflictsDetected}`
  );
}
