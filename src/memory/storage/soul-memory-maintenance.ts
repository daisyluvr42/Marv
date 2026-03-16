import { listAgentIds } from "../../agents/agent-scope.js";
import type { MarvConfig } from "../../core/config/config.js";
import { normalizeAgentId } from "../../routing/session-key.js";
import { compactP3Episodic } from "./soul-memory-compaction.js";
import { detectSoulMemoryConflicts, type SoulMemoryConflict } from "./soul-memory-conflict.js";
import { consolidateSoulMemories } from "./soul-memory-consolidation.js";
import { dedupeSoulMemories } from "./soul-memory-dedupe.js";
import {
  applySoulMemoryConfidenceDecay,
  promoteSoulMemories,
  type SoulMemoryConfig,
} from "./soul-memory-store.js";

export type SoulMemoryMaintenancePerAgent = {
  agentId: string;
  decayUpdated: number;
  decayDeleted: number;
  promotedToP1: number;
  promotedToP0: number;
  pendingP0Approvals: number;
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
    decayUpdated: number;
    decayDeleted: number;
    promotedToP1: number;
    promotedToP0: number;
    pendingP0Approvals: number;
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
  const memoryConfig = cfg.memory;
  if (!memoryConfig) {
    return undefined;
  }
  const legacyP0AllowedKinds = Array.isArray(memoryConfig.p0AllowedKinds)
    ? memoryConfig.p0AllowedKinds
    : undefined;
  if (!memoryConfig.soul) {
    return legacyP0AllowedKinds ? { p0AllowedKinds: legacyP0AllowedKinds } : undefined;
  }
  if (memoryConfig.soul.p0AllowedKinds || !legacyP0AllowedKinds) {
    return memoryConfig.soul;
  }
  return {
    ...memoryConfig.soul,
    p0AllowedKinds: legacyP0AllowedKinds,
  };
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

  let decayUpdated = 0;
  let decayDeleted = 0;
  let promotedToP1 = 0;
  let promotedToP0 = 0;
  let pendingP0Approvals = 0;
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
      const decay = applySoulMemoryConfidenceDecay({
        agentId,
        nowMs,
        soulConfig,
      });
      const promotion = promoteSoulMemories({
        agentId,
        nowMs,
        soulConfig,
      });
      const p3CompactionEnabled = soulConfig?.p3Compaction?.enabled === true;
      const dedupe = dedupeSoulMemories({
        agentId,
        p3CompactionEnabled,
      });
      const consolidation = consolidateSoulMemories({
        agentId,
        nowMs,
      });
      // P3 compaction: cluster episodic -> semantic, archive compacted footage
      const compaction = p3CompactionEnabled
        ? compactP3Episodic({
            agentId,
            config: {
              enabled: true,
              minClusterSize: soulConfig?.p3Compaction?.minClusterSize ?? 3,
              similarityMin: soulConfig?.p3Compaction?.similarityMin ?? 0.45,
              similarityMax: soulConfig?.p3Compaction?.similarityMax ?? 0.85,
              archiveAgeDays: soulConfig?.p3Compaction?.archiveAgeDays ?? 30,
              orphanAgeDays: soulConfig?.p3Compaction?.orphanAgeDays ?? 60,
              compactedDiscount: soulConfig?.p3Compaction?.compactedDiscount ?? 0.5,
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
        decayUpdated: decay.updated,
        decayDeleted: decay.deleted,
        promotedToP1: promotion.promotedToP1,
        promotedToP0: promotion.promotedToP0,
        pendingP0Approvals: promotion.p0ApprovalCandidates.length,
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

      decayUpdated += entry.decayUpdated;
      decayDeleted += entry.decayDeleted;
      promotedToP1 += entry.promotedToP1;
      promotedToP0 += entry.promotedToP0;
      pendingP0Approvals += entry.pendingP0Approvals;
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
        decayUpdated: 0,
        decayDeleted: 0,
        promotedToP1: 0,
        promotedToP0: 0,
        pendingP0Approvals: 0,
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
      decayUpdated,
      decayDeleted,
      promotedToP1,
      promotedToP0,
      pendingP0Approvals,
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
    `decayed=${report.totals.decayUpdated}, ` +
    `pruned=${report.totals.decayDeleted}, ` +
    `p2->p1=${report.totals.promotedToP1}, ` +
    `p1->p0=${report.totals.promotedToP0}, ` +
    `pendingP0=${report.totals.pendingP0Approvals}, ` +
    `deduped=${report.totals.deduplicationMergedPairs}, ` +
    `generalized=${report.totals.consolidationGeneralized}, ` +
    `compacted=${report.totals.compactionClusters}/${report.totals.compactionEpisodic}, ` +
    `evolved=${report.totals.compactionEvolved}, ` +
    `evo_conflicts=${report.totals.compactionEvolutionConflicts}, ` +
    `archived=${report.totals.compactionArchived}, ` +
    `conflicts=${report.totals.conflictsDetected}`
  );
}
