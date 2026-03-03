import { listAgentIds } from "../../agents/agent-scope.js";
import type { MarvConfig } from "../../core/config/config.js";
import { normalizeAgentId } from "../../routing/session-key.js";
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
      const dedupe = dedupeSoulMemories({
        agentId,
      });
      const consolidation = consolidateSoulMemories({
        agentId,
        nowMs,
      });
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
    `conflicts=${report.totals.conflictsDetected}`
  );
}
