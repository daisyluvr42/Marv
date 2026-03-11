import type { MarvConfig } from "../core/config/config.js";
import {
  countSoulArchiveEvents,
  countSoulMemoryItemsByRecordKind,
  countSoulMemoryItemsByTier,
} from "./storage/soul-memory-store.js";

export type MemoryStatusSnapshot = {
  agentId: string;
  backend: string;
  citations: string;
  autoRecallEnabled: boolean;
  knowledgeEnabled: boolean;
  runtimeIngestEnabled: boolean;
  totalItems: number;
  tiers: Record<"P0" | "P1" | "P2" | "P3", number>;
  recordKinds: Record<"fact" | "relationship" | "experience" | "soul", number>;
  archiveEvents: number;
};

export function getMemoryStatusSnapshot(params: {
  agentId: string;
  config?: MarvConfig;
}): MemoryStatusSnapshot {
  const cfg = params.config ?? {};
  const tiers = countSoulMemoryItemsByTier({ agentId: params.agentId });
  const recordKinds = countSoulMemoryItemsByRecordKind({ agentId: params.agentId });
  const totalItems = Object.values(tiers).reduce((sum, count) => sum + count, 0);
  return {
    agentId: params.agentId,
    backend: cfg.memory?.backend ?? "builtin",
    citations: cfg.memory?.citations ?? "auto",
    autoRecallEnabled: cfg.memory?.autoRecall?.enabled !== false,
    knowledgeEnabled: cfg.memory?.knowledge?.enabled === true,
    runtimeIngestEnabled: cfg.memory?.runtimeIngest !== false,
    totalItems,
    tiers,
    recordKinds,
    archiveEvents: countSoulArchiveEvents({ agentId: params.agentId }),
  };
}
