import type { MarvConfig } from "../core/config/config.js";
import {
  countSoulArchiveEvents,
  countSoulMemoryItems,
  countSoulMemoryItemsByRecordKind,
} from "./storage/soul-memory-store.js";

export type MemoryStatusSnapshot = {
  agentId: string;
  backend: string;
  citations: string;
  autoRecallEnabled: boolean;
  knowledgeEnabled: boolean;
  runtimeIngestEnabled: boolean;
  totalItems: number;
  recordKinds: Record<"fact" | "relationship" | "experience" | "soul", number>;
  archiveEvents: number;
};

export function getMemoryStatusSnapshot(params: {
  agentId: string;
  config?: MarvConfig;
}): MemoryStatusSnapshot {
  const cfg = params.config ?? {};
  const recordKinds = countSoulMemoryItemsByRecordKind({ agentId: params.agentId });
  const totalItems = countSoulMemoryItems({ agentId: params.agentId });
  return {
    agentId: params.agentId,
    backend: cfg.memory?.backend ?? "builtin",
    citations: cfg.memory?.citations ?? "auto",
    autoRecallEnabled: cfg.memory?.autoRecall?.enabled !== false,
    knowledgeEnabled: cfg.memory?.knowledge?.enabled === true,
    runtimeIngestEnabled: cfg.memory?.runtimeIngest !== false,
    totalItems,
    recordKinds,
    archiveEvents: countSoulArchiveEvents({ agentId: params.agentId }),
  };
}
