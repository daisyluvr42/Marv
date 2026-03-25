import type { SubagentAnnounceMode } from "../shared/subagent-metadata.js";
import type { DeliveryContext } from "../utils/delivery-context.js";

export type SubagentRunOutcome = {
  status: "ok" | "error" | "timeout" | "unknown";
  error?: string;
};

export type SubagentRunRecord = {
  runId: string;
  childSessionKey: string;
  requesterSessionKey: string;
  requesterOrigin?: DeliveryContext;
  requesterDisplayKey: string;
  task: string;
  cleanup: "delete" | "keep";
  label?: string;
  model?: string;
  role?: string;
  preset?: string;
  taskGroup?: string;
  dispatchId?: string;
  announceMode?: SubagentAnnounceMode;
  runTimeoutSeconds?: number;
  createdAt: number;
  startedAt?: number;
  endedAt?: number;
  outcome?: SubagentRunOutcome;
  archiveAtMs?: number;
  cleanupCompletedAt?: number;
  cleanupHandled?: boolean;
  suppressAnnounceReason?: "steer-restart" | "killed";
  expectsCompletionMessage?: boolean;
  /** Number of times announce delivery has been attempted and returned false (deferred). */
  announceRetryCount?: number;
  /** Timestamp of the last announce retry attempt (for backoff). */
  lastAnnounceRetryAt?: number;
  /** Orchestration contract ID if this run is managed by the orchestration loop. */
  contractId?: string;
};
