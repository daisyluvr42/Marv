import type { CronConfig } from "../../core/config/types.cron.js";
import type { HeartbeatRunResult } from "../../infra/heartbeat/heartbeat-wake.js";
import type { CronJobListEntry } from "../health.js";
import type {
  CronDeliveryMode,
  CronJob,
  CronJobCreate,
  CronJobPatch,
  CronSystemTask,
  CronRunOutcome,
  CronRunStatus,
  CronRunTelemetry,
  CronSessionTarget,
  CronStoreFile,
} from "../types.js";

export type CronEvent = {
  jobId: string;
  jobName?: string;
  action: "added" | "updated" | "removed" | "started" | "finished";
  agentId?: string;
  sessionTarget?: CronSessionTarget;
  deliveryMode?: CronDeliveryMode;
  runAtMs?: number;
  durationMs?: number;
  status?: CronRunStatus;
  error?: string;
  summary?: string;
  sessionId?: string;
  sessionKey?: string;
  nextRunAtMs?: number;
} & CronRunTelemetry;

export type Logger = {
  debug: (obj: unknown, msg?: string) => void;
  info: (obj: unknown, msg?: string) => void;
  warn: (obj: unknown, msg?: string) => void;
  error: (obj: unknown, msg?: string) => void;
};

export type CronServiceDeps = {
  nowMs?: () => number;
  log: Logger;
  storePath: string;
  cronEnabled: boolean;
  /** CronConfig for session retention settings. */
  cronConfig?: CronConfig;
  /** Default agent id for jobs without an agent id. */
  defaultAgentId?: string;
  /** Resolve session store path for a given agent id. */
  resolveSessionStorePath?: (agentId?: string) => string;
  /** Path to the session store (sessions.json) for reaper use. */
  sessionStorePath?: string;
  enqueueSystemEvent: (
    text: string,
    opts?: { agentId?: string; sessionKey?: string; contextKey?: string },
  ) => void;
  requestHeartbeatNow: (opts?: { reason?: string; agentId?: string; sessionKey?: string }) => void;
  runHeartbeatOnce?: (opts?: {
    reason?: string;
    agentId?: string;
    sessionKey?: string;
  }) => Promise<HeartbeatRunResult>;
  /**
   * WakeMode=now: max time to wait for runHeartbeatOnce to stop returning
   * { status:"skipped", reason:"requests-in-flight" } before falling back to
   * requestHeartbeatNow.
   */
  wakeNowHeartbeatBusyMaxWaitMs?: number;
  /** WakeMode=now: delay between runHeartbeatOnce retries while busy. */
  wakeNowHeartbeatBusyRetryDelayMs?: number;
  runIsolatedAgentJob: (params: { job: CronJob; message: string }) => Promise<
    {
      summary?: string;
      /** Last non-empty agent text output (not truncated). */
      outputText?: string;
      /**
       * `true` when the isolated run already delivered its output to the target
       * channel (including matching messaging-tool sends). See:
       * #15692
       */
      delivered?: boolean;
    } & CronRunOutcome &
      CronRunTelemetry
  >;
  runSystemTask?: (params: {
    job: CronJob;
    task: CronSystemTask;
  }) => Promise<CronRunOutcome & CronRunTelemetry>;
  onEvent?: (evt: CronEvent) => void;
};

export type CronServiceDepsInternal = Omit<CronServiceDeps, "nowMs"> & {
  nowMs: () => number;
};

export type CronServiceState = {
  deps: CronServiceDepsInternal;
  store: CronStoreFile | null;
  timer: NodeJS.Timeout | null;
  running: boolean;
  op: Promise<unknown>;
  warnedDisabled: boolean;
  storeLoadedAtMs: number | null;
  storeFileMtimeMs: number | null;
};

export function createCronServiceState(deps: CronServiceDeps): CronServiceState {
  return {
    deps: { ...deps, nowMs: deps.nowMs ?? (() => Date.now()) },
    store: null,
    timer: null,
    running: false,
    op: Promise.resolve(),
    warnedDisabled: false,
    storeLoadedAtMs: null,
    storeFileMtimeMs: null,
  };
}

export type CronRunMode = "due" | "force";
export type CronWakeMode = "now" | "next-heartbeat";

export type CronStatusSummary = {
  enabled: boolean;
  storePath: string;
  jobs: number;
  nextWakeAtMs: number | null;
  activeJobs: number;
  disabledJobs: number;
  degradedJobs: number;
  runningJobs: number;
  autoDisabledJobs: number;
};

export type CronRunResult =
  | { ok: true; ran: true }
  | { ok: true; ran: false; reason: "not-due" }
  | { ok: true; ran: false; reason: "already-running" }
  | { ok: false };

export type CronRemoveResult = { ok: true; removed: boolean } | { ok: false; removed: false };

export type CronAddResult = CronJob;
export type CronUpdateResult = CronJob;

export type CronListResult = CronJobListEntry[];
export type CronAddInput = CronJobCreate;
export type CronUpdateInput = CronJobPatch;
