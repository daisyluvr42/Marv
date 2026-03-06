import type { CronJob, CronLastRunSnapshot, CronRunStatus } from "./types.js";

export type CronJobHealth = {
  severity: "healthy" | "degraded" | "disabled";
  running: boolean;
  autoDisabled: boolean;
  disabledReason?:
    | "schedule-errors"
    | "one-shot-complete"
    | "one-shot-error"
    | "one-shot-skipped"
    | "manual-or-config";
  consecutiveErrors: number;
  scheduleErrorCount: number;
  lastStatus?: CronRunStatus;
};

export type CronJobListEntry = CronJob & {
  health: CronJobHealth;
  lastRun: CronLastRunSnapshot | null;
};

export type CronHealthSummary = {
  activeJobs: number;
  disabledJobs: number;
  degradedJobs: number;
  runningJobs: number;
  autoDisabledJobs: number;
};

function resolveDisabledReason(job: CronJob): CronJobHealth["disabledReason"] | undefined {
  if (job.enabled) {
    return undefined;
  }
  if ((job.state.scheduleErrorCount ?? 0) >= 3) {
    return "schedule-errors";
  }
  if (job.schedule.kind === "at") {
    if (job.state.lastStatus === "ok") {
      return "one-shot-complete";
    }
    if (job.state.lastStatus === "error") {
      return "one-shot-error";
    }
    if (job.state.lastStatus === "skipped") {
      return "one-shot-skipped";
    }
  }
  return "manual-or-config";
}

export function resolveCronJobHealth(job: CronJob): CronJobHealth {
  const running = typeof job.state.runningAtMs === "number";
  const consecutiveErrors = Math.max(0, job.state.consecutiveErrors ?? 0);
  const scheduleErrorCount = Math.max(0, job.state.scheduleErrorCount ?? 0);
  const disabledReason = resolveDisabledReason(job);
  const autoDisabled = disabledReason === "schedule-errors";
  const degraded =
    running || consecutiveErrors > 0 || scheduleErrorCount > 0 || job.state.lastStatus === "error";

  return {
    severity: !job.enabled ? "disabled" : degraded ? "degraded" : "healthy",
    running,
    autoDisabled,
    disabledReason,
    consecutiveErrors,
    scheduleErrorCount,
    lastStatus: job.state.lastStatus,
  };
}

export function resolveCronLastRunSnapshot(job: CronJob): CronLastRunSnapshot | null {
  if (
    job.state.lastRunAtMs === undefined &&
    job.state.lastStatus === undefined &&
    job.state.lastSummary === undefined &&
    job.state.lastSessionId === undefined &&
    job.state.lastSessionKey === undefined
  ) {
    return null;
  }
  return {
    status: job.state.lastStatus,
    error: job.state.lastError,
    summary: job.state.lastSummary,
    sessionId: job.state.lastSessionId,
    sessionKey: job.state.lastSessionKey,
    runAtMs: job.state.lastRunAtMs,
    durationMs: job.state.lastDurationMs,
    nextRunAtMs: job.state.nextRunAtMs,
    model: job.state.lastModel,
    provider: job.state.lastProvider,
    usage: job.state.lastUsage,
  };
}

export function enrichCronJob(job: CronJob): CronJobListEntry {
  return {
    ...job,
    health: resolveCronJobHealth(job),
    lastRun: resolveCronLastRunSnapshot(job),
  };
}

export function summarizeCronJobs(jobs: CronJob[]): CronHealthSummary {
  let activeJobs = 0;
  let disabledJobs = 0;
  let degradedJobs = 0;
  let runningJobs = 0;
  let autoDisabledJobs = 0;

  for (const job of jobs) {
    const health = resolveCronJobHealth(job);
    if (job.enabled) {
      activeJobs += 1;
    } else {
      disabledJobs += 1;
    }
    if (health.severity === "degraded") {
      degradedJobs += 1;
    }
    if (health.running) {
      runningJobs += 1;
    }
    if (health.autoDisabled) {
      autoDisabledJobs += 1;
    }
  }

  return {
    activeJobs,
    disabledJobs,
    degradedJobs,
    runningJobs,
    autoDisabledJobs,
  };
}
