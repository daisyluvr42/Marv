export type CronMutationNoticeAction = "added" | "updated" | "removed";

export type CronMutationNotice = {
  id: string;
  jobId: string;
  jobName: string | null;
  action: CronMutationNoticeAction;
  agentId: string | null;
  sessionKey: string | null;
  sessionTarget: string | null;
  deliveryMode: string | null;
  nextRunAtMs: number | null;
  createdAtMs: number;
  expiresAtMs: number;
};

const CRON_MUTATION_NOTICE_MS = 8000;
const CRON_MUTATION_NOTICE_LIMIT = 5;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function parseCronMutationNotice(payload: unknown, seq?: number): CronMutationNotice | null {
  if (!isRecord(payload)) {
    return null;
  }
  const action = payload.action;
  const jobId = typeof payload.jobId === "string" ? payload.jobId.trim() : "";
  if ((action !== "added" && action !== "updated" && action !== "removed") || !jobId) {
    return null;
  }
  const now = Date.now();
  const nextRunAtMs =
    typeof payload.nextRunAtMs === "number" && Number.isFinite(payload.nextRunAtMs)
      ? payload.nextRunAtMs
      : null;
  return {
    id: `cron:${seq ?? now}:${action}:${jobId}`,
    jobId,
    jobName: typeof payload.jobName === "string" && payload.jobName.trim() ? payload.jobName : null,
    action,
    agentId: typeof payload.agentId === "string" && payload.agentId.trim() ? payload.agentId : null,
    sessionKey:
      typeof payload.sessionKey === "string" && payload.sessionKey.trim()
        ? payload.sessionKey
        : null,
    sessionTarget:
      typeof payload.sessionTarget === "string" && payload.sessionTarget.trim()
        ? payload.sessionTarget
        : null,
    deliveryMode:
      typeof payload.deliveryMode === "string" && payload.deliveryMode.trim()
        ? payload.deliveryMode
        : null,
    nextRunAtMs,
    createdAtMs: now,
    expiresAtMs: now + CRON_MUTATION_NOTICE_MS,
  };
}

export function pruneCronMutationNotices(queue: CronMutationNotice[]): CronMutationNotice[] {
  const now = Date.now();
  return queue.filter((entry) => entry.expiresAtMs > now).slice(-CRON_MUTATION_NOTICE_LIMIT);
}

export function addCronMutationNotice(
  queue: CronMutationNotice[],
  entry: CronMutationNotice,
): CronMutationNotice[] {
  const next = pruneCronMutationNotices(queue).filter((item) => item.id !== entry.id);
  next.push(entry);
  return next.slice(-CRON_MUTATION_NOTICE_LIMIT);
}

export function removeCronMutationNotice(
  queue: CronMutationNotice[],
  id: string,
): CronMutationNotice[] {
  return pruneCronMutationNotices(queue).filter((entry) => entry.id !== id);
}
