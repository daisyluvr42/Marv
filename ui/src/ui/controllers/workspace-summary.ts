import type { GatewayBrowserClient } from "../gateway.js";
import type {
  CronJob,
  CronStatus,
  ProactiveStatusSnapshot,
  SessionsUsageResult,
} from "../types.js";
import type { WorkspaceSummarySnapshot } from "../workspace-types.js";
import { getRecentDateRange } from "./workspace-date.js";

export type WorkspaceSummaryState = {
  client: GatewayBrowserClient | null;
  connected: boolean;
  workspaceSummaryLoading: boolean;
  workspaceSummaryError: string | null;
  workspaceSummary: WorkspaceSummarySnapshot | null;
};

export async function loadWorkspaceSummary(state: WorkspaceSummaryState) {
  if (!state.client || !state.connected || state.workspaceSummaryLoading) {
    return;
  }
  state.workspaceSummaryLoading = true;
  state.workspaceSummaryError = null;
  const { startDate, endDate } = getRecentDateRange(7);
  const [usageResult, proactiveResult, cronStatusResult, cronJobsResult] = await Promise.allSettled(
    [
      state.client.request<SessionsUsageResult>("sessions.usage", {
        startDate,
        endDate,
        limit: 1000,
      }),
      state.client.request<ProactiveStatusSnapshot>("proactive.buffer", {}),
      state.client.request<CronStatus>("cron.status", {}),
      state.client.request<{ jobs?: CronJob[] }>("cron.list", { includeDisabled: true }),
    ],
  );
  const usage = usageResult.status === "fulfilled" ? usageResult.value : null;
  const proactive = proactiveResult.status === "fulfilled" ? proactiveResult.value : null;
  const cronStatus = cronStatusResult.status === "fulfilled" ? cronStatusResult.value : null;
  const cronJobs =
    cronJobsResult.status === "fulfilled" && Array.isArray(cronJobsResult.value.jobs)
      ? cronJobsResult.value.jobs
      : [];
  state.workspaceSummary = {
    startDate,
    endDate,
    sessionsTouched: usage?.sessions.length ?? 0,
    activeDays: usage?.aggregates.daily.filter((entry) => entry.messages > 0).length ?? 0,
    totalTokens: usage?.totals.totalTokens ?? 0,
    totalCost: usage?.totals.totalCost ?? 0,
    pendingProactive: proactive?.pendingEntries ?? 0,
    urgentProactive: proactive?.urgentEntries ?? 0,
    nextWakeAtMs: cronStatus?.nextWakeAtMs ?? null,
    failingJobs: cronJobs.filter((job) => job.state?.lastStatus === "error").length,
  };
  const errors = [usageResult, proactiveResult, cronStatusResult, cronJobsResult]
    .filter((result): result is PromiseRejectedResult => result.status === "rejected")
    .map((result) => String(result.reason));
  state.workspaceSummaryError = errors.length > 0 ? errors.join(" | ") : null;
  state.workspaceSummaryLoading = false;
}
