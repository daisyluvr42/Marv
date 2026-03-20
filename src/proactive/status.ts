import type { MarvConfig } from "../core/config/config.js";
import type { BudgetStatus, DailyTokenRecord } from "./budget.js";
import { getBudgetStatus, getUsageHistory } from "./budget.js";
import { listDeliverables, type DeliverableStatus } from "./deliverables.js";
import { readDigestBuffer } from "./digest-buffer.js";
import { listGoals, type GoalStatus } from "./goals.js";
import { listSources } from "./sources.js";
import { listTasks, type TaskStatus } from "./task-queue.js";

export type ProactiveStatusSnapshot = {
  agentId: string;
  enabled: boolean;
  checkEveryMinutes: number | null;
  digestTimes: string[];
  delivery: {
    channel: string;
    to: string | null;
  };
  totalEntries: number;
  pendingEntries: number;
  deliveredEntries: number;
  urgentEntries: number;
  lastFlushAt: number | null;
};

export async function getProactiveStatusSnapshot(params: {
  agentId: string;
  config?: MarvConfig;
}): Promise<ProactiveStatusSnapshot> {
  const cfg = params.config ?? {};
  const proactive = cfg.autonomy?.proactive;
  const buffer = await readDigestBuffer(params.agentId);
  const totalEntries = buffer.entries.length;
  const pendingEntries = buffer.entries.filter((entry) => !entry.delivered).length;
  const deliveredEntries = totalEntries - pendingEntries;
  const urgentEntries = buffer.entries.filter((entry) => entry.urgency === "urgent").length;
  return {
    agentId: params.agentId,
    enabled: proactive?.enabled === true,
    checkEveryMinutes:
      typeof proactive?.checkEveryMinutes === "number" &&
      Number.isFinite(proactive.checkEveryMinutes)
        ? Math.max(0, Math.floor(proactive.checkEveryMinutes))
        : null,
    digestTimes: Array.isArray(proactive?.digestTimes)
      ? proactive.digestTimes.filter(
          (value) => typeof value === "string" && value.trim().length > 0,
        )
      : [],
    delivery: {
      channel: proactive?.delivery?.channel?.trim() || "last",
      to: proactive?.delivery?.to?.trim() || null,
    },
    totalEntries,
    pendingEntries,
    deliveredEntries,
    urgentEntries,
    lastFlushAt: buffer.lastFlushAt > 0 ? buffer.lastFlushAt : null,
  };
}

// ── Continuous loop status ─────────────────────────────────────────────

export type ContinuousLoopStatus = {
  goals: { total: number; byStatus: Partial<Record<GoalStatus, number>> };
  tasks: { total: number; byStatus: Partial<Record<TaskStatus, number>> };
  sources: { total: number; enabled: number };
  deliverables: { total: number; byStatus: Partial<Record<DeliverableStatus, number>> };
  budget: BudgetStatus;
  usageHistory: DailyTokenRecord[];
};

/** Aggregate continuous-loop proactive subsystem stats for an agent. */
export async function getContinuousLoopStatus(
  agentId: string,
  dailyCloudTokenBudget = 0,
): Promise<ContinuousLoopStatus> {
  const [goals, tasks, sources, deliverables, budget, usageHistory] = await Promise.all([
    listGoals(agentId),
    listTasks(agentId),
    listSources(agentId),
    listDeliverables(agentId),
    getBudgetStatus(agentId, dailyCloudTokenBudget),
    getUsageHistory(agentId, 7),
  ]);

  const goalsByStatus: Partial<Record<GoalStatus, number>> = {};
  for (const g of goals) {
    goalsByStatus[g.status] = (goalsByStatus[g.status] ?? 0) + 1;
  }

  const tasksByStatus: Partial<Record<TaskStatus, number>> = {};
  for (const t of tasks) {
    tasksByStatus[t.status] = (tasksByStatus[t.status] ?? 0) + 1;
  }

  const deliverablesByStatus: Partial<Record<DeliverableStatus, number>> = {};
  for (const d of deliverables) {
    deliverablesByStatus[d.status] = (deliverablesByStatus[d.status] ?? 0) + 1;
  }

  return {
    goals: { total: goals.length, byStatus: goalsByStatus },
    tasks: { total: tasks.length, byStatus: tasksByStatus },
    sources: { total: sources.length, enabled: sources.filter((s) => s.enabled).length },
    deliverables: { total: deliverables.length, byStatus: deliverablesByStatus },
    budget,
    usageHistory,
  };
}
