import type { TaskStatus as TaskContextStatus } from "../memory/task-context/types.js";
import type { GoalStatus as ProactiveGoalStatus } from "../proactive/goals.js";
import type { TaskStatus as ProactiveTaskStatus } from "../proactive/task-queue.js";

export const UNIFIED_WORKBENCH_STATUSES = [
  "active",
  "paused",
  "blocked",
  "queued",
  "completed",
  "archived",
] as const;
export type UnifiedWorkbenchStatus = (typeof UNIFIED_WORKBENCH_STATUSES)[number];

export const WORKBENCH_ROW_SOURCES = ["task-context", "proactive-goal", "proactive-task"] as const;
export type WorkbenchRowSource = (typeof WORKBENCH_ROW_SOURCES)[number];

export const WORKBENCH_DEEP_LINK_VIEWS = ["session", "project", "task-context"] as const;
export type WorkbenchDeepLinkView = (typeof WORKBENCH_DEEP_LINK_VIEWS)[number];

export type WorkbenchDeepLink = {
  view: WorkbenchDeepLinkView;
  params: Record<string, string>;
};

export type WorkbenchRow = {
  id: string;
  source: WorkbenchRowSource;
  title: string;
  status: UnifiedWorkbenchStatus;
  updatedAt: string;
  summary: string;
  deepLink: WorkbenchDeepLink | null;
};

export type WorkbenchSnapshot = {
  agentId: string;
  rows: WorkbenchRow[];
  counts: Record<UnifiedWorkbenchStatus, number>;
  deliverableSummary: {
    total: number;
    completed: number;
  };
  fetchedAt: string;
};

export const WORKBENCH_TASK_CONTEXT_STATUS_MAP: Record<TaskContextStatus, UnifiedWorkbenchStatus> =
  {
    active: "active",
    paused: "paused",
    completed: "completed",
    archived: "archived",
  };

export const WORKBENCH_PROACTIVE_GOAL_STATUS_MAP: Record<
  ProactiveGoalStatus,
  UnifiedWorkbenchStatus
> = {
  active: "active",
  paused: "paused",
  completed: "completed",
};

export const WORKBENCH_PROACTIVE_TASK_STATUS_MAP: Record<
  ProactiveTaskStatus,
  UnifiedWorkbenchStatus
> = {
  pending: "queued",
  running: "active",
  paused: "paused",
  completed: "completed",
  failed: "blocked",
};

export const WORKBENCH_DEEP_LINK_SUPPORT: Record<
  WorkbenchDeepLinkView,
  { available: boolean; params: string[] }
> = {
  session: { available: true, params: ["sessionId"] },
  project: { available: true, params: ["projectId"] },
  "task-context": { available: false, params: ["taskId"] },
};

export const WORKBENCH_POLL_INTERVAL_MS = 30_000;
export const WORKBENCH_DEFAULT_ROW_LIMIT = 100;
export const WORKBENCH_SUMMARY_MAX_CHARS = 200;

export function createEmptyWorkbenchCounts(): Record<UnifiedWorkbenchStatus, number> {
  return {
    active: 0,
    paused: 0,
    blocked: 0,
    queued: 0,
    completed: 0,
    archived: 0,
  };
}

export function truncateWorkbenchSummary(
  text: string | null | undefined,
  maxChars = WORKBENCH_SUMMARY_MAX_CHARS,
): string {
  const normalized = text?.replace(/\s+/g, " ").trim() ?? "";
  if (!normalized) {
    return "";
  }
  if (normalized.length <= maxChars) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxChars - 1)).trimEnd()}…`;
}
