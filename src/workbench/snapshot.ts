import {
  getTaskContextRollingSummary,
  listTaskContextsForAgent,
  listTaskDecisionBookmarks,
} from "../memory/task-context/index.js";
import { listDeliverables } from "../proactive/deliverables.js";
import { listGoals } from "../proactive/goals.js";
import { listTasks } from "../proactive/task-queue.js";
import {
  WORKBENCH_DEFAULT_ROW_LIMIT,
  WORKBENCH_PROACTIVE_GOAL_STATUS_MAP,
  WORKBENCH_PROACTIVE_TASK_STATUS_MAP,
  WORKBENCH_TASK_CONTEXT_STATUS_MAP,
  createEmptyWorkbenchCounts,
  truncateWorkbenchSummary,
  type WorkbenchDeepLink,
  type WorkbenchRow,
  type WorkbenchSnapshot,
} from "./types.js";

function toIsoTimestamp(value: number | undefined): string {
  const safeValue = typeof value === "number" && Number.isFinite(value) && value > 0 ? value : 0;
  return new Date(safeValue).toISOString();
}

function createTaskContextSummary(params: {
  agentId: string;
  taskId: string;
  totalEntries: number;
  totalTokens: number;
}): string {
  const rollingSummary = getTaskContextRollingSummary({
    agentId: params.agentId,
    taskId: params.taskId,
  });
  if (rollingSummary) {
    return truncateWorkbenchSummary(rollingSummary);
  }
  const bookmarks = listTaskDecisionBookmarks({
    agentId: params.agentId,
    taskId: params.taskId,
    limit: 2,
  });
  if (bookmarks.length > 0) {
    return truncateWorkbenchSummary(bookmarks.map((bookmark) => bookmark.content).join(" | "));
  }
  return truncateWorkbenchSummary(
    `${params.totalEntries} entries · ${params.totalTokens.toLocaleString()} tokens`,
  );
}

function createProjectDeepLink(projectId: string): WorkbenchDeepLink {
  return {
    view: "project",
    params: { projectId },
  };
}

export async function getWorkbenchSnapshot(params: {
  agentId: string;
  rowLimit?: number;
}): Promise<WorkbenchSnapshot> {
  const rowLimit = Math.max(1, Math.floor(params.rowLimit ?? WORKBENCH_DEFAULT_ROW_LIMIT));
  const taskContexts = listTaskContextsForAgent({
    agentId: params.agentId,
    limit: rowLimit,
  });
  const [goals, tasks, deliverables] = await Promise.all([
    listGoals(params.agentId),
    listTasks(params.agentId),
    listDeliverables(params.agentId),
  ]);

  const rows: WorkbenchRow[] = [
    ...taskContexts.map((taskContext) => ({
      id: taskContext.taskId,
      source: "task-context" as const,
      title: taskContext.title,
      status: WORKBENCH_TASK_CONTEXT_STATUS_MAP[taskContext.status],
      updatedAt: toIsoTimestamp(taskContext.updatedAt),
      summary: createTaskContextSummary({
        agentId: params.agentId,
        taskId: taskContext.taskId,
        totalEntries: taskContext.totalEntries,
        totalTokens: taskContext.totalTokens,
      }),
      deepLink: createProjectDeepLink(taskContext.taskId),
    })),
    ...goals.map((goal) => ({
      id: goal.id,
      source: "proactive-goal" as const,
      title: goal.title,
      status: WORKBENCH_PROACTIVE_GOAL_STATUS_MAP[goal.status],
      updatedAt: toIsoTimestamp(goal.updatedAt),
      summary: truncateWorkbenchSummary(goal.description),
      deepLink: null,
    })),
    ...tasks.map((task) => ({
      id: task.id,
      source: "proactive-task" as const,
      title: task.title,
      status: WORKBENCH_PROACTIVE_TASK_STATUS_MAP[task.status],
      updatedAt: toIsoTimestamp(task.updatedAt),
      summary: truncateWorkbenchSummary(task.result || task.description),
      deepLink: task.goalId ? createProjectDeepLink(task.goalId) : null,
    })),
  ]
    .toSorted((left, right) => right.updatedAt.localeCompare(left.updatedAt))
    .slice(0, rowLimit);

  const counts = createEmptyWorkbenchCounts();
  for (const row of rows) {
    counts[row.status] += 1;
  }

  const completedDeliverables = deliverables.filter(
    (deliverable) => deliverable.status !== "failed",
  );

  return {
    agentId: params.agentId,
    rows,
    counts,
    deliverableSummary: {
      total: deliverables.length,
      completed: completedDeliverables.length,
    },
    fetchedAt: new Date().toISOString(),
  };
}
