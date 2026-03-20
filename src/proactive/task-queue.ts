import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { resolveStateDir } from "../core/config/paths.js";
import { withFileLock } from "../infra/file-lock.js";
import { createAsyncLock } from "../infra/json-files.js";

// ── Types ──────────────────────────────────────────────────────────────

export type TaskPriority = "urgent" | "high" | "normal" | "low";

export type TaskStatus = "pending" | "running" | "paused" | "completed" | "failed";

export type ProactiveTask = {
  id: string;
  goalId?: string;
  /**
   * Deterministic fingerprint for idempotent enqueue.
   * If set, `enqueueTaskIdempotent` will skip creating a duplicate when a
   * task with the same fingerprint is already pending/running/recently completed.
   */
  fingerprint?: string;
  title: string;
  description: string;
  priority: TaskPriority;
  status: TaskStatus;
  /** Opaque checkpoint data saved by the runner so the task can resume. */
  checkpoint?: unknown;
  /** Number of times this task has been started (incremented on each dequeue). */
  attemptCount: number;
  /** Timestamp when the task was last moved to "running". */
  lastStartedAt?: number;
  createdAt: number;
  updatedAt: number;
  /** Result summary written on completion or failure. */
  result?: string;
};

export type TaskQueueData = {
  tasks: ProactiveTask[];
};

// ── Priority ordering ──────────────────────────────────────────────────

const PRIORITY_ORDER: Record<TaskPriority, number> = {
  urgent: 0,
  high: 1,
  normal: 2,
  low: 3,
};

function compareTasks(a: ProactiveTask, b: ProactiveTask): number {
  const priorityDiff = PRIORITY_ORDER[a.priority] - PRIORITY_ORDER[b.priority];
  if (priorityDiff !== 0) {
    return priorityDiff;
  }
  return a.createdAt - b.createdAt; // FIFO within same priority
}

// ── File lock infrastructure (mirrors digest-buffer.ts) ────────────────

const TASK_QUEUE_LOCK_OPTIONS = {
  retries: {
    retries: 8,
    factor: 2,
    minTimeout: 25,
    maxTimeout: 1_000,
    randomize: true,
  },
  stale: 10_000,
} as const;

const withTaskQueueProcessLock = createAsyncLock();

function resolveTaskQueuePath(agentId: string): string {
  return path.join(resolveStateDir(process.env), "proactive", agentId, "tasks.json");
}

async function readTaskQueueFromPath(filePath: string): Promise<TaskQueueData> {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const parsed = JSON.parse(raw) as TaskQueueData;
    return {
      tasks: Array.isArray(parsed.tasks) ? [...parsed.tasks] : [],
    };
  } catch {
    return { tasks: [] };
  }
}

async function writeTaskQueueToPath(filePath: string, data: TaskQueueData): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(data, null, 2)}\n`, "utf-8");
}

async function withTaskQueueLock<T>(
  agentId: string,
  fn: (filePath: string) => Promise<T>,
): Promise<T> {
  const filePath = resolveTaskQueuePath(agentId);
  return await withTaskQueueProcessLock(
    async () =>
      await withFileLock(filePath, TASK_QUEUE_LOCK_OPTIONS, async () => await fn(filePath)),
  );
}

// ── Public API ─────────────────────────────────────────────────────────

export async function readTaskQueue(agentId: string): Promise<TaskQueueData> {
  const filePath = resolveTaskQueuePath(agentId);
  return await readTaskQueueFromPath(filePath);
}

/** Add a new task to the queue. Returns the created task. */
export async function enqueueTask(
  agentId: string,
  params: {
    goalId?: string;
    title: string;
    description: string;
    priority?: TaskPriority;
  },
): Promise<ProactiveTask> {
  return await withTaskQueueLock(agentId, async (filePath) => {
    const data = await readTaskQueueFromPath(filePath);
    const now = Date.now();
    const task: ProactiveTask = {
      id: `ptask_${crypto.randomUUID().replace(/-/g, "")}`,
      goalId: params.goalId,
      title: params.title,
      description: params.description,
      priority: params.priority ?? "normal",
      status: "pending",
      attemptCount: 0,
      createdAt: now,
      updatedAt: now,
    };
    data.tasks.push(task);
    await writeTaskQueueToPath(filePath, data);
    return task;
  });
}

/** Max age (ms) for a completed task to still block a duplicate fingerprint. */
const FINGERPRINT_COMPLETED_WINDOW_MS = 2 * 60 * 60_000; // 2 hours

/**
 * Enqueue a task only if no existing task with the same fingerprint is
 * pending, running, or was completed within the dedup window.
 * Returns the existing task if deduped, or the newly created task.
 */
export async function enqueueTaskIdempotent(
  agentId: string,
  params: {
    fingerprint: string;
    goalId?: string;
    title: string;
    description: string;
    priority?: TaskPriority;
  },
): Promise<{ task: ProactiveTask; created: boolean }> {
  return await withTaskQueueLock(agentId, async (filePath) => {
    const data = await readTaskQueueFromPath(filePath);
    const now = Date.now();
    const cutoff = now - FINGERPRINT_COMPLETED_WINDOW_MS;

    const existing = data.tasks.find(
      (t) =>
        t.fingerprint === params.fingerprint &&
        (t.status === "pending" ||
          t.status === "running" ||
          t.status === "paused" ||
          (t.status === "completed" && t.updatedAt > cutoff)),
    );
    if (existing) {
      return { task: { ...existing }, created: false };
    }

    const task: ProactiveTask = {
      id: `ptask_${crypto.randomUUID().replace(/-/g, "")}`,
      goalId: params.goalId,
      fingerprint: params.fingerprint,
      title: params.title,
      description: params.description,
      priority: params.priority ?? "normal",
      status: "pending",
      attemptCount: 0,
      createdAt: now,
      updatedAt: now,
    };
    data.tasks.push(task);
    await writeTaskQueueToPath(filePath, data);
    return { task, created: true };
  });
}

/**
 * Dequeue the highest-priority pending task and mark it as running.
 * Also dequeues paused tasks (resuming them) after all pending tasks.
 * Returns null if no actionable task exists.
 */
export async function dequeueTask(agentId: string): Promise<ProactiveTask | null> {
  return await withTaskQueueLock(agentId, async (filePath) => {
    const data = await readTaskQueueFromPath(filePath);
    const pending = data.tasks.filter((t) => t.status === "pending").toSorted(compareTasks);
    const paused = data.tasks.filter((t) => t.status === "paused").toSorted(compareTasks);
    const candidate = pending[0] ?? paused[0];
    if (!candidate) {
      return null;
    }

    const now = Date.now();
    candidate.status = "running";
    candidate.attemptCount = (candidate.attemptCount ?? 0) + 1;
    candidate.lastStartedAt = now;
    candidate.updatedAt = now;
    await writeTaskQueueToPath(filePath, data);
    return { ...candidate };
  });
}

/** Save checkpoint data for a running task so it can resume later. */
export async function checkpointTask(
  agentId: string,
  taskId: string,
  checkpoint: unknown,
): Promise<void> {
  await withTaskQueueLock(agentId, async (filePath) => {
    const data = await readTaskQueueFromPath(filePath);
    const task = data.tasks.find((t) => t.id === taskId);
    if (!task) {
      return;
    }
    task.checkpoint = checkpoint;
    task.status = "paused";
    task.updatedAt = Date.now();
    await writeTaskQueueToPath(filePath, data);
  });
}

/** Mark a running task as completed with an optional result summary. */
export async function completeTask(
  agentId: string,
  taskId: string,
  result?: string,
): Promise<void> {
  await withTaskQueueLock(agentId, async (filePath) => {
    const data = await readTaskQueueFromPath(filePath);
    const task = data.tasks.find((t) => t.id === taskId);
    if (!task) {
      return;
    }
    task.status = "completed";
    task.result = result;
    task.updatedAt = Date.now();
    await writeTaskQueueToPath(filePath, data);
  });
}

/** Mark a running task as failed with an error summary. */
export async function failTask(agentId: string, taskId: string, error?: string): Promise<void> {
  await withTaskQueueLock(agentId, async (filePath) => {
    const data = await readTaskQueueFromPath(filePath);
    const task = data.tasks.find((t) => t.id === taskId);
    if (!task) {
      return;
    }
    task.status = "failed";
    task.result = error;
    task.updatedAt = Date.now();
    await writeTaskQueueToPath(filePath, data);
  });
}

/** Pause a running task (without checkpoint data). */
export async function pauseTask(agentId: string, taskId: string): Promise<void> {
  await withTaskQueueLock(agentId, async (filePath) => {
    const data = await readTaskQueueFromPath(filePath);
    const task = data.tasks.find((t) => t.id === taskId);
    if (!task || task.status !== "running") {
      return;
    }
    task.status = "paused";
    task.updatedAt = Date.now();
    await writeTaskQueueToPath(filePath, data);
  });
}

/** List tasks, optionally filtered by status. */
export async function listTasks(
  agentId: string,
  filter?: { status?: TaskStatus },
): Promise<ProactiveTask[]> {
  const data = await readTaskQueue(agentId);
  let tasks = data.tasks;
  if (filter?.status) {
    tasks = tasks.filter((t) => t.status === filter.status);
  }
  return tasks.toSorted(compareTasks);
}

/**
 * Recover tasks stuck in "running" state (orphaned by crash/restart).
 * Tasks with `lastStartedAt` older than `staleThresholdMs` are moved back to "paused"
 * so they can be re-dequeued. Returns the number of recovered tasks.
 */
export async function recoverStaleTasks(
  agentId: string,
  staleThresholdMs: number,
): Promise<number> {
  return await withTaskQueueLock(agentId, async (filePath) => {
    const data = await readTaskQueueFromPath(filePath);
    const cutoff = Date.now() - staleThresholdMs;
    let recovered = 0;
    for (const task of data.tasks) {
      if (task.status !== "running") {
        continue;
      }
      const startedAt = task.lastStartedAt ?? task.updatedAt;
      if (startedAt < cutoff) {
        task.status = "paused";
        task.updatedAt = Date.now();
        recovered++;
      }
    }
    if (recovered > 0) {
      await writeTaskQueueToPath(filePath, data);
    }
    return recovered;
  });
}

/** Remove completed and failed tasks older than the given age in ms. */
export async function pruneFinishedTasks(agentId: string, maxAgeMs: number): Promise<number> {
  return await withTaskQueueLock(agentId, async (filePath) => {
    const data = await readTaskQueueFromPath(filePath);
    const cutoff = Date.now() - maxAgeMs;
    const before = data.tasks.length;
    data.tasks = data.tasks.filter(
      (t) => (t.status !== "completed" && t.status !== "failed") || t.updatedAt > cutoff,
    );
    const removed = before - data.tasks.length;
    if (removed > 0) {
      await writeTaskQueueToPath(filePath, data);
    }
    return removed;
  });
}
