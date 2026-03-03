import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  appendTaskContextEntry,
  createTaskContext,
  getTaskContext,
  listTaskContextEntries,
  listTaskContextsForAgent,
  normalizeTaskId,
  resolveTaskContextDbPath,
  updateTaskContextStatus,
} from "./store.js";

let stateDir = "";
let previousStateDir: string | undefined;

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-task-context-"));
  previousStateDir = process.env.MARV_STATE_DIR;
  process.env.MARV_STATE_DIR = stateDir;
});

afterEach(async () => {
  if (previousStateDir === undefined) {
    delete process.env.MARV_STATE_DIR;
  } else {
    process.env.MARV_STATE_DIR = previousStateDir;
  }
  if (stateDir) {
    await fs.rm(stateDir, { recursive: true, force: true });
  }
});

describe("task-context store", () => {
  it("stores each task in an isolated sqlite file and tracks totals", () => {
    const first = createTaskContext({
      agentId: "main",
      taskId: "task-a",
      title: "Task A",
      nowMs: 1000,
    });
    const second = createTaskContext({
      agentId: "main",
      taskId: "task-b",
      title: "Task B",
      nowMs: 1001,
    });

    expect(resolveTaskContextDbPath({ agentId: "main", taskId: first.taskId })).not.toBe(
      resolveTaskContextDbPath({ agentId: "main", taskId: second.taskId }),
    );

    appendTaskContextEntry({
      agentId: "main",
      taskId: first.taskId,
      role: "user",
      content: "First task request",
      tokenCount: 5,
      createdAt: 1100,
    });
    appendTaskContextEntry({
      agentId: "main",
      taskId: first.taskId,
      role: "assistant",
      content: "First task answer",
      tokenCount: 6,
      createdAt: 1200,
      metadata: { tool: "none", confidence: 0.8 },
    });
    appendTaskContextEntry({
      agentId: "main",
      taskId: second.taskId,
      role: "user",
      content: "Second task request",
      tokenCount: 7,
      createdAt: 1300,
    });

    const firstEntries = listTaskContextEntries({ agentId: "main", taskId: first.taskId });
    const secondEntries = listTaskContextEntries({ agentId: "main", taskId: second.taskId });

    expect(firstEntries).toHaveLength(2);
    expect(secondEntries).toHaveLength(1);
    expect(firstEntries[0]?.sequence).toBe(1);
    expect(firstEntries[1]?.sequence).toBe(2);
    expect(firstEntries[1]?.metadata).toContain('"tool":"none"');

    const firstTask = getTaskContext({ agentId: "main", taskId: first.taskId });
    const secondTask = getTaskContext({ agentId: "main", taskId: second.taskId });
    expect(firstTask?.totalEntries).toBe(2);
    expect(firstTask?.totalTokens).toBe(11);
    expect(secondTask?.totalEntries).toBe(1);
    expect(secondTask?.totalTokens).toBe(7);
  });

  it("supports status transitions and filtering by status", () => {
    createTaskContext({
      agentId: "main",
      taskId: "active-task",
      title: "Active Task",
      nowMs: 1000,
    });
    createTaskContext({
      agentId: "main",
      taskId: "pause-task",
      title: "Pause Task",
      nowMs: 1001,
    });

    const paused = updateTaskContextStatus({
      agentId: "main",
      taskId: "pause-task",
      status: "paused",
      updatedAt: 2000,
    });
    expect(paused?.status).toBe("paused");
    expect(paused?.completedAt).toBeUndefined();

    const completed = updateTaskContextStatus({
      agentId: "main",
      taskId: "active-task",
      status: "completed",
      updatedAt: 2100,
    });
    expect(completed?.status).toBe("completed");
    expect(completed?.completedAt).toBe(2100);

    const archived = updateTaskContextStatus({
      agentId: "main",
      taskId: "active-task",
      status: "archived",
      updatedAt: 2200,
    });
    expect(archived?.status).toBe("archived");
    expect(archived?.completedAt).toBe(2100);

    const pausedTasks = listTaskContextsForAgent({ agentId: "main", status: "paused" });
    const archivedTasks = listTaskContextsForAgent({ agentId: "main", status: "archived" });
    expect(pausedTasks).toHaveLength(1);
    expect(pausedTasks[0]?.taskId).toBe("pause-task");
    expect(archivedTasks).toHaveLength(1);
    expect(archivedTasks[0]?.taskId).toBe("active-task");
  });

  it("normalizes unsafe task ids and keeps entry sequence monotonic", () => {
    const task = createTaskContext({
      agentId: "Main Agent",
      taskId: " Feature/#1 ",
      title: "Feature Implementation",
      nowMs: 5000,
    });
    expect(task.agentId).toBe("main-agent");
    expect(task.taskId).toBe(normalizeTaskId(" Feature/#1 "));

    const one = appendTaskContextEntry({
      agentId: task.agentId,
      taskId: task.taskId,
      role: "user",
      content: "step one",
      createdAt: 5100,
    });
    const two = appendTaskContextEntry({
      agentId: task.agentId,
      taskId: task.taskId,
      role: "assistant",
      content: "step two",
      createdAt: 5200,
    });

    expect(one?.sequence).toBe(1);
    expect(two?.sequence).toBe(2);
  });
});
