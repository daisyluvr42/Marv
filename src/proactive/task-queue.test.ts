import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const testState = vi.hoisted(() => ({
  stateDir: "",
}));

vi.mock("../core/config/paths.js", () => ({
  resolveStateDir: () => testState.stateDir,
}));

import {
  checkpointTask,
  completeTask,
  dequeueTask,
  enqueueTask,
  failTask,
  listTasks,
  pauseTask,
  pruneFinishedTasks,
  readTaskQueue,
  recoverStaleTasks,
} from "./task-queue.js";

let stateDir = "";

function agentId(label: string): string {
  return `${label}-${randomUUID()}`;
}

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-taskq-"));
  testState.stateDir = stateDir;
});

afterEach(async () => {
  testState.stateDir = "";
  await fs.rm(stateDir, { recursive: true, force: true });
});

describe("task queue", () => {
  it("enqueues and dequeues a task", async () => {
    const id = agentId("enqueue");
    const task = await enqueueTask(id, {
      title: "Test task",
      description: "Do something",
    });
    expect(task.id).toMatch(/^ptask_/);
    expect(task.status).toBe("pending");
    expect(task.priority).toBe("normal");

    const dequeued = await dequeueTask(id);
    expect(dequeued).not.toBeNull();
    expect(dequeued!.id).toBe(task.id);
    expect(dequeued!.status).toBe("running");

    // Queue should be empty now.
    const next = await dequeueTask(id);
    expect(next).toBeNull();
  });

  it("dequeues by priority then FIFO", async () => {
    const id = agentId("priority");
    const low = await enqueueTask(id, {
      title: "Low",
      description: "low priority",
      priority: "low",
    });
    const urgent = await enqueueTask(id, {
      title: "Urgent",
      description: "urgent",
      priority: "urgent",
    });
    const normal = await enqueueTask(id, {
      title: "Normal",
      description: "normal",
      priority: "normal",
    });

    const first = await dequeueTask(id);
    expect(first!.id).toBe(urgent.id);
    const second = await dequeueTask(id);
    expect(second!.id).toBe(normal.id);
    const third = await dequeueTask(id);
    expect(third!.id).toBe(low.id);
  });

  it("checkpoint pauses a task and resumes on next dequeue", async () => {
    const id = agentId("checkpoint");
    const task = await enqueueTask(id, {
      title: "Checkpointable",
      description: "will be paused",
    });

    // Dequeue and start.
    await dequeueTask(id);

    // Checkpoint it.
    await checkpointTask(id, task.id, { progress: 50 });

    // Verify it's paused.
    const data = await readTaskQueue(id);
    const saved = data.tasks.find((t) => t.id === task.id);
    expect(saved!.status).toBe("paused");
    expect(saved!.checkpoint).toEqual({ progress: 50 });

    // Dequeue should resume it.
    const resumed = await dequeueTask(id);
    expect(resumed!.id).toBe(task.id);
    expect(resumed!.status).toBe("running");
    expect(resumed!.checkpoint).toEqual({ progress: 50 });
  });

  it("pending tasks are dequeued before paused tasks", async () => {
    const id = agentId("pending-before-paused");
    const paused = await enqueueTask(id, {
      title: "Will pause",
      description: "paused task",
    });
    await dequeueTask(id);
    await checkpointTask(id, paused.id, "partial");

    const pending = await enqueueTask(id, {
      title: "New pending",
      description: "fresh task",
    });

    const first = await dequeueTask(id);
    expect(first!.id).toBe(pending.id);
    const second = await dequeueTask(id);
    expect(second!.id).toBe(paused.id);
  });

  it("completes a task with result", async () => {
    const id = agentId("complete");
    const task = await enqueueTask(id, { title: "Complete me", description: "test" });
    await dequeueTask(id);
    await completeTask(id, task.id, "All done!");

    const data = await readTaskQueue(id);
    const done = data.tasks.find((t) => t.id === task.id);
    expect(done!.status).toBe("completed");
    expect(done!.result).toBe("All done!");
  });

  it("fails a task with error", async () => {
    const id = agentId("fail");
    const task = await enqueueTask(id, { title: "Fail me", description: "test" });
    await dequeueTask(id);
    await failTask(id, task.id, "Something broke");

    const data = await readTaskQueue(id);
    const failed = data.tasks.find((t) => t.id === task.id);
    expect(failed!.status).toBe("failed");
    expect(failed!.result).toBe("Something broke");
  });

  it("pauses a running task", async () => {
    const id = agentId("pause");
    const task = await enqueueTask(id, { title: "Pause me", description: "test" });
    await dequeueTask(id);
    await pauseTask(id, task.id);

    const data = await readTaskQueue(id);
    expect(data.tasks.find((t) => t.id === task.id)!.status).toBe("paused");
  });

  it("lists tasks with optional status filter", async () => {
    const id = agentId("list");
    await enqueueTask(id, { title: "A", description: "a" });
    const b = await enqueueTask(id, { title: "B", description: "b" });
    await dequeueTask(id); // dequeues A (FIFO)
    await completeTask(id, (await readTaskQueue(id)).tasks.find((t) => t.status === "running")!.id);

    const all = await listTasks(id);
    expect(all).toHaveLength(2);

    const pending = await listTasks(id, { status: "pending" });
    expect(pending).toHaveLength(1);
    expect(pending[0].id).toBe(b.id);

    const completed = await listTasks(id, { status: "completed" });
    expect(completed).toHaveLength(1);
  });

  it("prunes old finished tasks", async () => {
    const id = agentId("prune");
    const task = await enqueueTask(id, { title: "Old", description: "old" });
    await dequeueTask(id);
    await completeTask(id, task.id, "done");

    // Prune with 0ms max age should remove it.
    const removed = await pruneFinishedTasks(id, 0);
    expect(removed).toBe(1);

    const data = await readTaskQueue(id);
    expect(data.tasks).toHaveLength(0);
  });

  it("tracks attemptCount and lastStartedAt on dequeue", async () => {
    const id = agentId("attempt-tracking");
    const task = await enqueueTask(id, { title: "Track me", description: "test" });
    expect(task.attemptCount).toBe(0);

    const first = await dequeueTask(id);
    expect(first!.attemptCount).toBe(1);
    expect(first!.lastStartedAt).toBeGreaterThan(0);

    // Checkpoint and re-dequeue to verify increment.
    await checkpointTask(id, task.id, "progress");
    const second = await dequeueTask(id);
    expect(second!.attemptCount).toBe(2);
  });

  it("recovers stale running tasks", async () => {
    const id = agentId("recovery");
    const task = await enqueueTask(id, { title: "Orphan", description: "will be orphaned" });
    await dequeueTask(id);

    // Simulate stale: no recovery needed when threshold is large.
    const recoveredNone = await recoverStaleTasks(id, 999_999_999);
    expect(recoveredNone).toBe(0);

    // Wait 5ms then recover with 1ms threshold — the task is now stale.
    await new Promise((r) => setTimeout(r, 5));
    const recovered = await recoverStaleTasks(id, 1);
    expect(recovered).toBe(1);

    const data = await readTaskQueue(id);
    const t = data.tasks.find((t) => t.id === task.id);
    expect(t!.status).toBe("paused");

    // Should be dequeue-able again.
    const resumed = await dequeueTask(id);
    expect(resumed!.id).toBe(task.id);
  });

  it("returns empty queue for nonexistent agent", async () => {
    const data = await readTaskQueue("nonexistent-agent");
    expect(data.tasks).toHaveLength(0);
    const dequeued = await dequeueTask("nonexistent-agent");
    expect(dequeued).toBeNull();
  });
});
