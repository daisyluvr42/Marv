import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  listTaskContextsForAgent: vi.fn(),
  getTaskContextRollingSummary: vi.fn(),
  listTaskDecisionBookmarks: vi.fn(),
  listGoals: vi.fn(),
  listTasks: vi.fn(),
  listDeliverables: vi.fn(),
}));

vi.mock("../memory/task-context/index.js", () => ({
  listTaskContextsForAgent: mocks.listTaskContextsForAgent,
  getTaskContextRollingSummary: mocks.getTaskContextRollingSummary,
  listTaskDecisionBookmarks: mocks.listTaskDecisionBookmarks,
}));

vi.mock("../proactive/goals.js", () => ({
  listGoals: mocks.listGoals,
}));

vi.mock("../proactive/task-queue.js", () => ({
  listTasks: mocks.listTasks,
}));

vi.mock("../proactive/deliverables.js", () => ({
  listDeliverables: mocks.listDeliverables,
}));

import { getWorkbenchSnapshot } from "./snapshot.js";

describe("getWorkbenchSnapshot", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.listTaskContextsForAgent.mockReturnValue([
      {
        taskId: "task-alpha",
        agentId: "main",
        title: "Finish planner",
        status: "active",
        createdAt: 1,
        updatedAt: 3000,
        scopeId: "task:alpha",
        totalEntries: 4,
        totalTokens: 240,
      },
    ]);
    mocks.getTaskContextRollingSummary.mockReturnValue("Current work summary");
    mocks.listTaskDecisionBookmarks.mockReturnValue([]);
    mocks.listGoals.mockResolvedValue([
      {
        id: "goal-beta",
        title: "Ship workbench",
        description: "Expose task and proactive state in one dashboard surface.",
        priority: "high",
        status: "paused",
        createdAt: 2,
        updatedAt: 2000,
      },
    ]);
    mocks.listTasks.mockResolvedValue([
      {
        id: "ptask-gamma",
        goalId: "goal-beta",
        title: "Render overview card",
        description: "Add a summary card to overview.",
        priority: "normal",
        status: "pending",
        attemptCount: 0,
        createdAt: 3,
        updatedAt: 1000,
      },
    ]);
    mocks.listDeliverables.mockResolvedValue([
      {
        id: "deliv-1",
        title: "Plan",
        kind: "plan",
        status: "stored",
        createdAt: 10,
      },
      {
        id: "deliv-2",
        title: "Broken artifact",
        kind: "report",
        status: "failed",
        createdAt: 11,
      },
    ]);
  });

  it("aggregates task-context and proactive rows into one compact snapshot", async () => {
    const snapshot = await getWorkbenchSnapshot({ agentId: "main" });

    expect(snapshot.agentId).toBe("main");
    expect(snapshot.rows.map((row) => row.id)).toEqual(["task-alpha", "goal-beta", "ptask-gamma"]);
    expect(snapshot.rows[0]).toMatchObject({
      source: "task-context",
      title: "Finish planner",
      status: "active",
      summary: "Current work summary",
      deepLink: { view: "project", params: { projectId: "task-alpha" } },
    });
    expect(snapshot.rows[1]).toMatchObject({
      source: "proactive-goal",
      status: "paused",
    });
    expect(snapshot.rows[2]).toMatchObject({
      source: "proactive-task",
      status: "queued",
      deepLink: { view: "project", params: { projectId: "goal-beta" } },
    });
    expect(snapshot.counts).toEqual({
      active: 1,
      paused: 1,
      blocked: 0,
      queued: 1,
      completed: 0,
      archived: 0,
    });
    expect(snapshot.deliverableSummary).toEqual({
      total: 2,
      completed: 1,
    });
  });
});
