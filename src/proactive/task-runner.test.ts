import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Mock command queue to control yield behavior.
const mockGetQueueSize = vi.fn().mockReturnValue(0);
vi.mock("../process/command-queue.js", () => ({
  getQueueSize: (...args: unknown[]) => mockGetQueueSize(...args),
  setCommandLaneConcurrency: vi.fn(),
  enqueueCommandInLane: vi.fn(async (_lane: string, task: () => Promise<unknown>) => await task()),
}));

// Mock cron isolated agent turn.
const mockRunCronIsolatedAgentTurn = vi.fn().mockResolvedValue({
  status: "ok",
  summary: "Task completed",
  outputText: "Done",
});
vi.mock("../cron/isolated-agent.js", () => ({
  runCronIsolatedAgentTurn: (...args: unknown[]) => mockRunCronIsolatedAgentTurn(...args),
}));

// Mock sendMessage.
const mockSendMessage = vi.fn().mockResolvedValue(undefined);
vi.mock("../infra/outbound/message.js", () => ({
  sendMessage: (...args: unknown[]) => mockSendMessage(...args),
}));

// Mock deliverables.
const mockRegisterDeliverable = vi.fn();
const mockMarkAnnounced = vi.fn().mockResolvedValue(undefined);
const mockMarkAnnouncementFailed = vi.fn().mockResolvedValue(undefined);
const mockGetPendingAnnouncements = vi.fn().mockResolvedValue([]);
vi.mock("./deliverables.js", () => ({
  registerDeliverable: (...args: unknown[]) => mockRegisterDeliverable(...args),
  markAnnounced: (...args: unknown[]) => mockMarkAnnounced(...args),
  markAnnouncementFailed: (...args: unknown[]) => mockMarkAnnouncementFailed(...args),
  getPendingAnnouncements: (...args: unknown[]) => mockGetPendingAnnouncements(...args),
}));

// Mock remaining transitive deps.
vi.mock("../agents/auto-routing.js", () => ({
  classifyComplexityByRules: () => "simple",
}));
vi.mock("../agents/model/model-pool.js", () => ({
  resolveRuntimeModelPlan: () => ({ candidates: [] }),
}));
vi.mock("../experiments/protocol.js", () => ({
  runExperiment: vi.fn(),
  summarizeExperiment: vi.fn(),
}));
vi.mock("./budget.js", () => ({
  getBudgetStatus: vi.fn().mockResolvedValue({ exhausted: false, todayTokens: 0, dailyLimit: 0 }),
  recordTokenUsage: vi.fn(),
}));
vi.mock("./task-queue.js", () => ({
  dequeueTask: vi.fn().mockResolvedValue(null),
  completeTask: vi.fn(),
  failTask: vi.fn(),
  checkpointTask: vi.fn(),
  pruneFinishedTasks: vi.fn(),
  recoverStaleTasks: vi.fn().mockResolvedValue(0),
}));

// Mock loadConfig — use importOriginal to preserve re-exported constants (STATE_DIR, etc.).
vi.mock("../core/config/config.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../core/config/config.js")>();
  return {
    ...actual,
    loadConfig: () => ({
      autonomy: {
        proactive: {
          continuousLoop: true,
          maxConcurrentTasks: 1,
          yieldToUserMs: 10,
          taskPollIntervalMs: 10,
        },
      },
    }),
  };
});

// Mock agent scope.
vi.mock("../agents/agent-scope.js", () => ({
  resolveDefaultAgentId: () => "test-agent",
}));

// Mock logging.
vi.mock("../logging.js", () => ({
  getChildLogger: () => ({
    info: vi.fn(),
    warn: vi.fn(),
  }),
}));

import type { MarvConfig } from "../core/config/config.js";
import { CommandLane } from "../process/lanes.js";
import type { ProactiveTask } from "./task-queue.js";
import {
  announceTaskCompletion,
  drainPendingAnnouncements,
  shouldYieldToUser,
} from "./task-runner.js";

describe("shouldYieldToUser", () => {
  beforeEach(() => {
    mockGetQueueSize.mockReturnValue(0);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns false when main lane is empty", () => {
    mockGetQueueSize.mockReturnValue(0);
    expect(shouldYieldToUser()).toBe(false);
  });

  it("returns true when main lane has queued work", () => {
    mockGetQueueSize.mockReturnValue(1);
    expect(shouldYieldToUser()).toBe(true);
  });

  it("checks the main lane specifically", () => {
    shouldYieldToUser();
    expect(mockGetQueueSize).toHaveBeenCalledWith(CommandLane.Main);
  });
});

// ── announceTaskCompletion tests ───────────────────────────────────────

const baseCfg = {
  autonomy: {
    proactive: {
      delivery: { channel: "telegram", to: "123" },
    },
  },
} as MarvConfig;

const baseTask: ProactiveTask = {
  id: "task-1",
  title: "Test task",
  description: "Do something",
  priority: "normal",
  status: "running",
  attemptCount: 1,
  createdAt: Date.now(),
  updatedAt: Date.now(),
};

describe("announceTaskCompletion", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("skips already-announced deliverables", async () => {
    mockRegisterDeliverable.mockResolvedValueOnce({
      deliverable: { id: "d1", status: "announced" },
      created: false,
    });

    await announceTaskCompletion(baseCfg, "agent-1", baseTask);

    expect(mockSendMessage).not.toHaveBeenCalled();
    expect(mockMarkAnnounced).not.toHaveBeenCalled();
  });

  it("retries send for stored deliverable with created:false", async () => {
    mockRegisterDeliverable.mockResolvedValueOnce({
      deliverable: { id: "d2", status: "stored" },
      created: false,
    });

    await announceTaskCompletion(baseCfg, "agent-1", baseTask);

    expect(mockSendMessage).toHaveBeenCalledTimes(1);
    expect(mockMarkAnnounced).toHaveBeenCalledWith("agent-1", "d2");
  });

  it("marks announcement failed on send error (deliverable stays retryable)", async () => {
    mockRegisterDeliverable.mockResolvedValueOnce({
      deliverable: { id: "d3", status: "stored" },
      created: true,
    });
    mockSendMessage.mockRejectedValueOnce(new Error("network"));

    await announceTaskCompletion(baseCfg, "agent-1", baseTask);

    expect(mockMarkAnnouncementFailed).toHaveBeenCalledWith(
      "agent-1",
      "d3",
      expect.stringContaining("network"),
    );
    expect(mockMarkAnnounced).not.toHaveBeenCalled();
  });
});

describe("drainPendingAnnouncements", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("sends pending deliverables on idle tick", async () => {
    mockGetPendingAnnouncements.mockResolvedValueOnce([
      { id: "d10", title: "Pending task", status: "stored" },
    ]);

    await drainPendingAnnouncements(baseCfg, "agent-1");

    expect(mockSendMessage).toHaveBeenCalledTimes(1);
    expect(mockMarkAnnounced).toHaveBeenCalledWith("agent-1", "d10");
  });

  it("marks failed on send error during drain", async () => {
    mockGetPendingAnnouncements.mockResolvedValueOnce([
      { id: "d11", title: "Failing task", status: "stored" },
    ]);
    mockSendMessage.mockRejectedValueOnce(new Error("timeout"));

    await drainPendingAnnouncements(baseCfg, "agent-1");

    expect(mockMarkAnnouncementFailed).toHaveBeenCalledWith(
      "agent-1",
      "d11",
      expect.stringContaining("timeout"),
    );
    expect(mockMarkAnnounced).not.toHaveBeenCalled();
  });

  it("does nothing when no pending announcements", async () => {
    mockGetPendingAnnouncements.mockResolvedValueOnce([]);

    await drainPendingAnnouncements(baseCfg, "agent-1");

    expect(mockSendMessage).not.toHaveBeenCalled();
  });
});
