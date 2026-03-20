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

// Mock loadConfig.
vi.mock("../core/config/config.js", () => ({
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
}));

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

import { CommandLane } from "../process/lanes.js";
import { shouldYieldToUser } from "./task-runner.js";

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
