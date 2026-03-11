import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { loadGoalStrategyHints, persistGoalStrategyMemory } from "./goal-loop-memory.js";
import {
  buildGoalSteeringContext,
  createGoalLoopState,
  reviewGoalProgress,
  shouldSkipGoalFrame,
  type GoalLoopState,
} from "./goal-loop.js";
import type { EmbeddedRunAttemptResult } from "./run/types.js";

const ORIGINAL_STATE_DIR = process.env.MARV_STATE_DIR;

let stateDir = "";

function createAttemptResult(
  overrides: Partial<EmbeddedRunAttemptResult> = {},
): EmbeddedRunAttemptResult {
  return {
    aborted: false,
    timedOut: false,
    timedOutDuringCompaction: false,
    promptError: null,
    sessionIdUsed: "session-1",
    messagesSnapshot: [],
    assistantTexts: ["done"],
    toolMetas: [],
    lastAssistant: {
      role: "assistant",
      content: "done",
      stopReason: "end_turn",
    } as unknown as EmbeddedRunAttemptResult["lastAssistant"],
    didSendViaMessagingTool: false,
    messagingToolSentTexts: [],
    messagingToolSentMediaUrls: [],
    messagingToolSentTargets: [],
    cloudCodeAssistFormatError: false,
    ...overrides,
  };
}

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-goal-loop-"));
  process.env.MARV_STATE_DIR = stateDir;
});

afterEach(async () => {
  if (ORIGINAL_STATE_DIR === undefined) {
    delete process.env.MARV_STATE_DIR;
  } else {
    process.env.MARV_STATE_DIR = ORIGINAL_STATE_DIR;
  }
  if (stateDir) {
    await fs.rm(stateDir, { recursive: true, force: true });
  }
});

describe("goal-loop", () => {
  it("skips trivial greetings but keeps implementation prompts", () => {
    expect(shouldSkipGoalFrame("hi")).toBe(true);
    expect(shouldSkipGoalFrame("hello")).toBe(true);
    expect(shouldSkipGoalFrame("读取文档，完成这个计划，不需要brainstorm")).toBe(false);
  });

  it("creates a directional plan and initial steering context", () => {
    const state = createGoalLoopState({
      prompt: "Read the plan, implement the change, and keep user interruptions minimal.",
      priorStrategyHints: [
        {
          memoryId: "mem1",
          summary: "Smallest-change-first worked well for similar runtime edits.",
          strategyFamily: "try_alternative",
          problemShape: "implementation_blocked",
          score: 0.8,
        },
      ],
    });
    expect(state).not.toBeNull();
    if (!state) {
      throw new Error("goal loop state missing");
    }
    expect(state.directionNodes.length).toBeGreaterThanOrEqual(3);
    expect(state.directionNodes.length).toBeLessThanOrEqual(5);
    const steering = buildGoalSteeringContext(state, { includeAnchor: true });
    expect(steering).toContain("Objective:");
    expect(steering).toContain("Relevant prior strategy hints:");
  });

  it("escalates to force_shift on critical loop signals", () => {
    const initial = createGoalLoopState({
      prompt: "Fix the failing validation and keep going without asking me.",
    }) as GoalLoopState;
    const review = reviewGoalProgress({
      state: {
        ...initial,
        stuckCounter: 3,
        strategyFamily: "try_alternative",
      },
      attempt: createAttemptResult({
        assistantTexts: [],
        lastAssistant: {
          role: "assistant",
          content: "still failing",
          stopReason: "error",
          errorMessage: "validation failed again",
        } as unknown as EmbeddedRunAttemptResult["lastAssistant"],
      }),
      recentToolCalls: [
        {
          toolName: "test",
          argsHash: "a",
          resultHash: "same",
          timestamp: Date.now(),
        },
        {
          toolName: "test",
          argsHash: "a",
          resultHash: "same",
          timestamp: Date.now(),
        },
      ],
      priorResultHashes: new Set(["same"]),
      recentLoopEvents: [
        {
          level: "critical",
          detector: "known_poll_no_progress",
          count: 20,
          message: "loop",
          timestamp: Date.now(),
        },
      ],
      promptErrorText: "validation failed again",
    });
    expect(review.classification).toBe("stalled");
    expect(review.state.loopGuardLevel).toBe("force_shift");
    expect(review.state.shiftCount).toBeGreaterThan(0);
    expect(review.strategyFamily).toBe("delegated_subagent");
    expect(review.steeringContext).toContain("task_dispatch");
  });

  it("persists and reloads strategy hints through soul memory", () => {
    const state = createGoalLoopState({
      prompt: "Implement the smallest safe fix and validate it.",
    }) as GoalLoopState;
    const successfulState: GoalLoopState = {
      ...state,
      problemShape: "implementation_blocked",
      strategyFamily: "try_alternative",
      currentNodeIndex: 3,
      convergeReason: "sufficient_completion",
    };
    persistGoalStrategyMemory({
      agentId: "main",
      sessionKey: "agent:main:telegram:direct:u123",
      state: successfulState,
    });
    const hints = loadGoalStrategyHints({
      agentId: "main",
      sessionKey: "agent:main:telegram:direct:u123",
      objective: successfulState.goalFrame.objective,
      problemShape: "implementation_blocked",
    });
    expect(hints.length).toBeGreaterThan(0);
    expect(hints[0]?.strategyFamily).toBe("try_alternative");
    expect(hints[0]?.problemShape).toBe("implementation_blocked");
  });

  it("switches from requesting capability to synthesizing a tool after repeated boundary failures", () => {
    const initial = createGoalLoopState({
      prompt: "Read an unknown binary file, figure out the format, and keep going.",
    }) as GoalLoopState;
    const review = reviewGoalProgress({
      state: {
        ...initial,
        attemptCount: 2,
        strategyFamily: "request_capability",
      },
      attempt: createAttemptResult({
        assistantTexts: [],
        lastToolError: {
          toolName: "apply_patch",
          error: "write failed again",
          mutatingAction: true,
        },
        lastAssistant: {
          role: "assistant",
          content: "still blocked",
          stopReason: "error",
          errorMessage: "This is binary content and cannot display binary.",
        } as unknown as EmbeddedRunAttemptResult["lastAssistant"],
      }),
      recentToolCalls: [
        {
          toolName: "read",
          argsHash: "a",
          resultHash: "same",
          timestamp: Date.now(),
        },
      ],
      priorResultHashes: new Set(["same"]),
      recentLoopEvents: [],
      promptErrorText: "detectedMimeType application/zip",
    });

    expect(review.problemShape).toBe("tool_or_permission_limit");
    expect(review.strategyFamily).toBe("synthesize_tool");
    expect(review.visibility).toBe("building the missing tool");
    expect(review.steeringContext).toContain("Write a targeted script");
  });

  it("switches into delegated recovery after repeated stalled validation attempts", () => {
    const initial = createGoalLoopState({
      prompt: "Fix the failing tests and keep the scope tight.",
    }) as GoalLoopState;
    const review = reviewGoalProgress({
      state: {
        ...initial,
        attemptCount: 2,
        stuckCounter: 2,
        strategyFamily: "inspect_failure",
      },
      attempt: createAttemptResult({
        assistantTexts: [],
        lastAssistant: {
          role: "assistant",
          content: "still failing",
          stopReason: "error",
          errorMessage: "validation failed again",
        } as unknown as EmbeddedRunAttemptResult["lastAssistant"],
      }),
      recentToolCalls: [
        {
          toolName: "test",
          argsHash: "a",
          resultHash: "same",
          timestamp: Date.now(),
        },
      ],
      priorResultHashes: new Set(["same"]),
      recentLoopEvents: [],
      promptErrorText: "validation failed again",
      canDelegate: true,
    });

    expect(review.problemShape).toBe("validation_failure");
    expect(review.strategyFamily).toBe("delegated_subagent");
    expect(review.strategyTrack).toBe("delegated_subagent");
    expect(review.delegation?.roles).toEqual(["reviewer", "tester"]);
    expect(review.steeringContext).toContain("task_dispatch");
  });

  it("uses a final delegated recovery pass before stopping for stagnation", () => {
    const initial = createGoalLoopState({
      prompt: "Investigate the blocker and finish the fix without asking me.",
    }) as GoalLoopState;
    const review = reviewGoalProgress({
      state: {
        ...initial,
        shiftCount: 2,
        stuckCounter: 5,
        strategyFamily: "try_alternative",
      },
      attempt: createAttemptResult({
        assistantTexts: [],
        lastToolError: {
          toolName: "apply_patch",
          error: "write failed again",
          mutatingAction: true,
        },
        lastAssistant: {
          role: "assistant",
          content: "still blocked",
          stopReason: "error",
          errorMessage: "write failed again",
        } as unknown as EmbeddedRunAttemptResult["lastAssistant"],
      }),
      recentToolCalls: [
        {
          toolName: "apply_patch",
          argsHash: "a",
          resultHash: "same",
          timestamp: Date.now(),
        },
      ],
      priorResultHashes: new Set(["same"]),
      recentLoopEvents: [],
      promptErrorText: "write failed again",
      canDelegate: true,
    });

    expect(review.strategyFamily).toBe("delegated_subagent");
    expect(review.state.loopGuardLevel).toBe("force_shift");
    expect(review.delegation?.roles).toEqual(["debugger"]);
  });

  it("keeps internal-system runs out of delegated recovery", () => {
    const initial = createGoalLoopState({
      prompt: "Investigate the blocker and finish the fix without asking me.",
    }) as GoalLoopState;
    const review = reviewGoalProgress({
      state: {
        ...initial,
        shiftCount: 2,
        stuckCounter: 5,
        strategyFamily: "try_alternative",
      },
      attempt: createAttemptResult({
        assistantTexts: [],
        lastAssistant: {
          role: "assistant",
          content: "still blocked",
          stopReason: "error",
          errorMessage: "write failed again",
        } as unknown as EmbeddedRunAttemptResult["lastAssistant"],
      }),
      recentToolCalls: [
        {
          toolName: "apply_patch",
          argsHash: "a",
          resultHash: "same",
          timestamp: Date.now(),
        },
      ],
      priorResultHashes: new Set(["same"]),
      recentLoopEvents: [],
      promptErrorText: "write failed again",
      canDelegate: false,
    });

    expect(review.strategyFamily).not.toBe("delegated_subagent");
    expect(review.state.loopGuardLevel).toBe("stop");
  });
});
