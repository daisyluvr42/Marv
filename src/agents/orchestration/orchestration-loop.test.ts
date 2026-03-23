import { describe, expect, it, vi } from "vitest";
import type { OrchestrationDeps } from "./orchestration-loop.js";
import {
  checkBudget,
  deliverFeedback,
  handleSubagentCompletion,
  isTerminalPhase,
  runOrchestrationCycle,
  startOrchestrationWithContract,
} from "./orchestration-loop.js";
import type { GoalContract, OrchestrationEntry } from "./types.js";

// ── Test Helpers ───────────────────────────────────────────────────

function makeContract(overrides?: Partial<GoalContract>): GoalContract {
  return {
    id: "contract_test",
    objective: "Test objective",
    successCriteria: ["must include hello", "must include world"],
    auditStandards: [
      {
        id: "c_0",
        description: "Has hello",
        evaluator: { kind: "text_match", pattern: "hello" },
        required: true,
      },
      {
        id: "c_1",
        description: "Has world",
        evaluator: { kind: "text_match", pattern: "world" },
        required: true,
      },
    ],
    budget: { maxIterations: 3, maxDurationMs: 300_000 },
    ...overrides,
  };
}

function makeEntry(overrides?: Partial<OrchestrationEntry>): OrchestrationEntry {
  return {
    contractId: "contract_test",
    contract: makeContract(),
    runId: "run_1",
    childSessionKey: "agent:test:subagent:child1",
    phase: "monitoring",
    iteration: 0,
    evaluations: [],
    startedAt: Date.now(),
    ...overrides,
  };
}

function makeDeps(overrides?: Partial<OrchestrationDeps>): OrchestrationDeps {
  return {
    spawnSubagent: vi.fn().mockResolvedValue({
      status: "accepted",
      runId: "run_1",
      childSessionKey: "agent:test:subagent:child1",
    }),
    readSubagentResult: vi.fn().mockResolvedValue({
      status: "ok",
      text: "hello world",
      durationMs: 1000,
    }),
    steerSubagent: vi.fn().mockResolvedValue({
      runId: "run_2",
    }),
    ...overrides,
  };
}

// ── startOrchestrationWithContract ─────────────────────────────────

describe("startOrchestrationWithContract", () => {
  it("spawns subagent and returns entry in spawned phase", async () => {
    const deps = makeDeps();
    const contract = makeContract();

    const entry = await startOrchestrationWithContract({
      contract,
      spawnParams: { task: "Do the thing" },
      spawnCtx: {},
      deps,
    });

    expect(entry.phase).toBe("spawned");
    expect(entry.contractId).toBe("contract_test");
    expect(entry.runId).toBe("run_1");
    expect(entry.childSessionKey).toBe("agent:test:subagent:child1");
    expect(entry.iteration).toBe(0);
    expect(entry.evaluations).toHaveLength(0);
  });

  it("injects contract context into spawn params", async () => {
    const deps = makeDeps();
    const contract = makeContract();

    await startOrchestrationWithContract({
      contract,
      spawnParams: { task: "Do the thing" },
      spawnCtx: {},
      deps,
    });

    const spawnCall = (deps.spawnSubagent as ReturnType<typeof vi.fn>).mock.calls[0];
    const params = spawnCall[0];
    expect(params.contextBlock).toContain("[Audit Standards]");
    expect(params.contextBlock).toContain("Test objective");
  });

  it("appends to existing contextBlock", async () => {
    const deps = makeDeps();
    const contract = makeContract();

    await startOrchestrationWithContract({
      contract,
      spawnParams: { task: "Do the thing", contextBlock: "Existing context" },
      spawnCtx: {},
      deps,
    });

    const params = (deps.spawnSubagent as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(params.contextBlock).toContain("Existing context");
    expect(params.contextBlock).toContain("[Audit Standards]");
  });

  it("throws on spawn failure", async () => {
    const deps = makeDeps({
      spawnSubagent: vi.fn().mockResolvedValue({ status: "forbidden", error: "depth limit" }),
    });

    await expect(
      startOrchestrationWithContract({
        contract: makeContract(),
        spawnParams: { task: "Do the thing" },
        spawnCtx: {},
        deps,
      }),
    ).rejects.toThrow("depth limit");
  });
});

// ── handleSubagentCompletion ───────────────────────────────────────

describe("handleSubagentCompletion", () => {
  it("returns accepted when all criteria pass", async () => {
    const deps = makeDeps({
      readSubagentResult: vi.fn().mockResolvedValue({
        status: "ok",
        text: "hello world",
      }),
    });
    const entry = makeEntry();

    const result = await handleSubagentCompletion({ entry, deps });

    expect(result.phase).toBe("accepted");
    expect(result.evaluations).toHaveLength(1);
    expect(result.evaluations[0].verdict).toBe("accepted");
    expect(result.completedAt).toBeDefined();
  });

  it("returns feedback_delivered when criteria fail with budget remaining", async () => {
    const deps = makeDeps({
      readSubagentResult: vi.fn().mockResolvedValue({
        status: "ok",
        text: "hello but no second word",
      }),
    });
    const entry = makeEntry();

    const result = await handleSubagentCompletion({ entry, deps });

    expect(result.phase).toBe("feedback_delivered");
    expect(result.evaluations).toHaveLength(1);
    expect(result.evaluations[0].verdict).toBe("needs_revision");
    expect(result.evaluations[0].feedback).toBeDefined();
    expect(result.completedAt).toBeUndefined();
  });

  it("returns budget_exhausted at max iterations", async () => {
    const deps = makeDeps({
      readSubagentResult: vi.fn().mockResolvedValue({
        status: "ok",
        text: "nothing matches",
      }),
    });
    const contract = makeContract({ budget: { maxIterations: 2 } });
    const entry = makeEntry({
      contract,
      iteration: 1, // iteration 1 is the last (maxIterations=2)
    });

    const result = await handleSubagentCompletion({ entry, deps });

    expect(result.phase).toBe("budget_exhausted");
  });

  it("accumulates evaluations across iterations", async () => {
    const deps = makeDeps({
      readSubagentResult: vi.fn().mockResolvedValue({ status: "ok", text: "hello world" }),
    });
    const priorEval = {
      verdict: "needs_revision" as const,
      auditResults: [],
      overallScore: 0.5,
      summary: "prior",
    };
    const entry = makeEntry({ evaluations: [priorEval], iteration: 1 });

    const result = await handleSubagentCompletion({ entry, deps });

    expect(result.evaluations).toHaveLength(2);
    expect(result.evaluations[0]).toBe(priorEval);
    expect(result.evaluations[1].verdict).toBe("accepted");
  });
});

// ── deliverFeedback ────────────────────────────────────────────────

describe("deliverFeedback", () => {
  it("steers subagent and advances iteration", async () => {
    const deps = makeDeps();
    const entry = makeEntry({
      evaluations: [
        {
          verdict: "needs_revision",
          auditResults: [
            {
              criterionId: "c_0",
              passed: true,
              score: 1,
              evidence: "Has hello.",
            },
            {
              criterionId: "c_1",
              passed: false,
              score: 0,
              evidence: "Missing world.",
              suggestion: "Add world.",
            },
          ],
          overallScore: 0.5,
          summary: "1/2 passed",
          feedback: {
            failedCriteria: [
              {
                id: "c_1",
                description: "Missing world",
                evidence: "Missing world.",
                suggestion: "Add world.",
              },
            ],
            preserveAspects: ["Has hello."],
            revisionPrompt: "Your output was evaluated...",
          },
        },
      ],
    });

    const result = await deliverFeedback({ entry, deps });

    expect(result.phase).toBe("monitoring");
    expect(result.iteration).toBe(1);
    expect(result.runId).toBe("run_2");
    expect(deps.steerSubagent).toHaveBeenCalledWith({
      entry: expect.objectContaining({ runId: "run_1" }),
      message: "Your output was evaluated...",
    });
  });

  it("throws when no feedback available", async () => {
    const deps = makeDeps();
    const entry = makeEntry({ evaluations: [] });

    await expect(deliverFeedback({ entry, deps })).rejects.toThrow("No feedback available");
  });
});

// ── checkBudget ────────────────────────────────────────────────────

describe("checkBudget", () => {
  it("returns not exhausted when within limits", () => {
    const entry = makeEntry({ iteration: 1 });
    const result = checkBudget(entry);
    expect(result.exhausted).toBe(false);
  });

  it("returns exhausted when max iterations reached", () => {
    const entry = makeEntry({ iteration: 3 });
    const result = checkBudget(entry);
    expect(result.exhausted).toBe(true);
    expect(result.reason).toContain("Max iterations");
  });

  it("returns exhausted when max duration exceeded", () => {
    const entry = makeEntry({
      startedAt: Date.now() - 400_000, // 400s > 300s budget
    });
    const result = checkBudget(entry);
    expect(result.exhausted).toBe(true);
    expect(result.reason).toContain("Max duration");
  });
});

// ── runOrchestrationCycle ──────────────────────────────────────────

describe("runOrchestrationCycle", () => {
  it("accepts on first pass when all criteria met", async () => {
    const deps = makeDeps({
      readSubagentResult: vi.fn().mockResolvedValue({ status: "ok", text: "hello world" }),
    });
    const entry = makeEntry();

    const result = await runOrchestrationCycle({ entry, deps });

    expect(result.phase).toBe("accepted");
    expect(result.evaluations).toHaveLength(1);
    expect(deps.steerSubagent).not.toHaveBeenCalled();
  });

  it("iterates with feedback until accepted", async () => {
    let callCount = 0;
    const deps = makeDeps({
      readSubagentResult: vi.fn().mockImplementation(async () => {
        callCount++;
        // First call: partial output, second call: full output
        if (callCount === 1) {
          return { status: "ok", text: "hello only" };
        }
        return { status: "ok", text: "hello world" };
      }),
    });
    const entry = makeEntry();

    const result = await runOrchestrationCycle({ entry, deps });

    expect(result.phase).toBe("accepted");
    expect(result.evaluations).toHaveLength(2);
    expect(result.iteration).toBe(1);
    expect(deps.steerSubagent).toHaveBeenCalledTimes(1);
  });

  it("stops at budget_exhausted after max iterations", async () => {
    const deps = makeDeps({
      readSubagentResult: vi.fn().mockResolvedValue({ status: "ok", text: "hello only" }),
    });
    const contract = makeContract({ budget: { maxIterations: 2, maxDurationMs: 300_000 } });
    const entry = makeEntry({ contract });

    const result = await runOrchestrationCycle({ entry, deps });

    // iteration 0: evaluate → needs_revision → feedback
    // budget check: iteration 1 < 2, proceed
    // iteration 1: evaluate → budget_exhausted (iteration 1 >= maxIterations-1)
    expect(result.phase).toBe("budget_exhausted");
  });
});

// ── isTerminalPhase ────────────────────────────────────────────────

describe("isTerminalPhase", () => {
  it("identifies terminal phases", () => {
    expect(isTerminalPhase("accepted")).toBe(true);
    expect(isTerminalPhase("rejected")).toBe(true);
    expect(isTerminalPhase("budget_exhausted")).toBe(true);
  });

  it("identifies non-terminal phases", () => {
    expect(isTerminalPhase("spawned")).toBe(false);
    expect(isTerminalPhase("monitoring")).toBe(false);
    expect(isTerminalPhase("evaluating")).toBe(false);
    expect(isTerminalPhase("feedback_delivered")).toBe(false);
  });
});
