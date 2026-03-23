import { describe, expect, it, vi } from "vitest";
import type { GoalFrame } from "../pi-embedded-runner/goal-loop.js";
import {
  resolveGroupPhase,
  runGroupOrchestration,
  startOrchestrationGroup,
} from "./orchestration-group.js";
import type { OrchestrationDeps } from "./orchestration-loop.js";
import type { OrchestrationEntry, OrchestrationGroup } from "./types.js";

// ── Test Helpers ───────────────────────────────────────────────────

const baseFrame: GoalFrame = {
  objective: "Build the feature",
  successCriteria: ["tests pass", "code compiles"],
  constraints: [],
  complexity: "moderate",
  goalType: "mutation",
};

function makeDeps(overrides?: Partial<OrchestrationDeps>): OrchestrationDeps {
  let spawnCount = 0;
  return {
    spawnSubagent: vi.fn().mockImplementation(async () => {
      spawnCount++;
      return {
        status: "accepted",
        runId: `run_${spawnCount}`,
        childSessionKey: `agent:test:subagent:child${spawnCount}`,
      };
    }),
    readSubagentResult: vi.fn().mockResolvedValue({
      status: "ok",
      text: "tests pass and code compiles",
    }),
    steerSubagent: vi.fn().mockResolvedValue({ runId: "run_steered" }),
    ...overrides,
  };
}

function makeEntry(
  phase: OrchestrationEntry["phase"],
  overrides?: Partial<OrchestrationEntry>,
): OrchestrationEntry {
  return {
    contractId: "contract_test",
    contract: {
      id: "contract_test",
      objective: "Test",
      successCriteria: ["tests pass"],
      auditStandards: [
        {
          id: "c_0",
          description: "Tests pass",
          evaluator: { kind: "text_match", pattern: "tests pass" },
          required: true,
        },
      ],
      budget: { maxIterations: 3 },
    },
    runId: "run_1",
    childSessionKey: "agent:test:subagent:child1",
    phase,
    iteration: 0,
    evaluations: [],
    startedAt: Date.now(),
    ...overrides,
  };
}

// ── startOrchestrationGroup ────────────────────────────────────────

describe("startOrchestrationGroup", () => {
  it("spawns one entry per role", async () => {
    const deps = makeDeps();

    const group = await startOrchestrationGroup({
      goalFrame: baseFrame,
      roles: [
        { role: "coder", spawnParams: { task: "Write code" } },
        { role: "tester", spawnParams: { task: "Write tests" } },
      ],
      spawnCtx: {},
      deps,
    });

    expect(group.entries).toHaveLength(2);
    expect(group.groupPhase).toBe("running");
    expect(group.allMustPass).toBe(true);
    expect(deps.spawnSubagent).toHaveBeenCalledTimes(2);
  });

  it("tags contract objective with role", async () => {
    const deps = makeDeps();

    const group = await startOrchestrationGroup({
      goalFrame: baseFrame,
      roles: [{ role: "reviewer", spawnParams: { task: "Review" } }],
      spawnCtx: {},
      deps,
    });

    expect(group.entries[0].contract.objective).toContain("[reviewer]");
  });

  it("respects allMustPass flag", async () => {
    const deps = makeDeps();

    const group = await startOrchestrationGroup({
      goalFrame: baseFrame,
      roles: [{ role: "coder", spawnParams: { task: "Code" } }],
      spawnCtx: {},
      deps,
      allMustPass: false,
    });

    expect(group.allMustPass).toBe(false);
  });
});

// ── runGroupOrchestration ──────────────────────────────────────────

describe("runGroupOrchestration", () => {
  it("accepts when all entries pass", async () => {
    const deps = makeDeps({
      readSubagentResult: vi.fn().mockResolvedValue({
        status: "ok",
        text: "tests pass",
      }),
    });

    const group: OrchestrationGroup = {
      groupId: "group_test",
      entries: [
        makeEntry("monitoring"),
        makeEntry("monitoring", {
          runId: "run_2",
          childSessionKey: "agent:test:subagent:child2",
        }),
      ],
      allMustPass: true,
      groupPhase: "running",
    };

    const result = await runGroupOrchestration({ group, deps });

    expect(result.groupPhase).toBe("accepted");
    expect(result.entries.every((e) => e.phase === "accepted")).toBe(true);
  });

  it("stops on first failure in fail-fast mode", async () => {
    let callCount = 0;
    const deps = makeDeps({
      readSubagentResult: vi.fn().mockImplementation(async () => {
        callCount++;
        // First entry fails
        if (callCount === 1) {
          return { status: "ok", text: "nothing matches" };
        }
        return { status: "ok", text: "tests pass" };
      }),
    });

    const group: OrchestrationGroup = {
      groupId: "group_test",
      entries: [
        makeEntry("monitoring", {
          contract: {
            id: "c1",
            objective: "Test",
            successCriteria: ["x"],
            auditStandards: [
              {
                id: "c_0",
                description: "Match x",
                evaluator: { kind: "text_match", pattern: "never_match_xyz" },
                required: true,
              },
            ],
            budget: { maxIterations: 1 },
          },
        }),
        makeEntry("monitoring", {
          runId: "run_2",
          childSessionKey: "agent:test:subagent:child2",
        }),
      ],
      allMustPass: true,
      groupPhase: "running",
    };

    const result = await runGroupOrchestration({ group, deps });

    expect(result.groupPhase).toBe("budget_exhausted");
    // Second entry should not have been evaluated
    expect(result.entries[1].phase).toBe("monitoring");
  });
});

// ── resolveGroupPhase ──────────────────────────────────────────────

describe("resolveGroupPhase", () => {
  it("returns running when not all entries are terminal", () => {
    const entries = [makeEntry("accepted"), makeEntry("monitoring")];
    expect(resolveGroupPhase(entries, true)).toBe("running");
  });

  it("returns accepted when all entries accepted", () => {
    const entries = [makeEntry("accepted"), makeEntry("accepted")];
    expect(resolveGroupPhase(entries, true)).toBe("accepted");
  });

  it("returns rejected when any entry rejected (allMustPass)", () => {
    const entries = [makeEntry("accepted"), makeEntry("rejected")];
    expect(resolveGroupPhase(entries, true)).toBe("rejected");
  });

  it("returns budget_exhausted when any entry exhausted (allMustPass)", () => {
    const entries = [makeEntry("accepted"), makeEntry("budget_exhausted")];
    expect(resolveGroupPhase(entries, true)).toBe("budget_exhausted");
  });

  it("returns accepted in best-effort when at least one accepted", () => {
    const entries = [makeEntry("accepted"), makeEntry("rejected")];
    expect(resolveGroupPhase(entries, false)).toBe("accepted");
  });

  it("returns rejected in best-effort when none accepted", () => {
    const entries = [makeEntry("rejected"), makeEntry("budget_exhausted")];
    expect(resolveGroupPhase(entries, false)).toBe("rejected");
  });
});
