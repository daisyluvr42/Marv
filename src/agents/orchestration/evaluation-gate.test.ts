import { describe, expect, it } from "vitest";
import { evaluateAuditCriterion, evaluateContract } from "./evaluation-gate.js";
import type { AuditCriterion, GoalContract } from "./types.js";

// ── text_match evaluator ───────────────────────────────────────────

describe("evaluateAuditCriterion — text_match", () => {
  const criterion: AuditCriterion = {
    id: "tm_0",
    description: "Output mentions success",
    evaluator: { kind: "text_match", pattern: "success", flags: "i" },
    required: true,
  };

  it("passes when pattern matches", async () => {
    const result = await evaluateAuditCriterion(criterion, "Tests ran with Success!");
    expect(result.passed).toBe(true);
    expect(result.score).toBe(1);
    expect(result.suggestion).toBeUndefined();
  });

  it("fails when pattern does not match", async () => {
    const result = await evaluateAuditCriterion(criterion, "Tests failed.");
    expect(result.passed).toBe(false);
    expect(result.score).toBe(0);
    expect(result.suggestion).toBeDefined();
  });

  it("supports regex patterns", async () => {
    const regexCriterion: AuditCriterion = {
      id: "tm_1",
      description: "Has version number",
      evaluator: { kind: "text_match", pattern: "v\\d+\\.\\d+\\.\\d+" },
      required: true,
    };
    const pass = await evaluateAuditCriterion(regexCriterion, "Released v1.2.3 today");
    expect(pass.passed).toBe(true);

    const fail = await evaluateAuditCriterion(regexCriterion, "No version here");
    expect(fail.passed).toBe(false);
  });
});

// ── checklist evaluator ────────────────────────────────────────────

describe("evaluateAuditCriterion — checklist", () => {
  const criterion: AuditCriterion = {
    id: "cl_0",
    description: "Addresses all requirements",
    evaluator: {
      kind: "checklist",
      items: ["error handling", "input validation", "logging"],
    },
    required: true,
  };

  it("passes when all items found (case-insensitive)", async () => {
    const output = "Added Error Handling, Input Validation, and Logging to the module.";
    const result = await evaluateAuditCriterion(criterion, output);
    expect(result.passed).toBe(true);
    expect(result.score).toBe(1);
  });

  it("partially passes with correct score", async () => {
    const output = "Added error handling and logging.";
    const result = await evaluateAuditCriterion(criterion, output);
    expect(result.passed).toBe(false);
    expect(result.score).toBeCloseTo(2 / 3, 2);
    expect(result.evidence).toContain("input validation");
  });

  it("fails when no items found", async () => {
    const result = await evaluateAuditCriterion(criterion, "Did nothing.");
    expect(result.passed).toBe(false);
    expect(result.score).toBe(0);
  });

  it("passes with empty items list", async () => {
    const emptyCriterion: AuditCriterion = {
      id: "cl_1",
      description: "No items",
      evaluator: { kind: "checklist", items: [] },
      required: true,
    };
    const result = await evaluateAuditCriterion(emptyCriterion, "anything");
    expect(result.passed).toBe(true);
    expect(result.score).toBe(1);
  });
});

// ── evaluateContract ───────────────────────────────────────────────

describe("evaluateContract", () => {
  function makeContract(criteria: AuditCriterion[]): GoalContract {
    return {
      id: "contract_test",
      objective: "Test objective",
      successCriteria: criteria.map((c) => c.description),
      auditStandards: criteria,
      budget: { maxIterations: 3 },
    };
  }

  it("returns accepted when all required criteria pass", async () => {
    const contract = makeContract([
      {
        id: "c_0",
        description: "Has greeting",
        evaluator: { kind: "text_match", pattern: "hello" },
        required: true,
      },
      {
        id: "c_1",
        description: "Has farewell",
        evaluator: { kind: "text_match", pattern: "goodbye" },
        required: true,
      },
    ]);

    const result = await evaluateContract(contract, "hello and goodbye");
    expect(result.verdict).toBe("accepted");
    expect(result.overallScore).toBe(1);
  });

  it("returns needs_revision when required criteria fail with budget remaining", async () => {
    const contract = makeContract([
      {
        id: "c_0",
        description: "Has greeting",
        evaluator: { kind: "text_match", pattern: "hello" },
        required: true,
      },
      {
        id: "c_1",
        description: "Has farewell",
        evaluator: { kind: "text_match", pattern: "goodbye" },
        required: true,
      },
    ]);

    const result = await evaluateContract(contract, "hello but no farewell", {
      iteration: 0,
    });
    expect(result.verdict).toBe("needs_revision");
    expect(result.overallScore).toBe(0.5);
  });

  it("returns budget_exhausted at max iterations", async () => {
    const contract = makeContract([
      {
        id: "c_0",
        description: "Never passes",
        evaluator: { kind: "text_match", pattern: "impossible_string_xyz" },
        required: true,
      },
    ]);

    const result = await evaluateContract(contract, "some output", {
      iteration: 2, // maxIterations is 3, so iteration 2 is the last
    });
    expect(result.verdict).toBe("budget_exhausted");
  });

  it("returns rejected when score below threshold", async () => {
    const contract = makeContract([
      {
        id: "c_0",
        description: "Criterion 1",
        evaluator: { kind: "text_match", pattern: "abc" },
        required: true,
      },
      {
        id: "c_1",
        description: "Criterion 2",
        evaluator: { kind: "text_match", pattern: "def" },
        required: true,
      },
      {
        id: "c_2",
        description: "Criterion 3",
        evaluator: { kind: "text_match", pattern: "ghi" },
        required: true,
      },
      {
        id: "c_3",
        description: "Criterion 4",
        evaluator: { kind: "text_match", pattern: "jkl" },
        required: true,
      },
    ]);

    // 0/4 criteria pass → score 0, below rejectionThreshold 0.3
    const result = await evaluateContract(contract, "nothing matches", {
      iteration: 0,
      rejectionThreshold: 0.3,
    });
    expect(result.verdict).toBe("rejected");
  });

  it("returns rejected when score is declining", async () => {
    const contract = makeContract([
      {
        id: "c_0",
        description: "Match",
        evaluator: { kind: "text_match", pattern: "xyz" },
        required: true,
      },
    ]);

    const priorEvaluations = [
      { verdict: "needs_revision" as const, auditResults: [], overallScore: 0.6, summary: "" },
      { verdict: "needs_revision" as const, auditResults: [], overallScore: 0.5, summary: "" },
    ];

    const result = await evaluateContract(contract, "no match", {
      iteration: 0,
      priorEvaluations,
      rejectionThreshold: 0, // don't reject on absolute score
      minImprovementRatio: 0.05,
    });
    // Score is 0, prior scores were 0.6 and 0.5 — declining trend
    expect(result.verdict).toBe("rejected");
  });

  it("respects weights in score computation", async () => {
    const contract = makeContract([
      {
        id: "c_0",
        description: "Important",
        evaluator: { kind: "text_match", pattern: "found" },
        required: true,
        weight: 3,
      },
      {
        id: "c_1",
        description: "Minor",
        evaluator: { kind: "text_match", pattern: "missing" },
        required: false,
        weight: 1,
      },
    ]);

    // c_0 passes (weight 3, score 1), c_1 fails (weight 1, score 0)
    // Weighted: (3*1 + 1*0) / (3+1) = 0.75
    const result = await evaluateContract(contract, "found it");
    expect(result.overallScore).toBe(0.75);
    // All required passed → accepted
    expect(result.verdict).toBe("accepted");
  });

  it("non-required criteria don't block acceptance", async () => {
    const contract = makeContract([
      {
        id: "c_0",
        description: "Required",
        evaluator: { kind: "text_match", pattern: "must_have" },
        required: true,
      },
      {
        id: "c_1",
        description: "Nice to have",
        evaluator: { kind: "text_match", pattern: "optional_thing" },
        required: false,
      },
    ]);

    const result = await evaluateContract(contract, "must_have is here");
    expect(result.verdict).toBe("accepted");
    expect(result.auditResults[1].passed).toBe(false);
  });

  it("summary includes pass count and verdict", async () => {
    const contract = makeContract([
      {
        id: "c_0",
        description: "Check",
        evaluator: { kind: "text_match", pattern: "yes" },
        required: true,
      },
    ]);

    const result = await evaluateContract(contract, "yes");
    expect(result.summary).toContain("1/1");
    expect(result.summary).toContain("accepted");
  });
});
