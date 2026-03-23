import { describe, expect, it } from "vitest";
import type { GoalFrame } from "../pi-embedded-runner/goal-loop.js";
import { buildContractContextBlock, buildContractFromGoalFrame } from "./contract-builder.js";

describe("buildContractFromGoalFrame", () => {
  const baseFrame: GoalFrame = {
    objective: "Fix the login bug",
    successCriteria: ["tests pass", "no type errors"],
    constraints: ["do not modify the auth module"],
    complexity: "moderate",
    goalType: "mutation",
  };

  it("generates a contract with correct objective and criteria", () => {
    const contract = buildContractFromGoalFrame({ goalFrame: baseFrame });

    expect(contract.id).toMatch(/^contract_/);
    expect(contract.objective).toBe("Fix the login bug");
    expect(contract.successCriteria).toEqual(["tests pass", "no type errors"]);
    expect(contract.constraints).toEqual(["do not modify the auth module"]);
  });

  it("creates one AuditCriterion per successCriteria entry", () => {
    const contract = buildContractFromGoalFrame({ goalFrame: baseFrame });

    expect(contract.auditStandards).toHaveLength(2);
    expect(contract.auditStandards[0].id).toBe("c_0");
    expect(contract.auditStandards[0].description).toBe("tests pass");
    expect(contract.auditStandards[0].required).toBe(true);
    expect(contract.auditStandards[1].id).toBe("c_1");
    expect(contract.auditStandards[1].description).toBe("no type errors");
  });

  it("uses checklist evaluator by default for non-complex goals", () => {
    const contract = buildContractFromGoalFrame({ goalFrame: baseFrame });

    for (const criterion of contract.auditStandards) {
      expect(criterion.evaluator.kind).toBe("checklist");
      if (criterion.evaluator.kind === "checklist") {
        expect(criterion.evaluator.items).toHaveLength(1);
      }
    }
  });

  it("uses llm_judge evaluator for complex goals", () => {
    const complexFrame: GoalFrame = {
      ...baseFrame,
      complexity: "complex",
    };
    const contract = buildContractFromGoalFrame({ goalFrame: complexFrame });

    for (const criterion of contract.auditStandards) {
      expect(criterion.evaluator.kind).toBe("llm_judge");
      if (criterion.evaluator.kind === "llm_judge") {
        expect(criterion.evaluator.threshold).toBe(7);
        expect(criterion.evaluator.prompt).toContain(criterion.description);
      }
    }
  });

  it("respects config overrides for defaultAuditEvaluator", () => {
    const contract = buildContractFromGoalFrame({
      goalFrame: baseFrame,
      config: { defaultAuditEvaluator: "llm_judge" },
    });

    for (const criterion of contract.auditStandards) {
      expect(criterion.evaluator.kind).toBe("llm_judge");
    }
  });

  it("complex goals always use llm_judge even when config says checklist", () => {
    const complexFrame: GoalFrame = { ...baseFrame, complexity: "complex" };
    const contract = buildContractFromGoalFrame({
      goalFrame: complexFrame,
      config: { defaultAuditEvaluator: "checklist" },
    });

    for (const criterion of contract.auditStandards) {
      expect(criterion.evaluator.kind).toBe("llm_judge");
    }
  });

  it("uses config budget values", () => {
    const contract = buildContractFromGoalFrame({
      goalFrame: baseFrame,
      config: { maxIterations: 5, maxDurationMs: 600_000 },
    });

    expect(contract.budget.maxIterations).toBe(5);
    expect(contract.budget.maxDurationMs).toBe(600_000);
  });

  it("uses default budget when no config", () => {
    const contract = buildContractFromGoalFrame({ goalFrame: baseFrame });

    expect(contract.budget.maxIterations).toBe(3);
    expect(contract.budget.maxDurationMs).toBe(300_000);
  });

  it("omits constraints when goalFrame has empty constraints", () => {
    const frame: GoalFrame = { ...baseFrame, constraints: [] };
    const contract = buildContractFromGoalFrame({ goalFrame: frame });

    expect(contract.constraints).toBeUndefined();
  });

  it("handles empty successCriteria", () => {
    const frame: GoalFrame = { ...baseFrame, successCriteria: [] };
    const contract = buildContractFromGoalFrame({ goalFrame: frame });

    expect(contract.auditStandards).toHaveLength(0);
  });
});

describe("buildContractContextBlock", () => {
  it("produces a well-structured context block", () => {
    const frame: GoalFrame = {
      objective: "Refactor the API",
      successCriteria: ["tests pass", "type-check clean"],
      constraints: ["no breaking changes"],
      complexity: "moderate",
      goalType: "mutation",
    };
    const contract = buildContractFromGoalFrame({ goalFrame: frame });
    const block = buildContractContextBlock(contract);

    expect(block).toContain("[Audit Standards]");
    expect(block).toContain("Objective: Refactor the API");
    expect(block).toContain("- tests pass");
    expect(block).toContain("- type-check clean");
    expect(block).toContain("- no breaking changes");
    expect(block).toContain("evaluated against these criteria");
  });

  it("omits constraints section when none present", () => {
    const frame: GoalFrame = {
      objective: "Research topic",
      successCriteria: ["covers all points"],
      constraints: [],
      complexity: "trivial",
      goalType: "inquiry",
    };
    const contract = buildContractFromGoalFrame({ goalFrame: frame });
    const block = buildContractContextBlock(contract);

    expect(block).not.toContain("Constraints:");
  });
});
