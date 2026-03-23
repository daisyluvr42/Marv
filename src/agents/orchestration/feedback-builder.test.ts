import { describe, expect, it } from "vitest";
import { buildFeedback } from "./feedback-builder.js";
import type { EvaluationResult } from "./types.js";

describe("buildFeedback", () => {
  it("collects failed criteria into feedback", () => {
    const evaluation: EvaluationResult = {
      verdict: "needs_revision",
      overallScore: 0.5,
      summary: "1/2 passed",
      auditResults: [
        {
          criterionId: "c_0",
          passed: true,
          score: 1,
          evidence: "Tests pass successfully.",
        },
        {
          criterionId: "c_1",
          passed: false,
          score: 0,
          evidence: "Type errors found in output.",
          suggestion: "Fix type errors before submitting.",
        },
      ],
    };

    const feedback = buildFeedback(evaluation);

    expect(feedback.failedCriteria).toHaveLength(1);
    expect(feedback.failedCriteria[0].id).toBe("c_1");
    expect(feedback.failedCriteria[0].suggestion).toBe("Fix type errors before submitting.");
  });

  it("collects passing criteria as preserveAspects", () => {
    const evaluation: EvaluationResult = {
      verdict: "needs_revision",
      overallScore: 0.5,
      summary: "1/2 passed",
      auditResults: [
        {
          criterionId: "c_0",
          passed: true,
          score: 1,
          evidence: "Tests pass successfully.",
        },
        {
          criterionId: "c_1",
          passed: false,
          score: 0,
          evidence: "Missing docs.",
          suggestion: "Add documentation.",
        },
      ],
    };

    const feedback = buildFeedback(evaluation);
    expect(feedback.preserveAspects).toEqual(["Tests pass successfully."]);
  });

  it("builds a well-structured revision prompt", () => {
    const evaluation: EvaluationResult = {
      verdict: "needs_revision",
      overallScore: 0.33,
      summary: "1/3 passed",
      auditResults: [
        { criterionId: "c_0", passed: true, score: 1, evidence: "Good formatting." },
        {
          criterionId: "c_1",
          passed: false,
          score: 0,
          evidence: "No error handling.",
          suggestion: "Add try-catch blocks.",
        },
        {
          criterionId: "c_2",
          passed: false,
          score: 0,
          evidence: "No tests.",
          suggestion: "Write unit tests.",
        },
      ],
    };

    const feedback = buildFeedback(evaluation);
    const prompt = feedback.revisionPrompt;

    expect(prompt).toContain("1/3 criteria passed");
    expect(prompt).toContain("33%");
    expect(prompt).toContain("## Failed criteria");
    expect(prompt).toContain("No error handling.");
    expect(prompt).toContain("Add try-catch blocks.");
    expect(prompt).toContain("No tests.");
    expect(prompt).toContain("## Preserve these aspects");
    expect(prompt).toContain("Good formatting.");
    expect(prompt).toContain("Revise your output");
  });

  it("handles all-fail case", () => {
    const evaluation: EvaluationResult = {
      verdict: "needs_revision",
      overallScore: 0,
      summary: "0/2 passed",
      auditResults: [
        {
          criterionId: "c_0",
          passed: false,
          score: 0,
          evidence: "Fail A.",
          suggestion: "Fix A.",
        },
        {
          criterionId: "c_1",
          passed: false,
          score: 0,
          evidence: "Fail B.",
          suggestion: "Fix B.",
        },
      ],
    };

    const feedback = buildFeedback(evaluation);
    expect(feedback.failedCriteria).toHaveLength(2);
    expect(feedback.preserveAspects).toHaveLength(0);
    expect(feedback.revisionPrompt).not.toContain("## Preserve these aspects");
  });

  it("handles all-pass case (no failed criteria)", () => {
    const evaluation: EvaluationResult = {
      verdict: "accepted",
      overallScore: 1,
      summary: "2/2 passed",
      auditResults: [
        { criterionId: "c_0", passed: true, score: 1, evidence: "Good." },
        { criterionId: "c_1", passed: true, score: 1, evidence: "Also good." },
      ],
    };

    const feedback = buildFeedback(evaluation);
    expect(feedback.failedCriteria).toHaveLength(0);
    expect(feedback.preserveAspects).toHaveLength(2);
    expect(feedback.revisionPrompt).not.toContain("## Failed criteria");
  });

  it("provides default suggestion when criterion has no suggestion", () => {
    const evaluation: EvaluationResult = {
      verdict: "needs_revision",
      overallScore: 0,
      summary: "0/1 passed",
      auditResults: [
        {
          criterionId: "c_0",
          passed: false,
          score: 0,
          evidence: "Something wrong.",
          // no suggestion
        },
      ],
    };

    const feedback = buildFeedback(evaluation);
    expect(feedback.failedCriteria[0].suggestion).toBe("Address this criterion.");
  });
});
