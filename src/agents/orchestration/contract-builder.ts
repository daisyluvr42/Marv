import crypto from "node:crypto";
import type { GoalFrame } from "../pi-embedded-runner/goal-loop.js";
import {
  type AuditCriterion,
  type GoalContract,
  type OrchestrationConfig,
  ORCHESTRATION_DEFAULTS,
} from "./types.js";

/**
 * Build a GoalContract from an existing GoalFrame.
 *
 * Each GoalFrame.successCriteria entry becomes a checklist AuditCriterion.
 * For "complex" goals, criteria use llm_judge for stricter evaluation.
 */
export function buildContractFromGoalFrame(params: {
  goalFrame: GoalFrame;
  config?: OrchestrationConfig;
}): GoalContract {
  const { goalFrame, config } = params;
  const cfg = { ...ORCHESTRATION_DEFAULTS, ...config };

  const evaluatorKind = resolveEvaluatorKind(goalFrame.complexity, cfg.defaultAuditEvaluator);
  const auditStandards = goalFrame.successCriteria.map((criterion, idx) =>
    buildCriterionFromText(criterion, idx, evaluatorKind),
  );

  return {
    id: `contract_${crypto.randomUUID().slice(0, 12)}`,
    objective: goalFrame.objective,
    successCriteria: goalFrame.successCriteria,
    auditStandards,
    constraints: goalFrame.constraints.length > 0 ? goalFrame.constraints : undefined,
    budget: {
      maxIterations: cfg.maxIterations,
      maxDurationMs: cfg.maxDurationMs,
    },
  };
}

/**
 * Build the context block that gets injected into the subagent's task message.
 * This tells the subagent what it will be evaluated against.
 */
export function buildContractContextBlock(contract: GoalContract): string {
  const lines: string[] = ["[Audit Standards]"];

  lines.push(`Objective: ${contract.objective}`);

  if (contract.successCriteria.length > 0) {
    lines.push("Success Criteria:");
    for (const c of contract.successCriteria) {
      lines.push(`  - ${c}`);
    }
  }

  if (contract.constraints && contract.constraints.length > 0) {
    lines.push("Constraints:");
    for (const c of contract.constraints) {
      lines.push(`  - ${c}`);
    }
  }

  lines.push("");
  lines.push("Your output will be evaluated against these criteria. Address each one.");

  return lines.join("\n");
}

// ── Helpers ────────────────────────────────────────────────────────

function resolveEvaluatorKind(
  complexity: GoalFrame["complexity"],
  defaultKind: "checklist" | "llm_judge",
): "checklist" | "llm_judge" {
  // Complex goals get stricter evaluation via LLM judge
  if (complexity === "complex") {
    return "llm_judge";
  }
  return defaultKind;
}

function buildCriterionFromText(
  text: string,
  index: number,
  evaluatorKind: "checklist" | "llm_judge",
): AuditCriterion {
  const id = `c_${index}`;

  if (evaluatorKind === "llm_judge") {
    return {
      id,
      description: text,
      evaluator: {
        kind: "llm_judge",
        prompt: `Evaluate whether the following output satisfies this criterion: "${text}". Score from 1 (not at all) to 10 (fully satisfied). Output only the number.`,
        threshold: 7,
      },
      required: true,
    };
  }

  return {
    id,
    description: text,
    evaluator: {
      kind: "checklist",
      items: [text],
    },
    required: true,
  };
}
