import { runEvaluator } from "../../experiments/evaluator.js";
import type {
  AuditCriterion,
  AuditEvaluator,
  AuditResult,
  EvaluationResult,
  EvaluationVerdict,
  GoalContract,
} from "./types.js";

// ── Single Criterion Evaluation ────────────────────────────────────

export type EvaluateOptions = {
  cwd?: string;
  timeoutSeconds?: number;
};

/**
 * Evaluate a single audit criterion against subagent output.
 * Pure function — no side effects beyond command execution for "command" evaluators.
 */
export async function evaluateAuditCriterion(
  criterion: AuditCriterion,
  output: string,
  opts?: EvaluateOptions,
): Promise<AuditResult> {
  try {
    return await evaluateByKind(criterion.evaluator, criterion.id, output, opts);
  } catch (err) {
    return {
      criterionId: criterion.id,
      passed: false,
      score: 0,
      evidence: `Evaluation error: ${err instanceof Error ? err.message : String(err)}`,
      suggestion: "Fix the evaluator configuration and retry.",
    };
  }
}

async function evaluateByKind(
  evaluator: AuditEvaluator,
  criterionId: string,
  output: string,
  opts?: EvaluateOptions,
): Promise<AuditResult> {
  switch (evaluator.kind) {
    case "text_match":
      return evaluateTextMatch(evaluator, criterionId, output);
    case "command":
      return evaluateCommand(evaluator, criterionId, opts);
    case "llm_judge":
      return evaluateLlmJudge(evaluator, criterionId, output, opts);
    case "checklist":
      return evaluateChecklist(evaluator, criterionId, output);
  }
}

function evaluateTextMatch(
  evaluator: { kind: "text_match"; pattern: string; flags?: string },
  criterionId: string,
  output: string,
): AuditResult {
  const regex = new RegExp(evaluator.pattern, evaluator.flags);
  const matched = regex.test(output);
  return {
    criterionId,
    passed: matched,
    score: matched ? 1 : 0,
    evidence: matched
      ? `Pattern /${evaluator.pattern}/ matched in output.`
      : `Pattern /${evaluator.pattern}/ not found in output.`,
    suggestion: matched ? undefined : `Output should match pattern: ${evaluator.pattern}`,
  };
}

async function evaluateCommand(
  evaluator: { kind: "command"; spec: import("../../experiments/types.js").EvaluatorSpec },
  criterionId: string,
  opts?: EvaluateOptions,
): Promise<AuditResult> {
  const result = await runEvaluator(evaluator.spec, {
    cwd: opts?.cwd,
  });

  if (result.error) {
    return {
      criterionId,
      passed: false,
      score: 0,
      evidence: `Command evaluator error: ${result.error}`,
      suggestion: "Check the measure command and retry.",
    };
  }

  const threshold = evaluator.spec.threshold ?? 0;
  const direction = evaluator.spec.direction;
  const passed =
    direction === "higher_is_better" ? result.value >= threshold : result.value <= threshold;

  // Normalize score: 0-1 based on how close to threshold
  const score = normalizeCommandScore(result.value, threshold, direction);

  return {
    criterionId,
    passed,
    score,
    evidence: `Metric: ${result.value} (threshold: ${threshold}, direction: ${direction}).`,
    suggestion: passed
      ? undefined
      : `Metric ${result.value} does not meet ${direction === "higher_is_better" ? ">=" : "<="} ${threshold}.`,
  };
}

async function evaluateLlmJudge(
  evaluator: { kind: "llm_judge"; prompt: string; threshold: number },
  criterionId: string,
  output: string,
  opts?: EvaluateOptions,
): Promise<AuditResult> {
  // Build an EvaluatorSpec that pipes output + prompt to an LLM via stdin
  const result = await runEvaluator(
    {
      id: `judge_${criterionId}`,
      name: `LLM Judge for ${criterionId}`,
      measureCommand: "cat", // placeholder — in practice the LLM CLI
      metricParser: "first_number",
      direction: "higher_is_better",
      threshold: evaluator.threshold,
      timeoutSeconds: opts?.timeoutSeconds ?? 60,
      judgePrompt: evaluator.prompt,
      judgeFile: undefined, // output is passed via stdin construction
    },
    { cwd: opts?.cwd },
  );

  if (result.error) {
    return {
      criterionId,
      passed: false,
      score: 0,
      evidence: `LLM judge error: ${result.error}`,
      suggestion: "Check LLM judge configuration.",
    };
  }

  const normalizedScore = Math.max(0, Math.min(1, result.value / 10));
  const passed = result.value >= evaluator.threshold;

  return {
    criterionId,
    passed,
    score: normalizedScore,
    evidence: `LLM judge score: ${result.value}/10 (threshold: ${evaluator.threshold}).`,
    suggestion: passed
      ? undefined
      : `Score ${result.value} below threshold ${evaluator.threshold}.`,
  };
}

function evaluateChecklist(
  evaluator: { kind: "checklist"; items: string[] },
  criterionId: string,
  output: string,
): AuditResult {
  const lowerOutput = output.toLowerCase();
  const results = evaluator.items.map((item) => ({
    item,
    found: lowerOutput.includes(item.toLowerCase()),
  }));

  const passedCount = results.filter((r) => r.found).length;
  const totalCount = results.length;
  const allPassed = passedCount === totalCount;

  const missing = results.filter((r) => !r.found).map((r) => r.item);

  return {
    criterionId,
    passed: allPassed,
    score: totalCount > 0 ? passedCount / totalCount : 1,
    evidence: allPassed
      ? `All ${totalCount} checklist items addressed.`
      : `${passedCount}/${totalCount} items addressed. Missing: ${missing.join(", ")}.`,
    suggestion: allPassed ? undefined : `Address these items: ${missing.join("; ")}.`,
  };
}

// ── Contract-Level Evaluation ──────────────────────────────────────

export type ContractEvaluateOptions = EvaluateOptions & {
  /** Previous evaluation results for trend detection. */
  priorEvaluations?: EvaluationResult[];
  /** Current iteration (0-indexed). */
  iteration?: number;
  /** Config overrides. */
  rejectionThreshold?: number;
  minImprovementRatio?: number;
};

/**
 * Evaluate all criteria in a contract and produce an aggregate verdict.
 */
export async function evaluateContract(
  contract: GoalContract,
  output: string,
  opts?: ContractEvaluateOptions,
): Promise<EvaluationResult> {
  const auditResults = await Promise.all(
    contract.auditStandards.map((criterion) => evaluateAuditCriterion(criterion, output, opts)),
  );

  const overallScore = computeWeightedScore(contract.auditStandards, auditResults);
  const verdict = resolveVerdict(contract, auditResults, overallScore, opts);
  const summary = buildSummary(auditResults, overallScore, verdict);

  return {
    verdict,
    auditResults,
    overallScore,
    summary,
  };
}

// ── Verdict Resolution ─────────────────────────────────────────────

function resolveVerdict(
  contract: GoalContract,
  results: AuditResult[],
  score: number,
  opts?: ContractEvaluateOptions,
): EvaluationVerdict {
  const requiredCriteria = contract.auditStandards.filter((c) => c.required);
  const allRequiredPassed = requiredCriteria.every((criterion) => {
    const result = results.find((r) => r.criterionId === criterion.id);
    return result?.passed === true;
  });

  if (allRequiredPassed) {
    return "accepted";
  }

  const iteration = opts?.iteration ?? 0;
  const maxIterations = contract.budget.maxIterations;
  if (iteration >= maxIterations - 1) {
    return "budget_exhausted";
  }

  const rejectionThreshold = opts?.rejectionThreshold ?? 0.3;
  if (score < rejectionThreshold) {
    return "rejected";
  }

  // Check for declining trend (not converging)
  if (opts?.priorEvaluations && opts.priorEvaluations.length >= 2) {
    const recentScores = opts.priorEvaluations.slice(-2).map((e) => e.overallScore);
    const allDeclining = recentScores.every((s) => s >= score);
    if (allDeclining && recentScores.length >= 2) {
      const minImprovement = opts?.minImprovementRatio ?? 0.05;
      const bestPrior = Math.max(...recentScores);
      if (score <= bestPrior + minImprovement) {
        return "rejected";
      }
    }
  }

  return "needs_revision";
}

// ── Scoring ────────────────────────────────────────────────────────

function computeWeightedScore(criteria: AuditCriterion[], results: AuditResult[]): number {
  let totalWeight = 0;
  let weightedSum = 0;

  for (const criterion of criteria) {
    const weight = criterion.weight ?? 1;
    const result = results.find((r) => r.criterionId === criterion.id);
    const score = result?.score ?? 0;
    weightedSum += score * weight;
    totalWeight += weight;
  }

  return totalWeight > 0 ? weightedSum / totalWeight : 0;
}

function normalizeCommandScore(value: number, threshold: number, direction: string): number {
  if (!Number.isFinite(value) || !Number.isFinite(threshold) || threshold === 0) {
    return 0;
  }
  if (direction === "higher_is_better") {
    return Math.max(0, Math.min(1, value / threshold));
  }
  // lower_is_better: score is higher when value is lower
  return Math.max(0, Math.min(1, threshold / value));
}

function buildSummary(results: AuditResult[], score: number, verdict: EvaluationVerdict): string {
  const passed = results.filter((r) => r.passed).length;
  const total = results.length;
  return `${passed}/${total} criteria passed (score: ${(score * 100).toFixed(0)}%). Verdict: ${verdict}.`;
}
