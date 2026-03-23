import type { AuditResult, EvaluationResult, StructuredFeedback } from "./types.js";

/**
 * Build structured feedback from an evaluation result.
 * Only meaningful when verdict is "needs_revision".
 */
export function buildFeedback(evaluation: EvaluationResult): StructuredFeedback {
  const failed = evaluation.auditResults.filter((r) => !r.passed);
  const passed = evaluation.auditResults.filter((r) => r.passed);

  const failedCriteria = failed.map((r) => ({
    id: r.criterionId,
    description: r.evidence,
    evidence: r.evidence,
    suggestion: r.suggestion ?? "Address this criterion.",
  }));

  const preserveAspects = passed.map((r) => r.evidence);

  const revisionPrompt = buildRevisionPrompt({
    passedCount: passed.length,
    totalCount: evaluation.auditResults.length,
    score: evaluation.overallScore,
    failed,
    preserveAspects,
  });

  return { failedCriteria, preserveAspects, revisionPrompt };
}

// ── Revision Prompt Construction ───────────────────────────────────

function buildRevisionPrompt(params: {
  passedCount: number;
  totalCount: number;
  score: number;
  failed: AuditResult[];
  preserveAspects: string[];
}): string {
  const lines: string[] = [];

  lines.push(
    `Your output was evaluated: ${params.passedCount}/${params.totalCount} criteria passed (score: ${(params.score * 100).toFixed(0)}%).`,
  );
  lines.push("");

  if (params.failed.length > 0) {
    lines.push("## Failed criteria");
    for (const f of params.failed) {
      const suggestion = f.suggestion ? ` Fix: ${f.suggestion}` : "";
      lines.push(`- ${f.evidence}${suggestion}`);
    }
    lines.push("");
  }

  if (params.preserveAspects.length > 0) {
    lines.push("## Preserve these aspects");
    for (const aspect of params.preserveAspects) {
      lines.push(`- ${aspect}`);
    }
    lines.push("");
  }

  lines.push("Revise your output to address the failed criteria while keeping what worked.");

  return lines.join("\n");
}
