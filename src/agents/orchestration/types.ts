import type { EvaluatorSpec } from "../../experiments/types.js";

// ── Audit Evaluator Types ──────────────────────────────────────────

/**
 * How to evaluate a single audit criterion against subagent output.
 *
 * - text_match: regex against output text
 * - command: shell command that produces a metric (reuses EvaluatorSpec)
 * - llm_judge: LLM scores the output on a 1-10 scale
 * - checklist: all items must be addressed in the output
 */
export type AuditEvaluator =
  | { kind: "text_match"; pattern: string; flags?: string }
  | { kind: "command"; spec: EvaluatorSpec }
  | { kind: "llm_judge"; prompt: string; threshold: number }
  | { kind: "checklist"; items: string[] };

// ── Audit Criterion ────────────────────────────────────────────────

/** A single declarative acceptance criterion. */
export type AuditCriterion = {
  /** Unique identifier within the contract (e.g. "c_0"). */
  id: string;
  /** Human-readable description of what must be true. */
  description: string;
  /** Machine-evaluable check. */
  evaluator: AuditEvaluator;
  /** If true, this criterion must pass for the contract to be accepted. */
  required: boolean;
  /** Relative importance for weighted scoring (default 1). */
  weight?: number;
};

// ── Goal Contract ──────────────────────────────────────────────────

/**
 * A declarative specification of what a subagent must deliver.
 * Auto-generated from GoalFrame or manually specified.
 */
export type GoalContract = {
  /** Unique contract identifier (contract_<uuid>). */
  id: string;
  /** What the subagent should accomplish (from GoalFrame.objective). */
  objective: string;
  /** High-level success criteria (from GoalFrame.successCriteria). */
  successCriteria: string[];
  /** Machine-evaluable audit standards. */
  auditStandards: AuditCriterion[];
  /** Constraints the subagent must respect (from GoalFrame.constraints). */
  constraints?: string[];
  /** Resource limits for the orchestration loop. */
  budget: GoalContractBudget;
};

export type GoalContractBudget = {
  /** Max feedback iterations before giving up (default 3). */
  maxIterations: number;
  /** Max wall-clock time in ms for the full orchestration loop (default 300_000). */
  maxDurationMs?: number;
};

// ── Evaluation Results ─────────────────────────────────────────────

/** Result of evaluating a single audit criterion. */
export type AuditResult = {
  criterionId: string;
  passed: boolean;
  /** Normalized score 0-1 (if the evaluator produces a numeric result). */
  score?: number;
  /** What was observed during evaluation. */
  evidence: string;
  /** How to fix this (populated for failed criteria). */
  suggestion?: string;
};

export type EvaluationVerdict =
  | "accepted"
  | "needs_revision"
  | "rejected"
  | "budget_exhausted"
  | "error";

/** Aggregate result of evaluating all criteria in a contract. */
export type EvaluationResult = {
  verdict: EvaluationVerdict;
  auditResults: AuditResult[];
  /** Weighted aggregate score 0-1 across all criteria. */
  overallScore: number;
  /** Human-readable summary of the evaluation. */
  summary: string;
  /** Populated when verdict is "needs_revision". */
  feedback?: StructuredFeedback;
};

// ── Structured Feedback ────────────────────────────────────────────

/** Typed feedback delivered to a subagent for revision. */
export type StructuredFeedback = {
  failedCriteria: Array<{
    id: string;
    description: string;
    evidence: string;
    suggestion: string;
  }>;
  /** Aspects of the output that were good and should be preserved. */
  preserveAspects: string[];
  /** Synthesized steer message combining all feedback. */
  revisionPrompt: string;
};

// ── Orchestration State Machine ────────────────────────────────────

export type OrchestrationPhase =
  | "spawned"
  | "monitoring"
  | "evaluating"
  | "feedback_delivered"
  | "accepted"
  | "rejected"
  | "budget_exhausted";

/** Tracks a single subagent through the orchestration loop. */
export type OrchestrationEntry = {
  contractId: string;
  contract: GoalContract;
  runId: string;
  childSessionKey: string;
  phase: OrchestrationPhase;
  /** Current feedback iteration (0 = first attempt, 1 = first revision, ...). */
  iteration: number;
  /** History of evaluations across iterations. */
  evaluations: EvaluationResult[];
  startedAt: number;
  completedAt?: number;
};

// ── Multi-Subagent Group ───────────────────────────────────────────

/** Coordinates multiple orchestration entries for multi-role delegations. */
export type OrchestrationGroup = {
  groupId: string;
  entries: OrchestrationEntry[];
  /** If true, all entries must reach "accepted" for the group to succeed. */
  allMustPass: boolean;
  groupPhase: "running" | "accepted" | "rejected" | "budget_exhausted";
};

// ── Configuration ──────────────────────────────────────────────────

/** Orchestration config (lives under agents.defaults.subagents.orchestration). */
export type OrchestrationConfig = {
  enabled?: boolean;
  maxIterations?: number;
  maxDurationMs?: number;
  evaluationTimeoutSeconds?: number;
  defaultAuditEvaluator?: "checklist" | "llm_judge";
  rejectionThreshold?: number;
  minImprovementRatio?: number;
  groupMode?: "fail_fast" | "best_effort";
};

// ── Defaults ───────────────────────────────────────────────────────

export const ORCHESTRATION_DEFAULTS = {
  enabled: true,
  maxIterations: 3,
  maxDurationMs: 300_000,
  evaluationTimeoutSeconds: 60,
  defaultAuditEvaluator: "checklist" as const,
  rejectionThreshold: 0.3,
  minImprovementRatio: 0.05,
  groupMode: "best_effort" as const,
} satisfies Required<OrchestrationConfig>;
