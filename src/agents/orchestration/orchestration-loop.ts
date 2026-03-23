import type { GoalFrame } from "../pi-embedded-runner/goal-loop.js";
import type { SubagentRunRecord } from "../subagent-registry.js";
import type { SpawnSubagentContext, SpawnSubagentParams } from "../subagent-spawn.js";
import { buildContractContextBlock, buildContractFromGoalFrame } from "./contract-builder.js";
import { evaluateContract, type ContractEvaluateOptions } from "./evaluation-gate.js";
import { buildFeedback } from "./feedback-builder.js";
import {
  type EvaluationResult,
  type GoalContract,
  type OrchestrationConfig,
  type OrchestrationEntry,
  type OrchestrationPhase,
} from "./types.js";

// ── Deps (injected for testability) ────────────────────────────────

export type OrchestrationDeps = {
  /** Spawn a subagent. Returns runId + childSessionKey. */
  spawnSubagent: (
    params: SpawnSubagentParams,
    ctx: SpawnSubagentContext,
  ) => Promise<{ status: string; runId?: string; childSessionKey?: string; error?: string }>;

  /** Read the final output text from a completed subagent run. */
  readSubagentResult: (params: {
    runId: string;
    childSessionKey: string;
    waitTimeoutMs: number;
    alreadyEnded?: boolean;
  }) => Promise<{ status: string; text: string; durationMs?: number }>;

  /** Deliver a steer message to a running subagent. Returns new runId. */
  steerSubagent: (params: {
    entry: SubagentRunRecord;
    message: string;
  }) => Promise<{ runId: string }>;
};

// ── Create Entry ───────────────────────────────────────────────────

/**
 * Create a GoalContract from a GoalFrame and start orchestration.
 * Returns the initial OrchestrationEntry in "spawned" phase.
 */
export async function startOrchestration(params: {
  goalFrame: GoalFrame;
  spawnParams: SpawnSubagentParams;
  spawnCtx: SpawnSubagentContext;
  config?: OrchestrationConfig;
  deps: OrchestrationDeps;
}): Promise<OrchestrationEntry> {
  const { goalFrame, spawnParams, spawnCtx, config, deps } = params;
  const contract = buildContractFromGoalFrame({ goalFrame, config });

  return startOrchestrationWithContract({
    contract,
    spawnParams,
    spawnCtx,
    deps,
  });
}

/**
 * Start orchestration with a pre-built GoalContract.
 */
export async function startOrchestrationWithContract(params: {
  contract: GoalContract;
  spawnParams: SpawnSubagentParams;
  spawnCtx: SpawnSubagentContext;
  deps: OrchestrationDeps;
}): Promise<OrchestrationEntry> {
  const { contract, spawnParams, spawnCtx, deps } = params;

  // Inject contract context into the task message
  const contextBlock = buildContractContextBlock(contract);
  const augmentedParams: SpawnSubagentParams = {
    ...spawnParams,
    contextBlock: spawnParams.contextBlock
      ? `${spawnParams.contextBlock}\n\n${contextBlock}`
      : contextBlock,
  };

  const result = await deps.spawnSubagent(augmentedParams, spawnCtx);

  if (result.status !== "accepted" || !result.runId || !result.childSessionKey) {
    throw new Error(`Orchestration spawn failed: ${result.error ?? result.status}`);
  }

  return {
    contractId: contract.id,
    contract,
    runId: result.runId,
    childSessionKey: result.childSessionKey,
    phase: "spawned",
    iteration: 0,
    evaluations: [],
    startedAt: Date.now(),
  };
}

// ── Handle Completion ──────────────────────────────────────────────

/**
 * Called when a subagent run completes (lifecycle.end).
 * Reads output, evaluates against contract, returns updated entry.
 */
export async function handleSubagentCompletion(params: {
  entry: OrchestrationEntry;
  deps: OrchestrationDeps;
  evaluateOpts?: ContractEvaluateOptions;
}): Promise<OrchestrationEntry> {
  const { entry, deps } = params;

  // Read the subagent's final output
  const result = await deps.readSubagentResult({
    runId: entry.runId,
    childSessionKey: entry.childSessionKey,
    waitTimeoutMs: 10_000,
    alreadyEnded: true,
  });

  const output = result.text;

  // Evaluate against contract
  const evaluation = await evaluateContract(entry.contract, output, {
    ...params.evaluateOpts,
    iteration: entry.iteration,
    priorEvaluations: entry.evaluations,
  });

  const updatedEvaluations = [...entry.evaluations, evaluation];

  // Attach feedback if needs_revision
  if (evaluation.verdict === "needs_revision") {
    evaluation.feedback = buildFeedback(evaluation);
  }

  const phase = verdictToPhase(evaluation.verdict);

  return {
    ...entry,
    phase,
    evaluations: updatedEvaluations,
    completedAt: isTerminalPhase(phase) ? Date.now() : undefined,
  };
}

// ── Deliver Feedback ───────────────────────────────────────────────

/**
 * Steer the subagent with structured feedback and advance the iteration.
 * Returns the updated entry in "monitoring" phase with new runId.
 */
export async function deliverFeedback(params: {
  entry: OrchestrationEntry;
  deps: OrchestrationDeps;
}): Promise<OrchestrationEntry> {
  const { entry, deps } = params;
  const lastEvaluation = entry.evaluations[entry.evaluations.length - 1];
  const feedback = lastEvaluation?.feedback;

  if (!feedback) {
    throw new Error("No feedback available to deliver.");
  }

  // Use the steer mechanism to deliver the revision prompt
  const steerResult = await deps.steerSubagent({
    entry: {
      runId: entry.runId,
      childSessionKey: entry.childSessionKey,
    } as SubagentRunRecord,
    message: feedback.revisionPrompt,
  });

  return {
    ...entry,
    runId: steerResult.runId,
    phase: "monitoring",
    iteration: entry.iteration + 1,
  };
}

// ── Budget Check ───────────────────────────────────────────────────

/**
 * Check whether the orchestration has exceeded its budget.
 */
export function checkBudget(entry: OrchestrationEntry): {
  exhausted: boolean;
  reason?: string;
} {
  const { budget } = entry.contract;

  if (entry.iteration >= budget.maxIterations) {
    return {
      exhausted: true,
      reason: `Max iterations reached (${entry.iteration}/${budget.maxIterations}).`,
    };
  }

  if (budget.maxDurationMs) {
    const elapsed = Date.now() - entry.startedAt;
    if (elapsed >= budget.maxDurationMs) {
      return {
        exhausted: true,
        reason: `Max duration reached (${elapsed}ms/${budget.maxDurationMs}ms).`,
      };
    }
  }

  return { exhausted: false };
}

// ── Full Orchestration Cycle ───────────────────────────────────────

/**
 * Run the full orchestration cycle: evaluate → feedback → re-evaluate.
 * This is the main entry point for the goal loop integration.
 *
 * Returns the final entry in a terminal phase (accepted/rejected/budget_exhausted).
 */
export async function runOrchestrationCycle(params: {
  entry: OrchestrationEntry;
  deps: OrchestrationDeps;
  evaluateOpts?: ContractEvaluateOptions;
}): Promise<OrchestrationEntry> {
  let current = params.entry;

  // Evaluate the initial output
  current = await handleSubagentCompletion({
    entry: current,
    deps: params.deps,
    evaluateOpts: params.evaluateOpts,
  });

  // Loop: feedback → re-monitor → re-evaluate until terminal
  while (current.phase === "feedback_delivered") {
    // Check budget before delivering feedback
    const budgetCheck = checkBudget(current);
    if (budgetCheck.exhausted) {
      return {
        ...current,
        phase: "budget_exhausted",
        completedAt: Date.now(),
      };
    }

    // Deliver feedback and wait for next completion
    current = await deliverFeedback({
      entry: current,
      deps: params.deps,
    });

    // Now in "monitoring" phase — wait for the steered run to complete,
    // then evaluate again
    current = await handleSubagentCompletion({
      entry: current,
      deps: params.deps,
      evaluateOpts: params.evaluateOpts,
    });
  }

  return current;
}

// ── Helpers ────────────────────────────────────────────────────────

function verdictToPhase(verdict: EvaluationResult["verdict"]): OrchestrationPhase {
  switch (verdict) {
    case "accepted":
      return "accepted";
    case "needs_revision":
      return "feedback_delivered";
    case "rejected":
      return "rejected";
    case "budget_exhausted":
      return "budget_exhausted";
    case "error":
      return "rejected";
  }
}

export function isTerminalPhase(phase: OrchestrationPhase): boolean {
  return phase === "accepted" || phase === "rejected" || phase === "budget_exhausted";
}
