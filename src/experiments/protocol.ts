import { persistExperimentMemory } from "../agents/pi-embedded-runner/goal-loop-memory.js";
import { getChildLogger } from "../logging.js";
import { resolveCheckpointStrategy } from "./checkpoint.js";
import { compareAllResults, runAllEvaluators } from "./evaluator.js";
import { writeExperimentLog } from "./results.js";
import { createExperiment, generateExperimentId, updateExperiment } from "./store.js";
import type {
  CheckpointRef,
  CheckpointStrategy,
  EvaluatorResult,
  ExperimentIteration,
  ExperimentSpec,
  ExperimentState,
} from "./types.js";

const log = getChildLogger({ module: "experiment-protocol" });

// ── Types ──────────────────────────────────────────────────────────────

/** Callback to execute a single mutation turn. Returns agent summary text and tokens used. */
export type MutationRunner = (params: {
  /** The objective + context prompt for this iteration. */
  prompt: string;
  /** Iteration index. */
  iteration: number;
}) => Promise<{ summary?: string; tokensUsed: number }>;

/** Optional hooks for experiment lifecycle events. */
export type ExperimentHooks = {
  /** Called before each iteration. Return false to stop early. */
  beforeIteration?: (state: ExperimentState, iteration: number) => Promise<boolean>;
  /** Called after each iteration with the verdict. */
  afterIteration?: (state: ExperimentState, iteration: ExperimentIteration) => Promise<void>;
  /** Called when the experiment completes. */
  onComplete?: (state: ExperimentState) => Promise<void>;
};

export type RunExperimentParams = {
  spec: ExperimentSpec;
  agentId: string;
  /** Executes a single mutation agent turn. */
  runMutation: MutationRunner;
  /** Working directory for evaluators. */
  cwd?: string;
  /** Optional lifecycle hooks. */
  hooks?: ExperimentHooks;
  /** Check if we should yield to user. Returns true to pause. */
  shouldYield?: () => boolean;
};

// ── Core Experiment Loop ──────────────────────────────────────────────

export async function runExperiment(params: RunExperimentParams): Promise<ExperimentState> {
  const { spec, agentId, runMutation, cwd, hooks, shouldYield } = params;

  // Ensure spec has an id
  const specWithId: ExperimentSpec = {
    ...spec,
    id: spec.id || generateExperimentId(),
  };

  const checkpoint: CheckpointStrategy = resolveCheckpointStrategy(specWithId.checkpoint);

  log.info(
    {
      experimentId: specWithId.id,
      objective: specWithId.objective,
      checkpoint: checkpoint.describe(),
    },
    "starting experiment",
  );

  // Initialize state
  let state: ExperimentState = {
    spec: specWithId,
    status: "running",
    iterations: [],
    bestResult: null,
    bestIteration: null,
    totalTokensUsed: 0,
    startedAt: Date.now(),
  };

  // Persist initial state
  await createExperiment(agentId, state);

  // ── Measure baseline ──────────────────────────────────────────────
  const evalOpts = { cwd: cwd ?? process.cwd() };
  const baseline = await runAllEvaluators(specWithId.evaluators, evalOpts);

  const baselineErrors = baseline.filter((r) => r.error);
  if (baselineErrors.length > 0) {
    state = {
      ...state,
      status: "failed",
      completedAt: Date.now(),
      stopReason: `Baseline measurement failed: ${baselineErrors.map((e) => e.error).join("; ")}`,
    };
    await persistAndLog(agentId, state);
    return state;
  }

  state.bestResult = baseline;
  log.info(
    { metrics: baseline.map((r) => ({ id: r.evaluatorId, value: r.value })) },
    "baseline measured",
  );

  // ── Iteration loop ────────────────────────────────────────────────
  let currentBest = baseline;

  for (let i = 0; i < specWithId.maxIterations; i++) {
    // Check yield
    if (shouldYield?.()) {
      state = {
        ...state,
        status: "stopped",
        completedAt: Date.now(),
        stopReason: "yielded to user",
      };
      await persistAndLog(agentId, state);
      return state;
    }

    // Check time budget
    if (specWithId.constraints.timeBudgetSeconds) {
      const elapsed = (Date.now() - state.startedAt) / 1_000;
      if (elapsed >= specWithId.constraints.timeBudgetSeconds) {
        state = {
          ...state,
          status: "completed",
          completedAt: Date.now(),
          stopReason: "time budget exhausted",
        };
        await persistAndLog(agentId, state);
        return state;
      }
    }

    // Check token budget
    if (
      specWithId.constraints.tokenBudget &&
      state.totalTokensUsed >= specWithId.constraints.tokenBudget
    ) {
      state = {
        ...state,
        status: "completed",
        completedAt: Date.now(),
        stopReason: "token budget exhausted",
      };
      await persistAndLog(agentId, state);
      return state;
    }

    // Hook: before iteration
    if (hooks?.beforeIteration) {
      const shouldContinue = await hooks.beforeIteration(state, i);
      if (!shouldContinue) {
        state = {
          ...state,
          status: "stopped",
          completedAt: Date.now(),
          stopReason: "stopped by hook",
        };
        await persistAndLog(agentId, state);
        return state;
      }
    }

    const iterationStart = Date.now();
    let iteration: ExperimentIteration;

    try {
      // Save checkpoint before mutation
      let checkpointRef: CheckpointRef | undefined;
      try {
        checkpointRef = await checkpoint.save(`iteration-${i}`);
      } catch (err) {
        log.warn(
          { err: String(err), iteration: i },
          "checkpoint save failed, continuing without rollback",
        );
      }

      // Build mutation prompt with evaluator context + iteration history
      const prompt = buildMutationPrompt(specWithId, currentBest, i, state.iterations);

      // Run mutation agent turn
      const mutationResult = await runMutation({ prompt, iteration: i });

      // Measure candidate
      const candidate = await runAllEvaluators(specWithId.evaluators, evalOpts);

      // Compare with best-so-far
      const verdict = compareAllResults(currentBest, candidate, specWithId.evaluators);

      iteration = {
        index: i,
        baseline: currentBest,
        candidate,
        verdict,
        checkpointRef,
        agentSummary: mutationResult.summary,
        tokensUsed: mutationResult.tokensUsed,
        durationMs: Date.now() - iterationStart,
        startedAt: iterationStart,
      };

      // Act on verdict
      if (verdict === "improved" || verdict === "threshold_met") {
        currentBest = candidate;
        state.bestResult = candidate;
        state.bestIteration = i;
        log.info(
          {
            iteration: i,
            verdict,
            metrics: candidate.map((r) => ({ id: r.evaluatorId, value: r.value })),
          },
          "iteration kept",
        );
      } else {
        // Rollback
        if (checkpointRef) {
          try {
            await checkpoint.restore(checkpointRef);
            log.info({ iteration: i, verdict }, "iteration rolled back");
          } catch (err) {
            log.warn({ err: String(err), iteration: i }, "rollback failed");
          }
        } else {
          log.info({ iteration: i, verdict }, "iteration discarded (no checkpoint to restore)");
        }
      }
    } catch (err) {
      // Mutation or evaluation crashed
      iteration = {
        index: i,
        baseline: currentBest,
        candidate: null,
        verdict: "error",
        agentSummary: `Error: ${err instanceof Error ? err.message : String(err)}`,
        tokensUsed: 0,
        durationMs: Date.now() - iterationStart,
        startedAt: iterationStart,
      };
      log.warn({ err: String(err), iteration: i }, "iteration error");
    }

    // Record iteration
    state.iterations.push(iteration);
    state.totalTokensUsed += iteration.tokensUsed;

    // Persist progress after each iteration
    await updateExperiment(agentId, specWithId.id, () => state);

    // Hook: after iteration
    if (hooks?.afterIteration) {
      await hooks.afterIteration(state, iteration);
    }

    // Early stop on threshold_met
    if (iteration.verdict === "threshold_met") {
      state = {
        ...state,
        status: "completed",
        completedAt: Date.now(),
        stopReason: "threshold met",
      };
      await persistAndLog(agentId, state);
      await hooks?.onComplete?.(state);
      return state;
    }
  }

  // All iterations exhausted
  state = {
    ...state,
    status: "completed",
    completedAt: Date.now(),
    stopReason: "max iterations reached",
  };
  await persistAndLog(agentId, state);
  await hooks?.onComplete?.(state);
  return state;
}

// ── Helpers ─────────────────────────────────────────────────────────

/** Build the prompt for a mutation agent turn, injecting evaluator context + iteration history. */
function buildMutationPrompt(
  spec: ExperimentSpec,
  currentBest: EvaluatorResult[],
  iteration: number,
  priorIterations: ExperimentIteration[],
): string {
  const parts: string[] = [
    `You are running an experiment to optimize a measurable objective.`,
    ``,
    `## Objective`,
    spec.objective,
    ``,
    `## Current Metrics (iteration ${iteration})`,
  ];

  for (const result of currentBest) {
    const evalSpec = spec.evaluators.find((e) => e.id === result.evaluatorId);
    const direction =
      evalSpec?.direction === "higher_is_better" ? "higher is better" : "lower is better";
    const thresholdNote = evalSpec?.threshold != null ? ` (target: ${evalSpec.threshold})` : "";
    parts.push(
      `- **${evalSpec?.name ?? result.evaluatorId}**: ${result.value} (${direction}${thresholdNote})`,
    );
  }

  parts.push(``);

  // Inject iteration history so the agent knows what was already tried
  if (priorIterations.length > 0) {
    parts.push(`## Prior Iterations`);
    parts.push(``);
    // Show last 5 iterations to keep prompt manageable
    const recent = priorIterations.slice(-5);
    for (const iter of recent) {
      const verdictLabel =
        iter.verdict === "improved" || iter.verdict === "threshold_met" ? "KEPT" : "ROLLED BACK";
      const metricsStr = iter.candidate
        ? iter.candidate.map((r) => `${r.evaluatorId}=${r.value}`).join(", ")
        : "no measurement";
      const summary = iter.agentSummary ? iter.agentSummary.slice(0, 200) : "no summary";
      parts.push(`- **Iteration ${iter.index}** [${verdictLabel}]: ${summary} → ${metricsStr}`);
    }
    parts.push(``);
    parts.push(
      `Learn from these prior attempts: avoid repeating rolled-back approaches, build on what worked.`,
    );
    parts.push(``);
  }

  if (spec.constraints.allowedFiles?.length) {
    parts.push(
      `## Allowed Files`,
      `You may ONLY modify files matching: ${spec.constraints.allowedFiles.join(", ")}`,
      ``,
    );
  }
  if (spec.constraints.deniedFiles?.length) {
    parts.push(
      `## Denied Files`,
      `Do NOT modify files matching: ${spec.constraints.deniedFiles.join(", ")}`,
      ``,
    );
  }

  parts.push(
    `## Instructions`,
    ``,
    `- Make a targeted change to improve the metrics above.`,
    `- Focus on a single, clear improvement per iteration.`,
    `- After making changes, your work will be automatically evaluated.`,
    `- If the metrics improve, your changes will be kept. Otherwise, they will be rolled back.`,
    `- Be specific about what you changed and why in your response.`,
  );

  return parts.join("\n");
}

/** Persist final state, write human-readable log, and store to soul memory. */
async function persistAndLog(agentId: string, state: ExperimentState): Promise<void> {
  await updateExperiment(agentId, state.spec.id, () => state);
  try {
    await writeExperimentLog(agentId, state);
  } catch (err) {
    log.warn({ err: String(err) }, "failed to write experiment log");
  }
  try {
    persistExperimentMemory({ agentId, state });
  } catch (err) {
    log.warn({ err: String(err) }, "failed to persist experiment memory");
  }
  log.info(
    {
      experimentId: state.spec.id,
      status: state.status,
      iterations: state.iterations.length,
      kept: state.iterations.filter(
        (i) => i.verdict === "improved" || i.verdict === "threshold_met",
      ).length,
      stopReason: state.stopReason,
    },
    "experiment finished",
  );
}

/** Compute a summary of experiment results for announcements. */
export function summarizeExperiment(state: ExperimentState): string {
  const kept = state.iterations.filter(
    (i) => i.verdict === "improved" || i.verdict === "threshold_met",
  ).length;
  const rolled = state.iterations.filter(
    (i) => i.verdict === "regressed" || i.verdict === "no_change",
  ).length;
  const errored = state.iterations.filter((i) => i.verdict === "error").length;

  const baselineStr = state.iterations[0]?.baseline
    .map((r) => `${r.evaluatorId}: ${r.value}`)
    .join(", ");
  const bestStr = state.bestResult?.map((r) => `${r.evaluatorId}: ${r.value}`).join(", ");

  const parts = [
    `Experiment: **${state.spec.name}**`,
    `Status: ${state.status} (${state.stopReason ?? "unknown"})`,
    `Iterations: ${state.iterations.length} (${kept} kept, ${rolled} rolled back, ${errored} errors)`,
    `Baseline: ${baselineStr ?? "N/A"}`,
    `Best: ${bestStr ?? "N/A"} (iteration ${state.bestIteration ?? "N/A"})`,
    `Tokens used: ${state.totalTokensUsed.toLocaleString()}`,
  ];

  const durationMs = (state.completedAt ?? Date.now()) - state.startedAt;
  const durationStr =
    durationMs > 60_000
      ? `${(durationMs / 60_000).toFixed(1)}m`
      : `${(durationMs / 1_000).toFixed(1)}s`;
  parts.push(`Duration: ${durationStr}`);

  return parts.join("\n");
}
