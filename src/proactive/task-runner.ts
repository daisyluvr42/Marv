import { resolveDefaultAgentId } from "../agents/agent-scope.js";
import { classifyComplexityByRules } from "../agents/auto-routing.js";
import { resolveRuntimeModelPlan } from "../agents/model/model-pool.js";
import type { CliDeps } from "../cli/deps.js";
import type { MarvConfig } from "../core/config/config.js";
import { loadConfig } from "../core/config/config.js";
import { runCronIsolatedAgentTurn } from "../cron/isolated-agent.js";
import type { CronJob } from "../cron/types.js";
import { runExperiment, summarizeExperiment } from "../experiments/protocol.js";
import { sendMessage } from "../infra/outbound/message.js";
import { getChildLogger } from "../logging.js";
import {
  enqueueCommandInLane,
  getQueueSize,
  setCommandLaneConcurrency,
} from "../process/command-queue.js";
import { CommandLane } from "../process/lanes.js";
import { getBudgetStatus, recordTokenUsage } from "./budget.js";
import { markAnnounced, registerDeliverable } from "./deliverables.js";
import {
  checkpointTask,
  completeTask,
  dequeueTask,
  failTask,
  type ProactiveTask,
  pruneFinishedTasks,
  recoverStaleTasks,
} from "./task-queue.js";

const log = getChildLogger({ module: "proactive-runner" });

const DEFAULT_YIELD_TO_USER_MS = 500;
const DEFAULT_TASK_POLL_INTERVAL_MS = 5_000;
const DEFAULT_MAX_CONCURRENT = 1;
/** Tasks stuck in "running" longer than this are considered orphaned. */
const STALE_TASK_THRESHOLD_MS = 10 * 60_000; // 10 minutes
/** Cap proactive turns to avoid blocking resources for too long. */
const MAX_PROACTIVE_TURN_SECONDS = 120;
/** Prune completed/failed tasks older than this. */
const PRUNE_FINISHED_TASKS_AGE_MS = 24 * 60 * 60_000; // 24 hours
/** Run prune every N ticks to avoid doing it on every cycle. */
const PRUNE_EVERY_N_TICKS = 100;

// ── Runner state ───────────────────────────────────────────────────────

export type ProactiveTaskRunnerParams = {
  deps: CliDeps;
  /** Initial config snapshot; runner reloads on each iteration. */
  cfg: MarvConfig;
};

export type ProactiveTaskRunner = {
  start: () => void;
  stop: () => void;
  readonly running: boolean;
};

// ── Internal helpers ───────────────────────────────────────────────────

function resolveProactiveConfig(cfg: MarvConfig) {
  const p = cfg.autonomy?.proactive;
  return {
    yieldToUserMs: p?.yieldToUserMs ?? DEFAULT_YIELD_TO_USER_MS,
    taskPollIntervalMs: p?.taskPollIntervalMs ?? DEFAULT_TASK_POLL_INTERVAL_MS,
    maxConcurrent: p?.maxConcurrentTasks ?? DEFAULT_MAX_CONCURRENT,
    modelStrategy: p?.modelStrategy ?? "default",
    preferLocalModels: p?.preferLocalModels ?? true,
    dailyCloudTokenBudget: p?.dailyCloudTokenBudget ?? 0,
    cloudEscalationThreshold: p?.cloudEscalationThreshold ?? "expert",
    primaryModel: p?.primaryModel,
    escalationModel: p?.escalationModel,
  };
}

/**
 * Resolve the best local model ref for proactive work.
 * Returns undefined if no local model is available (falls back to default).
 */
function resolveLocalModelRef(cfg: MarvConfig, agentId: string): string | undefined {
  const plan = resolveRuntimeModelPlan({
    cfg,
    agentId,
    requirements: { requiredCapabilities: ["text"] },
  });
  const localCandidate = plan.candidates.find((c) => c.location === "local");
  return localCandidate?.ref;
}

/**
 * Returns true when the main lane has active or queued work,
 * meaning the proactive runner should yield resources.
 */
export function shouldYieldToUser(): boolean {
  return getQueueSize(CommandLane.Main) > 0;
}

type ResolvedProactiveConfig = ReturnType<typeof resolveProactiveConfig>;

/**
 * Resolve the model ref and whether we're using a cloud/paid model.
 *
 * "default" strategy: prefer local models, escalate to cloud when no local available and task is complex.
 * "custom" strategy: user-specified primaryModel for regular tasks, escalationModel for complex tasks.
 */
function resolveModelForTask(params: {
  cfg: MarvConfig;
  proactiveCfg: ResolvedProactiveConfig;
  agentId: string;
  needsEscalation: boolean;
}): { modelRef: string | undefined; useCloud: boolean } {
  const { cfg, proactiveCfg, agentId, needsEscalation } = params;

  if (proactiveCfg.modelStrategy === "custom") {
    // Custom strategy: user explicitly chose models.
    const ref = needsEscalation
      ? (proactiveCfg.escalationModel ?? proactiveCfg.primaryModel)
      : proactiveCfg.primaryModel;
    // In custom mode, budget applies to escalation model (assumed more expensive).
    const useCloud = needsEscalation && proactiveCfg.escalationModel !== proactiveCfg.primaryModel;
    return { modelRef: ref, useCloud };
  }

  // Default strategy: local-first, cloud escalation.
  const localModelRef = proactiveCfg.preferLocalModels
    ? resolveLocalModelRef(cfg, agentId)
    : undefined;
  if (!needsEscalation || localModelRef) {
    return { modelRef: localModelRef, useCloud: false };
  }
  // Needs escalation and no local model — fall through to cloud default.
  return { modelRef: undefined, useCloud: true };
}

/** Complexity tiers ordered low→high for threshold comparison. */
const COMPLEXITY_ORDER = ["simple", "moderate", "complex", "expert"] as const;

/**
 * Determine whether a task's complexity warrants cloud model escalation.
 * Returns true when the task complexity meets or exceeds the configured threshold.
 */
function shouldEscalateToCloud(
  taskDescription: string,
  threshold: "moderate" | "complex" | "expert",
): boolean {
  const complexity = classifyComplexityByRules({ prompt: taskDescription });
  const thresholdIdx = COMPLEXITY_ORDER.indexOf(threshold);
  const complexityIdx = COMPLEXITY_ORDER.indexOf(complexity);
  return complexityIdx >= thresholdIdx;
}

/** Map complexity tier to an appropriate thinking level. */
function resolveThinkingForTask(taskDescription: string, useCloud: boolean): string {
  if (!useCloud) {
    return "low";
  }
  const complexity = classifyComplexityByRules({ prompt: taskDescription });
  if (complexity === "expert") {
    return "high";
  }
  if (complexity === "complex") {
    return "medium";
  }
  return "low";
}

/**
 * Build a synthetic CronJob-like object so we can reuse
 * `runCronIsolatedAgentTurn` for proactive task execution.
 */
function buildProactiveJobShell(
  task: ProactiveTask,
  agentId: string,
  opts: { modelRef?: string; useCloud: boolean },
): CronJob {
  const thinking = resolveThinkingForTask(task.description, opts.useCloud);
  return {
    id: `proactive_${task.id}`,
    agentId,
    name: `Proactive: ${task.title}`,
    description: task.description,
    enabled: true,
    deleteAfterRun: true,
    createdAtMs: task.createdAt,
    updatedAtMs: task.updatedAt,
    schedule: { kind: "every", everyMs: 0 },
    sessionTarget: "isolated",
    wakeMode: "now",
    payload: {
      kind: "agentTurn",
      message: buildTaskPrompt(task),
      thinking,
      timeoutSeconds: MAX_PROACTIVE_TURN_SECONDS,
      ...(opts.modelRef ? { model: opts.modelRef } : {}),
    },
    state: {},
  };
}

function buildTaskPrompt(task: ProactiveTask): string {
  const parts = [
    `You are executing a proactive task autonomously.`,
    ``,
    `## Task: ${task.title}`,
    ``,
    task.description,
  ];
  if (task.checkpoint) {
    parts.push(
      ``,
      `## Checkpoint (previous progress)`,
      ``,
      typeof task.checkpoint === "string"
        ? task.checkpoint
        : JSON.stringify(task.checkpoint, null, 2),
    );
  }
  parts.push(
    ``,
    `## Instructions`,
    ``,
    `- Complete this task to the best of your ability.`,
    `- If you cannot finish in one turn, summarize your progress so far — it will be saved as a checkpoint for the next attempt.`,
    `- When done, state the result clearly.`,
  );
  return parts.join("\n");
}

// ── Main loop ──────────────────────────────────────────────────────────

export function createProactiveTaskRunner(params: ProactiveTaskRunnerParams): ProactiveTaskRunner {
  let running = false;
  let stopRequested = false;
  let tickCount = 0;

  const start = () => {
    if (running) {
      return;
    }
    running = true;
    stopRequested = false;

    const cfg = resolveProactiveConfig(params.cfg);
    setCommandLaneConcurrency(CommandLane.Proactive, cfg.maxConcurrent);

    // Recover orphaned tasks from previous crash/restart before looping.
    const agentId = resolveDefaultAgentId(params.cfg);
    void recoverStaleTasks(agentId, STALE_TASK_THRESHOLD_MS).then((recovered) => {
      if (recovered > 0) {
        log.info({ recovered }, "recovered stale running tasks on startup");
      }
    });

    log.info("proactive task runner started");

    // Fire-and-forget the loop; errors are caught inside.
    void runLoop();
  };

  const stop = () => {
    if (!running) {
      return;
    }
    stopRequested = true;
    log.info("proactive task runner stop requested");
  };

  async function runLoop() {
    while (!stopRequested) {
      try {
        await tick();
      } catch (err) {
        log.warn({ err: String(err) }, "proactive runner tick error");
      }
    }
    running = false;
    log.info("proactive task runner stopped");
  }

  async function tick() {
    tickCount++;
    const runtimeCfg = loadConfig();
    const proactiveCfg = resolveProactiveConfig(runtimeCfg);
    const agentId = resolveDefaultAgentId(runtimeCfg);

    // Periodic housekeeping: prune old completed/failed tasks.
    if (tickCount % PRUNE_EVERY_N_TICKS === 0) {
      await pruneFinishedTasks(agentId, PRUNE_FINISHED_TASKS_AGE_MS).catch(() => {});
    }

    // Yield to user conversations when the main lane is busy.
    if (shouldYieldToUser()) {
      await sleep(proactiveCfg.yieldToUserMs);
      return;
    }

    // Try to dequeue a task (cheap file read — do this before model resolution).
    const task = await dequeueTask(agentId);
    if (!task) {
      await sleep(proactiveCfg.taskPollIntervalMs);
      return;
    }

    // ── Model routing ──────────────────────────────────────────────────
    const needsEscalation = shouldEscalateToCloud(
      task.description,
      proactiveCfg.cloudEscalationThreshold,
    );
    const { modelRef, useCloud } = resolveModelForTask({
      cfg: runtimeCfg,
      proactiveCfg,
      agentId,
      needsEscalation,
    });

    // Check cloud token budget before committing to a cloud/paid model.
    if (useCloud && proactiveCfg.dailyCloudTokenBudget > 0) {
      const budget = await getBudgetStatus(agentId, proactiveCfg.dailyCloudTokenBudget);
      if (budget.exhausted) {
        await checkpointTask(agentId, task.id, task.checkpoint);
        log.info(
          { todayTokens: budget.todayTokens, limit: budget.dailyLimit },
          "cloud token budget exhausted, pausing proactive work",
        );
        await sleep(proactiveCfg.taskPollIntervalMs * 6);
        return;
      }
    }

    log.info(
      {
        taskId: task.id,
        title: task.title,
        model: modelRef ?? "pool-default",
        escalated: needsEscalation,
      },
      "executing proactive task",
    );

    // ── Experiment routing ────────────────────────────────────────────
    // Tasks with an experimentSpec run the evaluator-driven experiment
    // loop instead of a single agent turn.
    if (task.experimentSpec) {
      try {
        const expResult = await enqueueCommandInLane(CommandLane.Proactive, async () => {
          return await runExperiment({
            spec: task.experimentSpec!,
            agentId,
            cwd: process.cwd(),
            shouldYield: shouldYieldToUser,
            runMutation: async ({ prompt }) => {
              const job = buildProactiveJobShell({ ...task, description: prompt }, agentId, {
                modelRef,
                useCloud,
              });
              const turnResult = await runCronIsolatedAgentTurn({
                cfg: runtimeCfg,
                deps: params.deps,
                job,
                message: prompt,
                agentId,
                sessionKey: `experiment:${task.experimentSpec!.id}:${Date.now()}`,
                lane: CommandLane.Proactive,
              });
              const tokensUsed = turnResult.usage?.total_tokens ?? 0;
              if (tokensUsed > 0) {
                await recordTokenUsage(agentId, tokensUsed);
              }
              return { summary: turnResult.outputText, tokensUsed };
            },
          });
        });

        const taskResult = summarizeExperiment(expResult);
        if (expResult.status === "completed" || expResult.status === "stopped") {
          await completeTask(agentId, task.id, taskResult);
          await announceTaskCompletion(runtimeCfg, agentId, task, taskResult);
          log.info(
            { taskId: task.id, experimentId: task.experimentSpec.id, status: expResult.status },
            "experiment task completed",
          );
        } else {
          await failTask(agentId, task.id, taskResult);
          log.warn({ taskId: task.id, status: expResult.status }, "experiment task failed");
        }
      } catch (err) {
        await failTask(agentId, task.id, String(err));
        log.warn({ taskId: task.id, err: String(err) }, "experiment task error");
      }
      return;
    }

    // ── Regular task execution ──────────────────────────────────────────
    // Execute the task on the proactive lane.
    try {
      const result = await enqueueCommandInLane(CommandLane.Proactive, async () => {
        const job = buildProactiveJobShell(task, agentId, { modelRef, useCloud });
        return await runCronIsolatedAgentTurn({
          cfg: runtimeCfg,
          deps: params.deps,
          job,
          message: job.payload.kind === "agentTurn" ? job.payload.message : task.description,
          agentId,
          sessionKey: `proactive:${task.id}`,
          lane: CommandLane.Proactive,
        });
      });

      // Record token usage for budget tracking.
      const totalTokens = result.usage?.total_tokens;
      if (totalTokens && totalTokens > 0) {
        await recordTokenUsage(agentId, totalTokens);
      }

      if (result.status === "ok") {
        const taskResult = result.summary ?? result.outputText;
        await completeTask(agentId, task.id, taskResult);
        // Register deliverable and announce via system delivery pipeline.
        await announceTaskCompletion(runtimeCfg, agentId, task, taskResult);
        log.info({ taskId: task.id, tokens: totalTokens }, "proactive task completed");
      } else {
        // Partial progress — checkpoint for retry.
        await checkpointTask(agentId, task.id, result.outputText ?? result.summary);
        log.warn(
          { taskId: task.id, status: result.status, error: result.error },
          "proactive task did not complete, checkpointed",
        );
      }
    } catch (err) {
      await failTask(agentId, task.id, String(err));
      log.warn({ taskId: task.id, err: String(err) }, "proactive task failed");
    }
  }

  return {
    start,
    stop,
    get running() {
      return running;
    },
  };
}

/**
 * Register a deliverable for a completed task and announce it to the user
 * via the system delivery pipeline (same channel resolution as digest cron).
 */
async function announceTaskCompletion(
  cfg: MarvConfig,
  agentId: string,
  task: ProactiveTask,
  taskResult?: string,
): Promise<void> {
  try {
    const { deliverable, created } = await registerDeliverable(agentId, {
      taskId: task.id,
      goalId: task.goalId,
      title: task.title,
      kind: "other",
    });
    if (!created) {
      // Already registered (e.g. by the LLM during execution) — skip duplicate announce.
      return;
    }

    // Resolve delivery target from config.
    const delivery = cfg.autonomy?.proactive?.delivery;
    const channel = delivery?.channel?.trim() || undefined;
    const to = delivery?.to?.trim();

    if (!to && !channel) {
      // No delivery configured — mark as announced silently (no user notification).
      await markAnnounced(agentId, deliverable.id);
      log.info({ taskId: task.id }, "deliverable stored (no delivery target configured)");
      return;
    }

    // Build a concise notification.
    const summary = taskResult
      ? `✅ Proactive task completed: **${task.title}**\n\n${taskResult.slice(0, 500)}`
      : `✅ Proactive task completed: **${task.title}**`;

    await sendMessage({
      to: to ?? "",
      channel,
      content: summary,
      agentId,
      cfg,
      bestEffort: true,
    });
    await markAnnounced(agentId, deliverable.id);
    log.info({ taskId: task.id, channel, to }, "deliverable announced");
  } catch (err) {
    log.warn({ taskId: task.id, err: String(err) }, "failed to announce deliverable");
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
