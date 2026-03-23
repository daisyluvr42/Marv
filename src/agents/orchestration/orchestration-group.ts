import type { GoalFrame } from "../pi-embedded-runner/goal-loop.js";
import type { SpawnSubagentContext, SpawnSubagentParams } from "../subagent-spawn.js";
import { buildContractFromGoalFrame } from "./contract-builder.js";
import {
  type OrchestrationDeps,
  isTerminalPhase,
  runOrchestrationCycle,
  startOrchestrationWithContract,
} from "./orchestration-loop.js";
import type {
  GoalContract,
  OrchestrationConfig,
  OrchestrationEntry,
  OrchestrationGroup,
} from "./types.js";

/**
 * Start an orchestration group that coordinates multiple subagents.
 * Each role gets its own contract derived from the shared GoalFrame.
 */
export async function startOrchestrationGroup(params: {
  goalFrame: GoalFrame;
  roles: Array<{ role: string; spawnParams: SpawnSubagentParams }>;
  spawnCtx: SpawnSubagentContext;
  config?: OrchestrationConfig;
  deps: OrchestrationDeps;
  allMustPass?: boolean;
}): Promise<OrchestrationGroup> {
  const { goalFrame, roles, spawnCtx, config, deps } = params;
  const groupId = `group_${Date.now().toString(36)}`;

  const entries: OrchestrationEntry[] = [];
  for (const { role, spawnParams } of roles) {
    const contract = buildContractFromGoalFrame({ goalFrame, config });
    // Tag the contract with role info in the objective
    const roleContract: GoalContract = {
      ...contract,
      objective: `[${role}] ${contract.objective}`,
    };

    const entry = await startOrchestrationWithContract({
      contract: roleContract,
      spawnParams: { ...spawnParams, role },
      spawnCtx,
      deps,
    });
    entries.push(entry);
  }

  return {
    groupId,
    entries,
    allMustPass: params.allMustPass ?? true,
    groupPhase: "running",
  };
}

/**
 * Run the orchestration cycle for all entries in a group.
 * Respects the group mode (fail_fast vs best_effort).
 */
export async function runGroupOrchestration(params: {
  group: OrchestrationGroup;
  deps: OrchestrationDeps;
}): Promise<OrchestrationGroup> {
  const { group, deps } = params;
  const failFast = group.allMustPass;

  const results: OrchestrationEntry[] = [];
  for (const entry of group.entries) {
    const result = await runOrchestrationCycle({ entry, deps });
    results.push(result);

    // In fail-fast mode, stop as soon as any entry fails
    if (failFast && (result.phase === "rejected" || result.phase === "budget_exhausted")) {
      // Keep remaining entries as-is (still in "spawned" or "monitoring" phase)
      const remaining = group.entries.slice(results.length);
      return {
        ...group,
        entries: [...results, ...remaining],
        groupPhase: result.phase === "rejected" ? "rejected" : "budget_exhausted",
      };
    }
  }

  const groupPhase = resolveGroupPhase(results, group.allMustPass);

  return {
    ...group,
    entries: results,
    groupPhase,
  };
}

/**
 * Evaluate the group-level phase based on individual entry results.
 */
export function resolveGroupPhase(
  entries: OrchestrationEntry[],
  allMustPass: boolean,
): OrchestrationGroup["groupPhase"] {
  const allTerminal = entries.every((e) => isTerminalPhase(e.phase));
  if (!allTerminal) {
    return "running";
  }

  const allAccepted = entries.every((e) => e.phase === "accepted");
  if (allAccepted) {
    return "accepted";
  }

  if (allMustPass) {
    const hasBudgetExhausted = entries.some((e) => e.phase === "budget_exhausted");
    return hasBudgetExhausted ? "budget_exhausted" : "rejected";
  }

  // In best-effort mode, group is accepted if at least one entry succeeded
  const anyAccepted = entries.some((e) => e.phase === "accepted");
  return anyAccepted ? "accepted" : "rejected";
}
