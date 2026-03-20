import {
  querySoulMemoryMulti,
  writeSoulMemory,
  type SoulMemoryConfig,
  type SoulMemoryScope,
} from "../../memory/storage/soul-memory-store.js";
import { parseAgentSessionKey } from "../../routing/session-key.js";
import type { GoalLoopState, ProblemShape, StrategyFamily, StrategyHint } from "./goal-loop.js";

const STRATEGY_MEMORY_KIND = "agent_strategy";
const STRATEGY_MEMORY_SOURCE = "auto_extraction";

function dedupeScopes(scopes: SoulMemoryScope[]): SoulMemoryScope[] {
  const dedup = new Map<string, SoulMemoryScope>();
  for (const scope of scopes) {
    const scopeType = scope.scopeType.trim().toLowerCase();
    const scopeId = scope.scopeId.trim().toLowerCase();
    if (!scopeType || !scopeId) {
      continue;
    }
    const key = `${scopeType}:${scopeId}`;
    const existing = dedup.get(key);
    if (!existing || scope.weight > existing.weight) {
      dedup.set(key, { scopeType, scopeId, weight: scope.weight });
    }
  }
  return [...dedup.values()];
}

export function resolveGoalStrategyScopes(params: {
  agentId: string;
  sessionKey?: string;
}): SoulMemoryScope[] {
  const scopes: SoulMemoryScope[] = [{ scopeType: "agent", scopeId: params.agentId, weight: 1 }];
  const parsed = parseAgentSessionKey(params.sessionKey);
  if (!parsed?.rest) {
    return scopes;
  }
  scopes.unshift({
    scopeType: "session",
    scopeId: parsed.rest,
    weight: 1.1,
  });
  const tokens = parsed.rest.toLowerCase().split(":").filter(Boolean);
  if (tokens.length >= 3) {
    const channel = tokens[0] ?? "";
    const kind = tokens[1] ?? "";
    const peerId = tokens[2] ?? "";
    if (channel && peerId && kind === "direct") {
      scopes.push({
        scopeType: "user",
        scopeId: `${channel}:${peerId}`,
        weight: 1.05,
      });
    }
  }
  return dedupeScopes(scopes);
}

function asProblemShape(value: unknown): ProblemShape | null {
  if (
    value === "information_gap" ||
    value === "implementation_blocked" ||
    value === "validation_failure" ||
    value === "tool_or_permission_limit" ||
    value === "search_drift" ||
    value === "parallelizable_subproblems"
  ) {
    return value;
  }
  return null;
}

function asStrategyFamily(value: unknown): StrategyFamily | null {
  if (
    value === "read_context" ||
    value === "inspect_failure" ||
    value === "try_alternative" ||
    value === "validate_result" ||
    value === "request_capability" ||
    value === "synthesize_tool" ||
    value === "delegated_subagent" ||
    value === "recenter_goal" ||
    value === "split_subproblems" ||
    value === "wrap_up"
  ) {
    return value;
  }
  return null;
}

export function loadGoalStrategyHints(params: {
  agentId: string;
  sessionKey?: string;
  objective: string;
  problemShape?: ProblemShape | null;
  soulConfig?: SoulMemoryConfig;
}): StrategyHint[] {
  const query = [params.objective, params.problemShape ?? "", "strategy", "what worked"]
    .filter(Boolean)
    .join(" ");
  const results = querySoulMemoryMulti({
    agentId: params.agentId,
    scopes: resolveGoalStrategyScopes({
      agentId: params.agentId,
      sessionKey: params.sessionKey,
    }),
    query,
    topK: 4,
    minScore: 0.08,
    soulConfig: params.soulConfig,
  });

  return results
    .filter((entry) => entry.kind === STRATEGY_MEMORY_KIND)
    .map((entry) => ({
      memoryId: entry.id,
      summary: entry.summary?.trim() || entry.content.trim().split("\n")[0] || "Previous strategy",
      strategyFamily: asStrategyFamily(entry.metadata?.strategyFamily),
      problemShape: asProblemShape(entry.metadata?.problemShape),
      score: entry.score,
      tier: entry.tier,
    }))
    .toSorted((a, b) => b.score - a.score)
    .slice(0, 3);
}

export function persistGoalStrategyMemory(params: {
  agentId: string;
  sessionKey?: string;
  state: GoalLoopState;
  soulConfig?: SoulMemoryConfig;
}): void {
  const isSuccess = params.state.convergeReason === "sufficient_completion";
  const isFailure = params.state.convergeReason === "no_progress_no_new_strategy";
  if (!isSuccess && !isFailure) {
    return;
  }
  const scopes = resolveGoalStrategyScopes({
    agentId: params.agentId,
    sessionKey: params.sessionKey,
  });
  const primaryScope = scopes[0] ?? { scopeType: "agent", scopeId: params.agentId };
  const outcome = isSuccess ? "succeeded" : "failed";
  const content = [
    `Objective: ${params.state.goalFrame.objective}`,
    `Outcome: ${outcome}`,
    `Problem shape: ${params.state.problemShape ?? "unknown"}`,
    `Strategy family: ${params.state.strategyFamily}`,
    `Direction node: ${params.state.directionNodes[params.state.currentNodeIndex] ?? "wrap up"}`,
    `Converged because: ${params.state.convergeReason}`,
    `Shifts: ${params.state.shiftCount}`,
  ].join("\n");
  const summary = isSuccess
    ? `${params.state.strategyFamily} worked for ${params.state.goalFrame.objective}`
    : `${params.state.strategyFamily} failed for ${params.state.goalFrame.objective} (${params.state.problemShape ?? "unknown"})`;
  writeSoulMemory({
    agentId: params.agentId,
    scopeType: primaryScope.scopeType,
    scopeId: primaryScope.scopeId,
    kind: STRATEGY_MEMORY_KIND,
    content,
    summary,
    source: STRATEGY_MEMORY_SOURCE,
    tier: isSuccess ? "P2" : "P3",
    recordKind: "experience",
    metadata: {
      objective: params.state.goalFrame.objective,
      goalType: params.state.goalFrame.goalType,
      complexity: params.state.goalFrame.complexity,
      problemShape: params.state.problemShape,
      strategyFamily: params.state.strategyFamily,
      convergeReason: params.state.convergeReason,
      outcome,
      directionNode: params.state.directionNodes[params.state.currentNodeIndex] ?? null,
      shiftCount: params.state.shiftCount,
    },
    soulConfig: params.soulConfig,
  });
}
