/**
 * Unified "what model should I use right now?" entry point.
 *
 * Combines pool planning with availability state to return the single best
 * candidate, optionally biased by task complexity.
 */

import type { MarvConfig } from "../../core/config/config.js";
import type { AutoRoutingComplexity } from "../../core/config/types.agent-defaults.js";
import { getRuntimeModelAvailability } from "./model-availability-state.js";
import { resolveRuntimeModelPlan, type RuntimeModelRequirements } from "./model-pool.js";

export type EffectiveModelResult = {
  ref: string;
  provider: string;
  model: string;
  reason: "pool_candidate" | "all_cooling_down";
};

/**
 * Resolve the best available model right now.
 *
 * Pool candidates are sorted high → low tier by default.  For simple/moderate
 * tasks the list is reversed so cheaper low-tier models are tried first.
 * Candidates with an active `temporary_unavailable` cooldown (retryAfter in
 * the future) are skipped.
 */
export function resolveEffectiveModel(params: {
  cfg: MarvConfig;
  agentId?: string;
  agentDir?: string;
  complexityHint?: AutoRoutingComplexity;
  requirements?: RuntimeModelRequirements;
}): EffectiveModelResult | null {
  const plan = resolveRuntimeModelPlan({
    cfg: params.cfg,
    agentId: params.agentId,
    agentDir: params.agentDir,
    requirements: params.requirements,
  });

  let candidates = plan.candidates;

  // For simple tasks, prefer low-tier (cheaper) models first.
  if (params.complexityHint === "simple" || params.complexityHint === "moderate") {
    candidates = candidates.toReversed();
  }

  // Skip models with an active temporary cooldown.
  for (const c of candidates) {
    const avail = getRuntimeModelAvailability(c.ref);
    if (
      avail?.status === "temporary_unavailable" &&
      avail.retryAfter &&
      avail.retryAfter > Date.now()
    ) {
      continue;
    }
    return { ref: c.ref, provider: c.provider, model: c.model, reason: "pool_candidate" };
  }

  // All candidates temporarily unavailable — return first anyway so the
  // caller can attempt it (the fallback loop will handle the retry).
  const first = candidates[0] ?? plan.configured[0];
  if (first) {
    return {
      ref: first.ref,
      provider: first.provider,
      model: first.model,
      reason: "all_cooling_down",
    };
  }
  return null;
}
