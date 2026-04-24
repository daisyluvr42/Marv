import type { MarvConfig } from "../core/config/config.js";
import type {
  AutoRoutingComplexity,
  SubagentPresetConfig,
} from "../core/config/types.agent-defaults.js";
import { classifyComplexityByRules } from "./auto-routing.js";
import { resolveRuntimeModelPlan, type RuntimeConfiguredModel } from "./model/model-pool.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type SubagentAutoRouteResult = {
  /** Matched preset name (if any). */
  preset?: string;
  /** Expanded roles from the matched preset (or empty). */
  roles: string[];
  /** Classified task complexity. */
  complexity: AutoRoutingComplexity;
  /** Whether a preset was matched. */
  matched: boolean;
  /** Why the preset was matched. */
  matchReason?: "keywords" | "complexity" | "both";
  /** Task-aware model recommendation from the pool (provider/model ref). */
  recommendedModel?: string;
  /** Recommended thinking level based on complexity. */
  recommendedThinking?: string;
  /** Debug reason string for model selection. */
  modelReason?: string;
};

// Ordinal for complexity comparison.
const COMPLEXITY_ORDINAL: Record<AutoRoutingComplexity, number> = {
  simple: 0,
  moderate: 1,
  complex: 2,
  expert: 3,
};

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/**
 * Analyze a task string and auto-select the best preset/role and model.
 *
 * This bridges the existing complexity classifier (`classifyComplexityByRules`)
 * and model pool system (`resolveRuntimeModelPlan`) with the preset `autoTrigger`
 * config to provide a complete routing recommendation.
 */
export function resolveSubagentAutoRoute(params: {
  task: string;
  cfg: MarvConfig;
  hasImages?: boolean;
  /** Optional agent ID for pool resolution. */
  agentId?: string;
}): SubagentAutoRouteResult {
  const { task, cfg, hasImages } = params;
  const subagentsCfg = cfg.agents?.defaults?.subagents;

  // Step 1: Classify task complexity.
  const complexity = classifyComplexityByRules({
    prompt: task,
    hasImages,
    thresholds: cfg.agents?.defaults?.autoRouting?.thresholds,
  });

  // Step 2: Match preset via autoTrigger.
  const presetMatch = matchPreset({
    task,
    complexity,
    presets: subagentsCfg?.presets,
  });

  // Step 3: Resolve task-aware model.
  const modelResult = resolveTaskAwareModel({
    task,
    complexity,
    cfg,
    hasImages,
    agentId: params.agentId,
    presetAutoTrigger: presetMatch.matched
      ? subagentsCfg?.presets?.[presetMatch.preset!]?.autoTrigger
      : undefined,
  });

  return {
    ...presetMatch,
    complexity,
    recommendedModel: modelResult.model,
    recommendedThinking: modelResult.thinking,
    modelReason: modelResult.reason,
  };
}

// ---------------------------------------------------------------------------
// Preset matching
// ---------------------------------------------------------------------------

type PresetMatchResult = {
  preset?: string;
  roles: string[];
  matched: boolean;
  matchReason?: "keywords" | "complexity" | "both";
};

function matchPreset(params: {
  task: string;
  complexity: AutoRoutingComplexity;
  presets?: Record<string, SubagentPresetConfig>;
}): PresetMatchResult {
  const { task, complexity, presets } = params;
  if (!presets) {
    return { roles: [], matched: false };
  }

  const taskLower = task.toLowerCase();
  let bestPreset: string | undefined;
  let bestScore = 0;
  let bestReason: "keywords" | "complexity" | "both" | undefined;

  for (const [name, config] of Object.entries(presets)) {
    const trigger = config.autoTrigger;
    if (!trigger) {
      continue;
    }

    let score = 0;
    let keywordsMatched = false;
    let complexityMatched = false;

    // Check keywords.
    if (trigger.keywords && trigger.keywords.length > 0) {
      const matched = trigger.keywords.some((kw) => taskLower.includes(kw.toLowerCase()));
      if (matched) {
        score += 1;
        keywordsMatched = true;
      }
    }

    // Check complexity threshold.
    if (trigger.minComplexity) {
      const min = COMPLEXITY_ORDINAL[trigger.minComplexity] ?? 0;
      const actual = COMPLEXITY_ORDINAL[complexity] ?? 0;
      if (actual >= min) {
        score += 1;
        complexityMatched = true;
      }
    }

    if (score === 0) {
      continue;
    }

    // Prefer higher scores; break ties by preferring "both" over single match.
    if (score > bestScore) {
      bestScore = score;
      bestPreset = name;
      bestReason =
        keywordsMatched && complexityMatched ? "both" : keywordsMatched ? "keywords" : "complexity";
    }
  }

  if (!bestPreset) {
    return { roles: [], matched: false };
  }

  const roles = presets[bestPreset]?.roles ?? [];
  return {
    preset: bestPreset,
    roles,
    matched: true,
    matchReason: bestReason,
  };
}

// ---------------------------------------------------------------------------
// Task-aware model selection
// ---------------------------------------------------------------------------

/** Simple code/technical markers for detecting coding tasks. */
const CODE_TASK_MARKERS = [
  /\b(code|implement|refactor|debug|fix bug|compile|build|test|lint)\b/i,
  /\b(function|class|module|api|endpoint|database|query|migration)\b/i,
  /```/,
  /\.(ts|js|py|go|rs|java|cpp|swift)\b/,
];

export function resolveTaskAwareModel(params: {
  task: string;
  complexity: AutoRoutingComplexity;
  cfg: MarvConfig;
  hasImages?: boolean;
  agentId?: string;
  /** Auto-trigger config from matched preset (may override pool/thinking). */
  presetAutoTrigger?: {
    modelPool?: string;
    thinking?: string;
  };
}): { model?: string; thinking?: string; reason: string } {
  const { task, complexity, cfg, hasImages } = params;

  // Determine which pool to use.
  const poolOverrideCfg = params.presetAutoTrigger?.modelPool
    ? {
        ...cfg,
        agents: {
          ...cfg.agents,
          defaults: {
            ...cfg.agents?.defaults,
            modelPool: params.presetAutoTrigger.modelPool,
          },
        },
      }
    : cfg;

  // Build requirements based on task.
  const requiredCaps: string[] = [];
  if (hasImages) {
    requiredCaps.push("vision");
  }
  const isCodingTask = CODE_TASK_MARKERS.some((re) => re.test(task));
  if (isCodingTask) {
    requiredCaps.push("coding");
  }

  // Resolve pool candidates.
  const plan = resolveRuntimeModelPlan({
    cfg: poolOverrideCfg,
    agentId: params.agentId,
    requirements:
      requiredCaps.length > 0
        ? { requiredCapabilities: requiredCaps as ("text" | "vision" | "coding" | "tools")[] }
        : undefined,
  });

  if (plan.candidates.length === 0) {
    return {
      model: undefined,
      thinking: resolveThinkingForComplexity(complexity, params.presetAutoTrigger?.thinking),
      reason: "no candidates in pool",
    };
  }

  // Re-sort candidates based on complexity preference.
  const sorted = reorderForComplexity(plan.candidates, complexity);
  const selected = sorted[0];

  const thinking = resolveThinkingForComplexity(complexity, params.presetAutoTrigger?.thinking);
  const reasons: string[] = [];
  reasons.push(`complexity=${complexity}`);
  if (isCodingTask) {
    reasons.push("coding-task");
  }
  if (hasImages) {
    reasons.push("vision-required");
  }
  reasons.push(`pool=${plan.poolName}`);
  reasons.push(`selected=${selected.ref}`);
  reasons.push(`tier=${selected.tier}`);

  return {
    model: selected.ref,
    thinking,
    reason: reasons.join(", "),
  };
}

/**
 * Re-order candidates based on complexity:
 * - simple/moderate: reverse pool sort so low-tier (cheaper/faster) comes first
 * - complex/expert: keep default pool sort (high-tier first — most capable)
 *
 * The pool's default sort order is high → standard → low (descending tier weight).
 */
function reorderForComplexity(
  candidates: RuntimeConfiguredModel[],
  complexity: AutoRoutingComplexity,
): RuntimeConfiguredModel[] {
  if (complexity === "simple" || complexity === "moderate") {
    // Reverse the default tier sort so low-tier (cheaper) comes first.
    return [...candidates].toReversed();
  }
  // For complex/expert, pool's default sort already puts high-tier first.
  return candidates;
}

/**
 * Map complexity to a recommended thinking level.
 * Explicit override (from preset autoTrigger) takes precedence.
 */
function resolveThinkingForComplexity(
  complexity: AutoRoutingComplexity,
  override?: string,
): string | undefined {
  if (override) {
    return override;
  }
  switch (complexity) {
    case "expert":
      return "high";
    case "complex":
      return "medium";
    case "moderate":
      return "low";
    case "simple":
      return "off";
    default:
      return undefined;
  }
}
