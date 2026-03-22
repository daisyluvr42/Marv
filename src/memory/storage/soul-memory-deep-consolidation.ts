import { listAgentIds } from "../../agents/agent-scope.js";
import type { MarvConfig } from "../../core/config/config.js";
import type { DeepConsolidationModelConfig } from "../../core/config/types.memory.js";
import { normalizeAgentId } from "../../routing/session-key.js";
import { type ResolvedLocalLlmConfig, resolveLocalLlmConfig } from "./local-llm-client.js";

export const DEFAULT_DEEP_CONSOLIDATION_SCHEDULE = "20 4 * * 0";

export type ResolvedDeepConsolidationConfig = {
  enabled: boolean;
  schedule: string;
  maxItems: number;
  maxReflections: number;
  clusterSummarization: boolean;
  conflictJudgment: boolean;
  crossScopeReflection: boolean;
  model: DeepConsolidationModelConfig;
};

export type DeepConsolidationPerAgent = {
  agentId: string;
  llmConsolidated: number;
  llmConflictsDetected: number;
  crossScopeReflections: number;
  skippedStages: string[];
  error?: string;
};

export type DeepConsolidationReport = {
  agents: DeepConsolidationPerAgent[];
  totals: {
    llmConsolidated: number;
    llmConflictsDetected: number;
    crossScopeReflections: number;
  };
  model: {
    api: "ollama" | "openai-completions";
    baseUrl: string;
    model: string;
    available: boolean;
  };
  failedAgents: number;
};

const DEFAULT_MAX_ITEMS = 500;
const DEFAULT_MAX_REFLECTIONS = 5;

export function resolveDeepConsolidationConfig(cfg: MarvConfig): ResolvedDeepConsolidationConfig {
  const raw = cfg.memory?.soul?.deepConsolidation;
  return {
    enabled: raw?.enabled === true,
    schedule: raw?.schedule?.trim() || DEFAULT_DEEP_CONSOLIDATION_SCHEDULE,
    maxItems:
      typeof raw?.maxItems === "number" && Number.isFinite(raw.maxItems) && raw.maxItems > 0
        ? Math.floor(raw.maxItems)
        : DEFAULT_MAX_ITEMS,
    maxReflections:
      typeof raw?.maxReflections === "number" &&
      Number.isFinite(raw.maxReflections) &&
      raw.maxReflections > 0
        ? Math.floor(raw.maxReflections)
        : DEFAULT_MAX_REFLECTIONS,
    clusterSummarization: raw?.clusterSummarization !== false,
    conflictJudgment: raw?.conflictJudgment !== false,
    crossScopeReflection: raw?.crossScopeReflection !== false,
    model: raw?.model ?? {},
  };
}

/**
 * Weekly deep consolidation run.
 *
 * The old 3-stage LLM pipeline (cluster summarization, conflict judgment,
 * cross-scope reflection) has been replaced by EXPERIENCE.md weekly calibration.
 * P3 index maintenance (dedupe → consolidate → compact) runs via the standard
 * maintenance pipeline in soul-memory-maintenance.ts.
 */
export async function runSoulMemoryDeepConsolidation(params: {
  cfg: MarvConfig;
  nowMs?: number;
  agentId?: string;
  resolvedModel?: ResolvedLocalLlmConfig;
}): Promise<DeepConsolidationReport> {
  const config = resolveDeepConsolidationConfig(params.cfg);
  const resolvedModel =
    params.resolvedModel ?? resolveLocalLlmConfig({ cfg: params.cfg, model: config.model });
  const agentIds = resolveAgentIds(params.cfg, params.agentId);
  const agents: DeepConsolidationPerAgent[] = [];

  let failedAgents = 0;

  for (const agentId of agentIds) {
    const skippedStages: string[] = [];
    const errors: string[] = [];

    // Old 3-stage deep consolidation replaced by EXPERIENCE.md weekly calibration
    skippedStages.push(
      "clusterSummarization-replaced-by-experience-calibration",
      "conflictJudgment-replaced-by-experience-calibration",
      "crossScopeReflection-replaced-by-experience-calibration",
    );

    // EXPERIENCE.md weekly calibration (attribution-driven culling)
    try {
      const { weeklyCalibration } = await import("../experience/experience-rebuild.js");
      await weeklyCalibration({
        agentId,
        cfg: params.cfg,
      });
    } catch (err) {
      errors.push(`experienceCalibration: ${err instanceof Error ? err.message : String(err)}`);
    }

    if (errors.length > 0) {
      failedAgents += 1;
    }
    agents.push({
      agentId,
      llmConsolidated: 0,
      llmConflictsDetected: 0,
      crossScopeReflections: 0,
      skippedStages,
      error: errors.length > 0 ? errors.join("; ") : undefined,
    });
  }

  return {
    agents,
    totals: {
      llmConsolidated: 0,
      llmConflictsDetected: 0,
      crossScopeReflections: 0,
    },
    model: {
      api: resolvedModel.api,
      baseUrl: resolvedModel.baseUrl,
      model: resolvedModel.model,
      available: true,
    },
    failedAgents,
  };
}

export function formatSoulMemoryDeepConsolidationSummary(report: DeepConsolidationReport): string {
  return (
    "Deep consolidation complete: " +
    `agents=${report.agents.length}, ` +
    `failed=${report.failedAgents}, ` +
    `consolidated=${report.totals.llmConsolidated}, ` +
    `conflicts=${report.totals.llmConflictsDetected}, ` +
    `reflections=${report.totals.crossScopeReflections}`
  );
}

function resolveAgentIds(cfg: MarvConfig, agentId?: string): string[] {
  const single = agentId?.trim();
  if (single) {
    return [normalizeAgentId(single)];
  }
  return listAgentIds(cfg).map((entry) => normalizeAgentId(entry));
}
