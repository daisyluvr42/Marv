import fsSync from "node:fs";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { listAgentIds } from "../../agents/agent-scope.js";
import type { MarvConfig } from "../../core/config/config.js";
import type { DeepConsolidationModelConfig } from "../../core/config/types.memory.js";
import { normalizeAgentId } from "../../routing/session-key.js";
import {
  type ResolvedLocalLlmConfig,
  inferLocal,
  resolveLocalLlmConfig,
} from "./local-llm-client.js";
import {
  buildSoulMemoryConflictPairKey,
  detectSoulMemoryConflicts,
  listSoulMemoryConflictCandidates,
} from "./soul-memory-conflict.js";
import {
  buildConsolidationClusterKey,
  buildSimilarityClusters,
  consolidateSoulMemories,
  groupByScopeAndKind,
  listSoulMemoryConsolidationItems,
} from "./soul-memory-consolidation.js";
import { resolveSoulMemoryDbPath, writeSoulMemory } from "./soul-memory-store.js";
import { requireNodeSqlite } from "./sqlite.js";

export const DEFAULT_DEEP_CONSOLIDATION_SCHEDULE = "20 4 * * 0";
const DEFAULT_MAX_ITEMS = 500;
const DEFAULT_MAX_REFLECTIONS = 5;
const MIN_CLUSTER_SIZE = 3;
const NO_CONFLICT_SENTINEL = "NO_CONFLICT";
const NO_PATTERN_SENTINEL = "NO_PATTERN";
const CLUSTER_SUMMARY_SYSTEM_PROMPT =
  "Given a cluster of related memories, produce one concise summary of the shared stable pattern. Output only the summary and keep it under 200 characters.";
const CONFLICT_JUDGMENT_SYSTEM_PROMPT =
  "Given two memories, decide whether they genuinely conflict. If they do, output one very short reason. If they do not, output NO_CONFLICT.";
const CROSS_SCOPE_REFLECTION_SYSTEM_PROMPT =
  "Given memories from different contexts, identify one useful cross-cutting pattern or insight. Output only the insight under 250 characters. If none, output NO_PATTERN.";

type ReflectionRow = {
  id: string;
  scope_type: string;
  scope_id: string;
  kind: string;
  content: string;
  confidence: number;
  tier: string;
};

type ReflectionItem = {
  id: string;
  scopeType: string;
  scopeId: string;
  kind: string;
  content: string;
  confidence: number;
  tier: string;
};

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

export async function runSoulMemoryDeepConsolidation(params: {
  cfg: MarvConfig;
  nowMs?: number;
  agentId?: string;
  resolvedModel?: ResolvedLocalLlmConfig;
}): Promise<DeepConsolidationReport> {
  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  const config = resolveDeepConsolidationConfig(params.cfg);
  const resolvedModel =
    params.resolvedModel ?? resolveLocalLlmConfig({ cfg: params.cfg, model: config.model });
  const agentIds = resolveAgentIds(params.cfg, params.agentId);
  const agents: DeepConsolidationPerAgent[] = [];

  let llmConsolidated = 0;
  let llmConflictsDetected = 0;
  let crossScopeReflections = 0;
  let failedAgents = 0;

  for (const agentId of agentIds) {
    const skippedStages: string[] = [];
    const errors: string[] = [];
    let consolidatedCount = 0;
    let conflictCount = 0;
    let reflectionCount = 0;

    if (config.clusterSummarization) {
      try {
        const result = await runClusterSummarizationStage({
          cfg: params.cfg,
          agentId,
          nowMs,
          maxItems: config.maxItems,
          model: resolvedModel,
        });
        consolidatedCount = result.count;
        if (result.skippedReason) {
          skippedStages.push(result.skippedReason);
        }
      } catch (err) {
        errors.push(`clusterSummarization: ${err instanceof Error ? err.message : String(err)}`);
      }
    } else {
      skippedStages.push("clusterSummarization-disabled");
    }

    if (config.conflictJudgment) {
      try {
        const result = await runConflictJudgmentStage({
          cfg: params.cfg,
          agentId,
          nowMs,
          maxItems: config.maxItems,
          model: resolvedModel,
        });
        conflictCount = result.count;
        if (result.skippedReason) {
          skippedStages.push(result.skippedReason);
        }
      } catch (err) {
        errors.push(`conflictJudgment: ${err instanceof Error ? err.message : String(err)}`);
      }
    } else {
      skippedStages.push("conflictJudgment-disabled");
    }

    if (config.crossScopeReflection) {
      try {
        const result = await runCrossScopeReflectionStage({
          cfg: params.cfg,
          agentId,
          nowMs,
          maxItems: config.maxItems,
          maxReflections: config.maxReflections,
          model: resolvedModel,
        });
        reflectionCount = result.count;
        if (result.skippedReason) {
          skippedStages.push(result.skippedReason);
        }
      } catch (err) {
        errors.push(`crossScopeReflection: ${err instanceof Error ? err.message : String(err)}`);
      }
    } else {
      skippedStages.push("crossScopeReflection-disabled");
    }

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
    llmConsolidated += consolidatedCount;
    llmConflictsDetected += conflictCount;
    crossScopeReflections += reflectionCount;
    agents.push({
      agentId,
      llmConsolidated: consolidatedCount,
      llmConflictsDetected: conflictCount,
      crossScopeReflections: reflectionCount,
      skippedStages,
      error: errors.length > 0 ? errors.join("; ") : undefined,
    });
  }

  return {
    agents,
    totals: {
      llmConsolidated,
      llmConflictsDetected,
      crossScopeReflections,
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

async function runClusterSummarizationStage(params: {
  cfg: MarvConfig;
  agentId: string;
  nowMs: number;
  maxItems: number;
  model: ResolvedLocalLlmConfig;
}): Promise<{ count: number; skippedReason?: string }> {
  const items = listSoulMemoryConsolidationItems({
    agentId: params.agentId,
    limit: params.maxItems,
  });
  if (items.length < MIN_CLUSTER_SIZE) {
    return { count: 0, skippedReason: "clusterSummarization-no-candidates" };
  }

  const summaries = new Map<string, string>();
  const grouped = groupByScopeAndKind(items);
  for (const group of grouped.values()) {
    if (group.length < MIN_CLUSTER_SIZE) {
      continue;
    }
    const clusters = buildSimilarityClusters(group, {
      minSimilarity: 0.6,
      maxSimilarity: 0.9,
    });
    for (const cluster of clusters) {
      if (cluster.length < MIN_CLUSTER_SIZE) {
        continue;
      }
      const prompt = buildClusterSummarizationPrompt(cluster);
      const result = await inferLocal({
        cfg: params.cfg,
        model: params.model,
        system: CLUSTER_SUMMARY_SYSTEM_PROMPT,
        prompt,
      });
      if (!result.ok) {
        continue;
      }
      const summary = normalizeGeneratedText(result.text, {
        maxChars: 200,
      });
      if (!summary || summary.length < 12) {
        continue;
      }
      summaries.set(buildConsolidationClusterKey(cluster), summary);
    }
  }

  if (summaries.size === 0) {
    return { count: 0, skippedReason: "clusterSummarization-no-llm-output" };
  }

  const result = consolidateSoulMemories({
    agentId: params.agentId,
    nowMs: params.nowMs,
    maxItems: params.maxItems,
    summarizeCluster: (input) => summaries.get(buildConsolidationClusterKey(input.items)) ?? "",
  });
  return {
    count: result.generalizedCount,
    skippedReason: result.generalizedCount > 0 ? undefined : "clusterSummarization-no-new-memory",
  };
}

async function runConflictJudgmentStage(params: {
  cfg: MarvConfig;
  agentId: string;
  nowMs: number;
  maxItems: number;
  model: ResolvedLocalLlmConfig;
}): Promise<{ count: number; skippedReason?: string }> {
  const candidates = listSoulMemoryConflictCandidates({
    agentId: params.agentId,
    minConfidence: 0.7,
    overlapThreshold: 0.2,
    unresolvedOnly: true,
  })
    .filter((candidate) => !candidate.ruleBasedConflictReason)
    .slice(0, params.maxItems);
  if (candidates.length === 0) {
    return { count: 0, skippedReason: "conflictJudgment-no-candidates" };
  }

  const judgments = new Map<string, string | null>();
  for (const candidate of candidates) {
    const result = await inferLocal({
      cfg: params.cfg,
      model: params.model,
      system: CONFLICT_JUDGMENT_SYSTEM_PROMPT,
      prompt: buildConflictJudgmentPrompt(candidate.left.content, candidate.right.content),
    });
    if (!result.ok) {
      continue;
    }
    const normalized = normalizeConflictReason(result.text);
    judgments.set(candidate.pairKey, normalized);
  }

  if (judgments.size === 0) {
    return { count: 0, skippedReason: "conflictJudgment-no-llm-output" };
  }

  const conflictResult = detectSoulMemoryConflicts({
    agentId: params.agentId,
    nowMs: params.nowMs,
    judgeConflictPair: ({ pairKey }) => judgments.get(pairKey) ?? null,
  });
  const llmDetected = conflictResult.conflicts.filter((conflict) => {
    const pairKey = buildSoulMemoryConflictPairKey(conflict.memoryIdA, conflict.memoryIdB);
    return judgments.get(pairKey) != null;
  }).length;
  return {
    count: llmDetected,
    skippedReason: llmDetected > 0 ? undefined : "conflictJudgment-no-new-conflicts",
  };
}

async function runCrossScopeReflectionStage(params: {
  cfg: MarvConfig;
  agentId: string;
  nowMs: number;
  maxItems: number;
  maxReflections: number;
  model: ResolvedLocalLlmConfig;
}): Promise<{ count: number; skippedReason?: string }> {
  const candidateItems = listReflectionItems(params.agentId, params.maxItems);
  if (candidateItems.length < 2) {
    return { count: 0, skippedReason: "crossScopeReflection-no-candidates" };
  }
  const grouped = groupReflectionItems(candidateItems);
  if (grouped.length < 2) {
    return { count: 0, skippedReason: "crossScopeReflection-single-scope" };
  }

  const existingInsights = new Set(listExistingCrossScopeInsights(params.agentId));
  let inserted = 0;
  const maxPairs = Math.max(params.maxReflections * 3, params.maxReflections);
  const pairCandidates = buildScopePairCandidates(grouped, maxPairs);
  for (const pair of pairCandidates) {
    if (inserted >= params.maxReflections) {
      break;
    }
    const result = await inferLocal({
      cfg: params.cfg,
      model: params.model,
      system: CROSS_SCOPE_REFLECTION_SYSTEM_PROMPT,
      prompt: buildCrossScopeReflectionPrompt(pair.left, pair.right),
    });
    if (!result.ok) {
      continue;
    }
    const insight = normalizeReflectionInsight(result.text);
    if (!insight || existingInsights.has(normalizeStoredInsight(insight))) {
      continue;
    }
    const written = writeSoulMemory({
      agentId: params.agentId,
      scopeType: "global",
      scopeId: "cross-scope",
      kind: "insight",
      content: insight,
      recordKind: "fact",
      source: "auto_extraction",
      confidence: 0.58,
      metadata: {
        generatedBy: "deep-consolidation",
        stage: "cross-scope-reflection",
        scopes: [pair.left.scopeKey, pair.right.scopeKey],
      },
      nowMs: params.nowMs,
    });
    if (!written) {
      continue;
    }
    inserted += 1;
    existingInsights.add(normalizeStoredInsight(insight));
  }

  return {
    count: inserted,
    skippedReason: inserted > 0 ? undefined : "crossScopeReflection-no-new-insights",
  };
}

function buildClusterSummarizationPrompt(
  items: Array<{
    kind: string;
    content: string;
  }>,
): string {
  const lines = items
    .slice(0, 6)
    .map((item, index) => `${index + 1}. (${item.kind}) ${item.content.trim()}`);
  return `Cluster memories:\n${lines.join("\n")}`;
}

function buildConflictJudgmentPrompt(left: string, right: string): string {
  return `Memory A:\n${left.trim()}\n\nMemory B:\n${right.trim()}`;
}

function buildCrossScopeReflectionPrompt(
  left: { scopeKey: string; items: ReflectionItem[] },
  right: { scopeKey: string; items: ReflectionItem[] },
): string {
  const leftLines = left.items
    .slice(0, 3)
    .map((item, index) => `${index + 1}. (${item.kind}) ${item.content.trim()}`);
  const rightLines = right.items
    .slice(0, 3)
    .map((item, index) => `${index + 1}. (${item.kind}) ${item.content.trim()}`);
  return [
    `Context A: ${left.scopeKey}`,
    leftLines.join("\n"),
    "",
    `Context B: ${right.scopeKey}`,
    rightLines.join("\n"),
  ].join("\n");
}

function normalizeGeneratedText(
  value: string,
  opts: {
    maxChars: number;
  },
): string {
  const normalized = value
    .replace(/^["'`]+/, "")
    .replace(/["'`]+$/, "")
    .replace(/\s+/g, " ")
    .trim();
  if (!normalized) {
    return "";
  }
  return normalized.length <= opts.maxChars
    ? normalized
    : normalized.slice(0, opts.maxChars).trim();
}

function normalizeConflictReason(value: string): string | null {
  const normalized = normalizeGeneratedText(value, { maxChars: 120 });
  if (!normalized) {
    return null;
  }
  return normalized.toUpperCase() === NO_CONFLICT_SENTINEL ? null : normalized;
}

function normalizeReflectionInsight(value: string): string | null {
  const normalized = normalizeGeneratedText(value, { maxChars: 250 });
  if (!normalized) {
    return null;
  }
  return normalized.toUpperCase() === NO_PATTERN_SENTINEL ? null : normalized;
}

function normalizeStoredInsight(value: string): string {
  return value.toLowerCase().replace(/\s+/g, " ").trim();
}

function groupReflectionItems(items: ReflectionItem[]): Array<{
  scopeKey: string;
  items: ReflectionItem[];
}> {
  const grouped = new Map<string, ReflectionItem[]>();
  for (const item of items) {
    if (item.scopeType === "global" && item.scopeId === "cross-scope" && item.kind === "insight") {
      continue;
    }
    const scopeKey = `${item.scopeType}:${item.scopeId}`;
    const bucket = grouped.get(scopeKey) ?? [];
    if (bucket.length < 3) {
      bucket.push(item);
    }
    grouped.set(scopeKey, bucket);
  }
  return Array.from(grouped.entries())
    .map(([scopeKey, bucket]) => ({ scopeKey, items: bucket }))
    .filter((entry) => entry.items.length > 0)
    .toSorted((left, right) => {
      const leftScore = Math.max(...left.items.map((item) => item.confidence));
      const rightScore = Math.max(...right.items.map((item) => item.confidence));
      return rightScore - leftScore;
    });
}

function buildScopePairCandidates(
  grouped: Array<{
    scopeKey: string;
    items: ReflectionItem[];
  }>,
  limit: number,
): Array<{
  left: { scopeKey: string; items: ReflectionItem[] };
  right: { scopeKey: string; items: ReflectionItem[] };
}> {
  const pairs: Array<{
    left: { scopeKey: string; items: ReflectionItem[] };
    right: { scopeKey: string; items: ReflectionItem[] };
  }> = [];
  for (let i = 0; i < grouped.length; i += 1) {
    const left = grouped[i];
    if (!left) {
      continue;
    }
    for (let j = i + 1; j < grouped.length; j += 1) {
      const right = grouped[j];
      if (!right) {
        continue;
      }
      pairs.push({ left, right });
      if (pairs.length >= limit) {
        return pairs;
      }
    }
  }
  return pairs;
}

function listReflectionItems(agentId: string, limit: number): ReflectionItem[] {
  const db = openSoulMemoryDb(agentId);
  try {
    const rows = db
      .prepare(
        "SELECT id, scope_type, scope_id, kind, content, confidence, tier " +
          "FROM memory_items WHERE confidence >= 0.7 ORDER BY confidence DESC, created_at DESC LIMIT ?",
      )
      .all(limit) as ReflectionRow[];
    return rows.map((row) => ({
      id: row.id,
      scopeType: row.scope_type,
      scopeId: row.scope_id,
      kind: row.kind,
      content: row.content,
      confidence: Number(row.confidence ?? 0),
      tier: row.tier,
    }));
  } finally {
    db.close();
  }
}

function listExistingCrossScopeInsights(agentId: string): string[] {
  const db = openSoulMemoryDb(agentId);
  try {
    const rows = db
      .prepare(
        "SELECT content FROM memory_items WHERE scope_type = 'global' AND scope_id = 'cross-scope' AND kind = 'insight'",
      )
      .all() as Array<{ content?: string }>;
    return rows
      .map((row) => normalizeStoredInsight(typeof row.content === "string" ? row.content : ""))
      .filter(Boolean);
  } finally {
    db.close();
  }
}

function openSoulMemoryDb(agentId: string): DatabaseSync {
  const dbPath = resolveSoulMemoryDbPath(agentId);
  fsSync.mkdirSync(path.dirname(dbPath), { recursive: true });
  const { DatabaseSync } = requireNodeSqlite();
  return new DatabaseSync(dbPath);
}
