import {
  getTaskContextRollingSummary,
  listUnsummarizedTaskContextEntries,
  markTaskContextEntriesSummarized,
  setTaskContextRollingSummary,
} from "./state.js";
import { listTaskContextEntries } from "./store.js";
import type { TaskContextEntry } from "./types.js";

export type TaskSummaryBatch = {
  existingSummary?: string;
  entries: TaskContextEntry[];
  nowMs: number;
};

export type TaskSummaryGenerator = (batch: TaskSummaryBatch) => Promise<string> | string;

export type TaskCompressionResult = {
  compressed: boolean;
  reason?: string;
  summarizedEntries: number;
  summarizedTokens: number;
  batchSummary?: string;
  rollingSummary?: string;
};

export async function compressTaskContext(params: {
  agentId: string;
  taskId: string;
  recentTokenThreshold?: number;
  batchTokenTarget?: number;
  keepRecentTurns?: number;
  nowMs?: number;
  summaryGenerator?: TaskSummaryGenerator;
  env?: NodeJS.ProcessEnv;
}): Promise<TaskCompressionResult> {
  const recentTokenThreshold = Math.max(2000, Math.floor(params.recentTokenThreshold ?? 60_000));
  const batchTokenTarget = Math.max(500, Math.floor(params.batchTokenTarget ?? 20_000));
  const keepRecentTurns = Math.max(2, Math.floor(params.keepRecentTurns ?? 8));
  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();

  const unsummarized = listUnsummarizedTaskContextEntries({
    agentId: params.agentId,
    taskId: params.taskId,
    limit: 5000,
    env: params.env,
  });
  if (unsummarized.length === 0) {
    return {
      compressed: false,
      reason: "no-unsummarized-entries",
      summarizedEntries: 0,
      summarizedTokens: 0,
      rollingSummary:
        getTaskContextRollingSummary({
          agentId: params.agentId,
          taskId: params.taskId,
          env: params.env,
        }) ?? undefined,
    };
  }

  const recentEntries = unsummarized.slice(-keepRecentTurns);
  const recentTokens = sumTokens(recentEntries);
  if (recentTokens <= recentTokenThreshold) {
    return {
      compressed: false,
      reason: "below-threshold",
      summarizedEntries: 0,
      summarizedTokens: 0,
      rollingSummary:
        getTaskContextRollingSummary({
          agentId: params.agentId,
          taskId: params.taskId,
          env: params.env,
        }) ?? undefined,
    };
  }

  const candidates = selectCompressionBatch({
    entries: unsummarized,
    batchTokenTarget,
    keepRecentTurns,
  });
  if (candidates.length === 0) {
    return {
      compressed: false,
      reason: "no-eligible-batch",
      summarizedEntries: 0,
      summarizedTokens: 0,
      rollingSummary:
        getTaskContextRollingSummary({
          agentId: params.agentId,
          taskId: params.taskId,
          env: params.env,
        }) ?? undefined,
    };
  }

  const existingSummary =
    getTaskContextRollingSummary({
      agentId: params.agentId,
      taskId: params.taskId,
      env: params.env,
    }) ?? undefined;
  const batchSummaryRaw = await resolveSummaryGenerator(params.summaryGenerator)({
    existingSummary,
    entries: candidates,
    nowMs,
  });
  const batchSummary = normalizeSummary(batchSummaryRaw);
  if (!batchSummary) {
    return {
      compressed: false,
      reason: "empty-summary",
      summarizedEntries: 0,
      summarizedTokens: 0,
      rollingSummary: existingSummary,
    };
  }

  const rollingSummary = mergeRollingSummary(existingSummary, batchSummary);
  const updated = markTaskContextEntriesSummarized({
    agentId: params.agentId,
    taskId: params.taskId,
    entryIds: candidates.map((entry) => entry.id),
    summary: batchSummary,
    env: params.env,
  });
  setTaskContextRollingSummary({
    agentId: params.agentId,
    taskId: params.taskId,
    summary: rollingSummary,
    updatedAt: nowMs,
    env: params.env,
  });

  return {
    compressed: updated > 0,
    summarizedEntries: updated,
    summarizedTokens: sumTokens(candidates),
    batchSummary,
    rollingSummary,
  };
}

export function buildHeuristicBatchSummary(batch: TaskSummaryBatch): string {
  const lines: string[] = [];
  for (const entry of batch.entries) {
    const compact = compactLine(entry.content);
    if (!compact) {
      continue;
    }
    const rolePrefix =
      entry.role === "user"
        ? "User"
        : entry.role === "assistant"
          ? "Assistant"
          : entry.role === "tool"
            ? "Tool"
            : "System";
    lines.push(`- ${rolePrefix}: ${compact}`);
    if (lines.length >= 18) {
      break;
    }
  }
  if (lines.length === 0) {
    return "";
  }
  return lines.join("\n");
}

export function estimateTextTokens(input: string): number {
  return Math.max(1, Math.ceil(input.trim().length / 4));
}

function selectCompressionBatch(params: {
  entries: TaskContextEntry[];
  batchTokenTarget: number;
  keepRecentTurns: number;
}): TaskContextEntry[] {
  const maxIndex = Math.max(0, params.entries.length - params.keepRecentTurns);
  if (maxIndex <= 0) {
    return [];
  }
  const candidates = params.entries.slice(0, maxIndex);
  const selected: TaskContextEntry[] = [];
  let tokens = 0;
  for (const entry of candidates) {
    selected.push(entry);
    tokens += Math.max(1, entry.tokenCount);
    if (tokens >= params.batchTokenTarget) {
      break;
    }
  }
  return selected;
}

function mergeRollingSummary(existingSummary: string | undefined, batchSummary: string): string {
  if (!existingSummary?.trim()) {
    return batchSummary.trim();
  }
  const merged = `${existingSummary.trim()}\n${batchSummary.trim()}`.trim();
  const maxTokens = 4000;
  if (estimateTextTokens(merged) <= maxTokens) {
    return merged;
  }
  const lines = merged
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  let acc = 0;
  const kept: string[] = [];
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const line = lines[i];
    const lineTokens = estimateTextTokens(line);
    if (acc + lineTokens > maxTokens) {
      break;
    }
    kept.unshift(line);
    acc += lineTokens;
  }
  return kept.join("\n");
}

function resolveSummaryGenerator(
  generator: TaskSummaryGenerator | undefined,
): TaskSummaryGenerator {
  if (generator) {
    return generator;
  }
  return (batch) => buildHeuristicBatchSummary(batch);
}

function sumTokens(entries: TaskContextEntry[]): number {
  return entries.reduce((sum, entry) => sum + Math.max(1, entry.tokenCount), 0);
}

function normalizeSummary(input: string): string {
  return input
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .join("\n")
    .trim();
}

function compactLine(input: string): string {
  const normalized = input.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return "";
  }
  return normalized.length > 180 ? `${normalized.slice(0, 177)}...` : normalized;
}

export async function maybeCompressTaskContext(params: {
  agentId: string;
  taskId: string;
  thresholdTokens?: number;
  env?: NodeJS.ProcessEnv;
}): Promise<TaskCompressionResult> {
  const entries = listTaskContextEntries({
    agentId: params.agentId,
    taskId: params.taskId,
    limit: 2000,
    env: params.env,
  });
  if (entries.length === 0) {
    return {
      compressed: false,
      reason: "no-entries",
      summarizedEntries: 0,
      summarizedTokens: 0,
    };
  }
  const recentTokens = sumTokens(entries.slice(-8));
  const threshold = Math.max(1000, Math.floor(params.thresholdTokens ?? 60_000));
  if (recentTokens <= threshold) {
    return {
      compressed: false,
      reason: "below-threshold",
      summarizedEntries: 0,
      summarizedTokens: 0,
      rollingSummary:
        getTaskContextRollingSummary({
          agentId: params.agentId,
          taskId: params.taskId,
          env: params.env,
        }) ?? undefined,
    };
  }
  return await compressTaskContext({
    agentId: params.agentId,
    taskId: params.taskId,
    recentTokenThreshold: threshold,
    env: params.env,
  });
}
