import { filterTaskContextEntriesForReplyPreferences } from "../../agents/context-pollution.js";
import { querySoulMemoryMulti } from "../storage/soul-memory-store.js";
import { listTaskDecisionBookmarks } from "./bookmark.js";
import { estimateTextTokens } from "./compressor.js";
import { getTaskContextRollingSummary } from "./state.js";
import { getTaskContext, listTaskContextEntries } from "./store.js";
import type { TaskContextEntry } from "./types.js";

export type TaskContextMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type TaskContextBuildResult = {
  messages: TaskContextMessage[];
  layers: {
    p0Memory: string[];
    rollingSummary?: string;
    keyDecisions: string[];
    recentEntries: TaskContextEntry[];
    currentQuery: string;
    toolContext?: string;
  };
  tokenUsage: {
    p0Memory: number;
    rollingSummary: number;
    keyDecisions: number;
    recentEntries: number;
    currentQuery: number;
    toolContext: number;
    total: number;
    budget: number;
  };
};

const DEFAULT_BUDGET = 128_000;
const DEFAULT_LAYER_BUDGETS = {
  p0Memory: 4_000,
  rollingSummary: 4_000,
  keyDecisions: 4_000,
  recentEntries: 96_000,
  currentAndTools: 20_000,
};

export function buildTaskContextWindow(params: {
  agentId: string;
  taskId: string;
  currentQuery: string;
  toolContext?: string;
  totalBudgetTokens?: number;
  layerBudgets?: Partial<{
    p0Memory: number;
    rollingSummary: number;
    keyDecisions: number;
    recentEntries: number;
    currentAndTools: number;
  }>;
  env?: NodeJS.ProcessEnv;
}): TaskContextBuildResult {
  const task = getTaskContext({
    agentId: params.agentId,
    taskId: params.taskId,
    env: params.env,
  });
  const scopeId = task?.scopeId ?? `task:${params.agentId}:${params.taskId}`;
  const currentQuery = params.currentQuery.trim();
  const toolContext = params.toolContext?.trim();

  const budgets = {
    p0Memory: Math.max(
      500,
      Math.floor(params.layerBudgets?.p0Memory ?? DEFAULT_LAYER_BUDGETS.p0Memory),
    ),
    rollingSummary: Math.max(
      500,
      Math.floor(params.layerBudgets?.rollingSummary ?? DEFAULT_LAYER_BUDGETS.rollingSummary),
    ),
    keyDecisions: Math.max(
      500,
      Math.floor(params.layerBudgets?.keyDecisions ?? DEFAULT_LAYER_BUDGETS.keyDecisions),
    ),
    recentEntries: Math.max(
      1_000,
      Math.floor(params.layerBudgets?.recentEntries ?? DEFAULT_LAYER_BUDGETS.recentEntries),
    ),
    currentAndTools: Math.max(
      500,
      Math.floor(params.layerBudgets?.currentAndTools ?? DEFAULT_LAYER_BUDGETS.currentAndTools),
    ),
  };

  const totalBudget = Math.max(2_000, Math.floor(params.totalBudgetTokens ?? DEFAULT_BUDGET));

  const p0Memories = querySoulMemoryMulti({
    agentId: params.agentId,
    scopes: [
      { scopeType: "task", scopeId, weight: 1 },
      { scopeType: "agent", scopeId: params.agentId, weight: 0.85 },
    ],
    query: currentQuery || "task context",
    topK: 12,
    minScore: 0.5,
  })
    .map((item) => item.content)
    .filter(Boolean);
  const p0Memory = clampTextListByTokens(p0Memories, budgets.p0Memory);

  const rollingSummaryRaw =
    getTaskContextRollingSummary({
      agentId: params.agentId,
      taskId: params.taskId,
      env: params.env,
    }) ?? "";
  const rollingSummary = clampTextByTokens(rollingSummaryRaw, budgets.rollingSummary);

  const keyDecisionTexts = listTaskDecisionBookmarks({
    agentId: params.agentId,
    taskId: params.taskId,
    limit: 80,
    env: params.env,
  }).map((bookmark) => bookmark.content);
  const keyDecisions = clampTextListByTokens(keyDecisionTexts, budgets.keyDecisions);

  const allEntries = listTaskContextEntries({
    agentId: params.agentId,
    taskId: params.taskId,
    limit: 5000,
    env: params.env,
  });
  const sanitizedEntries = filterTaskContextEntriesForReplyPreferences(allEntries).entries;
  const recentEntries = selectRecentEntriesByTokenBudget(sanitizedEntries, budgets.recentEntries);

  const currentAndTools = [currentQuery, toolContext].filter(Boolean).join("\n\n").trim();
  const currentAndToolsClamped = clampTextByTokens(currentAndTools, budgets.currentAndTools);

  const messages: TaskContextMessage[] = [];
  if (p0Memory.length > 0) {
    messages.push({
      role: "system",
      content: `Recalled Memory:\n${p0Memory.map((item) => `- ${item}`).join("\n")}`,
    });
  }
  if (rollingSummary) {
    messages.push({
      role: "system",
      content: `Task Summary:\n${rollingSummary}`,
    });
  }
  if (keyDecisions.length > 0) {
    messages.push({
      role: "system",
      content: `Key Decisions:\n${keyDecisions.map((item) => `- ${item}`).join("\n")}`,
    });
  }
  for (const entry of recentEntries) {
    const role = entry.role === "user" ? "user" : "assistant";
    const content = entry.role === "tool" ? `[tool]\n${entry.content}` : entry.content;
    messages.push({ role, content });
  }
  if (currentAndToolsClamped) {
    messages.push({ role: "user", content: currentAndToolsClamped });
  }

  const tokenUsage = {
    p0Memory: sumTextTokens(p0Memory),
    rollingSummary: estimateTextTokens(rollingSummary),
    keyDecisions: sumTextTokens(keyDecisions),
    recentEntries: recentEntries.reduce((sum, entry) => sum + Math.max(1, entry.tokenCount), 0),
    currentQuery: estimateTextTokens(currentQuery),
    toolContext: estimateTextTokens(toolContext ?? ""),
    total: 0,
    budget: totalBudget,
  };
  tokenUsage.total =
    tokenUsage.p0Memory +
    tokenUsage.rollingSummary +
    tokenUsage.keyDecisions +
    tokenUsage.recentEntries +
    tokenUsage.currentQuery +
    tokenUsage.toolContext;

  if (tokenUsage.total > totalBudget) {
    const trimmedRecent = selectRecentEntriesByTokenBudget(
      recentEntries,
      Math.max(500, budgets.recentEntries - (tokenUsage.total - totalBudget)),
    );
    const trimmedMessages = messages.filter(
      (message, index) =>
        !(
          index >= messages.length - recentEntries.length - 1 &&
          index < messages.length - 1 &&
          message.role !== "system"
        ),
    );
    for (const entry of trimmedRecent) {
      trimmedMessages.push({
        role: entry.role === "user" ? "user" : "assistant",
        content: entry.role === "tool" ? `[tool]\n${entry.content}` : entry.content,
      });
    }
    if (currentAndToolsClamped) {
      trimmedMessages.push({ role: "user", content: currentAndToolsClamped });
    }
    return {
      messages: trimmedMessages,
      layers: {
        p0Memory,
        rollingSummary: rollingSummary || undefined,
        keyDecisions,
        recentEntries: trimmedRecent,
        currentQuery,
        toolContext,
      },
      tokenUsage: {
        ...tokenUsage,
        recentEntries: trimmedRecent.reduce((sum, entry) => sum + Math.max(1, entry.tokenCount), 0),
        total:
          tokenUsage.p0Memory +
          tokenUsage.rollingSummary +
          tokenUsage.keyDecisions +
          trimmedRecent.reduce((sum, entry) => sum + Math.max(1, entry.tokenCount), 0) +
          tokenUsage.currentQuery +
          tokenUsage.toolContext,
      },
    };
  }

  return {
    messages,
    layers: {
      p0Memory,
      rollingSummary: rollingSummary || undefined,
      keyDecisions,
      recentEntries,
      currentQuery,
      toolContext,
    },
    tokenUsage,
  };
}

export function buildTaskContextPrelude(params: {
  agentId: string;
  taskId: string;
  currentQuery: string;
  toolContext?: string;
  env?: NodeJS.ProcessEnv;
}): string {
  const built = buildTaskContextWindow(params);
  const preludeParts: string[] = [];
  if (built.layers.rollingSummary?.trim()) {
    preludeParts.push(`Task Summary:\n${built.layers.rollingSummary.trim()}`);
  }
  if (built.layers.keyDecisions.length > 0) {
    preludeParts.push(
      `Key Decisions:\n${built.layers.keyDecisions.map((line) => `- ${line}`).join("\n")}`,
    );
  }
  if (built.layers.p0Memory.length > 0) {
    preludeParts.push(
      `Memory Constraints:\n${built.layers.p0Memory.map((line) => `- ${line}`).join("\n")}`,
    );
  }
  return preludeParts.join("\n\n").trim();
}

function selectRecentEntriesByTokenBudget(
  entries: TaskContextEntry[],
  budget: number,
): TaskContextEntry[] {
  const maxBudget = Math.max(1, Math.floor(budget));
  let used = 0;
  const selected: TaskContextEntry[] = [];
  for (let i = entries.length - 1; i >= 0; i -= 1) {
    const entry = entries[i];
    const entryTokens = Math.max(1, entry?.tokenCount ?? 1);
    if (used + entryTokens > maxBudget && selected.length > 0) {
      break;
    }
    if (!entry) {
      continue;
    }
    selected.unshift(entry);
    used += entryTokens;
  }
  return selected;
}

function clampTextByTokens(input: string, tokenBudget: number): string {
  const tokens = estimateTextTokens(input);
  if (tokens <= tokenBudget) {
    return input.trim();
  }
  const chars = Math.max(0, tokenBudget * 4);
  return input.trim().slice(0, chars).trim();
}

function clampTextListByTokens(values: string[], tokenBudget: number): string[] {
  const out: string[] = [];
  let used = 0;
  for (const value of values) {
    const normalized = value.trim();
    if (!normalized) {
      continue;
    }
    const tokens = estimateTextTokens(normalized);
    if (used + tokens > tokenBudget && out.length > 0) {
      break;
    }
    out.push(normalized);
    used += tokens;
  }
  return out;
}

function sumTextTokens(values: string[]): number {
  return values.reduce((sum, value) => sum + estimateTextTokens(value), 0);
}
