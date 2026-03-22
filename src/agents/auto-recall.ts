import path from "node:path";
import type { MarvConfig } from "../core/config/config.js";
import { syncConfiguredKnowledgeBases } from "../knowledge/indexer.js";
import {
  querySoulArchive,
  querySoulMemoryMulti,
  type SoulArchiveQueryResult,
  type SoulMemoryQueryResult,
} from "../memory/storage/soul-memory-store.js";
import { resolveSoulScopes } from "./memory-soul-scopes.js";
import type { EmbeddedContextFile } from "./runner/pi-embedded-helpers.js";

export type AutoRecallConfig = {
  enabled?: boolean;
  maxResults?: number;
  minScore?: number;
  maxContextChars?: number;
  includeConversationContext?: boolean;
};

export type AutoRecallResult = {
  contextFile: EmbeddedContextFile | null;
  queryUsed: string;
  resultCount: number;
  durationMs: number;
};

const DEFAULT_AUTO_RECALL_CONFIG: Required<AutoRecallConfig> = {
  enabled: true,
  maxResults: 8,
  minScore: 0.3,
  maxContextChars: 8_000,
  includeConversationContext: true,
};

type RecallEntry =
  | { kind: "memory"; score: number; item: SoulMemoryQueryResult }
  | { kind: "archive"; score: number; item: SoulArchiveQueryResult };

export function resolveAutoRecallConfig(cfg?: MarvConfig): Required<AutoRecallConfig> {
  const raw = cfg?.memory?.autoRecall;
  return {
    enabled: raw?.enabled ?? DEFAULT_AUTO_RECALL_CONFIG.enabled,
    maxResults: clampInt(raw?.maxResults, DEFAULT_AUTO_RECALL_CONFIG.maxResults, 1, 16),
    minScore: clampNumber(raw?.minScore, DEFAULT_AUTO_RECALL_CONFIG.minScore, 0, 1.5),
    maxContextChars: clampInt(
      raw?.maxContextChars,
      DEFAULT_AUTO_RECALL_CONFIG.maxContextChars,
      500,
      20_000,
    ),
    includeConversationContext:
      raw?.includeConversationContext ?? DEFAULT_AUTO_RECALL_CONFIG.includeConversationContext,
  };
}

export async function autoRecallFromSoulMemory(params: {
  agentId: string;
  sessionKey?: string;
  incomingMessage: string;
  recentContext?: string[];
  config?: AutoRecallConfig;
  marvConfig?: MarvConfig;
}): Promise<AutoRecallResult> {
  const startedAt = Date.now();
  const config = {
    ...DEFAULT_AUTO_RECALL_CONFIG,
    ...params.config,
  };
  const queryUsed = buildAutoRecallQuery({
    incomingMessage: params.incomingMessage,
    recentContext: config.includeConversationContext ? params.recentContext : undefined,
  });
  if (!config.enabled || !queryUsed) {
    return {
      contextFile: null,
      queryUsed,
      resultCount: 0,
      durationMs: Date.now() - startedAt,
    };
  }

  await syncConfiguredKnowledgeBases({
    agentId: params.agentId,
    config: params.marvConfig,
  });

  const scopes = resolveSoulScopes({
    agentId: params.agentId,
    sessionKey: params.sessionKey,
  });
  const active = querySoulMemoryMulti({
    agentId: params.agentId,
    scopes,
    query: queryUsed,
    topK: config.maxResults,
    minScore: config.minScore,
  });
  const archive = querySoulArchive({
    agentId: params.agentId,
    scopes,
    query: queryUsed,
    topK: config.maxResults,
    minScore: Math.max(0.12, config.minScore * 0.6),
  });
  const merged = mergeRecallEntries(active, archive, config.maxResults);
  const contextFile = buildRecalledContextFile(merged, config.maxContextChars);
  return {
    contextFile,
    queryUsed,
    resultCount: merged.length,
    durationMs: Date.now() - startedAt,
  };
}

function buildAutoRecallQuery(params: {
  incomingMessage: string;
  recentContext?: string[];
}): string {
  const parts = [
    params.incomingMessage.trim(),
    ...(params.recentContext ?? []).map((entry) => entry.trim()).filter(Boolean),
  ].filter(Boolean);
  if (parts.length === 0) {
    return "";
  }
  return parts.join("\n\n").slice(0, 512).trim();
}

function mergeRecallEntries(
  active: SoulMemoryQueryResult[],
  archive: SoulArchiveQueryResult[],
  maxResults: number,
): RecallEntry[] {
  const merged: RecallEntry[] = [
    ...active.map((item) => ({ kind: "memory" as const, score: adjustedRecallScore(item), item })),
    ...archive.map((item) => ({ kind: "archive" as const, score: item.score, item })),
  ].toSorted((a, b) => b.score - a.score);

  const dedup = new Set<string>();
  const results: RecallEntry[] = [];
  for (const entry of merged) {
    const content = entry.item.content.trim().toLowerCase().replace(/\s+/g, " ");
    if (!content || dedup.has(content)) {
      continue;
    }
    dedup.add(content);
    results.push(entry);
    if (results.length >= maxResults) {
      break;
    }
  }
  return results;
}

function buildRecalledContextFile(
  entries: RecallEntry[],
  maxContextChars: number,
): EmbeddedContextFile | null {
  if (entries.length === 0) {
    return null;
  }
  const sections: string[] = [
    "The following memories were automatically recalled for the current conversation.",
    "Use them as context, but do not quote them verbatim unless needed.",
    "",
  ];
  let used = sections.join("\n").length;

  for (const entry of entries) {
    const block = formatRecallEntry(entry);
    if (!block) {
      continue;
    }
    const next = `${block}\n`;
    if (used + next.length > maxContextChars) {
      break;
    }
    sections.push(block, "");
    used += next.length;
  }

  const content = sections.join("\n").trim();
  if (!content) {
    return null;
  }
  return {
    path: path.join("/", "virtual", "RECALLED_CONTEXT.md"),
    content,
  };
}

function formatRecallEntry(entry: RecallEntry): string | null {
  const text = entry.item.content.trim();
  if (!text) {
    return null;
  }
  const title =
    entry.kind === "memory"
      ? entry.item.scopeType === "document"
        ? describeDocumentEntry(entry.item.metadata)
        : entry.item.kind
      : entry.item.kind;
  const confidence =
    entry.kind === "memory"
      ? clampNumber(entry.item.score, 0, 0, 9_999).toFixed(2)
      : clampNumber(entry.item.score, 0, 0, 9_999).toFixed(2);
  const source = entry.kind === "memory" ? "memory" : "archive";
  return [`### ${title} (score: ${confidence}, source: ${source})`, text].join("\n");
}

function describeDocumentEntry(metadata: Record<string, unknown> | undefined): string {
  const relativePath =
    typeof metadata?.relativePath === "string" ? metadata.relativePath.trim() : "document";
  const heading = typeof metadata?.heading === "string" ? metadata.heading.trim() : "";
  return heading ? `[doc] ${relativePath} > ${heading}` : `[doc] ${relativePath}`;
}

function adjustedRecallScore(item: SoulMemoryQueryResult): number {
  if (item.scopeType !== "document") {
    return item.score;
  }
  return item.score * 0.85;
}

function clampInt(value: unknown, fallback: number, min: number, max: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }
  const numeric = Math.floor(value);
  if (numeric < min || numeric > max) {
    return fallback;
  }
  return numeric;
}

function clampNumber(value: unknown, fallback: number, min: number, max: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }
  if (value < min || value > max) {
    return fallback;
  }
  return value;
}
