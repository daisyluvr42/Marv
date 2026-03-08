import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../core/config/config.js";
import type { MemoryCitationsMode } from "../../core/config/types.memory.js";
import { appendLedgerEvent } from "../../ledger/event-store.js";
import { resolveMemoryBackendConfig } from "../../memory/backend-config.js";
import { getMemorySearchManager } from "../../memory/index.js";
import { SMALL_TALK_RE } from "../../memory/small-talk.js";
import {
  buildSoulMemoryPath,
  buildSoulArchivePath,
  getSoulMemoryItem,
  getSoulArchiveEvent,
  listSoulMemoryReferences,
  parseSoulArchivePath,
  parseSoulMemoryPath,
  querySoulArchive,
  querySoulMemoryMulti,
  type SoulMemoryConfig,
  writeSoulMemory,
  type SoulMemoryItem,
  type SoulMemoryQueryResult,
  type SoulArchiveQueryResult,
  type SoulMemoryScope,
} from "../../memory/storage/soul-memory-store.js";
import type { MemorySearchResult } from "../../memory/types.js";
import { evaluateMemoryWriteHeuristics } from "../../memory/write-heuristics.js";
import { parseAgentSessionKey } from "../../routing/session-key.js";
import { truncateUtf16Safe } from "../../utils.js";
import { resolveSessionAgentId } from "../agent-scope.js";
import { resolveMemorySearchConfig } from "../memory-search.js";
import type { AnyAgentTool } from "./common.js";
import { jsonResult, readNumberParam, readStringParam } from "./common.js";

const MemorySearchSchema = Type.Object({
  query: Type.String(),
  maxResults: Type.Optional(Type.Number()),
  minScore: Type.Optional(Type.Number()),
});

const MemoryGetSchema = Type.Object({
  path: Type.String(),
  from: Type.Optional(Type.Number()),
  lines: Type.Optional(Type.Number()),
});

const MemoryWriteSchema = Type.Object({
  content: Type.String(),
  kind: Type.Optional(Type.String()),
  scopeType: Type.Optional(Type.String()),
  scopeId: Type.Optional(Type.String()),
  confidence: Type.Optional(Type.Number()),
  source: Type.Optional(Type.String()),
});

function resolveMemoryToolContext(options: { config?: MarvConfig; agentSessionKey?: string }) {
  const cfg = options.config;
  if (!cfg) {
    return null;
  }
  const agentId = resolveSessionAgentId({
    sessionKey: options.agentSessionKey,
    config: cfg,
  });
  if (!resolveMemorySearchConfig(cfg, agentId)) {
    return null;
  }
  return { cfg, agentId };
}

function resolveMemorySoulConfig(cfg: MarvConfig): SoulMemoryConfig | undefined {
  const memoryConfig = cfg.memory;
  if (!memoryConfig) {
    return undefined;
  }
  const legacyP0AllowedKinds = Array.isArray(memoryConfig.p0AllowedKinds)
    ? memoryConfig.p0AllowedKinds
    : undefined;
  if (!memoryConfig.soul) {
    return legacyP0AllowedKinds ? { p0AllowedKinds: legacyP0AllowedKinds } : undefined;
  }
  if (memoryConfig.soul.p0AllowedKinds || !legacyP0AllowedKinds) {
    return memoryConfig.soul;
  }
  return {
    ...memoryConfig.soul,
    p0AllowedKinds: legacyP0AllowedKinds,
  };
}

export function createMemorySearchTool(options: {
  config?: MarvConfig;
  agentSessionKey?: string;
}): AnyAgentTool | null {
  const ctx = resolveMemoryToolContext(options);
  if (!ctx) {
    return null;
  }
  const { cfg, agentId } = ctx;
  const soulConfig = resolveMemorySoulConfig(cfg);
  const memorySearchCfg = resolveMemorySearchConfig(cfg, agentId);
  if (!memorySearchCfg) {
    return null;
  }
  return {
    label: "Memory Search",
    name: "memory_search",
    description:
      "Mandatory recall step: search structured memory first across active tiers and archived history, then use legacy Markdown search only as compatibility fallback. Use before answering questions about prior work, decisions, dates, people, preferences, or todos.",
    parameters: MemorySearchSchema,
    execute: async (_toolCallId, params) => {
      const query = readStringParam(params, "query", { required: true });
      const precheckDecision = evaluateMemorySearchPrecheck({
        query,
        cfg: memorySearchCfg,
      });
      if (!precheckDecision.shouldSearch) {
        return jsonResult({
          results: [],
          skipped: true,
          reason: precheckDecision.reason,
          mode: "precheck_skip",
        });
      }
      const effectiveQuery = precheckDecision.query;
      const requestedMaxResults = readNumberParam(params, "maxResults", { integer: true });
      const maxResults = Math.max(1, Math.min(32, Math.trunc(requestedMaxResults ?? 6)));
      const minScore = readNumberParam(params, "minScore");
      const citationsMode = resolveMemoryCitationsMode(cfg);
      const includeCitations = shouldIncludeCitations({
        mode: citationsMode,
        sessionKey: options.agentSessionKey,
      });

      const soulResults = querySoulMemoryMulti({
        agentId,
        scopes: resolveSoulScopes({ agentId, sessionKey: options.agentSessionKey }),
        query: effectiveQuery,
        topK: maxResults,
        minScore: minScore ?? undefined,
        soulConfig,
      }).map((entry) => toSoulSearchResult(entry));
      const archiveResults = querySoulArchive({
        agentId,
        scopes: resolveSoulScopes({ agentId, sessionKey: options.agentSessionKey }),
        query: effectiveQuery,
        topK: maxResults,
        minScore: Math.max(0.12, minScore ?? 0.12),
      }).map((entry) => toSoulArchiveSearchResult(entry));

      const managerResult = await getMemorySearchManager({
        cfg,
        agentId,
      });
      const manager = managerResult.manager;

      let managerResults: MemorySearchResult[] = [];
      let managerStatus: ReturnType<NonNullable<typeof manager>["status"]> | undefined;
      if (manager && soulResults.length + archiveResults.length < maxResults) {
        try {
          managerResults = await manager.search(effectiveQuery, {
            maxResults,
            minScore: minScore ?? undefined,
            sessionKey: options.agentSessionKey,
          });
          managerStatus = manager.status();
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          if (soulResults.length === 0) {
            return jsonResult({ results: [], disabled: true, error: message });
          }
        }
      } else if (manager) {
        managerStatus = manager.status();
      }

      const merged = mergeMemoryResults({
        primary: [...soulResults, ...archiveResults],
        secondary: managerResults,
        maxResults,
      });
      if (merged.length === 0 && !manager) {
        return jsonResult({
          results: [],
          disabled: true,
          error: managerResult.error ?? "memory search unavailable",
        });
      }

      const decorated = decorateCitations(merged, includeCitations);
      const resolved = resolveMemoryBackendConfig({ cfg, agentId });
      const finalResults =
        managerStatus?.backend === "qmd"
          ? clampResultsByInjectedChars(decorated, resolved.qmd?.limits.maxInjectedChars)
          : decorated;
      const managerMode = (managerStatus?.custom as { searchMode?: string } | undefined)
        ?.searchMode;
      const usedStructured = merged.some((entry) => isStructuredMemoryPath(entry.path));
      const mode = usedStructured
        ? managerMode
          ? `structured+${managerMode}`
          : "structured"
        : managerMode;
      return jsonResult({
        results: finalResults,
        provider: managerStatus?.provider ?? "soul-memory",
        model: managerStatus?.model ?? "deterministic-v1",
        fallback: managerStatus?.fallback,
        citations: citationsMode,
        mode,
        rewrittenQuery: precheckDecision.rewritten ? effectiveQuery : undefined,
      });
    },
  };
}

type MemorySearchPrecheckDecision = {
  shouldSearch: boolean;
  reason: "disabled" | "memory-cue" | "general-query" | "small-talk" | "query-too-short";
  query: string;
  rewritten: boolean;
};

function evaluateMemorySearchPrecheck(params: {
  query: string;
  cfg: ReturnType<typeof resolveMemorySearchConfig>;
}): MemorySearchPrecheckDecision {
  const normalized = params.query.trim().replace(/\s+/g, " ");
  const precheck = params.cfg?.query.precheck;
  if (!precheck?.enabled) {
    return {
      shouldSearch: true,
      reason: "disabled",
      query: normalized,
      rewritten: false,
    };
  }
  const lowered = normalized.toLowerCase();
  if (SMALL_TALK_QUERY_RE.test(lowered)) {
    return {
      shouldSearch: false,
      reason: "small-talk",
      query: normalized,
      rewritten: false,
    };
  }
  const hasMemoryCue = MEMORY_CUE_RE.test(lowered);
  if (!hasMemoryCue && normalized.length < precheck.minQueryChars) {
    return {
      shouldSearch: false,
      reason: "query-too-short",
      query: normalized,
      rewritten: false,
    };
  }
  const rewrittenQuery = precheck.rewrite ? rewriteMemorySearchQuery(normalized) : normalized;
  return {
    shouldSearch: true,
    reason: hasMemoryCue ? "memory-cue" : "general-query",
    query: rewrittenQuery,
    rewritten: rewrittenQuery !== normalized,
  };
}

// Imported via shared small-talk module; kept as alias for search precheck.
const SMALL_TALK_QUERY_RE = SMALL_TALK_RE;
const MEMORY_CUE_RE =
  /\b(remember|recall|previous|earlier|history|past|before|last time|last week|last month|what did we|we discussed|my preference|my preferences|todo|decision|decisions|profile|habit|habits|identity|policy|guardrail)\b/i;

function rewriteMemorySearchQuery(query: string): string {
  let rewritten = query
    .replace(/^(please|can you|could you|would you|kindly)\s+/i, "")
    .replace(/^(do you\s+)?(remember|recall)\s+/i, "")
    .replace(/[?]+\s*$/g, "")
    .trim();
  if (!rewritten) {
    return query;
  }
  const lowered = rewritten.toLowerCase();
  if (
    /\b(what|which|when|where|who|how)\b/.test(lowered) &&
    /\b(we|i|my|our|me)\b/.test(lowered) &&
    !/\b(previous|earlier|history|memory|remember|recall)\b/.test(lowered)
  ) {
    rewritten = `${rewritten} from prior conversations and saved memory`;
  }
  return rewritten;
}

export function createMemoryGetTool(options: {
  config?: MarvConfig;
  agentSessionKey?: string;
}): AnyAgentTool | null {
  const ctx = resolveMemoryToolContext(options);
  if (!ctx) {
    return null;
  }
  const { cfg, agentId } = ctx;
  return {
    label: "Memory Get",
    name: "memory_get",
    description:
      "Safe snippet read by memory path. Works for structured active/archive memory entries and legacy MEMORY.md/memory/*.md paths. Use after memory_search to fetch only the minimum needed context.",
    parameters: MemoryGetSchema,
    execute: async (_toolCallId, params) => {
      const relPath = readStringParam(params, "path", { required: true });
      const from = readNumberParam(params, "from", { integer: true });
      const lines = readNumberParam(params, "lines", { integer: true });
      const soulId = parseSoulMemoryPath(relPath);
      if (soulId) {
        const item = getSoulMemoryItem({ agentId, itemId: soulId });
        if (!item) {
          return jsonResult({
            path: relPath,
            text: "",
            disabled: true,
            error: "path required",
          });
        }
        const references = listSoulMemoryReferences({ agentId, itemId: soulId });
        return jsonResult({
          path: relPath,
          text: sliceTextByLines(item.content, from ?? undefined, lines ?? undefined),
          references,
        });
      }
      const archiveId = parseSoulArchivePath(relPath);
      if (archiveId) {
        const item = getSoulArchiveEvent({ agentId, eventId: archiveId });
        if (!item) {
          return jsonResult({
            path: relPath,
            text: "",
            disabled: true,
            error: "path required",
          });
        }
        return jsonResult({
          path: relPath,
          text: sliceTextByLines(item.content, from ?? undefined, lines ?? undefined),
          summary: item.summary,
        });
      }

      const { manager, error } = await getMemorySearchManager({
        cfg,
        agentId,
      });
      if (!manager) {
        return jsonResult({ path: relPath, text: "", disabled: true, error });
      }
      try {
        const result = await manager.readFile({
          relPath,
          from: from ?? undefined,
          lines: lines ?? undefined,
        });
        return jsonResult(result);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return jsonResult({ path: relPath, text: "", disabled: true, error: message });
      }
    },
  };
}

export function createMemoryWriteTool(options: {
  config?: MarvConfig;
  agentSessionKey?: string;
}): AnyAgentTool | null {
  const ctx = resolveMemoryToolContext(options);
  if (!ctx) {
    return null;
  }
  const { agentId, cfg } = ctx;
  const soulConfig = resolveMemorySoulConfig(cfg);
  return {
    label: "Memory Write",
    name: "memory_write",
    description:
      "Write a durable structured memory entry (scope/kind/source/confidence) into structured memory. Prefer this over editing MEMORY.md or memory/*.md.",
    parameters: MemoryWriteSchema,
    execute: async (_toolCallId, params) => {
      const content = readStringParam(params, "content", { required: true });
      const kind = readStringParam(params, "kind") ?? "note";
      const scopeTypeRaw = readStringParam(params, "scopeType");
      const scopeIdRaw = readStringParam(params, "scopeId");
      const source = readStringParam(params, "source");
      const confidence = readNumberParam(params, "confidence");

      const fallbackScope = resolveDefaultWriteScope({
        agentId,
        sessionKey: options.agentSessionKey,
      });
      const scopeType = (scopeTypeRaw ?? fallbackScope.scopeType).toLowerCase().trim();
      const scopeId = (scopeIdRaw ?? fallbackScope.scopeId).toLowerCase().trim();

      if (!scopeType || !scopeId) {
        return jsonResult({
          ok: false,
          error: "scopeType and scopeId required",
        });
      }

      const heuristics = evaluateMemoryWriteHeuristics({
        content,
        kind,
      });
      if (!heuristics.shouldWrite) {
        return jsonResult({
          ok: false,
          skipped: true,
          classification: heuristics.classification,
          error: "memory write skipped by heuristics",
        });
      }

      const item = writeSoulMemory({
        agentId,
        scopeType,
        scopeId,
        kind,
        content: heuristics.normalizedContent,
        confidence: confidence ?? undefined,
        source: source ?? undefined,
        soulConfig,
      });
      if (!item) {
        return jsonResult({
          ok: false,
          error: "content required",
        });
      }
      appendMemoryWriteLedgerEvent({
        item,
        agentId,
        sessionKey: options.agentSessionKey,
      });
      const references = listSoulMemoryReferences({ agentId, itemId: item.id });
      return jsonResult({
        ok: true,
        id: item.id,
        path: buildSoulMemoryPath(item.id),
        scopeType: item.scopeType,
        scopeId: item.scopeId,
        kind: item.kind,
        tier: item.tier,
        source: item.source,
        confidence: item.confidence,
        classification: heuristics.classification,
        references,
      });
    },
  };
}

function resolveDefaultWriteScope(params: { agentId: string; sessionKey?: string }): {
  scopeType: string;
  scopeId: string;
} {
  const parsed = parseAgentSessionKey(params.sessionKey);
  if (parsed?.rest) {
    return { scopeType: "session", scopeId: parsed.rest };
  }
  return { scopeType: "agent", scopeId: params.agentId };
}

function resolveSoulScopes(params: { agentId: string; sessionKey?: string }): SoulMemoryScope[] {
  const scopes: SoulMemoryScope[] = [{ scopeType: "agent", scopeId: params.agentId, weight: 1 }];
  const parsed = parseAgentSessionKey(params.sessionKey);
  if (!parsed?.rest) {
    return scopes;
  }
  scopes.unshift({
    scopeType: "session",
    scopeId: parsed.rest,
    weight: 1.15,
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
    if (channel && peerId && (kind === "group" || kind === "channel")) {
      scopes.push({
        scopeType: "channel",
        scopeId: `${channel}:${peerId}`,
        weight: 0.9,
      });
    }
  }
  return dedupeScopes(scopes);
}

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

function toSoulSearchResult(entry: SoulMemoryQueryResult): MemorySearchResult {
  const path = buildSoulMemoryPath(entry.id);
  const lines = entry.content.split("\n");
  const endLine = Math.max(1, lines.length);
  const refSuffix =
    entry.references.length > 0
      ? `\n\nRefs: ${entry.references.map((refId) => `[ref:${refId}]`).join(" ")}`
      : "";
  const snippet = truncateUtf16Safe(`${entry.content.trim()}${refSuffix}`, 700);
  return {
    path,
    startLine: 1,
    endLine,
    score: entry.score,
    snippet,
    salienceScore: entry.salienceScore,
    salienceDecay: entry.salienceDecay,
    salienceReinforcement: entry.salienceReinforcement,
    referenceBoost: entry.referenceBoost,
    references: entry.references,
    // Keep compatibility with existing MemorySource union.
    source: "memory",
  };
}

function toSoulArchiveSearchResult(entry: SoulArchiveQueryResult): MemorySearchResult {
  const path = buildSoulArchivePath(entry.id);
  const snippet = truncateUtf16Safe(`${entry.summary}\n\n${entry.content.trim()}`, 700);
  return {
    path,
    startLine: 1,
    endLine: Math.max(1, entry.content.split("\n").length),
    score: entry.score,
    snippet,
    source: "memory",
  };
}

function isSoulMemoryPath(pathname: string): boolean {
  return pathname.trim().toLowerCase().startsWith("soul-memory/");
}

function isSoulArchivePath(pathname: string): boolean {
  return pathname.trim().toLowerCase().startsWith("soul-archive/");
}

function isStructuredMemoryPath(pathname: string): boolean {
  return isSoulMemoryPath(pathname) || isSoulArchivePath(pathname);
}

function mergeMemoryResults(params: {
  primary: MemorySearchResult[];
  secondary: MemorySearchResult[];
  maxResults: number;
}): MemorySearchResult[] {
  const combined = [...params.primary, ...params.secondary].toSorted((a, b) => {
    const scoreDelta = (b.score ?? 0) - (a.score ?? 0);
    if (Math.abs(scoreDelta) > 1e-9) {
      return scoreDelta;
    }
    if (isStructuredMemoryPath(a.path) && !isStructuredMemoryPath(b.path)) {
      return -1;
    }
    if (!isStructuredMemoryPath(a.path) && isStructuredMemoryPath(b.path)) {
      return 1;
    }
    return a.path.localeCompare(b.path);
  });
  const out: MemorySearchResult[] = [];
  const seen = new Set<string>();
  for (const entry of combined) {
    const key = `${entry.path}:${entry.startLine}:${entry.endLine}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(entry);
    if (out.length >= params.maxResults) {
      break;
    }
  }
  return out;
}

function sliceTextByLines(text: string, from?: number, lines?: number): string {
  if (!from && !lines) {
    return text;
  }
  const items = text.split("\n");
  const start = Math.max(1, Math.trunc(from ?? 1));
  const count = Math.max(1, Math.trunc(lines ?? items.length));
  return items.slice(start - 1, start - 1 + count).join("\n");
}

function resolveMemoryCitationsMode(cfg: MarvConfig): MemoryCitationsMode {
  const mode = cfg.memory?.citations;
  if (mode === "on" || mode === "off" || mode === "auto") {
    return mode;
  }
  return "auto";
}

function decorateCitations(results: MemorySearchResult[], include: boolean): MemorySearchResult[] {
  if (!include) {
    return results.map((entry) => ({ ...entry, citation: undefined }));
  }
  return results.map((entry) => {
    const citation = formatCitation(entry);
    const snippet = `${entry.snippet.trim()}\n\nSource: ${citation}`;
    return { ...entry, citation, snippet };
  });
}

function formatCitation(entry: MemorySearchResult): string {
  const lineRange =
    entry.startLine === entry.endLine
      ? `#L${entry.startLine}`
      : `#L${entry.startLine}-L${entry.endLine}`;
  return `${entry.path}${lineRange}`;
}

function clampResultsByInjectedChars(
  results: MemorySearchResult[],
  budget?: number,
): MemorySearchResult[] {
  if (!budget || budget <= 0) {
    return results;
  }
  let remaining = budget;
  const clamped: MemorySearchResult[] = [];
  for (const entry of results) {
    if (remaining <= 0) {
      break;
    }
    const snippet = entry.snippet ?? "";
    if (snippet.length <= remaining) {
      clamped.push(entry);
      remaining -= snippet.length;
    } else {
      const trimmed = snippet.slice(0, Math.max(0, remaining));
      clamped.push({ ...entry, snippet: trimmed });
      break;
    }
  }
  return clamped;
}

function shouldIncludeCitations(params: {
  mode: MemoryCitationsMode;
  sessionKey?: string;
}): boolean {
  if (params.mode === "on") {
    return true;
  }
  if (params.mode === "off") {
    return false;
  }
  // auto: show citations in direct chats; suppress in groups/channels by default.
  const chatType = deriveChatTypeFromSessionKey(params.sessionKey);
  return chatType === "direct";
}

function deriveChatTypeFromSessionKey(sessionKey?: string): "direct" | "group" | "channel" {
  const parsed = parseAgentSessionKey(sessionKey);
  if (!parsed?.rest) {
    return "direct";
  }
  const tokens = new Set(parsed.rest.toLowerCase().split(":").filter(Boolean));
  if (tokens.has("channel")) {
    return "channel";
  }
  if (tokens.has("group")) {
    return "group";
  }
  return "direct";
}

function appendMemoryWriteLedgerEvent(params: {
  item: SoulMemoryItem;
  agentId: string;
  sessionKey?: string;
}): void {
  try {
    appendLedgerEvent({
      conversationId: params.sessionKey?.trim() || `memory:agent:${params.agentId}`,
      type: "MemoryWrittenEvent",
      payload: {
        memoryId: params.item.id,
        scopeType: params.item.scopeType,
        scopeId: params.item.scopeId,
        kind: params.item.kind,
        tier: params.item.tier,
        source: params.item.source,
        confidence: params.item.confidence,
      },
    });
  } catch {
    // Ledger writes are best-effort and must not break memory_write.
  }
}
