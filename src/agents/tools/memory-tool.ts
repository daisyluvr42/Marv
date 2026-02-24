import { Type } from "@sinclair/typebox";
import type { OpenClawConfig } from "../../config/config.js";
import type { MemoryCitationsMode } from "../../config/types.memory.js";
import { appendLedgerEvent } from "../../ledger/event-store.js";
import { resolveMemoryBackendConfig } from "../../memory/backend-config.js";
import { getMemorySearchManager } from "../../memory/index.js";
import {
  buildSoulMemoryPath,
  getSoulMemoryItem,
  parseSoulMemoryPath,
  querySoulMemoryMulti,
  writeSoulMemory,
  type SoulMemoryItem,
  type SoulMemoryQueryResult,
  type SoulMemoryScope,
} from "../../memory/soul-memory-store.js";
import type { MemorySearchResult } from "../../memory/types.js";
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

function resolveMemoryToolContext(options: { config?: OpenClawConfig; agentSessionKey?: string }) {
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

export function createMemorySearchTool(options: {
  config?: OpenClawConfig;
  agentSessionKey?: string;
}): AnyAgentTool | null {
  const ctx = resolveMemoryToolContext(options);
  if (!ctx) {
    return null;
  }
  const { cfg, agentId } = ctx;
  return {
    label: "Memory Search",
    name: "memory_search",
    description:
      "Mandatory recall step: search structured soul memory first (with scope/tier/confidence), then merge legacy memory files when needed. Use before answering questions about prior work, decisions, dates, people, preferences, or todos.",
    parameters: MemorySearchSchema,
    execute: async (_toolCallId, params) => {
      const query = readStringParam(params, "query", { required: true });
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
        query,
        topK: maxResults,
        minScore: minScore ?? undefined,
      }).map((entry) => toSoulSearchResult(entry));

      const managerResult = await getMemorySearchManager({
        cfg,
        agentId,
      });
      const manager = managerResult.manager;

      let managerResults: MemorySearchResult[] = [];
      let managerStatus: ReturnType<NonNullable<typeof manager>["status"]> | undefined;
      if (manager && soulResults.length < maxResults) {
        try {
          managerResults = await manager.search(query, {
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
        primary: soulResults,
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
      const usedSoul = merged.some((entry) => isSoulMemoryPath(entry.path));
      const mode = usedSoul ? (managerMode ? `soul+${managerMode}` : "soul") : managerMode;
      return jsonResult({
        results: finalResults,
        provider: managerStatus?.provider ?? "soul-memory",
        model: managerStatus?.model ?? "deterministic-v1",
        fallback: managerStatus?.fallback,
        citations: citationsMode,
        mode,
      });
    },
  };
}

export function createMemoryGetTool(options: {
  config?: OpenClawConfig;
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
      "Safe snippet read by memory path. Works for structured soul-memory entries and legacy MEMORY.md/memory/*.md paths. Use after memory_search to fetch only the minimum needed context.",
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
        return jsonResult({
          path: relPath,
          text: sliceTextByLines(item.content, from ?? undefined, lines ?? undefined),
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
  config?: OpenClawConfig;
  agentSessionKey?: string;
}): AnyAgentTool | null {
  const ctx = resolveMemoryToolContext(options);
  if (!ctx) {
    return null;
  }
  const { agentId } = ctx;
  return {
    label: "Memory Write",
    name: "memory_write",
    description:
      "Write a durable structured memory entry (scope/kind/source/confidence) into soul memory. Prefer this over editing MEMORY.md or memory/*.md.",
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

      const item = writeSoulMemory({
        agentId,
        scopeType,
        scopeId,
        kind,
        content,
        confidence: confidence ?? undefined,
        source: source ?? undefined,
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
  const snippet = truncateUtf16Safe(entry.content.trim(), 700);
  return {
    path,
    startLine: 1,
    endLine,
    score: entry.score,
    snippet,
    // Keep compatibility with existing MemorySource union.
    source: "memory",
  };
}

function isSoulMemoryPath(pathname: string): boolean {
  return pathname.trim().toLowerCase().startsWith("soul-memory/");
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
    if (isSoulMemoryPath(a.path) && !isSoulMemoryPath(b.path)) {
      return -1;
    }
    if (!isSoulMemoryPath(a.path) && isSoulMemoryPath(b.path)) {
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

function resolveMemoryCitationsMode(cfg: OpenClawConfig): MemoryCitationsMode {
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
