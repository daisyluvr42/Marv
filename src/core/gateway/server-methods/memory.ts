import { resolveSoulScopes } from "../../../agents/memory-soul-scopes.js";
import {
  listSoulMemoryItems,
  querySoulMemoryMulti,
  type SoulMemoryRecordKind,
  type SoulMemoryScope,
} from "../../../memory/storage/soul-memory-store.js";
import {
  ErrorCodes,
  errorShape,
  formatValidationErrors,
  validateMemoryListParams,
  validateMemorySearchParams,
} from "../protocol/index.js";
import type { GatewayRequestHandlers } from "./types.js";
import { resolveWorkspaceAgent } from "./workspace-agent.js";

function sanitizeScopes(scopes: unknown): SoulMemoryScope[] | null {
  if (!Array.isArray(scopes)) {
    return null;
  }
  return scopes
    .map((scope) => {
      if (!scope || typeof scope !== "object") {
        return null;
      }
      const candidate = scope as Record<string, unknown>;
      const scopeType =
        typeof candidate.scopeType === "string" ? candidate.scopeType.trim().toLowerCase() : "";
      const scopeId =
        typeof candidate.scopeId === "string" ? candidate.scopeId.trim().toLowerCase() : "";
      const weight =
        typeof candidate.weight === "number" ? candidate.weight : Number(candidate.weight);
      if (!scopeType || !scopeId || !Number.isFinite(weight) || weight <= 0) {
        return null;
      }
      return { scopeType, scopeId, weight };
    })
    .filter((scope): scope is SoulMemoryScope => scope !== null);
}

export const memoryHandlers: GatewayRequestHandlers = {
  "memory.list": ({ params, respond }) => {
    if (!validateMemoryListParams(params)) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          `invalid memory.list params: ${formatValidationErrors(validateMemoryListParams.errors)}`,
        ),
      );
      return;
    }
    const resolved = resolveWorkspaceAgent(params);
    if (!resolved.ok) {
      respond(false, undefined, resolved.error);
      return;
    }
    const items = listSoulMemoryItems({
      agentId: resolved.agentId,
      scopeType: typeof params.scopeType === "string" ? params.scopeType.trim() : undefined,
      scopeId: typeof params.scopeId === "string" ? params.scopeId.trim() : undefined,
      kind: typeof params.kind === "string" ? params.kind.trim() : undefined,
      tier: typeof params.tier === "string" ? params.tier.trim().toLowerCase() : undefined,
      recordKind:
        typeof params.recordKind === "string"
          ? (params.recordKind.trim().toLowerCase() as SoulMemoryRecordKind)
          : undefined,
      limit: typeof params.limit === "number" ? params.limit : undefined,
    });
    respond(true, { agentId: resolved.agentId, items }, undefined);
  },
  "memory.search": ({ params, respond }) => {
    if (!validateMemorySearchParams(params)) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          `invalid memory.search params: ${formatValidationErrors(validateMemorySearchParams.errors)}`,
        ),
      );
      return;
    }
    const resolved = resolveWorkspaceAgent(params);
    if (!resolved.ok) {
      respond(false, undefined, resolved.error);
      return;
    }
    const sessionKey = typeof params.sessionKey === "string" ? params.sessionKey.trim() : undefined;
    const parsedScopes = sanitizeScopes(params.scopes);
    const scopes =
      parsedScopes && parsedScopes.length > 0
        ? parsedScopes
        : resolveSoulScopes({
            agentId: resolved.agentId,
            sessionKey,
          });
    const items = querySoulMemoryMulti({
      agentId: resolved.agentId,
      query: String(params.query),
      scopes,
      topK: typeof params.topK === "number" ? params.topK : 20,
      minScore: typeof params.minScore === "number" ? params.minScore : undefined,
      ttlDays: typeof params.ttlDays === "number" ? params.ttlDays : undefined,
    });
    respond(
      true,
      {
        agentId: resolved.agentId,
        query: String(params.query),
        scopes,
        items,
      },
      undefined,
    );
  },
};
