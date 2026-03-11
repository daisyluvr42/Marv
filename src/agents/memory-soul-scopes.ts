import type { SoulMemoryScope } from "../memory/storage/soul-memory-store.js";
import { parseAgentSessionKey } from "../routing/session-key.js";

export function resolveSoulScopes(params: {
  agentId: string;
  sessionKey?: string;
}): SoulMemoryScope[] {
  const scopes: SoulMemoryScope[] = [{ scopeType: "agent", scopeId: params.agentId, weight: 1 }];
  const parsed = parseAgentSessionKey(params.sessionKey);
  if (!parsed?.rest) {
    return dedupeScopes(scopes);
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

export function dedupeScopes(scopes: SoulMemoryScope[]): SoulMemoryScope[] {
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
