import type { TurnContext } from "../../../auto-reply/support/templating.js";
import type { SessionEntry } from "../../../core/config/sessions.js";
import type { ExternalCliAdapterId } from "../../../core/config/types.tools.js";
import { sessionsHandlers } from "../../../core/gateway/server-methods/sessions.js";
import { computeNextRunAtMs } from "../../../cron/schedule.js";
import type { GatewayMessageChannel } from "../../../utils/message-channel.js";
import { normalizeExternalCliId } from "../cli/external-cli-adapters.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const GENERIC_DENIED_MESSAGE = "This setting request cannot be applied right now.";
export const GENERIC_INVALID_MESSAGE = "I can't apply that session setting directly.";
export const REDACTED_VALUE = "[redacted]";

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

export type MemorySearchProvider = "openai" | "gemini" | "local" | "voyage";
export type MemorySearchFallback = MemorySearchProvider | "none";
export type HeartbeatFileAction = "replace" | "append" | "clear";
export type SelfSettingsArgs = Record<string, unknown>;

// ---------------------------------------------------------------------------
// Result helpers
// ---------------------------------------------------------------------------

export function buildGenericDeniedResult() {
  return {
    content: [{ type: "text" as const, text: GENERIC_DENIED_MESSAGE }],
    details: { ok: false, applied: false, denied: true },
  };
}

export function buildInvalidResult(reason?: string) {
  const text = reason ? `${GENERIC_INVALID_MESSAGE} Reason: ${reason}` : GENERIC_INVALID_MESSAGE;
  return {
    content: [{ type: "text" as const, text }],
    details: { ok: false, applied: false, invalid: true, ...(reason ? { reason } : {}) },
  };
}

// ---------------------------------------------------------------------------
// Normalizers
// ---------------------------------------------------------------------------

export function normalizeSessionAction(raw?: string): "new" | "reset" | undefined {
  const normalized = raw?.trim().toLowerCase();
  if (normalized === "new" || normalized === "reset") {
    return normalized;
  }
  return undefined;
}

export function normalizeModelRegistryAction(raw?: string): "refresh" | undefined {
  const normalized = raw?.trim().toLowerCase();
  if (normalized === "refresh" || normalized === "update" || normalized === "sync") {
    return "refresh";
  }
  return undefined;
}

export function normalizePatchString(raw: string | undefined): string | null | undefined {
  if (raw === undefined) {
    return undefined;
  }
  const trimmed = raw.trim();
  if (!trimmed) {
    return undefined;
  }
  const normalized = trimmed.toLowerCase();
  if (normalized === "default" || normalized === "clear" || normalized === "reset") {
    return null;
  }
  return trimmed;
}

export function normalizeDeepMemoryApi(
  raw: string | undefined,
): "ollama" | "openai-completions" | null | undefined {
  const normalized = normalizePatchString(raw);
  if (normalized == null) {
    return normalized;
  }
  const lower = normalized.trim().toLowerCase();
  if (lower === "ollama") {
    return "ollama";
  }
  if (lower === "openai-completions" || lower === "openai" || lower === "openai-compatible") {
    return "openai-completions";
  }
  return undefined;
}

export function normalizeMemorySearchProvider(
  raw: string | undefined,
): MemorySearchProvider | null | undefined {
  const normalized = normalizePatchString(raw);
  if (normalized == null) {
    return normalized;
  }
  const lower = normalized.toLowerCase();
  if (lower === "openai" || lower === "gemini" || lower === "local" || lower === "voyage") {
    return lower;
  }
  return undefined;
}

export function normalizeMemorySearchFallback(
  raw: string | undefined,
): MemorySearchFallback | null | undefined {
  const normalized = normalizePatchString(raw);
  if (normalized == null) {
    return normalized;
  }
  const lower = normalized.toLowerCase();
  if (
    lower === "openai" ||
    lower === "gemini" ||
    lower === "local" ||
    lower === "voyage" ||
    lower === "none"
  ) {
    return lower;
  }
  return undefined;
}

export function readBooleanParam(
  params: Record<string, unknown>,
  key: string,
): boolean | undefined {
  const raw = params[key];
  if (typeof raw === "boolean") {
    return raw;
  }
  if (typeof raw !== "string") {
    return undefined;
  }
  const normalized = raw.trim().toLowerCase();
  if (["true", "yes", "on", "enabled"].includes(normalized)) {
    return true;
  }
  if (["false", "no", "off", "disabled"].includes(normalized)) {
    return false;
  }
  return undefined;
}

export function normalizeAuthProfile(raw: string | undefined): string | null | undefined {
  if (raw === undefined) {
    return undefined;
  }
  const trimmed = raw.trim();
  if (!trimmed) {
    return undefined;
  }
  const normalized = trimmed.toLowerCase();
  if (
    normalized === "default" ||
    normalized === "clear" ||
    normalized === "reset" ||
    normalized === "auto"
  ) {
    return null;
  }
  return trimmed;
}

export function normalizeExternalCliDefault(
  raw: string | undefined,
): ExternalCliAdapterId | null | undefined {
  const normalized = normalizePatchString(raw);
  if (normalized == null) {
    return normalized;
  }
  return normalizeExternalCliId(normalized) ?? undefined;
}

export function normalizeExternalCliAvailableBrands(
  raw: string | undefined,
): ExternalCliAdapterId[] | undefined {
  if (raw === undefined) {
    return undefined;
  }
  const trimmed = raw.trim();
  if (!trimmed) {
    return [];
  }
  const parts = trimmed
    .split(/[,\n]/g)
    .flatMap((part) => part.split(/\s+/g))
    .map((part) => part.trim())
    .filter(Boolean);
  const out: ExternalCliAdapterId[] = [];
  const seen = new Set<string>();
  for (const part of parts) {
    const normalized = normalizeExternalCliId(part);
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

export function normalizeHeartbeatFileAction(raw?: string): HeartbeatFileAction | undefined {
  const normalized = raw?.trim().toLowerCase();
  if (
    normalized === "replace" ||
    normalized === "set" ||
    normalized === "rewrite" ||
    normalized === "overwrite"
  ) {
    return "replace";
  }
  if (normalized === "append" || normalized === "add") {
    return "append";
  }
  if (normalized === "clear" || normalized === "empty" || normalized === "reset") {
    return "clear";
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Elevated context builder
// ---------------------------------------------------------------------------

export function buildElevatedTurnContext(opts: {
  agentChannel?: GatewayMessageChannel;
  agentAccountId?: string;
  agentTo?: string;
  senderId?: string;
  senderName?: string;
  senderUsername?: string;
  senderE164?: string;
}): TurnContext {
  return {
    Provider: opts.agentChannel,
    AccountId: opts.agentAccountId,
    To: opts.agentTo,
    From: opts.senderE164 ?? opts.senderId,
    SenderId: opts.senderId,
    SenderName: opts.senderName,
    SenderUsername: opts.senderUsername,
    SenderE164: opts.senderE164,
  };
}

// ---------------------------------------------------------------------------
// Session helpers
// ---------------------------------------------------------------------------

export async function invokeSessionReset(params: {
  key: string;
  reason: "new" | "reset";
}): Promise<{ ok: true; entry?: SessionEntry } | { ok: false }> {
  let response:
    | { ok: true; result?: { entry?: SessionEntry } }
    | { ok: false; error?: unknown }
    | undefined;
  await sessionsHandlers["sessions.reset"]({
    req: {
      id: "self-settings-reset",
      method: "sessions.reset",
      params: { key: params.key, reason: params.reason },
    } as never,
    params: { key: params.key, reason: params.reason },
    client: null,
    isWebchatConnect: () => false,
    respond: (ok, result, error) => {
      response = ok
        ? { ok: true, result: result as { entry?: SessionEntry } }
        : { ok: false, error };
    },
    context: {} as never,
  });
  if (!response?.ok) {
    return { ok: false };
  }
  return { ok: true, entry: response.result?.entry };
}

export function resolveCurrentSessionEntry(params: {
  store: Record<string, SessionEntry>;
  sessionKeys: string[];
}): SessionEntry | undefined {
  for (const key of params.sessionKeys) {
    const entry = params.store[key];
    if (entry) {
      return entry;
    }
  }
  return undefined;
}

export function isValidCronExpr(expr: string): boolean {
  try {
    const next = computeNextRunAtMs({ kind: "cron", expr }, Date.now());
    return next !== undefined;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Config patch helpers
// ---------------------------------------------------------------------------

export function applyOptionalStringPatch(
  target: Record<string, unknown>,
  key: string,
  value: string | null | undefined,
) {
  if (value === undefined) {
    return;
  }
  if (value === null) {
    delete target[key];
    return;
  }
  target[key] = value;
}

export function pruneEmptyObject(
  value: Record<string, unknown> | undefined,
): Record<string, unknown> | undefined {
  if (!value) {
    return undefined;
  }
  return Object.keys(value).length > 0 ? value : undefined;
}

export function summarizeSensitivePatch(
  label: string,
  value: string | null | undefined,
): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  return `${label} ${value === null ? "default" : "configured"}`;
}

// ---------------------------------------------------------------------------
// Shared opts type for all tool creators
// ---------------------------------------------------------------------------

export type SelfSettingsToolOpts = {
  agentSessionKey?: string;
  config?: import("../../../core/config/config.js").MarvConfig;
  agentChannel?: GatewayMessageChannel;
  agentAccountId?: string;
  agentTo?: string;
  senderId?: string;
  senderName?: string;
  senderUsername?: string;
  senderE164?: string;
  directUserInstruction?: boolean;
};
