import type { VerboseLevel } from "../auto-reply/support/thinking.js";
import type { SpecialRunMode } from "../contracts/run-mode.js";

export const AGENT_EVENT_CONTRACT_VERSION = 1;
export const AGENT_EVENT_REQUIRED_FIELDS = ["runId", "seq", "stream", "ts", "data"] as const;
export const AGENT_EVENT_STREAMS = [
  "lifecycle",
  "tool",
  "assistant",
  "error",
  "compaction",
] as const;
export const AGENT_EVENT_CONTEXT_FIELDS = [
  "sessionKey",
  "verboseLevel",
  "isHeartbeat",
  "runModeKind",
] as const;

export type AgentEventCoreStream = (typeof AGENT_EVENT_STREAMS)[number];
export type AgentEventStream = AgentEventCoreStream | (string & {});

export type AgentEventPayload = {
  runId: string;
  seq: number;
  stream: AgentEventStream;
  ts: number;
  data: Record<string, unknown>;
  sessionKey?: string;
  context?: AgentRunContext;
};

export type AgentRunContext = {
  sessionKey?: string;
  verboseLevel?: VerboseLevel;
  isHeartbeat?: boolean;
  runModeKind?: SpecialRunMode["kind"];
};

// Keep per-run counters so streams stay strictly monotonic per runId.
const seqByRun = new Map<string, number>();
const listeners = new Set<(evt: AgentEventPayload) => void>();
const runContextById = new Map<string, AgentRunContext>();

export function registerAgentRunContext(runId: string, context: AgentRunContext) {
  if (!runId) {
    return;
  }
  const existing = runContextById.get(runId);
  if (!existing) {
    runContextById.set(runId, { ...context });
    return;
  }
  if (context.sessionKey && existing.sessionKey !== context.sessionKey) {
    existing.sessionKey = context.sessionKey;
  }
  if (context.verboseLevel && existing.verboseLevel !== context.verboseLevel) {
    existing.verboseLevel = context.verboseLevel;
  }
  if (context.isHeartbeat !== undefined && existing.isHeartbeat !== context.isHeartbeat) {
    existing.isHeartbeat = context.isHeartbeat;
  }
  if (context.runModeKind && existing.runModeKind !== context.runModeKind) {
    existing.runModeKind = context.runModeKind;
  }
}

export function getAgentRunContext(runId: string) {
  return runContextById.get(runId);
}

export function isHeartbeatRunContext(
  context?: Pick<AgentRunContext, "isHeartbeat" | "runModeKind"> | null,
) {
  if (!context) {
    return false;
  }
  return context.runModeKind === "heartbeat" || context.isHeartbeat === true;
}

export function clearAgentRunContext(runId: string) {
  runContextById.delete(runId);
}

export function resetAgentRunContextForTest() {
  runContextById.clear();
}

export function emitAgentEvent(event: Omit<AgentEventPayload, "seq" | "ts">) {
  const nextSeq = (seqByRun.get(event.runId) ?? 0) + 1;
  seqByRun.set(event.runId, nextSeq);
  const context = runContextById.get(event.runId);
  const sessionKey =
    typeof event.sessionKey === "string" && event.sessionKey.trim()
      ? event.sessionKey
      : context?.sessionKey;
  const enriched: AgentEventPayload = {
    ...event,
    sessionKey,
    context:
      context || sessionKey
        ? {
            ...context,
            ...(sessionKey ? { sessionKey } : {}),
          }
        : undefined,
    seq: nextSeq,
    ts: Date.now(),
  };
  for (const listener of listeners) {
    try {
      listener(enriched);
    } catch {
      /* ignore */
    }
  }
}

export function onAgentEvent(listener: (evt: AgentEventPayload) => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
