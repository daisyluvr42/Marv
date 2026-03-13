import { describe, expect, test } from "vitest";
import {
  AGENT_EVENT_CONTRACT_VERSION,
  AGENT_EVENT_CONTEXT_FIELDS,
  AGENT_EVENT_REQUIRED_FIELDS,
  AGENT_EVENT_STREAMS,
  clearAgentRunContext,
  emitAgentEvent,
  getAgentRunContext,
  isHeartbeatRunContext,
  onAgentEvent,
  registerAgentRunContext,
  resetAgentRunContextForTest,
} from "./agent-events.js";

describe("agent-events sequencing", () => {
  test("exports a stable contract version and core stream list", () => {
    expect(AGENT_EVENT_CONTRACT_VERSION).toBe(1);
    expect(AGENT_EVENT_REQUIRED_FIELDS).toEqual(["runId", "seq", "stream", "ts", "data"]);
    expect(AGENT_EVENT_STREAMS).toEqual(["lifecycle", "tool", "assistant", "error", "compaction"]);
    expect(AGENT_EVENT_CONTEXT_FIELDS).toEqual([
      "sessionKey",
      "verboseLevel",
      "isHeartbeat",
      "runModeKind",
    ]);
  });

  test("stores and clears run context", async () => {
    resetAgentRunContextForTest();
    registerAgentRunContext("run-1", { sessionKey: "main" });
    expect(getAgentRunContext("run-1")?.sessionKey).toBe("main");
    clearAgentRunContext("run-1");
    expect(getAgentRunContext("run-1")).toBeUndefined();
  });

  test("recognizes heartbeat semantics from runModeKind or legacy heartbeat flag", () => {
    expect(isHeartbeatRunContext(undefined)).toBe(false);
    expect(isHeartbeatRunContext({ isHeartbeat: true })).toBe(true);
    expect(isHeartbeatRunContext({ runModeKind: "heartbeat" })).toBe(true);
    expect(isHeartbeatRunContext({ isHeartbeat: false, runModeKind: "user" })).toBe(false);
  });

  test("maintains monotonic seq per runId", async () => {
    const seen: Record<string, number[]> = {};
    const stop = onAgentEvent((evt) => {
      const list = seen[evt.runId] ?? [];
      seen[evt.runId] = list;
      list.push(evt.seq);
    });

    emitAgentEvent({ runId: "run-1", stream: "lifecycle", data: {} });
    emitAgentEvent({ runId: "run-1", stream: "lifecycle", data: {} });
    emitAgentEvent({ runId: "run-2", stream: "lifecycle", data: {} });
    emitAgentEvent({ runId: "run-1", stream: "lifecycle", data: {} });

    stop();

    expect(seen["run-1"]).toEqual([1, 2, 3]);
    expect(seen["run-2"]).toEqual([1]);
  });

  test("preserves compaction ordering on the event bus", async () => {
    const phases: Array<string> = [];
    const stop = onAgentEvent((evt) => {
      if (evt.runId !== "run-1") {
        return;
      }
      if (evt.stream !== "compaction") {
        return;
      }
      if (typeof evt.data?.phase === "string") {
        phases.push(evt.data.phase);
      }
    });

    emitAgentEvent({ runId: "run-1", stream: "compaction", data: { phase: "start" } });
    emitAgentEvent({
      runId: "run-1",
      stream: "compaction",
      data: { phase: "end", willRetry: false },
    });

    stop();

    expect(phases).toEqual(["start", "end"]);
  });

  test("enriches emitted payloads with registered run context", async () => {
    resetAgentRunContextForTest();
    registerAgentRunContext("run-ctx", {
      sessionKey: "session-ctx",
      verboseLevel: "on",
      isHeartbeat: true,
      runModeKind: "heartbeat",
    });
    const seen: Array<{
      sessionKey?: string;
      context?: {
        sessionKey?: string;
        verboseLevel?: string;
        isHeartbeat?: boolean;
        runModeKind?: string;
      };
    }> = [];
    const stop = onAgentEvent((evt) => {
      if (evt.runId === "run-ctx") {
        seen.push({
          sessionKey: evt.sessionKey,
          context: evt.context as {
            sessionKey?: string;
            verboseLevel?: string;
            isHeartbeat?: boolean;
            runModeKind?: string;
          },
        });
      }
    });

    emitAgentEvent({ runId: "run-ctx", stream: "assistant", data: { text: "hello" } });

    stop();

    expect(seen).toEqual([
      {
        sessionKey: "session-ctx",
        context: {
          sessionKey: "session-ctx",
          verboseLevel: "on",
          isHeartbeat: true,
          runModeKind: "heartbeat",
        },
      },
    ]);
  });
});
