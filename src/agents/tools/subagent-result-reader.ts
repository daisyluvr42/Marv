import { callGateway } from "../../core/gateway/call.js";
import { extractAssistantText, stripToolMessages } from "./sessions-helpers.js";

/**
 * Shared helper to wait for a subagent run to complete and extract its final output.
 *
 * Used by `sessions_spawn` (waitForResult), `parallel_spawn`, and `task_dispatch` (waitForAll).
 */
export async function readSubagentResult(params: {
  runId: string;
  childSessionKey: string;
  waitTimeoutMs: number;
  /** If the run has already ended, skip the wait RPC. */
  alreadyEnded?: boolean;
}): Promise<SubagentResultOutput> {
  const startMs = Date.now();

  if (!params.alreadyEnded) {
    try {
      const waitResponse = await callGateway<{
        status?: string;
        error?: string;
      }>({
        method: "agent.wait",
        params: {
          runId: params.runId,
          timeoutMs: params.waitTimeoutMs,
        },
        timeoutMs: params.waitTimeoutMs + 5_000,
      });

      if (waitResponse?.status === "timeout") {
        return {
          status: "timeout",
          text: waitResponse.error ?? "Run timed out waiting for completion.",
          durationMs: Date.now() - startMs,
        };
      }
      if (waitResponse?.status === "error") {
        return {
          status: "error",
          text: waitResponse.error ?? "Run ended with an error.",
          durationMs: Date.now() - startMs,
        };
      }
    } catch (err) {
      return {
        status: "error",
        text: err instanceof Error ? err.message : String(err),
        durationMs: Date.now() - startMs,
      };
    }
  }

  // Extract the last assistant message from the child session.
  const history = await callGateway<{ messages: unknown[] }>({
    method: "chat.history",
    params: {
      sessionKey: params.childSessionKey,
      limit: 12,
    },
    timeoutMs: 10_000,
  });
  const messages = stripToolMessages(Array.isArray(history?.messages) ? history.messages : []);
  const lastAssistant = [...messages].toReversed().find((message) => {
    return (message as { role?: unknown })?.role === "assistant";
  });
  const text = extractAssistantText(lastAssistant);

  return {
    status: "ok",
    text: text || "(no assistant output)",
    durationMs: Date.now() - startMs,
  };
}

export type SubagentResultOutput = {
  status: "ok" | "error" | "timeout";
  text: string;
  durationMs?: number;
};
