import { Type } from "@sinclair/typebox";
import { loadConfig } from "../../core/config/config.js";
import type { GatewayMessageChannel } from "../../utils/message-channel.js";
import { optionalStringEnum } from "../schema/typebox.js";
import { buildSubagentContext, type SubagentContextSpec } from "../subagent-context-builder.js";
import { spawnSubagentDirect } from "../subagent-spawn.js";
import { resolveWorkspaceRoot } from "../workspace-dir.js";
import type { AnyAgentTool } from "./common.js";
import { jsonResult, readStringParam } from "./common.js";
import { readSubagentResult } from "./subagent-result-reader.js";

const SubagentContextSpecSchema = Type.Object({
  recentTurns: Type.Optional(Type.Number({ minimum: 0, maximum: 10 })),
  includeToolResults: Type.Optional(Type.Array(Type.String())),
  includeFiles: Type.Optional(Type.Array(Type.String())),
  maxContextChars: Type.Optional(Type.Number({ minimum: 0 })),
  preamble: Type.Optional(Type.String()),
});

const ParallelSpawnTaskSchema = Type.Object({
  task: Type.String(),
  label: Type.Optional(Type.String()),
  role: Type.Optional(Type.String()),
  model: Type.Optional(Type.String()),
  thinking: Type.Optional(Type.String()),
  context: Type.Optional(SubagentContextSpecSchema),
});

const ParallelSpawnToolSchema = Type.Object({
  tasks: Type.Array(ParallelSpawnTaskSchema, { minItems: 1, maxItems: 20 }),
  waitForAll: Type.Optional(Type.Boolean()),
  timeoutSeconds: Type.Optional(Type.Number({ minimum: 0 })),
  cleanup: optionalStringEnum(["delete", "keep"] as const),
});

export { SubagentContextSpecSchema };

export function createParallelSpawnTool(opts?: {
  agentSessionKey?: string;
  agentChannel?: GatewayMessageChannel;
  agentAccountId?: string;
  agentTo?: string;
  agentThreadId?: string | number;
  agentGroupId?: string | null;
  agentGroupChannel?: string | null;
  agentGroupSpace?: string | null;
  sandboxed?: boolean;
  requesterAgentIdOverride?: string;
}): AnyAgentTool {
  return {
    label: "Subagents",
    name: "parallel_spawn",
    description:
      "Spawn multiple heterogeneous sub-agent tasks in parallel. Each task gets its own session with optional role, model, and context. By default waits for all results and returns them inline.",
    parameters: ParallelSpawnToolSchema,
    execute: async (_toolCallId, args) => {
      const params = args as Record<string, unknown>;
      const tasks = params.tasks as Array<Record<string, unknown>>;
      const waitForAll = params.waitForAll !== false; // default: true
      const cfg = loadConfig();
      const maxChildren = cfg.agents?.defaults?.subagents?.maxChildrenPerAgent ?? 5;

      if (tasks.length > maxChildren) {
        return jsonResult({
          status: "error",
          error: `parallel_spawn received ${tasks.length} tasks but maxChildrenPerAgent is ${maxChildren}. Reduce the number of tasks or increase the limit.`,
        });
      }

      const timeoutSecondsCandidate =
        typeof params.timeoutSeconds === "number" ? params.timeoutSeconds : undefined;
      const runTimeoutSeconds =
        typeof timeoutSecondsCandidate === "number" && Number.isFinite(timeoutSecondsCandidate)
          ? Math.max(0, Math.floor(timeoutSecondsCandidate))
          : undefined;
      const cleanup =
        params.cleanup === "delete" || params.cleanup === "keep" ? params.cleanup : "keep";

      const workspaceDir = resolveWorkspaceRoot();
      const parentSessionKey = opts?.agentSessionKey;

      // Build context blocks for each task (in parallel).
      const contextBlocks = await Promise.all(
        tasks.map(async (t) => {
          const contextSpec = t.context as SubagentContextSpec | undefined;
          if (!contextSpec || !parentSessionKey) {
            return undefined;
          }
          try {
            return await buildSubagentContext({
              spec: contextSpec,
              parentSessionKey,
              workspaceDir,
            });
          } catch {
            return undefined;
          }
        }),
      );

      // Spawn all tasks in parallel.
      const spawnResults = await Promise.all(
        tasks.map(async (t, index) => {
          const task = readStringParam(t, "task", { required: true });
          const label = typeof t.label === "string" ? t.label.trim() : `parallel-${index}`;
          const role = typeof t.role === "string" ? t.role.trim() : undefined;
          const model = typeof t.model === "string" ? t.model.trim() : undefined;
          const thinking = typeof t.thinking === "string" ? t.thinking.trim() : undefined;
          const contextBlock = contextBlocks[index];

          const result = await spawnSubagentDirect(
            {
              task: contextBlock ? `[Context]\n${contextBlock}\n\n[Subagent Task]: ${task}` : task,
              label,
              model,
              thinking,
              role,
              announceMode: "aggregate",
              runTimeoutSeconds,
              cleanup,
              expectsCompletionMessage: false,
            },
            {
              agentSessionKey: opts?.agentSessionKey,
              agentChannel: opts?.agentChannel,
              agentAccountId: opts?.agentAccountId,
              agentTo: opts?.agentTo,
              agentThreadId: opts?.agentThreadId,
              agentGroupId: opts?.agentGroupId,
              agentGroupChannel: opts?.agentGroupChannel,
              agentGroupSpace: opts?.agentGroupSpace,
              requesterAgentIdOverride: opts?.requesterAgentIdOverride,
            },
          );

          return { index, label, role, result };
        }),
      );

      // If not waiting, return immediately with spawn status.
      if (!waitForAll) {
        const spawned: Array<{
          index: number;
          label: string;
          role?: string;
          status: string;
          runId?: string;
          sessionKey?: string;
          error?: string;
        }> = [];
        for (const { index, label, role, result } of spawnResults) {
          spawned.push({
            index,
            label,
            role,
            status: result.status,
            runId: result.runId,
            sessionKey: result.childSessionKey,
            error: result.error,
          });
        }
        return jsonResult({
          status: spawned.every((s) => s.status === "accepted") ? "accepted" : "partial",
          spawned,
          text: `Spawned ${spawned.filter((s) => s.status === "accepted").length}/${tasks.length} tasks (async).`,
        });
      }

      // Wait for all results in parallel.
      const waitTimeoutMs =
        typeof runTimeoutSeconds === "number" && runTimeoutSeconds > 0
          ? runTimeoutSeconds * 1000
          : 60_000;

      const resultPromises = spawnResults.map(async ({ index, label, role, result }) => {
        if (result.status !== "accepted" || !result.runId || !result.childSessionKey) {
          return {
            index,
            label,
            role,
            status: "error" as const,
            text: result.error ?? "spawn failed",
            runId: result.runId,
            sessionKey: result.childSessionKey,
          };
        }

        const output = await readSubagentResult({
          runId: result.runId,
          childSessionKey: result.childSessionKey,
          waitTimeoutMs,
        });

        return {
          index,
          label,
          role,
          status: output.status,
          text: output.text,
          runId: result.runId,
          sessionKey: result.childSessionKey,
          durationMs: output.durationMs,
        };
      });

      const results = await Promise.allSettled(resultPromises);
      const collected: Array<{
        index: number;
        label: string;
        role?: string;
        status: string;
        text: string;
        runId?: string;
        sessionKey?: string;
        durationMs?: number;
      }> = [];

      for (const settled of results) {
        if (settled.status === "fulfilled") {
          collected.push(settled.value);
        } else {
          collected.push({
            index: -1,
            label: "unknown",
            status: "error",
            text: settled.reason instanceof Error ? settled.reason.message : String(settled.reason),
          });
        }
      }

      // Sort by original index.
      collected.sort((a, b) => a.index - b.index);

      const okCount = collected.filter((r) => r.status === "ok").length;
      return jsonResult({
        status: okCount === collected.length ? "ok" : "partial",
        results: collected,
        text: buildParallelSummary(collected),
      });
    },
  };
}

function buildParallelSummary(
  results: Array<{ label: string; status: string; text: string; role?: string }>,
): string {
  const lines = [`Parallel spawn complete (${results.length} tasks):`];
  for (const r of results) {
    const roleTag = r.role ? ` [${r.role}]` : "";
    const preview = r.text.length > 200 ? `${r.text.slice(0, 200)}...` : r.text;
    lines.push(`- ${r.label}${roleTag}: ${r.status} — ${preview}`);
  }
  return lines.join("\n");
}
