import { Type } from "@sinclair/typebox";
import { loadConfig } from "../../../core/config/config.js";
import { logDebug } from "../../../logger.js";
import type { GatewayMessageChannel } from "../../../utils/message-channel.js";
import { optionalStringEnum } from "../../schema/typebox.js";
import { resolveSubagentAutoRoute } from "../../subagent-auto-route.js";
import { buildSubagentContext, type SubagentContextSpec } from "../../subagent-context-builder.js";
import { spawnSubagentDirect } from "../../subagent-spawn.js";
import { resolveWorkspaceRoot } from "../../workspace-dir.js";
import type { AnyAgentTool } from "../common.js";
import { jsonResult, readStringParam } from "../common.js";
import { SubagentContextSpecSchema } from "../parallel-spawn-tool.js";
import { readSubagentResult } from "../subagent-result-reader.js";

const SessionsSpawnToolSchema = Type.Object({
  task: Type.String(),
  label: Type.Optional(Type.String()),
  agentId: Type.Optional(Type.String()),
  model: Type.Optional(Type.String()),
  thinking: Type.Optional(Type.String()),
  role: Type.Optional(Type.String()),
  preset: Type.Optional(Type.String()),
  taskGroup: Type.Optional(Type.String()),
  dispatchId: Type.Optional(Type.String()),
  announceMode: optionalStringEnum(["child", "aggregate"] as const),
  runTimeoutSeconds: Type.Optional(Type.Number({ minimum: 0 })),
  // Back-compat: older callers used timeoutSeconds for this tool.
  timeoutSeconds: Type.Optional(Type.Number({ minimum: 0 })),
  cleanup: optionalStringEnum(["delete", "keep"] as const),
  waitForResult: Type.Optional(Type.Boolean()),
  context: Type.Optional(SubagentContextSpecSchema),
});

export function createSessionsSpawnTool(opts?: {
  agentSessionKey?: string;
  agentChannel?: GatewayMessageChannel;
  agentAccountId?: string;
  agentTo?: string;
  agentThreadId?: string | number;
  agentGroupId?: string | null;
  agentGroupChannel?: string | null;
  agentGroupSpace?: string | null;
  sandboxed?: boolean;
  /** Explicit agent ID override for cron/hook sessions where session key parsing may not work. */
  requesterAgentIdOverride?: string;
}): AnyAgentTool {
  return {
    label: "Sessions",
    name: "sessions_spawn",
    description:
      "Spawn a background sub-agent run in an isolated session and announce the result back to the requester chat.",
    parameters: SessionsSpawnToolSchema,
    execute: async (_toolCallId, args) => {
      const params = args as Record<string, unknown>;
      const task = readStringParam(params, "task", { required: true });
      const label = typeof params.label === "string" ? params.label.trim() : "";
      const requestedAgentId = readStringParam(params, "agentId");
      const modelOverride = readStringParam(params, "model");
      const thinkingOverrideRaw = readStringParam(params, "thinking");
      const role = readStringParam(params, "role");
      const preset = readStringParam(params, "preset");
      const taskGroup = readStringParam(params, "taskGroup");
      const dispatchId = readStringParam(params, "dispatchId");
      const announceMode =
        params.announceMode === "child" || params.announceMode === "aggregate"
          ? params.announceMode
          : undefined;
      const cleanup =
        params.cleanup === "keep" || params.cleanup === "delete" ? params.cleanup : "keep";
      // Back-compat: older callers used timeoutSeconds for this tool.
      const timeoutSecondsCandidate =
        typeof params.runTimeoutSeconds === "number"
          ? params.runTimeoutSeconds
          : typeof params.timeoutSeconds === "number"
            ? params.timeoutSeconds
            : undefined;
      const runTimeoutSeconds =
        typeof timeoutSecondsCandidate === "number" && Number.isFinite(timeoutSecondsCandidate)
          ? Math.max(0, Math.floor(timeoutSecondsCandidate))
          : undefined;

      const waitForResult = params.waitForResult === true;

      // Auto-route: when no explicit role/preset/model is provided and auto-routing
      // is enabled, analyze the task and auto-select based on configured presets and
      // model pool. Respects the agents.defaults.autoRouting.enabled toggle set
      // during onboarding.
      let autoRoutedPreset: string | undefined;
      let autoRoutedModel: string | undefined;
      let autoRoutedThinking: string | undefined;
      if (!role && !preset && !modelOverride) {
        const cfg = loadConfig();
        const autoRoutingEnabled = cfg.agents?.defaults?.autoRouting?.enabled === true;
        if (!autoRoutingEnabled) {
          logDebug("[sessions_spawn] auto-routing disabled by config; skipping.");
        }
        const route = autoRoutingEnabled ? resolveSubagentAutoRoute({ task, cfg }) : null;
        if (route?.matched && route.preset) {
          autoRoutedPreset = route.preset;
        }
        if (route?.recommendedModel) {
          autoRoutedModel = route.recommendedModel;
        }
        if (route?.recommendedThinking && !thinkingOverrideRaw) {
          autoRoutedThinking = route.recommendedThinking;
        }
        if (route) {
          logDebug(
            `[sessions_spawn] auto-route: complexity=${route.complexity}, matched=${route.matched}, preset=${autoRoutedPreset ?? "none"}, model=${autoRoutedModel ?? "default"}, thinking=${autoRoutedThinking ?? "default"}, reason=${route.modelReason ?? "n/a"}`,
          );
        }
      }

      // Build context block if requested.
      let contextBlock: string | undefined;
      const contextSpec = params.context as SubagentContextSpec | undefined;
      if (contextSpec && opts?.agentSessionKey) {
        try {
          const block = await buildSubagentContext({
            spec: contextSpec,
            parentSessionKey: opts.agentSessionKey,
            workspaceDir: resolveWorkspaceRoot(),
          });
          if (block) {
            contextBlock = block;
          }
        } catch {
          // Context building failure is non-fatal; proceed without context.
        }
      }

      const result = await spawnSubagentDirect(
        {
          task,
          contextBlock,
          label: label || undefined,
          agentId: requestedAgentId,
          model: modelOverride || autoRoutedModel,
          thinking: thinkingOverrideRaw || autoRoutedThinking,
          role,
          preset: preset || autoRoutedPreset,
          taskGroup,
          dispatchId,
          // When waiting for inline result, suppress async announce to avoid duplicate delivery.
          announceMode: waitForResult ? "aggregate" : announceMode,
          runTimeoutSeconds,
          cleanup,
          expectsCompletionMessage: !waitForResult,
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

      if (!waitForResult || result.status !== "accepted") {
        return jsonResult(result);
      }

      // Block and return the subagent's output inline as the tool result.
      const waitTimeoutMs =
        typeof runTimeoutSeconds === "number" && runTimeoutSeconds > 0
          ? runTimeoutSeconds * 1000
          : 60_000;
      const output = await readSubagentResult({
        runId: result.runId!,
        childSessionKey: result.childSessionKey!,
        waitTimeoutMs,
      });

      return jsonResult({
        status: output.status,
        runId: result.runId,
        sessionKey: result.childSessionKey,
        text: output.text,
        durationMs: output.durationMs,
        model: result.modelApplied ? undefined : undefined,
      });
    },
  };
}
