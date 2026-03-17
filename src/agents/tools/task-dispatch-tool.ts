import crypto from "node:crypto";
import { Type } from "@sinclair/typebox";
import { loadConfig } from "../../core/config/config.js";
import { optionalStringEnum } from "../schema/typebox.js";
import { listSubagentRunsForRequester, type SubagentRunRecord } from "../subagent-registry.js";
import { spawnSubagentDirect } from "../subagent-spawn.js";
import type { AnyAgentTool } from "./common.js";
import { jsonResult, readStringArrayParam, readStringParam } from "./common.js";
import { resolveInternalSessionKey, resolveMainSessionAlias } from "./sessions-helpers.js";
import { readSubagentResult } from "./subagent-result-reader.js";

const TaskDispatchToolSchema = Type.Object({
  task: Type.String(),
  roles: Type.Optional(Type.Array(Type.String(), { minItems: 1 })),
  preset: Type.Optional(Type.String()),
  agentId: Type.Optional(Type.String()),
  waitForAll: Type.Optional(Type.Boolean()),
  taskGroup: Type.Optional(Type.String()),
  dispatchId: Type.Optional(Type.String()),
  announceMode: optionalStringEnum(["child", "aggregate"] as const),
  runTimeoutSeconds: Type.Optional(Type.Number({ minimum: 0 })),
  timeoutSeconds: Type.Optional(Type.Number({ minimum: 0 })),
  cleanup: optionalStringEnum(["delete", "keep"] as const),
});

function buildDispatchId(params: {
  requesterSessionKey: string;
  agentId?: string;
  task: string;
  roles: string[];
  preset?: string;
  taskGroup?: string;
}): string {
  const hash = crypto.createHash("sha256");
  hash.update(
    JSON.stringify({
      requesterSessionKey: params.requesterSessionKey,
      agentId: params.agentId ?? "",
      task: params.task,
      roles: params.roles,
      preset: params.preset ?? "",
      taskGroup: params.taskGroup ?? "",
    }),
  );
  return hash.digest("hex").slice(0, 16);
}

function normalizeRoleList(values: string[] | undefined): string[] {
  if (!values) {
    return [];
  }
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of values) {
    const role = raw.trim();
    if (!role || seen.has(role)) {
      continue;
    }
    seen.add(role);
    out.push(role);
  }
  return out;
}

function resolveDispatchRoles(params: {
  cfg: ReturnType<typeof loadConfig>;
  roles?: string[];
  preset?: string;
}): { preset?: string; roles: string[] } {
  const preset =
    params.preset?.trim() ||
    params.cfg.agents?.defaults?.subagents?.defaultPreset?.trim() ||
    undefined;
  const presetRoles =
    preset && params.cfg.agents?.defaults?.subagents?.presets?.[preset]?.roles
      ? normalizeRoleList(params.cfg.agents.defaults?.subagents?.presets?.[preset]?.roles)
      : [];
  const explicitRoles = normalizeRoleList(params.roles);
  return {
    preset,
    roles: explicitRoles.length > 0 ? explicitRoles : presetRoles,
  };
}

function pickDispatchRunForRole(
  runs: SubagentRunRecord[],
  role: string,
): SubagentRunRecord | undefined {
  const matching = runs
    .filter((entry) => entry.role === role)
    .toSorted((a, b) => {
      if (!a.endedAt && b.endedAt) {
        return -1;
      }
      if (a.endedAt && !b.endedAt) {
        return 1;
      }
      return (b.createdAt ?? 0) - (a.createdAt ?? 0);
    });
  return matching[0];
}

function buildAggregateText(
  results: Array<{ role: string; status: string; text?: string }>,
): string {
  const lines = ["Dispatch complete:"];
  for (const result of results) {
    const summary = result.text?.trim() ? result.text.trim() : "(no assistant summary)";
    lines.push(`- ${result.role}: ${result.status} - ${summary}`);
  }
  return lines.join("\n");
}

export function createTaskDispatchTool(opts?: {
  agentSessionKey?: string;
  agentChannel?: string;
  agentAccountId?: string;
  agentTo?: string;
  agentThreadId?: string | number;
  agentGroupId?: string | null;
  agentGroupChannel?: string | null;
  agentGroupSpace?: string | null;
  requesterAgentIdOverride?: string;
}): AnyAgentTool {
  return {
    label: "Subagents",
    name: "task_dispatch",
    description:
      "Dispatch the same task to multiple role-based subagents with optional wait-for-all aggregation and retry dedupe.",
    parameters: TaskDispatchToolSchema,
    execute: async (_toolCallId, args) => {
      const params = args as Record<string, unknown>;
      const task = readStringParam(params, "task", { required: true });
      const cfg = loadConfig();
      const { mainKey, alias } = resolveMainSessionAlias(cfg);
      const requesterSessionKey = resolveInternalSessionKey({
        key: opts?.agentSessionKey?.trim() || alias,
        alias,
        mainKey,
      });
      const requestedRoles = readStringArrayParam(params, "roles");
      const requestedPreset = readStringParam(params, "preset");
      const { preset, roles } = resolveDispatchRoles({
        cfg,
        roles: requestedRoles,
        preset: requestedPreset,
      });
      if (roles.length === 0) {
        return jsonResult({
          status: "error",
          error: "task_dispatch requires at least one role or a preset that expands to roles.",
        });
      }

      const waitForAll = params.waitForAll === true;
      const requestedDispatchId = readStringParam(params, "dispatchId");
      const requestedTaskGroup = readStringParam(params, "taskGroup");
      const requestedAgentId = readStringParam(params, "agentId");
      const announceMode =
        params.announceMode === "child" || params.announceMode === "aggregate"
          ? params.announceMode
          : waitForAll
            ? "aggregate"
            : "child";
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
      const cleanup =
        params.cleanup === "delete" || params.cleanup === "keep"
          ? params.cleanup
          : waitForAll
            ? "keep"
            : "keep";

      const dispatchId =
        requestedDispatchId ||
        buildDispatchId({
          requesterSessionKey,
          agentId: requestedAgentId,
          task,
          roles,
          preset,
          taskGroup: requestedTaskGroup,
        });
      const taskGroup = requestedTaskGroup || dispatchId;

      const existingDispatchRuns = listSubagentRunsForRequester(requesterSessionKey).filter(
        (entry) => entry.dispatchId === dispatchId,
      );
      const reused: Array<{ role: string; runId: string; sessionKey: string }> = [];
      const spawned: Array<{ role: string; runId?: string; sessionKey?: string }> = [];
      const runs: SubagentRunRecord[] = [];
      const errors: Array<{ role: string; error: string }> = [];

      for (const role of roles) {
        const existing = pickDispatchRunForRole(existingDispatchRuns, role);
        if (existing) {
          reused.push({
            role,
            runId: existing.runId,
            sessionKey: existing.childSessionKey,
          });
          runs.push(existing);
          continue;
        }

        const result = await spawnSubagentDirect(
          {
            task,
            label: role,
            agentId: requestedAgentId,
            role,
            preset,
            taskGroup,
            dispatchId,
            announceMode: announceMode,
            runTimeoutSeconds,
            cleanup,
            expectsCompletionMessage: !waitForAll,
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
        if (result.status !== "accepted") {
          errors.push({ role, error: result.error ?? "dispatch failed" });
          continue;
        }
        spawned.push({
          role,
          runId: result.runId,
          sessionKey: result.childSessionKey,
        });
        const spawnedRun = listSubagentRunsForRequester(requesterSessionKey)
          .filter((entry) => entry.dispatchId === dispatchId && entry.role === role)
          .toSorted((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0))[0];
        if (spawnedRun) {
          runs.push(spawnedRun);
        }
      }

      if (!waitForAll) {
        return jsonResult({
          status: errors.length > 0 ? "partial" : "accepted",
          dispatchId,
          taskGroup,
          preset,
          announceMode,
          roles,
          reused,
          spawned,
          errors: errors.length > 0 ? errors : undefined,
          text:
            errors.length > 0
              ? `dispatched ${runs.length} role worker(s) with ${errors.length} error(s).`
              : `dispatched ${runs.length} role worker(s).`,
        });
      }

      const waitTimeoutMs =
        typeof runTimeoutSeconds === "number" ? Math.max(1, runTimeoutSeconds * 1000) : 60_000;
      const results: Array<{
        role: string;
        runId: string;
        sessionKey: string;
        status: string;
        text?: string;
      }> = [];

      for (const run of runs) {
        try {
          const output = await readSubagentResult({
            runId: run.runId,
            childSessionKey: run.childSessionKey,
            waitTimeoutMs,
            alreadyEnded: !!run.endedAt,
          });
          results.push({
            role: run.role ?? "worker",
            runId: run.runId,
            sessionKey: run.childSessionKey,
            status: output.status,
            text: output.text || run.outcome?.error,
          });
        } catch (err) {
          results.push({
            role: run.role ?? "worker",
            runId: run.runId,
            sessionKey: run.childSessionKey,
            status: "error",
            text: err instanceof Error ? err.message : String(err),
          });
        }
      }

      return jsonResult({
        status: errors.length > 0 ? "partial" : "ok",
        dispatchId,
        taskGroup,
        preset,
        announceMode,
        roles,
        reused,
        spawned,
        errors: errors.length > 0 ? errors : undefined,
        results,
        text: buildAggregateText(results),
      });
    },
  };
}
