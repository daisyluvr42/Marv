import path from "node:path";
import { resolveConfigPath, resolveStateDir } from "../../core/config/config.js";
import type { ToolLoopDetectionConfig } from "../../core/config/types.tools.js";
import { ensureWorkspaceSnapshotBeforeMutation } from "../../infra/workspace-rollback.js";
import type { SessionState } from "../../logging/diagnostic-session-state.js";
import { createSubsystemLogger } from "../../logging/subsystem.js";
import { getGlobalHookRunner } from "../../plugins/hook-runner-global.js";
import { isPlainObject } from "../../utils.js";
import type { AnyAgentTool } from "./common.js";
import {
  buildEscalationBlockReason,
  classifyEscalationRequirement,
} from "./policy/escalation-policy.js";
import { getEscalationManager } from "./policy/permission-escalation.js";
import { normalizeToolName } from "./policy/tool-policy.js";

export type HookContext = {
  agentId?: string;
  sessionKey?: string;
  taskId?: string;
  workspaceDir?: string;
  loopDetection?: ToolLoopDetectionConfig;
  directUserInstruction?: boolean;
};

type HookOutcome = { blocked: true; reason: string } | { blocked: false; params: unknown };

const log = createSubsystemLogger("agents/tools");
const BEFORE_TOOL_CALL_WRAPPED = Symbol("beforeToolCallWrapped");
const LOOP_WARNING_BUCKET_SIZE = 10;
const MAX_LOOP_WARNING_KEYS = 256;

function resolveActiveConfigPath(): string {
  return path.resolve(resolveConfigPath(process.env, resolveStateDir(process.env)));
}

function resolveMutationPath(params: unknown, workspaceDir?: string): string | undefined {
  if (!params || typeof params !== "object") {
    return undefined;
  }
  const record = params as Record<string, unknown>;
  const rawPath =
    typeof record.path === "string"
      ? record.path.trim()
      : typeof record.file_path === "string"
        ? record.file_path.trim()
        : "";
  if (!rawPath) {
    return undefined;
  }
  return path.resolve(workspaceDir ?? process.cwd(), rawPath);
}

function buildActiveConfigMutationError(toolName: string, activeConfigPath: string): string {
  return [
    `Refusing to modify the active config file with ${toolName}: ${activeConfigPath}`,
    'Call `gateway` with `action: "config.get"` to confirm the active config path, then use `config.patch`, `config.apply`, or `config.patches.propose` instead of direct file edits.',
  ].join("\n");
}

function shouldEmitLoopWarning(state: SessionState, warningKey: string, count: number): boolean {
  if (!state.toolLoopWarningBuckets) {
    state.toolLoopWarningBuckets = new Map();
  }
  const bucket = Math.floor(count / LOOP_WARNING_BUCKET_SIZE);
  const lastBucket = state.toolLoopWarningBuckets.get(warningKey) ?? 0;
  if (bucket <= lastBucket) {
    return false;
  }
  state.toolLoopWarningBuckets.set(warningKey, bucket);
  if (state.toolLoopWarningBuckets.size > MAX_LOOP_WARNING_KEYS) {
    const oldest = state.toolLoopWarningBuckets.keys().next().value;
    if (oldest) {
      state.toolLoopWarningBuckets.delete(oldest);
    }
  }
  return true;
}

async function recordLoopOutcome(args: {
  ctx?: HookContext;
  toolName: string;
  toolParams: unknown;
  toolCallId?: string;
  result?: unknown;
  error?: unknown;
}): Promise<void> {
  if (!args.ctx?.sessionKey) {
    return;
  }
  try {
    const { getDiagnosticSessionState } = await import("../../logging/diagnostic-session-state.js");
    const { recordToolCallOutcome } = await import("./meta/tool-loop-detection.js");
    const sessionState = getDiagnosticSessionState({
      sessionKey: args.ctx.sessionKey,
      sessionId: args.ctx?.agentId,
    });
    recordToolCallOutcome(sessionState, {
      toolName: args.toolName,
      toolParams: args.toolParams,
      toolCallId: args.toolCallId,
      result: args.result,
      error: args.error,
      config: args.ctx.loopDetection,
    });
  } catch (err) {
    log.warn(`tool loop outcome tracking failed: tool=${args.toolName} error=${String(err)}`);
  }
}

function resolveEscalationTaskId(params: unknown, ctx?: HookContext): string | undefined {
  if (ctx?.taskId?.trim()) {
    return ctx.taskId.trim();
  }
  if (params && typeof params === "object" && !Array.isArray(params)) {
    const rawTaskId = (params as Record<string, unknown>).taskId;
    if (typeof rawTaskId === "string" && rawTaskId.trim()) {
      return rawTaskId.trim();
    }
  }
  return ctx?.sessionKey?.trim() || undefined;
}

export async function runBeforeToolCallHook(args: {
  toolName: string;
  params: unknown;
  toolCallId?: string;
  ctx?: HookContext;
}): Promise<HookOutcome> {
  const toolName = normalizeToolName(args.toolName || "tool");
  const params = args.params;
  if (toolName === "write" || toolName === "edit") {
    const mutationPath = resolveMutationPath(params, args.ctx?.workspaceDir);
    if (mutationPath) {
      const activeConfigPath = resolveActiveConfigPath();
      if (mutationPath === activeConfigPath) {
        return {
          blocked: true,
          reason: buildActiveConfigMutationError(toolName, activeConfigPath),
        };
      }
    }
  }

  const escalationRequirement = classifyEscalationRequirement({
    toolName,
    params,
  });
  if (escalationRequirement.category !== "none") {
    const taskId = resolveEscalationTaskId(params, args.ctx);
    const manager = getEscalationManager();
    const granted = taskId
      ? manager.checkPermission(taskId, escalationRequirement.requiredLevel)
      : false;
    if (!granted) {
      return {
        blocked: true,
        reason: buildEscalationBlockReason({
          requirement: escalationRequirement,
          taskId: taskId ?? "current-task",
          directUserInstruction: args.ctx?.directUserInstruction,
        }),
      };
    }
  }

  if (args.ctx?.workspaceDir) {
    try {
      await ensureWorkspaceSnapshotBeforeMutation({
        workspaceDir: args.ctx.workspaceDir,
        sessionKey: args.ctx.sessionKey,
        toolName,
      });
    } catch (err) {
      log.warn(`workspace auto-snapshot failed: tool=${toolName} error=${String(err)}`);
    }
  }

  if (args.ctx?.sessionKey) {
    const { appendDiagnosticToolLoopEvent, getDiagnosticSessionState } =
      await import("../../logging/diagnostic-session-state.js");
    const { logToolLoopAction } = await import("../../logging/diagnostic.js");
    const { detectToolCallLoop, recordToolCall } = await import("./meta/tool-loop-detection.js");

    const sessionState = getDiagnosticSessionState({
      sessionKey: args.ctx.sessionKey,
      sessionId: args.ctx?.agentId,
    });

    const loopResult = detectToolCallLoop(sessionState, toolName, params, args.ctx.loopDetection);

    if (loopResult.stuck) {
      appendDiagnosticToolLoopEvent(sessionState, {
        level: loopResult.level,
        detector: loopResult.detector,
        count: loopResult.count,
        message: loopResult.message,
      });
      if (loopResult.level === "critical") {
        log.error(`Blocking ${toolName} due to critical loop: ${loopResult.message}`);
        logToolLoopAction({
          sessionKey: args.ctx.sessionKey,
          sessionId: args.ctx?.agentId,
          toolName,
          level: "critical",
          action: "block",
          detector: loopResult.detector,
          count: loopResult.count,
          message: loopResult.message,
          pairedToolName: loopResult.pairedToolName,
        });
        return {
          blocked: true,
          reason: loopResult.message,
        };
      } else {
        const warningKey = loopResult.warningKey ?? `${loopResult.detector}:${toolName}`;
        if (shouldEmitLoopWarning(sessionState, warningKey, loopResult.count)) {
          log.warn(`Loop warning for ${toolName}: ${loopResult.message}`);
          logToolLoopAction({
            sessionKey: args.ctx.sessionKey,
            sessionId: args.ctx?.agentId,
            toolName,
            level: "warning",
            action: "warn",
            detector: loopResult.detector,
            count: loopResult.count,
            message: loopResult.message,
            pairedToolName: loopResult.pairedToolName,
          });
        }
      }
    }

    recordToolCall(sessionState, toolName, params, args.toolCallId, args.ctx.loopDetection);
  }

  const hookRunner = getGlobalHookRunner();
  if (!hookRunner?.hasHooks("before_tool_call")) {
    return { blocked: false, params: args.params };
  }

  try {
    const normalizedParams = isPlainObject(params) ? params : {};
    const hookResult = await hookRunner.runBeforeToolCall(
      {
        toolName,
        params: normalizedParams,
      },
      {
        toolName,
        agentId: args.ctx?.agentId,
        sessionKey: args.ctx?.sessionKey,
      },
    );

    if (hookResult?.block) {
      return {
        blocked: true,
        reason: hookResult.blockReason || "Tool call blocked by plugin hook",
      };
    }

    if (hookResult?.params && isPlainObject(hookResult.params)) {
      if (isPlainObject(params)) {
        return { blocked: false, params: { ...params, ...hookResult.params } };
      }
      return { blocked: false, params: hookResult.params };
    }
  } catch (err) {
    const toolCallId = args.toolCallId ? ` toolCallId=${args.toolCallId}` : "";
    log.warn(`before_tool_call hook failed: tool=${toolName}${toolCallId} error=${String(err)}`);
  }

  return { blocked: false, params };
}

export function wrapToolWithBeforeToolCallHook(
  tool: AnyAgentTool,
  ctx?: HookContext,
): AnyAgentTool {
  const execute = tool.execute;
  if (!execute) {
    return tool;
  }
  const toolName = tool.name || "tool";
  const wrappedTool: AnyAgentTool = {
    ...tool,
    execute: async (toolCallId, params, signal, onUpdate) => {
      const outcome = await runBeforeToolCallHook({
        toolName,
        params,
        toolCallId,
        ctx,
      });
      if (outcome.blocked) {
        throw new Error(outcome.reason);
      }
      const normalizedToolName = normalizeToolName(toolName || "tool");
      try {
        const result = await execute(toolCallId, outcome.params, signal, onUpdate);
        await recordLoopOutcome({
          ctx,
          toolName: normalizedToolName,
          toolParams: outcome.params,
          toolCallId,
          result,
        });
        return result;
      } catch (err) {
        await recordLoopOutcome({
          ctx,
          toolName: normalizedToolName,
          toolParams: outcome.params,
          toolCallId,
          error: err,
        });
        throw err;
      }
    },
  };
  Object.defineProperty(wrappedTool, BEFORE_TOOL_CALL_WRAPPED, {
    value: true,
    enumerable: true,
  });
  return wrappedTool;
}

export function isToolWrappedWithBeforeToolCallHook(tool: AnyAgentTool): boolean {
  const taggedTool = tool as unknown as Record<symbol, unknown>;
  return taggedTool[BEFORE_TOOL_CALL_WRAPPED] === true;
}

export const __testing = {
  BEFORE_TOOL_CALL_WRAPPED,
  runBeforeToolCallHook,
  isPlainObject,
};
