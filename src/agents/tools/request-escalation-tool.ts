import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../core/config/config.js";
import { resolveAgentIdFromSessionKey } from "../../routing/session-key.js";
import { optionalStringEnum, stringEnum } from "../schema/typebox.js";
import type { AnyAgentTool } from "./common.js";
import { jsonResult, readStringParam } from "./common.js";
import { callGatewayTool, readGatewayCallOptions } from "./gateway.js";
import { type EscalationLevel, getEscalationManager } from "./permission-escalation.js";

const ESCALATION_LEVELS = ["read", "write", "execute", "admin"] as const;

const RequestEscalationSchema = Type.Object(
  {
    requestedLevel: stringEnum(ESCALATION_LEVELS),
    currentLevel: optionalStringEnum(ESCALATION_LEVELS),
    reason: Type.String({ minLength: 1 }),
    scope: Type.Optional(Type.String()),
    toolName: Type.Optional(Type.String()),
    taskId: Type.Optional(Type.String()),
    gatewayUrl: Type.Optional(Type.String()),
    gatewayToken: Type.Optional(Type.String()),
    timeoutMs: Type.Optional(Type.Integer({ minimum: 1 })),
  },
  { additionalProperties: false },
);

function resolveApprovalTimeoutMs(config?: MarvConfig): number {
  const configured = config?.autonomy?.escalation?.approvalTimeoutSeconds;
  const seconds = typeof configured === "number" && Number.isFinite(configured) ? configured : 120;
  return Math.max(1_000, Math.floor(seconds * 1000));
}

async function requestEscalationApproval(params: {
  requestId: string;
  requestedLevel: EscalationLevel;
  reason: string;
  scope?: string;
  taskId: string;
  agentSessionKey?: string;
  config?: MarvConfig;
  gatewayOptions: ReturnType<typeof readGatewayCallOptions>;
}): Promise<"allow-once" | "allow-always" | "deny" | "unavailable"> {
  try {
    const result = await callGatewayTool<{ decision?: string }>(
      "exec.approval.request",
      params.gatewayOptions,
      {
        id: params.requestId,
        command: `request_escalation ${params.requestedLevel}: ${params.reason}`,
        kind: "permission-escalation",
        ask: "always",
        sessionKey: params.agentSessionKey ?? null,
        resolvedPath: params.scope ?? null,
        timeoutMs: resolveApprovalTimeoutMs(params.config),
      },
      { expectFinal: true },
    );
    if (result?.decision === "allow-once" || result?.decision === "allow-always") {
      return result.decision;
    }
    if (result?.decision === "deny") {
      return "deny";
    }
    return "unavailable";
  } catch {
    return "unavailable";
  }
}

export function createRequestEscalationTool(opts?: {
  agentSessionKey?: string;
  config?: MarvConfig;
}): AnyAgentTool {
  return {
    label: "Permission Escalation",
    name: "request_escalation",
    description: "Request elevated task-scoped permissions for sensitive operations.",
    parameters: RequestEscalationSchema,
    execute: async (_toolCallId, rawParams) => {
      const params =
        rawParams && typeof rawParams === "object" && !Array.isArray(rawParams)
          ? (rawParams as Record<string, unknown>)
          : {};
      const requestedLevel = readStringParam(params, "requestedLevel", {
        required: true,
      }) as EscalationLevel;
      const currentLevel = (readStringParam(params, "currentLevel") ?? "read") as EscalationLevel;
      const reason = readStringParam(params, "reason", {
        required: true,
      });
      const scope = readStringParam(params, "scope");
      const toolName = readStringParam(params, "toolName");
      const taskId =
        readStringParam(params, "taskId") ?? opts?.agentSessionKey ?? `task-${Date.now()}`;
      const agentId = resolveAgentIdFromSessionKey(opts?.agentSessionKey) || "main";
      const manager = getEscalationManager();
      const request = manager.createRequest({
        agentId,
        taskId,
        currentLevel,
        requestedLevel,
        reason,
        toolName: toolName ?? undefined,
        scope: scope ?? undefined,
      });

      const gatewayOptions = readGatewayCallOptions(params);
      const decision = await requestEscalationApproval({
        requestId: request.requestId,
        requestedLevel,
        reason,
        scope: scope ?? undefined,
        taskId,
        agentSessionKey: opts?.agentSessionKey,
        config: opts?.config,
        gatewayOptions,
      });

      if (decision === "allow-always") {
        manager.recordDecision(request.requestId, "approve");
      } else if (decision === "allow-once") {
        manager.recordDecision(request.requestId, "approve-once");
      } else {
        manager.recordDecision(request.requestId, "deny");
      }

      return jsonResult({
        requestId: request.requestId,
        taskId,
        requestedLevel,
        decision,
        granted:
          decision === "allow-once" || decision === "allow-always"
            ? manager.checkPermission(taskId, requestedLevel, scope ?? undefined)
            : false,
      });
    },
  };
}
