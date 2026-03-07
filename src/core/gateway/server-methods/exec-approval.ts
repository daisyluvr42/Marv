import type { ExecApprovalForwarder } from "../../../infra/exec-approval-forwarder.js";
import {
  DEFAULT_EXEC_APPROVAL_TIMEOUT_MS,
  type ExecApprovalDecision,
} from "../../../infra/exec-approvals.js";
import { parseAgentSessionKey } from "../../../routing/session-key.js";
import type { ExecApprovalManager } from "../exec-approval-manager.js";
import {
  ErrorCodes,
  errorShape,
  formatValidationErrors,
  validateExecApprovalRequestParams,
  validateExecApprovalResolveParams,
} from "../protocol/index.js";
import type { GatewayRequestHandlers } from "./types.js";

function formatApprovalIdentifierError(params: {
  identifier: string;
  kind: "taskId" | "slug";
  matches: Array<{ id: string }>;
}): string {
  const shown = params.matches
    .slice(0, 5)
    .map((match) => match.id)
    .join(", ");
  const suffix = params.matches.length > 5 ? ", ..." : "";
  return `approval identifier "${params.identifier}" is ambiguous by ${params.kind}; use requestId instead (${shown}${suffix})`;
}

export function createExecApprovalHandlers(
  manager: ExecApprovalManager,
  opts?: { forwarder?: ExecApprovalForwarder },
): GatewayRequestHandlers {
  return {
    "exec.approval.request": async ({ params, respond, context, client }) => {
      if (!validateExecApprovalRequestParams(params)) {
        respond(
          false,
          undefined,
          errorShape(
            ErrorCodes.INVALID_REQUEST,
            `invalid exec.approval.request params: ${formatValidationErrors(
              validateExecApprovalRequestParams.errors,
            )}`,
          ),
        );
        return;
      }
      const p = params as {
        id?: string;
        command: string;
        kind?: string | null;
        taskId?: string | null;
        cwd?: string;
        host?: string;
        security?: string;
        ask?: string;
        agentId?: string;
        resolvedPath?: string;
        sessionKey?: string;
        timeoutMs?: number;
        twoPhase?: boolean;
      };
      const twoPhase = p.twoPhase === true;
      const timeoutMs =
        typeof p.timeoutMs === "number" ? p.timeoutMs : DEFAULT_EXEC_APPROVAL_TIMEOUT_MS;
      const explicitId = typeof p.id === "string" && p.id.trim().length > 0 ? p.id.trim() : null;
      if (explicitId && manager.getSnapshot(explicitId)) {
        respond(
          false,
          undefined,
          errorShape(ErrorCodes.INVALID_REQUEST, "approval id already pending"),
        );
        return;
      }
      const normalizedTaskId =
        typeof p.taskId === "string" && p.taskId.trim().length > 0 ? p.taskId.trim() : null;
      const inferredAgentId =
        (typeof p.agentId === "string" && p.agentId.trim().length > 0 ? p.agentId.trim() : null) ??
        parseAgentSessionKey(p.sessionKey ?? null)?.agentId ??
        null;
      const request = {
        command: p.command,
        kind: p.kind ?? null,
        taskId: normalizedTaskId,
        cwd: p.cwd ?? null,
        host: p.host ?? null,
        security: p.security ?? null,
        ask: p.ask ?? null,
        agentId: inferredAgentId,
        resolvedPath: p.resolvedPath ?? null,
        sessionKey: p.sessionKey ?? null,
      };
      const record = manager.create(request, timeoutMs, explicitId);
      record.requestedByConnId = client?.connId ?? null;
      record.requestedByDeviceId = client?.connect?.device?.id ?? null;
      record.requestedByClientId = client?.connect?.client?.id ?? null;
      // Use register() to synchronously add to pending map before sending any response.
      // This ensures the approval ID is valid immediately after the "accepted" response.
      let decisionPromise: Promise<
        import("../../../infra/exec-approvals.js").ExecApprovalDecision | null
      >;
      try {
        decisionPromise = manager.register(record, timeoutMs);
      } catch (err) {
        respond(
          false,
          undefined,
          errorShape(ErrorCodes.INVALID_REQUEST, `registration failed: ${String(err)}`),
        );
        return;
      }
      context.broadcast(
        "exec.approval.requested",
        {
          id: record.id,
          request: record.request,
          createdAtMs: record.createdAtMs,
          expiresAtMs: record.expiresAtMs,
        },
        { dropIfSlow: true },
      );
      void opts?.forwarder
        ?.handleRequested({
          id: record.id,
          request: record.request,
          createdAtMs: record.createdAtMs,
          expiresAtMs: record.expiresAtMs,
        })
        .catch((err) => {
          context.logGateway?.error?.(`exec approvals: forward request failed: ${String(err)}`);
        });

      // Only send immediate "accepted" response when twoPhase is requested.
      // This preserves single-response semantics for existing callers.
      if (twoPhase) {
        respond(
          true,
          {
            status: "accepted",
            id: record.id,
            createdAtMs: record.createdAtMs,
            expiresAtMs: record.expiresAtMs,
          },
          undefined,
        );
      }

      const decision = await decisionPromise;
      // Send final response with decision for callers using expectFinal:true.
      respond(
        true,
        {
          id: record.id,
          decision,
          createdAtMs: record.createdAtMs,
          expiresAtMs: record.expiresAtMs,
        },
        undefined,
      );
    },
    "exec.approval.waitDecision": async ({ params, respond }) => {
      const p = params as { id?: string };
      const requestedId = typeof p.id === "string" ? p.id.trim() : "";
      if (!requestedId) {
        respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "id is required"));
        return;
      }
      const resolved = manager.resolveIdentifier(requestedId);
      if (resolved.status === "missing") {
        respond(
          false,
          undefined,
          errorShape(ErrorCodes.INVALID_REQUEST, "approval expired or not found"),
        );
        return;
      }
      if (resolved.status === "ambiguous") {
        respond(
          false,
          undefined,
          errorShape(
            ErrorCodes.INVALID_REQUEST,
            formatApprovalIdentifierError({
              identifier: requestedId,
              kind: resolved.kind,
              matches: resolved.matches,
            }),
          ),
        );
        return;
      }
      const decisionPromise = manager.awaitDecision(resolved.record.id);
      if (!decisionPromise) {
        respond(
          false,
          undefined,
          errorShape(ErrorCodes.INVALID_REQUEST, "approval expired or not found"),
        );
        return;
      }
      // Capture snapshot before await (entry may be deleted after grace period)
      const snapshot = manager.getSnapshot(resolved.record.id);
      const decision = await decisionPromise;
      // Return decision (can be null on timeout) - let clients handle via askFallback
      respond(
        true,
        {
          id: resolved.record.id,
          decision,
          createdAtMs: snapshot?.createdAtMs,
          expiresAtMs: snapshot?.expiresAtMs,
        },
        undefined,
      );
    },
    "exec.approval.resolve": async ({ params, respond, client, context }) => {
      if (!validateExecApprovalResolveParams(params)) {
        respond(
          false,
          undefined,
          errorShape(
            ErrorCodes.INVALID_REQUEST,
            `invalid exec.approval.resolve params: ${formatValidationErrors(
              validateExecApprovalResolveParams.errors,
            )}`,
          ),
        );
        return;
      }
      const p = params as { id: string; decision: string };
      const decision = p.decision as ExecApprovalDecision;
      if (decision !== "allow-once" && decision !== "allow-always" && decision !== "deny") {
        respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "invalid decision"));
        return;
      }
      const resolved = manager.resolveIdentifier(p.id);
      if (resolved.status === "missing") {
        respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "unknown approval id"));
        return;
      }
      if (resolved.status === "ambiguous") {
        respond(
          false,
          undefined,
          errorShape(
            ErrorCodes.INVALID_REQUEST,
            formatApprovalIdentifierError({
              identifier: p.id,
              kind: resolved.kind,
              matches: resolved.matches,
            }),
          ),
        );
        return;
      }
      const resolvedBy = client?.connect?.client?.displayName ?? client?.connect?.client?.id;
      const ok = manager.resolve(resolved.record.id, decision, resolvedBy ?? null);
      if (!ok) {
        respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "unknown approval id"));
        return;
      }
      context.broadcast(
        "exec.approval.resolved",
        { id: resolved.record.id, decision, resolvedBy, ts: Date.now() },
        { dropIfSlow: true },
      );
      void opts?.forwarder
        ?.handleResolved({ id: resolved.record.id, decision, resolvedBy, ts: Date.now() })
        .catch((err) => {
          context.logGateway?.error?.(`exec approvals: forward resolve failed: ${String(err)}`);
        });
      respond(true, { ok: true }, undefined);
    },
  };
}
