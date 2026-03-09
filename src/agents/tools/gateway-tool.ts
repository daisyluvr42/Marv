import path from "node:path";
import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../core/config/config.js";
import { resolveConfigSnapshotHash } from "../../core/config/io.js";
import { extractDeliveryInfo } from "../../core/config/sessions.js";
import {
  formatDoctorNonInteractiveHint,
  type RestartSentinelPayload,
  writeRestartSentinel,
} from "../../infra/restart-sentinel.js";
import { scheduleGatewaySigusr1Restart } from "../../infra/restart.js";
import { stringEnum } from "../schema/typebox.js";
import { type AnyAgentTool, jsonResult, readStringParam } from "./common.js";
import { callGatewayTool } from "./gateway.js";

const DEFAULT_UPDATE_TIMEOUT_MS = 20 * 60_000;

function resolveBaseHashFromSnapshot(snapshot: unknown): string | undefined {
  if (!snapshot || typeof snapshot !== "object") {
    return undefined;
  }
  const hashValue = (snapshot as { hash?: unknown }).hash;
  const rawValue = (snapshot as { raw?: unknown }).raw;
  const hash = resolveConfigSnapshotHash({
    hash: typeof hashValue === "string" ? hashValue : undefined,
    raw: typeof rawValue === "string" ? rawValue : undefined,
  });
  return hash ?? undefined;
}

const GATEWAY_ACTIONS = [
  "restart",
  "config.get",
  "config.schema",
  "config.apply",
  "config.patch",
  "config.patches.propose",
  "config.patches.commit",
  "config.revisions.rollback",
  "config.revisions.list",
  "ledger.events.query",
  "update.status",
  "update.run",
  "update.rollback",
] as const;

// NOTE: Using a flattened object schema instead of Type.Union([Type.Object(...), ...])
// because Claude API on Vertex AI rejects nested anyOf schemas as invalid JSON Schema.
// The discriminator (action) determines which properties are relevant; runtime validates.
const GatewayToolSchema = Type.Object({
  action: stringEnum(GATEWAY_ACTIONS),
  // restart
  delayMs: Type.Optional(Type.Number()),
  reason: Type.Optional(Type.String()),
  // config.get, config.schema, config.apply, update.run, update.status, update.rollback
  gatewayUrl: Type.Optional(Type.String()),
  gatewayToken: Type.Optional(Type.String()),
  timeoutMs: Type.Optional(Type.Number()),
  // config.apply, config.patch
  raw: Type.Optional(Type.String()),
  baseHash: Type.Optional(Type.String()),
  // semantic config patching
  naturalLanguage: Type.Optional(Type.String()),
  proposalId: Type.Optional(Type.String()),
  revision: Type.Optional(Type.String()),
  scopeType: Type.Optional(Type.String()),
  scopeId: Type.Optional(Type.String()),
  autoCommit: Type.Optional(Type.Boolean()),
  // ledger query
  conversationId: Type.Optional(Type.String()),
  taskId: Type.Optional(Type.String()),
  type: Type.Optional(Type.String()),
  fromTs: Type.Optional(Type.Number()),
  toTs: Type.Optional(Type.Number()),
  limit: Type.Optional(Type.Number()),
  // config.apply, config.patch, update.run
  sessionKey: Type.Optional(Type.String()),
  note: Type.Optional(Type.String()),
  restartDelayMs: Type.Optional(Type.Number()),
});
// NOTE: We intentionally avoid top-level `allOf`/`anyOf`/`oneOf` conditionals here:
// - OpenAI rejects tool schemas that include these keywords at the *top-level*.
// - Claude/Vertex has other JSON Schema quirks.
// Conditional requirements (like `raw` for config.apply) are enforced at runtime.

export function createGatewayTool(opts?: {
  agentSessionKey?: string;
  config?: MarvConfig;
}): AnyAgentTool {
  return {
    label: "Gateway",
    name: "gateway",
    description:
      "Restart/update the gateway, inspect deploy status, roll back to the last known good deploy, apply config changes, and run semantic config patch lifecycle (propose/commit/rollback). Use config.patch for direct partial updates, config.apply for full replacement, and ledger.events.query for config audit trails.",
    parameters: GatewayToolSchema,
    execute: async (_toolCallId, args) => {
      const params = args as Record<string, unknown>;
      const action = readStringParam(params, "action", { required: true });
      if (action === "restart") {
        if (opts?.config?.commands?.restart !== true) {
          throw new Error("Gateway restart is disabled. Set commands.restart=true to enable.");
        }
        const sessionKey =
          typeof params.sessionKey === "string" && params.sessionKey.trim()
            ? params.sessionKey.trim()
            : opts?.agentSessionKey?.trim() || undefined;
        const delayMs =
          typeof params.delayMs === "number" && Number.isFinite(params.delayMs)
            ? Math.floor(params.delayMs)
            : undefined;
        const reason =
          typeof params.reason === "string" && params.reason.trim()
            ? params.reason.trim().slice(0, 200)
            : undefined;
        const note =
          typeof params.note === "string" && params.note.trim() ? params.note.trim() : undefined;
        // Extract channel + threadId for routing after restart
        // Supports both :thread: (most channels) and :topic: (Telegram)
        const { deliveryContext, threadId } = extractDeliveryInfo(sessionKey);
        const payload: RestartSentinelPayload = {
          kind: "restart",
          status: "ok",
          ts: Date.now(),
          sessionKey,
          deliveryContext,
          threadId,
          message: note ?? reason ?? null,
          doctorHint: formatDoctorNonInteractiveHint(),
          stats: {
            mode: "gateway.restart",
            reason,
          },
        };
        try {
          await writeRestartSentinel(payload);
        } catch {
          // ignore: sentinel is best-effort
        }
        console.info(
          `gateway tool: restart requested (delayMs=${delayMs ?? "default"}, reason=${reason ?? "none"})`,
        );
        const scheduled = scheduleGatewaySigusr1Restart({
          delayMs,
          reason,
        });
        return jsonResult(scheduled);
      }

      const gatewayUrl =
        typeof params.gatewayUrl === "string" && params.gatewayUrl.trim()
          ? params.gatewayUrl.trim()
          : undefined;
      const gatewayToken =
        typeof params.gatewayToken === "string" && params.gatewayToken.trim()
          ? params.gatewayToken.trim()
          : undefined;
      const timeoutMs =
        typeof params.timeoutMs === "number" && Number.isFinite(params.timeoutMs)
          ? Math.max(1, Math.floor(params.timeoutMs))
          : undefined;
      const gatewayOpts = { gatewayUrl, gatewayToken, timeoutMs };

      const resolveGatewayWriteMeta = (): {
        sessionKey: string | undefined;
        note: string | undefined;
        restartDelayMs: number | undefined;
      } => {
        const sessionKey =
          typeof params.sessionKey === "string" && params.sessionKey.trim()
            ? params.sessionKey.trim()
            : opts?.agentSessionKey?.trim() || undefined;
        const note =
          typeof params.note === "string" && params.note.trim() ? params.note.trim() : undefined;
        const restartDelayMs =
          typeof params.restartDelayMs === "number" && Number.isFinite(params.restartDelayMs)
            ? Math.floor(params.restartDelayMs)
            : undefined;
        return { sessionKey, note, restartDelayMs };
      };

      const resolveConfigWriteParams = async (): Promise<{
        raw: string;
        baseHash: string;
        sessionKey: string | undefined;
        note: string | undefined;
        restartDelayMs: number | undefined;
      }> => {
        const raw = readStringParam(params, "raw", { required: true });
        let baseHash = readStringParam(params, "baseHash");
        if (!baseHash) {
          const snapshot = await callGatewayTool("config.get", gatewayOpts, {});
          baseHash = resolveBaseHashFromSnapshot(snapshot);
        }
        if (!baseHash) {
          throw new Error("Missing baseHash from config snapshot.");
        }
        return { raw, baseHash, ...resolveGatewayWriteMeta() };
      };

      if (action === "config.get") {
        const result = await callGatewayTool("config.get", gatewayOpts, {});
        if (result && typeof result === "object") {
          const pathValue =
            (result as { path?: unknown; configPath?: unknown }).path ??
            (result as { path?: unknown; configPath?: unknown }).configPath;
          const activeConfigPath =
            typeof pathValue === "string" && pathValue.trim() ? pathValue.trim() : undefined;
          const activeStateDir = activeConfigPath ? path.dirname(activeConfigPath) : undefined;
          return jsonResult({
            ok: true,
            result: {
              ...result,
              ...(activeConfigPath ? { activeConfigPath } : {}),
              ...(activeStateDir ? { activeStateDir } : {}),
            },
          });
        }
        return jsonResult({ ok: true, result });
      }
      if (action === "config.schema") {
        const result = await callGatewayTool("config.schema", gatewayOpts, {});
        return jsonResult({ ok: true, result });
      }
      if (action === "update.status") {
        const result = await callGatewayTool(
          "update.status",
          gatewayOpts,
          timeoutMs ? { timeoutMs } : {},
        );
        return jsonResult({ ok: true, result });
      }
      if (action === "config.apply") {
        const { raw, baseHash, sessionKey, note, restartDelayMs } =
          await resolveConfigWriteParams();
        const result = await callGatewayTool("config.apply", gatewayOpts, {
          raw,
          baseHash,
          sessionKey,
          note,
          restartDelayMs,
        });
        return jsonResult({ ok: true, result });
      }
      if (action === "config.patch") {
        const { raw, baseHash, sessionKey, note, restartDelayMs } =
          await resolveConfigWriteParams();
        const result = await callGatewayTool("config.patch", gatewayOpts, {
          raw,
          baseHash,
          sessionKey,
          note,
          restartDelayMs,
        });
        return jsonResult({ ok: true, result });
      }
      if (action === "config.patches.propose") {
        const naturalLanguage = readStringParam(params, "naturalLanguage", { required: true });
        const scopeType = readStringParam(params, "scopeType");
        const scopeId = readStringParam(params, "scopeId");
        const autoCommit = typeof params.autoCommit === "boolean" ? params.autoCommit : undefined;
        const { sessionKey, note, restartDelayMs } = resolveGatewayWriteMeta();
        const result = await callGatewayTool("config.patches.propose", gatewayOpts, {
          naturalLanguage,
          scopeType,
          scopeId,
          autoCommit,
          sessionKey,
          note,
          restartDelayMs,
        });
        return jsonResult({ ok: true, result });
      }
      if (action === "config.patches.commit") {
        const proposalId = readStringParam(params, "proposalId", { required: true });
        const { sessionKey, note, restartDelayMs } = resolveGatewayWriteMeta();
        const result = await callGatewayTool("config.patches.commit", gatewayOpts, {
          proposalId,
          sessionKey,
          note,
          restartDelayMs,
        });
        return jsonResult({ ok: true, result });
      }
      if (action === "config.revisions.rollback") {
        const revision = readStringParam(params, "revision", { required: true });
        const { sessionKey, note, restartDelayMs } = resolveGatewayWriteMeta();
        const result = await callGatewayTool("config.revisions.rollback", gatewayOpts, {
          revision,
          sessionKey,
          note,
          restartDelayMs,
        });
        return jsonResult({ ok: true, result });
      }
      if (action === "config.revisions.list") {
        const scopeType = readStringParam(params, "scopeType");
        const scopeId = readStringParam(params, "scopeId");
        const limit =
          typeof params.limit === "number" && Number.isFinite(params.limit)
            ? Math.floor(params.limit)
            : undefined;
        const result = await callGatewayTool("config.revisions.list", gatewayOpts, {
          scopeType,
          scopeId,
          limit,
        });
        return jsonResult({ ok: true, result });
      }
      if (action === "ledger.events.query") {
        const conversationId = readStringParam(params, "conversationId", { required: true });
        const taskId = readStringParam(params, "taskId");
        const type = readStringParam(params, "type");
        const fromTs =
          typeof params.fromTs === "number" && Number.isFinite(params.fromTs)
            ? Math.floor(params.fromTs)
            : undefined;
        const toTs =
          typeof params.toTs === "number" && Number.isFinite(params.toTs)
            ? Math.floor(params.toTs)
            : undefined;
        const limit =
          typeof params.limit === "number" && Number.isFinite(params.limit)
            ? Math.floor(params.limit)
            : undefined;
        const result = await callGatewayTool("ledger.events.query", gatewayOpts, {
          conversationId,
          taskId,
          type,
          fromTs,
          toTs,
          limit,
        });
        return jsonResult({ ok: true, result });
      }
      if (action === "update.run") {
        const { sessionKey, note, restartDelayMs } = resolveGatewayWriteMeta();
        const updateGatewayOpts = {
          ...gatewayOpts,
          timeoutMs: timeoutMs ?? DEFAULT_UPDATE_TIMEOUT_MS,
        };
        const result = await callGatewayTool("update.run", updateGatewayOpts, {
          sessionKey,
          note,
          restartDelayMs,
          timeoutMs: timeoutMs ?? DEFAULT_UPDATE_TIMEOUT_MS,
        });
        return jsonResult({ ok: true, result });
      }
      if (action === "update.rollback") {
        const { sessionKey, note, restartDelayMs } = resolveGatewayWriteMeta();
        const result = await callGatewayTool("update.rollback", gatewayOpts, {
          ...(timeoutMs ? { timeoutMs } : {}),
          ...(sessionKey ? { sessionKey } : {}),
          ...(note ? { note } : {}),
          ...(restartDelayMs !== undefined ? { restartDelayMs } : {}),
        });
        return jsonResult({ ok: true, result });
      }

      throw new Error(`Unknown action: ${action}`);
    },
  };
}
