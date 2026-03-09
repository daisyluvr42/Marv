import { resolveMarvPackageRoot } from "../../../infra/marv-root.js";
import {
  formatDoctorNonInteractiveHint,
  type RestartSentinelPayload,
  writeRestartSentinel,
} from "../../../infra/restart-sentinel.js";
import { scheduleGatewaySigusr1Restart } from "../../../infra/restart.js";
import { readDeployState, resolveDeployStatePath } from "../../../infra/update/deploy-state.js";
import { normalizeUpdateChannel } from "../../../infra/update/update-channels.js";
import { checkForUpdate } from "../../../infra/update/update-notify.js";
import { runGatewayRollback, runGatewayUpdate } from "../../../infra/update/update-runner.js";
import { loadConfig } from "../../config/config.js";
import { extractDeliveryInfo } from "../../config/sessions.js";
import {
  validateUpdateRollbackParams,
  validateUpdateRunParams,
  validateUpdateStatusParams,
} from "../protocol/index.js";
import { parseRestartRequestParams } from "./restart-request.js";
import type { GatewayRequestHandlers } from "./types.js";
import { assertValidParams } from "./validation.js";

async function resolveUpdateRoot(): Promise<string> {
  return (
    (await resolveMarvPackageRoot({
      moduleUrl: import.meta.url,
      argv1: process.argv[1],
      cwd: process.cwd(),
    })) ?? process.cwd()
  );
}

function normalizeTimeout(params: unknown): number | undefined {
  const timeoutMsRaw = (params as { timeoutMs?: unknown }).timeoutMs;
  return typeof timeoutMsRaw === "number" && Number.isFinite(timeoutMsRaw)
    ? Math.max(1000, Math.floor(timeoutMsRaw))
    : undefined;
}

function buildUpdateSentinel(params: {
  sessionKey: string | undefined;
  note: string | undefined;
  result: Awaited<ReturnType<typeof runGatewayUpdate>>;
}): RestartSentinelPayload {
  const { deliveryContext, threadId } = extractDeliveryInfo(params.sessionKey);
  return {
    kind: "update",
    status: params.result.status,
    ts: Date.now(),
    sessionKey: params.sessionKey,
    deliveryContext,
    threadId,
    message: params.note ?? null,
    doctorHint: formatDoctorNonInteractiveHint(),
    stats: {
      mode: params.result.mode,
      root: params.result.root ?? undefined,
      before: params.result.before ?? null,
      after: params.result.after ?? null,
      steps: params.result.steps.map((step) => ({
        name: step.name,
        command: step.command,
        cwd: step.cwd,
        durationMs: step.durationMs,
        log: {
          stdoutTail: step.stdoutTail ?? null,
          stderrTail: step.stderrTail ?? null,
          exitCode: step.exitCode ?? null,
        },
      })),
      reason: params.result.reason ?? null,
      durationMs: params.result.durationMs,
    },
  };
}

export const updateHandlers: GatewayRequestHandlers = {
  "update.run": async ({ params, respond }) => {
    if (!assertValidParams(params, validateUpdateRunParams, "update.run", respond)) {
      return;
    }
    const { sessionKey, note, restartDelayMs } = parseRestartRequestParams(params);
    const timeoutMs = normalizeTimeout(params);

    let result: Awaited<ReturnType<typeof runGatewayUpdate>>;
    try {
      const config = loadConfig();
      const configChannel = normalizeUpdateChannel(config.update?.channel);
      const root = await resolveUpdateRoot();
      result = await runGatewayUpdate({
        timeoutMs,
        cwd: root,
        argv1: process.argv[1],
        channel: configChannel ?? undefined,
        approval: config.update?.approval,
      });
    } catch (err) {
      result = {
        status: "error",
        mode: "unknown",
        reason: String(err),
        steps: [],
        durationMs: 0,
      };
    }

    const payload = buildUpdateSentinel({ sessionKey, note, result });

    let sentinelPath: string | null = null;
    try {
      sentinelPath = await writeRestartSentinel(payload);
    } catch {
      sentinelPath = null;
    }

    // Only restart the gateway when the update actually succeeded.
    // Restarting after a failed update leaves the process in a broken state
    // (corrupted node_modules, partial builds) and causes a crash loop.
    const restart =
      result.status === "ok"
        ? scheduleGatewaySigusr1Restart({
            delayMs: restartDelayMs,
            reason: "update.run",
          })
        : null;

    respond(
      true,
      {
        ok: result.status !== "error",
        result,
        restart,
        sentinel: {
          path: sentinelPath,
          payload,
        },
      },
      undefined,
    );
  },
  "update.status": async ({ params, respond }) => {
    if (!assertValidParams(params, validateUpdateStatusParams, "update.status", respond)) {
      return;
    }
    const timeoutMs = normalizeTimeout(params);
    const cfg = loadConfig();
    const root = await resolveUpdateRoot();
    const [state, update] = await Promise.all([
      readDeployState({ root }),
      checkForUpdate({
        cfg,
        timeoutMs,
        root,
        fetchGit: true,
      }).catch(() => null),
    ]);
    respond(true, {
      ok: true,
      root,
      statePath: resolveDeployStatePath(root),
      trackedBranch: cfg.update?.approval?.branch?.trim() || "main",
      autoApplyCron: cfg.update?.autoApplyCron === true,
      deployApprovalRequired: cfg.update?.approval?.required === true,
      state,
      update,
    });
  },
  "update.rollback": async ({ params, respond }) => {
    if (!assertValidParams(params, validateUpdateRollbackParams, "update.rollback", respond)) {
      return;
    }
    const { sessionKey, note, restartDelayMs } = parseRestartRequestParams(params);
    const timeoutMs = normalizeTimeout(params);

    let result: Awaited<ReturnType<typeof runGatewayRollback>>;
    try {
      const root = await resolveUpdateRoot();
      result = await runGatewayRollback({
        timeoutMs,
        cwd: root,
        argv1: process.argv[1],
      });
    } catch (err) {
      result = {
        status: "error",
        mode: "unknown",
        reason: String(err),
        steps: [],
        durationMs: 0,
      };
    }

    const payload = buildUpdateSentinel({ sessionKey, note, result });
    let sentinelPath: string | null = null;
    try {
      sentinelPath = await writeRestartSentinel(payload);
    } catch {
      sentinelPath = null;
    }

    const restart =
      result.status === "ok"
        ? scheduleGatewaySigusr1Restart({
            delayMs: restartDelayMs,
            reason: "update.rollback",
          })
        : null;

    respond(true, {
      ok: result.status !== "error",
      result,
      restart,
      sentinel: {
        path: sentinelPath,
        payload,
      },
    });
  },
};
