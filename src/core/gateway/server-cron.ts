import { resolveDefaultAgentId } from "../../agents/agent-scope.js";
import type { CliDeps } from "../../cli/deps.js";
import { runCronIsolatedAgentTurn } from "../../cron/isolated-agent.js";
import { appendCronRunLog, resolveCronRunLogPath } from "../../cron/run-log.js";
import { CronService } from "../../cron/service.js";
import { resolveCronStorePath } from "../../cron/store.js";
import type { CronJobCreate } from "../../cron/types.js";
import { normalizeHttpWebhookUrl } from "../../cron/webhook-url.js";
import { formatErrorMessage } from "../../infra/errors.js";
import { runHeartbeatOnce } from "../../infra/heartbeat/heartbeat-runner.js";
import { requestHeartbeatNow } from "../../infra/heartbeat/heartbeat-wake.js";
import { resolveMarvPackageRoot } from "../../infra/marv-root.js";
import { fetchWithSsrFGuard } from "../../infra/net/fetch-guard.js";
import { SsrFBlockedError } from "../../infra/net/ssrf.js";
import { scheduleGatewaySigusr1Restart } from "../../infra/restart.js";
import { enqueueSystemEvent } from "../../infra/system-events.js";
import { normalizeUpdateChannel } from "../../infra/update/update-channels.js";
import {
  buildUpdateNotificationPrompt,
  checkForUpdate,
  formatAvailableUpdateSummary,
  resolveUpdateCheckEnabled,
  resolveUpdateCheckIntervalMs,
  shouldNotifyForVersion,
} from "../../infra/update/update-notify.js";
import { runGatewayUpdate } from "../../infra/update/update-runner.js";
import { getChildLogger } from "../../logging.js";
import {
  formatSoulMemoryMaintenanceSummary,
  runSoulMemoryMaintenance,
} from "../../memory/storage/soul-memory-maintenance.js";
import { normalizeAgentId, toAgentStoreSessionKey } from "../../routing/session-key.js";
import { defaultRuntime } from "../../runtime.js";
import { loadConfig } from "../config/config.js";
import {
  canonicalizeMainSessionAlias,
  resolveAgentIdFromSessionKey,
  resolveAgentMainSessionKey,
} from "../config/sessions.js";
import { resolveStorePath } from "../config/sessions/paths.js";

export type GatewayCronState = {
  cron: CronService;
  storePath: string;
  cronEnabled: boolean;
};

const CRON_WEBHOOK_TIMEOUT_MS = 10_000;
const SOUL_MAINTENANCE_JOB_NAME = "Soul Memory Maintenance";
const SOUL_MAINTENANCE_JOB_DESCRIPTION =
  "Daily deterministic maintenance for soul memory (decay, promotion, dedupe, consolidation, conflict detection).";
const SOUL_MAINTENANCE_CRON_EXPR = "20 3 * * *";
const SOUL_MAINTENANCE_TASK = "soulMemoryNightlyMaintenance";
const SUPPORTED_SOUL_MAINTENANCE_TASKS = new Set([
  "soulMemoryMaintenance",
  "soulMemoryNightlyMaintenance",
]);
const UPDATE_CHECK_JOB_NAME = "Update Check";
const UPDATE_CHECK_JOB_DESCRIPTION =
  "Periodically check for new Marv versions and notify the user through the last active channel.";
const UPDATE_CHECK_TASK = "updateCheck";

function redactWebhookUrl(url: string): string {
  try {
    const parsed = new URL(url);
    return `${parsed.origin}${parsed.pathname}`;
  } catch {
    return "<invalid-webhook-url>";
  }
}

type CronWebhookTarget = {
  url: string;
  source: "delivery" | "legacy";
};

function resolveCronWebhookTarget(params: {
  delivery?: { mode?: string; to?: string };
  legacyNotify?: boolean;
  legacyWebhook?: string;
}): CronWebhookTarget | null {
  const mode = params.delivery?.mode?.trim().toLowerCase();
  if (mode === "webhook") {
    const url = normalizeHttpWebhookUrl(params.delivery?.to);
    return url ? { url, source: "delivery" } : null;
  }

  if (params.legacyNotify) {
    const legacyUrl = normalizeHttpWebhookUrl(params.legacyWebhook);
    if (legacyUrl) {
      return { url: legacyUrl, source: "legacy" };
    }
  }

  return null;
}

export function buildGatewayCronService(params: {
  cfg: ReturnType<typeof loadConfig>;
  deps: CliDeps;
  broadcast: (event: string, payload: unknown, opts?: { dropIfSlow?: boolean }) => void;
}): GatewayCronState {
  const cronLogger = getChildLogger({ module: "cron" });
  const storePath = resolveCronStorePath(params.cfg.cron?.store);
  const cronEnabled = process.env.MARV_SKIP_CRON !== "1" && params.cfg.cron?.enabled !== false;

  const resolveCronAgent = (requested?: string | null) => {
    const runtimeConfig = loadConfig();
    const normalized =
      typeof requested === "string" && requested.trim() ? normalizeAgentId(requested) : undefined;
    const hasAgent =
      normalized !== undefined &&
      Array.isArray(runtimeConfig.agents?.list) &&
      runtimeConfig.agents.list.some(
        (entry) =>
          entry && typeof entry.id === "string" && normalizeAgentId(entry.id) === normalized,
      );
    const agentId = hasAgent ? normalized : resolveDefaultAgentId(runtimeConfig);
    return { agentId, cfg: runtimeConfig };
  };

  const resolveCronSessionKey = (params: {
    runtimeConfig: ReturnType<typeof loadConfig>;
    agentId: string;
    requestedSessionKey?: string | null;
  }) => {
    const requested = params.requestedSessionKey?.trim();
    if (!requested) {
      return resolveAgentMainSessionKey({
        cfg: params.runtimeConfig,
        agentId: params.agentId,
      });
    }
    const candidate = toAgentStoreSessionKey({
      agentId: params.agentId,
      requestKey: requested,
      mainKey: params.runtimeConfig.session?.mainKey,
    });
    const canonical = canonicalizeMainSessionAlias({
      cfg: params.runtimeConfig,
      agentId: params.agentId,
      sessionKey: candidate,
    });
    if (canonical !== "global") {
      const sessionAgentId = resolveAgentIdFromSessionKey(canonical);
      if (normalizeAgentId(sessionAgentId) !== normalizeAgentId(params.agentId)) {
        return resolveAgentMainSessionKey({
          cfg: params.runtimeConfig,
          agentId: params.agentId,
        });
      }
    }
    return canonical;
  };

  const resolveCronWakeTarget = (opts?: { agentId?: string; sessionKey?: string | null }) => {
    const runtimeConfig = loadConfig();
    const requestedAgentId = opts?.agentId ? resolveCronAgent(opts.agentId).agentId : undefined;
    const derivedAgentId =
      requestedAgentId ??
      (opts?.sessionKey
        ? normalizeAgentId(resolveAgentIdFromSessionKey(opts.sessionKey))
        : undefined);
    const agentId = derivedAgentId || undefined;
    const sessionKey =
      opts?.sessionKey && agentId
        ? resolveCronSessionKey({
            runtimeConfig,
            agentId,
            requestedSessionKey: opts.sessionKey,
          })
        : undefined;
    return { runtimeConfig, agentId, sessionKey };
  };

  const defaultAgentId = resolveDefaultAgentId(params.cfg);
  const resolveSessionStorePath = (agentId?: string) =>
    resolveStorePath(params.cfg.session?.store, {
      agentId: agentId ?? defaultAgentId,
    });
  const sessionStorePath = resolveSessionStorePath(defaultAgentId);
  const warnedLegacyWebhookJobs = new Set<string>();

  const cron = new CronService({
    storePath,
    cronEnabled,
    cronConfig: params.cfg.cron,
    defaultAgentId,
    resolveSessionStorePath,
    sessionStorePath,
    enqueueSystemEvent: (text, opts) => {
      const { agentId, cfg: runtimeConfig } = resolveCronAgent(opts?.agentId);
      const sessionKey = resolveCronSessionKey({
        runtimeConfig,
        agentId,
        requestedSessionKey: opts?.sessionKey,
      });
      enqueueSystemEvent(text, { sessionKey, contextKey: opts?.contextKey });
    },
    requestHeartbeatNow: (opts) => {
      const { agentId, sessionKey } = resolveCronWakeTarget(opts);
      requestHeartbeatNow({
        reason: opts?.reason,
        agentId,
        sessionKey,
      });
    },
    runHeartbeatOnce: async (opts) => {
      const { runtimeConfig, agentId, sessionKey } = resolveCronWakeTarget(opts);
      return await runHeartbeatOnce({
        cfg: runtimeConfig,
        reason: opts?.reason,
        agentId,
        sessionKey,
        deps: { ...params.deps, runtime: defaultRuntime },
      });
    },
    runIsolatedAgentJob: async ({ job, message }) => {
      const { agentId, cfg: runtimeConfig } = resolveCronAgent(job.agentId);
      return await runCronIsolatedAgentTurn({
        cfg: runtimeConfig,
        deps: params.deps,
        job,
        message,
        agentId,
        sessionKey: `cron:${job.id}`,
        lane: "cron",
      });
    },
    runSystemTask: async ({ job, task }) => {
      if (SUPPORTED_SOUL_MAINTENANCE_TASKS.has(task)) {
        const runtimeConfig = loadConfig();
        const report = runSoulMemoryMaintenance({
          cfg: runtimeConfig,
          nowMs: Date.now(),
          agentId: job.agentId,
        });
        if (report.agents.length > 0 && report.failedAgents >= report.agents.length) {
          return {
            status: "error",
            error: "soul maintenance failed for all agents",
            summary: formatSoulMemoryMaintenanceSummary(report),
          };
        }
        return {
          status: "ok",
          summary: formatSoulMemoryMaintenanceSummary(report),
        };
      }

      if (task === UPDATE_CHECK_TASK) {
        const runtimeConfig = loadConfig();
        const update = await checkForUpdate({
          cfg: runtimeConfig,
          timeoutMs: 2_500,
          fetchGit: true,
        });
        if (!update.available || !update.latestVersion) {
          return { status: "skipped", summary: "Marv is up to date." };
        }
        if (update.installKind === "git" && runtimeConfig.update?.autoApplyCron === true) {
          const root =
            (await resolveMarvPackageRoot({
              moduleUrl: import.meta.url,
              argv1: process.argv[1],
              cwd: process.cwd(),
            })) ?? process.cwd();
          const updateResult = await runGatewayUpdate({
            timeoutMs: 20 * 60_000,
            cwd: root,
            argv1: process.argv[1],
            channel: normalizeUpdateChannel(runtimeConfig.update?.channel) ?? undefined,
            approval: runtimeConfig.update?.approval,
          });
          if (updateResult.status === "ok") {
            job.state.lastNotifiedVersion = update.latestVersion;
            job.state.lastNotifiedTag = update.tag;
            scheduleGatewaySigusr1Restart({
              reason: "cron.updateCheck",
            });
          }
          return {
            status:
              updateResult.status === "ok"
                ? "ok"
                : updateResult.status === "skipped"
                  ? "skipped"
                  : "error",
            error: updateResult.status === "error" ? updateResult.reason : undefined,
            summary:
              updateResult.status === "ok"
                ? `Applied git update ${update.currentVersion} -> ${update.latestVersion}.`
                : (updateResult.reason ?? formatAvailableUpdateSummary(update)),
          };
        }
        if (
          !shouldNotifyForVersion({
            update,
            lastNotifiedVersion: job.state.lastNotifiedVersion,
            lastNotifiedTag: job.state.lastNotifiedTag,
          })
        ) {
          return {
            status: "skipped",
            summary: `Already notified about ${formatAvailableUpdateSummary(update)}`,
          };
        }

        const result = await runCronIsolatedAgentTurn({
          cfg: runtimeConfig,
          deps: params.deps,
          job,
          message: buildUpdateNotificationPrompt(update),
          agentId: job.agentId,
          sessionKey: `cron:${job.id}`,
          lane: "cron",
        });
        if (result.status === "ok") {
          job.state.lastNotifiedVersion = update.latestVersion;
          job.state.lastNotifiedTag = update.tag;
        }
        return {
          status: result.status,
          error: result.error,
          summary: result.summary ?? formatAvailableUpdateSummary(update),
          sessionId: result.sessionId,
          sessionKey: result.sessionKey,
          model: result.model,
          provider: result.provider,
          usage: result.usage,
        };
      }

      return { status: "skipped", error: "unsupported system task" };
    },
    log: getChildLogger({ module: "cron", storePath }),
    onEvent: (evt) => {
      params.broadcast("cron", evt, { dropIfSlow: true });
      if (evt.action === "finished") {
        const webhookToken = params.cfg.cron?.webhookToken?.trim();
        const legacyWebhook = params.cfg.cron?.webhook?.trim();
        const job = cron.getJob(evt.jobId);
        const legacyNotify = (job as { notify?: unknown } | undefined)?.notify === true;
        const webhookTarget = resolveCronWebhookTarget({
          delivery:
            job?.delivery && typeof job.delivery.mode === "string"
              ? { mode: job.delivery.mode, to: job.delivery.to }
              : undefined,
          legacyNotify,
          legacyWebhook,
        });

        if (!webhookTarget && job?.delivery?.mode === "webhook") {
          cronLogger.warn(
            {
              jobId: evt.jobId,
              deliveryTo: job.delivery.to,
            },
            "cron: skipped webhook delivery, delivery.to must be a valid http(s) URL",
          );
        }

        if (webhookTarget?.source === "legacy" && !warnedLegacyWebhookJobs.has(evt.jobId)) {
          warnedLegacyWebhookJobs.add(evt.jobId);
          cronLogger.warn(
            {
              jobId: evt.jobId,
              legacyWebhook: redactWebhookUrl(webhookTarget.url),
            },
            "cron: deprecated notify+cron.webhook fallback in use, migrate to delivery.mode=webhook with delivery.to",
          );
        }

        if (webhookTarget && evt.summary) {
          const headers: Record<string, string> = {
            "Content-Type": "application/json",
          };
          if (webhookToken) {
            headers.Authorization = `Bearer ${webhookToken}`;
          }
          const abortController = new AbortController();
          const timeout = setTimeout(() => {
            abortController.abort();
          }, CRON_WEBHOOK_TIMEOUT_MS);

          void (async () => {
            try {
              const result = await fetchWithSsrFGuard({
                url: webhookTarget.url,
                init: {
                  method: "POST",
                  headers,
                  body: JSON.stringify(evt),
                  signal: abortController.signal,
                },
              });
              await result.release();
            } catch (err) {
              if (err instanceof SsrFBlockedError) {
                cronLogger.warn(
                  {
                    reason: formatErrorMessage(err),
                    jobId: evt.jobId,
                    webhookUrl: redactWebhookUrl(webhookTarget.url),
                  },
                  "cron: webhook delivery blocked by SSRF guard",
                );
              } else {
                cronLogger.warn(
                  {
                    err: formatErrorMessage(err),
                    jobId: evt.jobId,
                    webhookUrl: redactWebhookUrl(webhookTarget.url),
                  },
                  "cron: webhook delivery failed",
                );
              }
            } finally {
              clearTimeout(timeout);
            }
          })();
        }
        const logPath = resolveCronRunLogPath({
          storePath,
          jobId: evt.jobId,
        });
        void appendCronRunLog(logPath, {
          ts: Date.now(),
          jobId: evt.jobId,
          action: "finished",
          status: evt.status,
          error: evt.error,
          summary: evt.summary,
          sessionId: evt.sessionId,
          sessionKey: evt.sessionKey,
          runAtMs: evt.runAtMs,
          durationMs: evt.durationMs,
          nextRunAtMs: evt.nextRunAtMs,
          model: evt.model,
          provider: evt.provider,
          usage: evt.usage,
        }).catch((err) => {
          cronLogger.warn({ err: String(err), logPath }, "cron: run log append failed");
        });
      }
    },
  });

  return { cron, storePath, cronEnabled };
}

export async function ensureSoulMemoryMaintenanceCronJob(params: {
  cron: CronService;
  log?: { info: (obj: unknown, msg?: string) => void; warn: (obj: unknown, msg?: string) => void };
}) {
  const logger = params.log ?? getChildLogger({ module: "cron" });
  const status = await params.cron.status();
  if (!status.enabled) {
    return;
  }

  const existingJobs = await params.cron.list({ includeDisabled: true });
  const managed = existingJobs.find(
    (job) =>
      job.payload.kind === "systemTask" && SUPPORTED_SOUL_MAINTENANCE_TASKS.has(job.payload.task),
  );

  if (!managed) {
    const desired: CronJobCreate = {
      name: SOUL_MAINTENANCE_JOB_NAME,
      description: SOUL_MAINTENANCE_JOB_DESCRIPTION,
      enabled: true,
      deleteAfterRun: false,
      schedule: {
        kind: "cron",
        expr: SOUL_MAINTENANCE_CRON_EXPR,
        staggerMs: 0,
      },
      sessionTarget: "main",
      wakeMode: "next-heartbeat",
      payload: {
        kind: "systemTask",
        task: SOUL_MAINTENANCE_TASK,
      },
    };
    const created = await params.cron.add(desired);
    logger.info(
      { jobId: created.id, nextRunAtMs: created.state.nextRunAtMs ?? null },
      "cron: added managed soul-memory maintenance job",
    );
    return;
  }

  const needsScheduleUpdate =
    managed.schedule.kind !== "cron" ||
    managed.schedule.expr !== SOUL_MAINTENANCE_CRON_EXPR ||
    (managed.schedule.staggerMs ?? 0) !== 0;
  const needsShapeUpdate =
    !managed.enabled ||
    managed.deleteAfterRun !== false ||
    managed.sessionTarget !== "main" ||
    managed.wakeMode !== "next-heartbeat" ||
    managed.payload.kind !== "systemTask" ||
    managed.payload.task !== SOUL_MAINTENANCE_TASK ||
    managed.delivery !== undefined;
  const needsMetadataUpdate =
    managed.name !== SOUL_MAINTENANCE_JOB_NAME ||
    managed.description !== SOUL_MAINTENANCE_JOB_DESCRIPTION;

  if (!needsScheduleUpdate && !needsShapeUpdate && !needsMetadataUpdate) {
    return;
  }

  const patch: Parameters<CronService["update"]>[1] = {};
  if (needsMetadataUpdate) {
    patch.name = SOUL_MAINTENANCE_JOB_NAME;
    patch.description = SOUL_MAINTENANCE_JOB_DESCRIPTION;
  }
  if (needsScheduleUpdate) {
    patch.schedule = {
      kind: "cron",
      expr: SOUL_MAINTENANCE_CRON_EXPR,
      staggerMs: 0,
    };
  }
  if (needsShapeUpdate) {
    patch.enabled = true;
    patch.deleteAfterRun = false;
    patch.sessionTarget = "main";
    patch.wakeMode = "next-heartbeat";
    patch.payload = {
      kind: "systemTask",
      task: SOUL_MAINTENANCE_TASK,
    };
    patch.delivery = { mode: "none" };
  }

  const updated = await params.cron.update(managed.id, patch);
  logger.info(
    { jobId: updated.id, nextRunAtMs: updated.state.nextRunAtMs ?? null },
    "cron: updated managed soul-memory maintenance job",
  );
}

export async function ensureUpdateCheckCronJob(params: {
  cron: CronService;
  cfg?: ReturnType<typeof loadConfig>;
  log?: { info: (obj: unknown, msg?: string) => void; warn: (obj: unknown, msg?: string) => void };
}) {
  const logger = params.log ?? getChildLogger({ module: "cron" });
  const status = await params.cron.status();
  if (!status.enabled) {
    return;
  }

  const runtimeConfig = params.cfg ?? loadConfig();
  const desiredIntervalMs = resolveUpdateCheckIntervalMs(runtimeConfig);
  const desiredEnabled = resolveUpdateCheckEnabled(runtimeConfig);
  const existingJobs = await params.cron.list({ includeDisabled: true });
  const managed = existingJobs.find(
    (job) => job.payload.kind === "systemTask" && job.payload.task === UPDATE_CHECK_TASK,
  );

  if (!managed) {
    const created = await params.cron.add({
      name: UPDATE_CHECK_JOB_NAME,
      description: UPDATE_CHECK_JOB_DESCRIPTION,
      enabled: desiredEnabled,
      deleteAfterRun: false,
      schedule: {
        kind: "every",
        everyMs: desiredIntervalMs,
      },
      sessionTarget: "isolated",
      wakeMode: "next-heartbeat",
      payload: {
        kind: "systemTask",
        task: UPDATE_CHECK_TASK,
      },
      delivery: {
        mode: "announce",
        channel: "last",
      },
    });
    logger.info(
      { jobId: created.id, nextRunAtMs: created.state.nextRunAtMs ?? null },
      "cron: added managed update-check job",
    );
    return;
  }

  const needsScheduleUpdate =
    managed.schedule.kind !== "every" || managed.schedule.everyMs !== desiredIntervalMs;
  const needsShapeUpdate =
    managed.enabled !== desiredEnabled ||
    managed.deleteAfterRun !== false ||
    managed.sessionTarget !== "isolated" ||
    managed.wakeMode !== "next-heartbeat" ||
    managed.payload.kind !== "systemTask" ||
    managed.payload.task !== UPDATE_CHECK_TASK ||
    managed.delivery?.mode !== "announce" ||
    managed.delivery?.channel !== "last" ||
    managed.delivery?.to !== undefined;
  const needsMetadataUpdate =
    managed.name !== UPDATE_CHECK_JOB_NAME || managed.description !== UPDATE_CHECK_JOB_DESCRIPTION;

  if (!needsScheduleUpdate && !needsShapeUpdate && !needsMetadataUpdate) {
    return;
  }

  const patch: Parameters<CronService["update"]>[1] = {};
  if (needsMetadataUpdate) {
    patch.name = UPDATE_CHECK_JOB_NAME;
    patch.description = UPDATE_CHECK_JOB_DESCRIPTION;
  }
  if (needsScheduleUpdate) {
    patch.schedule = {
      kind: "every",
      everyMs: desiredIntervalMs,
    };
  }
  if (needsShapeUpdate) {
    patch.enabled = desiredEnabled;
    patch.deleteAfterRun = false;
    patch.sessionTarget = "isolated";
    patch.wakeMode = "next-heartbeat";
    patch.payload = {
      kind: "systemTask",
      task: UPDATE_CHECK_TASK,
    };
    patch.delivery = {
      mode: "announce",
      channel: "last",
      to: "",
    };
  }

  const updated = await params.cron.update(managed.id, patch);
  logger.info(
    { jobId: updated.id, nextRunAtMs: updated.state.nextRunAtMs ?? null },
    "cron: updated managed update-check job",
  );
}
