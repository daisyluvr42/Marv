import os from "node:os";
import path from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { CliDeps } from "../../cli/deps.js";
import { SsrFBlockedError } from "../../infra/net/ssrf.js";
import type { MarvConfig } from "../config/config.js";

const enqueueSystemEventMock = vi.fn();
const requestHeartbeatNowMock = vi.fn();
const loadConfigMock = vi.fn();
const fetchWithSsrFGuardMock = vi.fn();
const runCronIsolatedAgentTurnMock = vi.fn();
const checkForUpdateMock = vi.fn();
const runGatewayUpdateMock = vi.fn();
const scheduleGatewaySigusr1RestartMock = vi.fn();
const probeLocalModelMock = vi.fn();
const runSoulMemoryDeepConsolidationMock = vi.fn();

vi.mock("../../infra/system-events.js", () => ({
  enqueueSystemEvent: (...args: unknown[]) => enqueueSystemEventMock(...args),
}));

vi.mock("../../infra/heartbeat/heartbeat-wake.js", () => ({
  requestHeartbeatNow: (...args: unknown[]) => requestHeartbeatNowMock(...args),
}));

vi.mock("../config/config.js", async () => {
  const actual = await vi.importActual<typeof import("../config/config.js")>("../config/config.js");
  return {
    ...actual,
    loadConfig: () => loadConfigMock(),
  };
});

vi.mock("../../infra/net/fetch-guard.js", () => ({
  fetchWithSsrFGuard: (...args: unknown[]) => fetchWithSsrFGuardMock(...args),
}));

vi.mock("../../cron/isolated-agent.js", () => ({
  runCronIsolatedAgentTurn: (...args: unknown[]) => runCronIsolatedAgentTurnMock(...args),
}));

vi.mock("../../infra/update/update-notify.js", () => ({
  buildUpdateNotificationPrompt: vi.fn(() => "notify about update"),
  checkForUpdate: (...args: unknown[]) => checkForUpdateMock(...args),
  formatAvailableUpdateSummary: vi.fn(
    (update: { currentVersion: string; latestVersion: string }) =>
      `Marv v${update.latestVersion} is available (current v${update.currentVersion}). Run marv update.`,
  ),
  resolveUpdateCheckEnabled: vi.fn(
    (cfg: { update?: { checkOnStart?: boolean } }) => cfg.update?.checkOnStart !== false,
  ),
  resolveUpdateCheckIntervalMs: vi.fn(
    (cfg: { update?: { autoCheckIntervalMs?: number } }) =>
      cfg.update?.autoCheckIntervalMs ?? 24 * 60 * 60 * 1000,
  ),
  shouldNotifyForVersion: vi.fn(
    (params: {
      update: { available: boolean; latestVersion: string | null; tag: string };
      lastNotifiedVersion?: string;
      lastNotifiedTag?: string;
    }) =>
      Boolean(params.update.available) &&
      Boolean(params.update.latestVersion) &&
      (params.lastNotifiedVersion !== params.update.latestVersion ||
        params.lastNotifiedTag !== params.update.tag),
  ),
}));

vi.mock("../../infra/update/update-runner.js", () => ({
  runGatewayUpdate: (...args: unknown[]) => runGatewayUpdateMock(...args),
}));

vi.mock("../../infra/restart.js", () => ({
  scheduleGatewaySigusr1Restart: (...args: unknown[]) => scheduleGatewaySigusr1RestartMock(...args),
}));

vi.mock("../../infra/update/update-channels.js", () => ({
  normalizeUpdateChannel: vi.fn(() => undefined),
}));

vi.mock("../../infra/marv-root.js", () => ({
  resolveMarvPackageRoot: vi.fn(async () => "/tmp/marv"),
}));

vi.mock("../../memory/storage/local-llm-client.js", () => ({
  probeLocalModel: (...args: unknown[]) => probeLocalModelMock(...args),
}));

vi.mock("../../memory/storage/soul-memory-deep-consolidation.js", () => ({
  DEFAULT_DEEP_CONSOLIDATION_SCHEDULE: "20 4 * * 0",
  resolveDeepConsolidationConfig: (
    cfg: MarvConfig & {
      memory?: {
        soul?: {
          deepConsolidation?: {
            enabled?: boolean;
            schedule?: string;
            maxItems?: number;
            maxReflections?: number;
            clusterSummarization?: boolean;
            conflictJudgment?: boolean;
            crossScopeReflection?: boolean;
            model?: Record<string, unknown>;
          };
        };
      };
    },
  ) => ({
    enabled: cfg.memory?.soul?.deepConsolidation?.enabled === true,
    schedule: cfg.memory?.soul?.deepConsolidation?.schedule ?? "20 4 * * 0",
    maxItems: cfg.memory?.soul?.deepConsolidation?.maxItems ?? 500,
    maxReflections: cfg.memory?.soul?.deepConsolidation?.maxReflections ?? 5,
    clusterSummarization: cfg.memory?.soul?.deepConsolidation?.clusterSummarization !== false,
    conflictJudgment: cfg.memory?.soul?.deepConsolidation?.conflictJudgment !== false,
    crossScopeReflection: cfg.memory?.soul?.deepConsolidation?.crossScopeReflection !== false,
    model: cfg.memory?.soul?.deepConsolidation?.model ?? {},
  }),
  formatSoulMemoryDeepConsolidationSummary: vi.fn(
    (report: {
      agents: unknown[];
      failedAgents: number;
      totals: {
        llmConsolidated: number;
        llmConflictsDetected: number;
        crossScopeReflections: number;
      };
    }) =>
      `Deep consolidation complete: agents=${report.agents.length}, failed=${report.failedAgents}, consolidated=${report.totals.llmConsolidated}, conflicts=${report.totals.llmConflictsDetected}, reflections=${report.totals.crossScopeReflections}`,
  ),
  runSoulMemoryDeepConsolidation: (...args: unknown[]) =>
    runSoulMemoryDeepConsolidationMock(...args),
}));

import {
  buildGatewayCronService,
  ensureDeepConsolidationCronJob,
  ensureProactiveCheckCronJob,
  ensureProactiveDigestCronJobs,
  ensureSoulMemoryMaintenanceCronJob,
  ensureUpdateCheckCronJob,
} from "./server-cron.js";

describe("buildGatewayCronService", () => {
  beforeEach(() => {
    enqueueSystemEventMock.mockReset();
    requestHeartbeatNowMock.mockReset();
    loadConfigMock.mockReset();
    fetchWithSsrFGuardMock.mockReset();
    runCronIsolatedAgentTurnMock.mockReset();
    checkForUpdateMock.mockReset();
    runGatewayUpdateMock.mockReset();
    scheduleGatewaySigusr1RestartMock.mockReset();
    probeLocalModelMock.mockReset();
    runSoulMemoryDeepConsolidationMock.mockReset();
    runCronIsolatedAgentTurnMock.mockResolvedValue({ status: "ok", summary: "update delivered" });
    runGatewayUpdateMock.mockResolvedValue({
      status: "ok",
      mode: "git",
      steps: [],
      durationMs: 100,
    });
    scheduleGatewaySigusr1RestartMock.mockReturnValue({ scheduled: true });
    probeLocalModelMock.mockResolvedValue({
      ok: true,
      resolved: {
        api: "ollama",
        baseUrl: "http://127.0.0.1:11434",
        model: "qwen2.5:3b",
        timeoutMs: 30000,
        headers: {},
      },
    });
    runSoulMemoryDeepConsolidationMock.mockResolvedValue({
      agents: [
        {
          agentId: "main",
          llmConsolidated: 1,
          llmConflictsDetected: 0,
          crossScopeReflections: 1,
          skippedStages: [],
        },
      ],
      totals: {
        llmConsolidated: 1,
        llmConflictsDetected: 0,
        crossScopeReflections: 1,
      },
      model: {
        api: "ollama",
        baseUrl: "http://127.0.0.1:11434",
        model: "qwen2.5:3b",
        available: true,
      },
      failedAgents: 0,
    });
  });

  it("canonicalizes non-agent sessionKey to agent store key for enqueue + wake", async () => {
    const tmpDir = path.join(os.tmpdir(), `server-cron-${Date.now()}`);
    const cfg = {
      session: {
        mainKey: "main",
      },
      cron: {
        store: path.join(tmpDir, "cron.json"),
      },
    } as MarvConfig;
    loadConfigMock.mockReturnValue(cfg);

    const state = buildGatewayCronService({
      cfg,
      deps: {} as CliDeps,
      broadcast: () => {},
    });
    try {
      const job = await state.cron.add({
        name: "canonicalize-session-key",
        enabled: true,
        schedule: { kind: "at", at: new Date(1).toISOString() },
        sessionTarget: "main",
        wakeMode: "next-heartbeat",
        sessionKey: "discord:channel:ops",
        payload: { kind: "systemEvent", text: "hello" },
      });

      await state.cron.run(job.id, "force");

      expect(enqueueSystemEventMock).toHaveBeenCalledWith(
        "hello",
        expect.objectContaining({
          sessionKey: "agent:main:discord:channel:ops",
        }),
      );
      expect(requestHeartbeatNowMock).toHaveBeenCalledWith(
        expect.objectContaining({
          sessionKey: "agent:main:discord:channel:ops",
        }),
      );
    } finally {
      state.cron.stop();
    }
  });

  it("blocks private webhook URLs via SSRF-guarded fetch", async () => {
    const tmpDir = path.join(os.tmpdir(), `server-cron-ssrf-${Date.now()}`);
    const cfg = {
      session: {
        mainKey: "main",
      },
      cron: {
        store: path.join(tmpDir, "cron.json"),
      },
    } as MarvConfig;

    loadConfigMock.mockReturnValue(cfg);
    fetchWithSsrFGuardMock.mockRejectedValue(
      new SsrFBlockedError("Blocked: private/internal IP address"),
    );

    const state = buildGatewayCronService({
      cfg,
      deps: {} as CliDeps,
      broadcast: () => {},
    });
    try {
      const job = await state.cron.add({
        name: "ssrf-webhook-blocked",
        enabled: true,
        schedule: { kind: "at", at: new Date(1).toISOString() },
        sessionTarget: "main",
        wakeMode: "next-heartbeat",
        payload: { kind: "systemEvent", text: "hello" },
        delivery: {
          mode: "webhook",
          to: "http://127.0.0.1:8080/cron-finished",
        },
      });

      await state.cron.run(job.id, "force");

      expect(fetchWithSsrFGuardMock).toHaveBeenCalledOnce();
      expect(fetchWithSsrFGuardMock).toHaveBeenCalledWith({
        url: "http://127.0.0.1:8080/cron-finished",
        init: {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: expect.stringContaining('"action":"finished"'),
          signal: expect.any(AbortSignal),
        },
      });
    } finally {
      state.cron.stop();
    }
  });

  it("ensures managed soul-memory maintenance cron job exists", async () => {
    const tmpDir = path.join(os.tmpdir(), `server-cron-maintenance-${Date.now()}`);
    const cfg = {
      session: {
        mainKey: "main",
      },
      cron: {
        store: path.join(tmpDir, "cron.json"),
      },
    } as MarvConfig;
    loadConfigMock.mockReturnValue(cfg);

    const state = buildGatewayCronService({
      cfg,
      deps: {} as CliDeps,
      broadcast: () => {},
    });
    try {
      await state.cron.start();
      await ensureSoulMemoryMaintenanceCronJob({ cron: state.cron });
      const jobs = await state.cron.list({ includeDisabled: true });
      const maintenance = jobs.find(
        (job) =>
          job.payload.kind === "systemTask" && job.payload.task === "soulMemoryNightlyMaintenance",
      );
      expect(maintenance).toBeDefined();
      if (!maintenance) {
        throw new Error("maintenance job missing");
      }
      expect(maintenance.enabled).toBe(true);
      expect(maintenance.sessionTarget).toBe("main");
      expect(maintenance.wakeMode).toBe("next-heartbeat");
      expect(maintenance.delivery).toBeUndefined();
      expect(maintenance.schedule.kind).toBe("cron");
      if (maintenance.schedule.kind === "cron") {
        expect(maintenance.schedule.expr).toBe("20 3 * * *");
        expect(maintenance.schedule.staggerMs).toBe(0);
      }
    } finally {
      state.cron.stop();
    }
  });

  it("repairs drifted managed soul-memory maintenance job shape", async () => {
    const tmpDir = path.join(os.tmpdir(), `server-cron-maintenance-repair-${Date.now()}`);
    const cfg = {
      session: {
        mainKey: "main",
      },
      cron: {
        store: path.join(tmpDir, "cron.json"),
      },
    } as MarvConfig;
    loadConfigMock.mockReturnValue(cfg);

    const state = buildGatewayCronService({
      cfg,
      deps: {} as CliDeps,
      broadcast: () => {},
    });
    try {
      await state.cron.start();
      await state.cron.add({
        name: "bad maintenance",
        enabled: false,
        deleteAfterRun: true,
        schedule: { kind: "cron", expr: "0 2 * * *", staggerMs: 60_000 },
        sessionTarget: "main",
        wakeMode: "now",
        payload: { kind: "systemTask", task: "soulMemoryMaintenance" },
        delivery: { mode: "webhook", to: "https://example.invalid/bad" },
      });

      await ensureSoulMemoryMaintenanceCronJob({ cron: state.cron });

      const jobs = await state.cron.list({ includeDisabled: true });
      const maintenance = jobs.find(
        (job) =>
          job.payload.kind === "systemTask" && job.payload.task === "soulMemoryNightlyMaintenance",
      );
      expect(maintenance).toBeDefined();
      if (!maintenance) {
        throw new Error("maintenance job missing");
      }
      expect(maintenance.name).toBe("Soul Memory Maintenance");
      expect(maintenance.enabled).toBe(true);
      expect(maintenance.deleteAfterRun).toBe(false);
      expect(maintenance.wakeMode).toBe("next-heartbeat");
      expect(maintenance.delivery).toBeUndefined();
      expect(maintenance.schedule.kind).toBe("cron");
      if (maintenance.schedule.kind === "cron") {
        expect(maintenance.schedule.expr).toBe("20 3 * * *");
        expect(maintenance.schedule.staggerMs).toBe(0);
      }
    } finally {
      state.cron.stop();
    }
  });

  it("ensures managed update-check cron job exists", async () => {
    const tmpDir = path.join(os.tmpdir(), `server-cron-update-check-${Date.now()}`);
    const cfg = {
      session: {
        mainKey: "main",
      },
      cron: {
        store: path.join(tmpDir, "cron.json"),
      },
      update: {
        autoCheckIntervalMs: 12_000,
      },
    } as MarvConfig;
    loadConfigMock.mockReturnValue(cfg);

    const state = buildGatewayCronService({
      cfg,
      deps: {} as CliDeps,
      broadcast: () => {},
    });
    try {
      await state.cron.start();
      await ensureUpdateCheckCronJob({ cron: state.cron, cfg });
      const jobs = await state.cron.list({ includeDisabled: true });
      const updateCheck = jobs.find(
        (job) => job.payload.kind === "systemTask" && job.payload.task === "updateCheck",
      );
      expect(updateCheck).toBeDefined();
      if (!updateCheck) {
        throw new Error("update check job missing");
      }
      expect(updateCheck.enabled).toBe(true);
      expect(updateCheck.sessionTarget).toBe("isolated");
      expect(updateCheck.wakeMode).toBe("next-heartbeat");
      expect(updateCheck.delivery).toEqual({ mode: "announce", channel: "last" });
      expect(updateCheck.schedule).toEqual({
        kind: "every",
        everyMs: 12_000,
        anchorMs: expect.any(Number),
      });
    } finally {
      state.cron.stop();
    }
  });

  it("ensures managed deep-consolidation cron job exists", async () => {
    const tmpDir = path.join(os.tmpdir(), `server-cron-deep-consolidation-${Date.now()}`);
    const cfg = {
      session: {
        mainKey: "main",
      },
      cron: {
        store: path.join(tmpDir, "cron.json"),
      },
      memory: {
        soul: {
          deepConsolidation: {
            enabled: true,
            schedule: "15 5 * * 0",
            model: {
              provider: "ollama",
              api: "ollama",
              model: "qwen2.5:3b",
            },
          },
        },
      },
    } as MarvConfig;
    loadConfigMock.mockReturnValue(cfg);

    const state = buildGatewayCronService({
      cfg,
      deps: {} as CliDeps,
      broadcast: () => {},
    });
    try {
      await state.cron.start();
      await ensureDeepConsolidationCronJob({ cron: state.cron, cfg });
      const jobs = await state.cron.list({ includeDisabled: true });
      const deepJob = jobs.find(
        (job) =>
          job.payload.kind === "systemTask" && job.payload.task === "soulMemoryDeepConsolidation",
      );
      expect(deepJob).toBeDefined();
      if (!deepJob) {
        throw new Error("deep-consolidation job missing");
      }
      expect(deepJob.enabled).toBe(true);
      expect(deepJob.sessionTarget).toBe("main");
      expect(deepJob.wakeMode).toBe("next-heartbeat");
      expect(deepJob.schedule.kind).toBe("cron");
      if (deepJob.schedule.kind === "cron") {
        expect(deepJob.schedule.expr).toBe("15 5 * * 0");
        expect(deepJob.schedule.staggerMs).toBe(0);
      }
    } finally {
      state.cron.stop();
    }
  });

  it("runs deep consolidation system tasks through the local-model pipeline", async () => {
    const tmpDir = path.join(os.tmpdir(), `server-cron-deep-run-${Date.now()}`);
    const cfg = {
      session: {
        mainKey: "main",
      },
      cron: {
        store: path.join(tmpDir, "cron.json"),
      },
      memory: {
        soul: {
          deepConsolidation: {
            enabled: true,
            schedule: "20 4 * * 0",
            model: {
              provider: "ollama",
              api: "ollama",
              model: "qwen2.5:3b",
            },
          },
        },
      },
    } as MarvConfig;
    loadConfigMock.mockReturnValue(cfg);

    const state = buildGatewayCronService({
      cfg,
      deps: {} as CliDeps,
      broadcast: () => {},
    });
    try {
      await state.cron.start();
      await ensureDeepConsolidationCronJob({ cron: state.cron, cfg });
      const jobs = await state.cron.list({ includeDisabled: true });
      const deepJob = jobs.find(
        (job) =>
          job.payload.kind === "systemTask" && job.payload.task === "soulMemoryDeepConsolidation",
      );
      if (!deepJob) {
        throw new Error("deep-consolidation job missing");
      }

      await state.cron.run(deepJob.id, "force");

      expect(probeLocalModelMock).toHaveBeenCalledTimes(1);
      expect(runSoulMemoryDeepConsolidationMock).toHaveBeenCalledTimes(1);
    } finally {
      state.cron.stop();
    }
  });

  it("ensures managed proactive check and digest cron jobs exist", async () => {
    const tmpDir = path.join(os.tmpdir(), `server-cron-proactive-${Date.now()}`);
    const cfg = {
      session: {
        mainKey: "main",
      },
      cron: {
        store: path.join(tmpDir, "cron.json"),
      },
      autonomy: {
        proactive: {
          enabled: true,
          checkEveryMinutes: 30,
          digestTimes: ["08:00", "20:00"],
          delivery: {
            channel: "telegram",
            to: "123",
          },
        },
      },
    } as MarvConfig;
    loadConfigMock.mockReturnValue(cfg);

    const state = buildGatewayCronService({
      cfg,
      deps: {} as CliDeps,
      broadcast: () => {},
    });
    try {
      await state.cron.start();
      await ensureProactiveCheckCronJob({ cron: state.cron, cfg });
      await ensureProactiveDigestCronJobs({ cron: state.cron, cfg });
      const jobs = await state.cron.list({ includeDisabled: true });
      const proactiveCheck = jobs.find((job) => job.name === "Proactive Check");
      expect(proactiveCheck).toBeDefined();
      expect(proactiveCheck?.payload.kind).toBe("agentTurn");
      expect(proactiveCheck?.delivery).toEqual({ mode: "none" });

      const digests = jobs.filter((job) => job.name.startsWith("Proactive Digest "));
      expect(digests).toHaveLength(2);
      expect(digests[0]?.delivery).toEqual({ mode: "announce", channel: "telegram", to: "123" });
    } finally {
      state.cron.stop();
    }
  });

  it("notifies only once per available version", async () => {
    const tmpDir = path.join(os.tmpdir(), `server-cron-update-dedupe-${Date.now()}`);
    const cfg = {
      session: {
        mainKey: "main",
      },
      cron: {
        store: path.join(tmpDir, "cron.json"),
      },
      update: {
        autoCheckIntervalMs: 60_000,
      },
    } as MarvConfig;
    loadConfigMock.mockReturnValue(cfg);
    checkForUpdateMock.mockResolvedValue({
      available: true,
      currentVersion: "1.0.0",
      latestVersion: "2.0.0",
      channel: "stable",
      tag: "latest",
      installKind: "package",
    });

    const state = buildGatewayCronService({
      cfg,
      deps: {} as CliDeps,
      broadcast: () => {},
    });
    try {
      await state.cron.start();
      await ensureUpdateCheckCronJob({ cron: state.cron, cfg });
      const jobs = await state.cron.list({ includeDisabled: true });
      const updateCheck = jobs.find(
        (job) => job.payload.kind === "systemTask" && job.payload.task === "updateCheck",
      );
      if (!updateCheck) {
        throw new Error("update check job missing");
      }

      await state.cron.run(updateCheck.id, "force");
      await state.cron.run(updateCheck.id, "force");

      expect(runCronIsolatedAgentTurnMock).toHaveBeenCalledTimes(1);
      const refreshed = state.cron.getJob(updateCheck.id);
      expect(refreshed?.state.lastNotifiedVersion).toBe("2.0.0");
      expect(refreshed?.state.lastNotifiedTag).toBe("latest");
    } finally {
      state.cron.stop();
    }
  });

  it("auto-applies git updates when update.autoApplyCron is enabled", async () => {
    const tmpDir = path.join(os.tmpdir(), `server-cron-update-apply-${Date.now()}`);
    const cfg = {
      session: {
        mainKey: "main",
      },
      cron: {
        store: path.join(tmpDir, "cron.json"),
      },
      update: {
        autoApplyCron: true,
      },
    } as MarvConfig;
    loadConfigMock.mockReturnValue(cfg);
    checkForUpdateMock.mockResolvedValue({
      available: true,
      currentVersion: "abc123",
      latestVersion: "def456",
      channel: "dev",
      tag: "dev",
      installKind: "git",
    });

    const state = buildGatewayCronService({
      cfg,
      deps: {} as CliDeps,
      broadcast: () => {},
    });
    try {
      await state.cron.start();
      await ensureUpdateCheckCronJob({ cron: state.cron, cfg });
      const jobs = await state.cron.list({ includeDisabled: true });
      const updateCheck = jobs.find(
        (job) => job.payload.kind === "systemTask" && job.payload.task === "updateCheck",
      );
      if (!updateCheck) {
        throw new Error("update check job missing");
      }

      await state.cron.run(updateCheck.id, "force");

      expect(runGatewayUpdateMock).toHaveBeenCalledTimes(1);
      expect(runCronIsolatedAgentTurnMock).not.toHaveBeenCalled();
      expect(scheduleGatewaySigusr1RestartMock).toHaveBeenCalledWith(
        expect.objectContaining({ reason: "cron.updateCheck" }),
      );
    } finally {
      state.cron.stop();
    }
  });
});
