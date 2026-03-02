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

import { buildGatewayCronService, ensureSoulMemoryMaintenanceCronJob } from "./server-cron.js";

describe("buildGatewayCronService", () => {
  beforeEach(() => {
    enqueueSystemEventMock.mockReset();
    requestHeartbeatNowMock.mockReset();
    loadConfigMock.mockReset();
    fetchWithSsrFGuardMock.mockReset();
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
        (job) => job.payload.kind === "systemTask" && job.payload.task === "soulMemoryMaintenance",
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
        (job) => job.payload.kind === "systemTask" && job.payload.task === "soulMemoryMaintenance",
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
});
