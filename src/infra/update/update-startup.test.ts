import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./update-notify.js", () => ({
  checkForUpdate: vi.fn(),
  DEFAULT_UPDATE_CHECK_INTERVAL_MS: 24 * 60 * 60 * 1000,
  formatAvailableUpdateSummary: vi.fn(
    (update: { currentVersion: string; latestVersion: string }) =>
      `Marv v${update.latestVersion} is available (current v${update.currentVersion}). Run marv update.`,
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

describe("update-startup", () => {
  let suiteRoot = "";
  let suiteCase = 0;
  let tempDir: string;
  let prevStateDir: string | undefined;
  let prevNodeEnv: string | undefined;
  let prevVitest: string | undefined;
  let hadStateDir = false;
  let hadNodeEnv = false;
  let hadVitest = false;

  let checkForUpdate: (typeof import("./update-notify.js"))["checkForUpdate"];
  let runGatewayUpdateCheck: (typeof import("./update-startup.js"))["runGatewayUpdateCheck"];
  let loaded = false;

  beforeAll(async () => {
    suiteRoot = await fs.mkdtemp(path.join(os.tmpdir(), "marv-update-check-suite-"));
  });

  beforeEach(async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-01-17T10:00:00Z"));
    tempDir = path.join(suiteRoot, `case-${++suiteCase}`);
    await fs.mkdir(tempDir);
    hadStateDir = Object.prototype.hasOwnProperty.call(process.env, "MARV_STATE_DIR");
    prevStateDir = process.env.MARV_STATE_DIR;
    process.env.MARV_STATE_DIR = tempDir;

    hadNodeEnv = Object.prototype.hasOwnProperty.call(process.env, "NODE_ENV");
    prevNodeEnv = process.env.NODE_ENV;
    process.env.NODE_ENV = "test";

    hadVitest = Object.prototype.hasOwnProperty.call(process.env, "VITEST");
    prevVitest = process.env.VITEST;
    delete process.env.VITEST;

    if (!loaded) {
      ({ checkForUpdate } = await import("./update-notify.js"));
      ({ runGatewayUpdateCheck } = await import("./update-startup.js"));
      loaded = true;
    }
  });

  afterEach(async () => {
    vi.useRealTimers();
    if (hadStateDir) {
      process.env.MARV_STATE_DIR = prevStateDir;
    } else {
      delete process.env.MARV_STATE_DIR;
    }
    if (hadNodeEnv) {
      process.env.NODE_ENV = prevNodeEnv;
    } else {
      delete process.env.NODE_ENV;
    }
    if (hadVitest) {
      process.env.VITEST = prevVitest;
    } else {
      delete process.env.VITEST;
    }
  });

  afterAll(async () => {
    if (suiteRoot) {
      await fs.rm(suiteRoot, { recursive: true, force: true });
    }
    suiteRoot = "";
    suiteCase = 0;
  });

  async function runUpdateCheckAndReadState(channel: "stable" | "beta") {
    vi.mocked(checkForUpdate).mockResolvedValue({
      available: true,
      currentVersion: "1.0.0",
      latestVersion: "2.0.0",
      channel,
      tag: "latest",
      installKind: "package",
    } as Awaited<ReturnType<typeof checkForUpdate>>);

    const log = { info: vi.fn() };
    await runGatewayUpdateCheck({
      cfg: { update: { channel } },
      log,
      isNixMode: false,
      allowInTests: true,
    });

    const statePath = path.join(tempDir, "update-check.json");
    const parsed = JSON.parse(await fs.readFile(statePath, "utf-8")) as {
      lastNotifiedVersion?: string;
      lastNotifiedTag?: string;
    };
    return { log, parsed };
  }

  it("logs update hint for npm installs when newer tag exists", async () => {
    const { log, parsed } = await runUpdateCheckAndReadState("stable");

    expect(log.info).toHaveBeenCalledWith(expect.stringContaining("Marv v2.0.0 is available"));
    expect(parsed.lastNotifiedVersion).toBe("2.0.0");
  });

  it("uses latest when beta tag is older than release", async () => {
    const { log, parsed } = await runUpdateCheckAndReadState("beta");

    expect(log.info).toHaveBeenCalledWith(expect.stringContaining("Marv v2.0.0 is available"));
    expect(parsed.lastNotifiedTag).toBe("latest");
  });

  it("skips update check when disabled in config", async () => {
    const log = { info: vi.fn() };

    await runGatewayUpdateCheck({
      cfg: { update: { checkOnStart: false } },
      log,
      isNixMode: false,
      allowInTests: true,
    });

    expect(log.info).not.toHaveBeenCalled();
    await expect(fs.stat(path.join(tempDir, "update-check.json"))).rejects.toThrow();
  });
});
