import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import type { MarvConfig, ConfigFileSnapshot } from "../core/config/types.marv.js";
import type { UpdateRunResult } from "../infra/update/update-runner.js";
import { captureEnv } from "../test-utils/env.js";

const confirm = vi.fn();
const select = vi.fn();
const spinner = vi.fn(() => ({ start: vi.fn(), stop: vi.fn() }));
const isCancel = (value: unknown) => value === "cancel";

const readPackageName = vi.fn();
const readPackageVersion = vi.fn();
const resolveGlobalManager = vi.fn();
const serviceLoaded = vi.fn();
const prepareRestartScript = vi.fn();
const runRestartScript = vi.fn();

vi.mock("@clack/prompts", () => ({
  confirm,
  select,
  isCancel,
  spinner,
}));

// Mock the update-runner module
vi.mock("../infra/update/update-runner.js", () => ({
  runGatewayUpdate: vi.fn(),
  runGatewayRollback: vi.fn(),
}));

vi.mock("../infra/marv-root.js", () => ({
  resolveMarvPackageRoot: vi.fn(),
}));

vi.mock("../core/config/config.js", () => ({
  readConfigFileSnapshot: vi.fn(),
  writeConfigFile: vi.fn(),
  STATE_DIR: "/tmp/marv-test-state",
}));

vi.mock("../infra/update/update-check.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../infra/update/update-check.js")>();
  return {
    ...actual,
    checkUpdateStatus: vi.fn(),
    fetchNpmTagVersion: vi.fn(),
    resolveNpmChannelTag: vi.fn(),
  };
});

vi.mock("node:child_process", async () => {
  const actual = await vi.importActual<typeof import("node:child_process")>("node:child_process");
  return {
    ...actual,
    spawnSync: vi.fn(() => ({
      pid: 0,
      output: [],
      stdout: "",
      stderr: "",
      status: 0,
      signal: null,
    })),
  };
});

vi.mock("../process/exec.js", () => ({
  runCommandWithTimeout: vi.fn(),
}));

vi.mock("./update-cli/shared.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./update-cli/shared.js")>();
  return {
    ...actual,
    readPackageName,
    readPackageVersion,
    resolveGlobalManager,
  };
});

vi.mock("../infra/daemon/service.js", () => ({
  resolveGatewayService: vi.fn(() => ({
    isLoaded: (...args: unknown[]) => serviceLoaded(...args),
  })),
}));

vi.mock("./update-cli/restart-helper.js", () => ({
  prepareRestartScript: (...args: unknown[]) => prepareRestartScript(...args),
  runRestartScript: (...args: unknown[]) => runRestartScript(...args),
}));

vi.mock("./completion-utils.js", () => ({
  installCompletion: vi.fn(),
}));

// Mock doctor (heavy module; should not run in unit tests)
vi.mock("../commands/doctor.js", () => ({
  doctorCommand: vi.fn(),
}));
// Mock the daemon-cli module
vi.mock("./daemon-cli.js", () => ({
  runDaemonRestart: vi.fn(),
}));

// Mock the runtime
vi.mock("../runtime.js", () => ({
  defaultRuntime: {
    log: vi.fn(),
    error: vi.fn(),
    exit: vi.fn(),
  },
}));

const { runGatewayRollback, runGatewayUpdate } = await import("../infra/update/update-runner.js");
const { resolveMarvPackageRoot } = await import("../infra/marv-root.js");
const { readConfigFileSnapshot, writeConfigFile } = await import("../core/config/config.js");
const { checkUpdateStatus, fetchNpmTagVersion, resolveNpmChannelTag } =
  await import("../infra/update/update-check.js");
const { runCommandWithTimeout } = await import("../process/exec.js");
const { runDaemonRestart } = await import("./daemon-cli.js");
const { doctorCommand } = await import("../commands/doctor.js");
const { defaultRuntime } = await import("../runtime.js");
const {
  updateCommand,
  updateRollbackCommand,
  registerUpdateCli,
  updateStatusCommand,
  updateWizardCommand,
} = await import("./update-cli.js");

describe("update-cli", () => {
  let fixtureRoot = "";
  let fixtureCount = 0;

  const createCaseDir = async (prefix: string) => {
    const dir = path.join(fixtureRoot, `${prefix}-${fixtureCount++}`);
    // Tests only need a stable path; the directory does not have to exist because all I/O is mocked.
    return dir;
  };

  beforeAll(async () => {
    fixtureRoot = await fs.mkdtemp(path.join(os.tmpdir(), "marv-update-tests-"));
  });

  afterAll(async () => {
    await fs.rm(fixtureRoot, { recursive: true, force: true });
  });

  const baseConfig = {} as MarvConfig;
  const baseSnapshot: ConfigFileSnapshot = {
    path: "/tmp/marv-config.json",
    exists: true,
    raw: "{}",
    parsed: {},
    resolved: baseConfig,
    valid: true,
    config: baseConfig,
    issues: [],
    warnings: [],
    legacyIssues: [],
  };

  const setTty = (value: boolean | undefined) => {
    Object.defineProperty(process.stdin, "isTTY", {
      value,
      configurable: true,
    });
  };

  const setStdoutTty = (value: boolean | undefined) => {
    Object.defineProperty(process.stdout, "isTTY", {
      value,
      configurable: true,
    });
  };

  const mockPackageInstallStatus = (root: string) => {
    vi.mocked(resolveMarvPackageRoot).mockResolvedValue(root);
    vi.mocked(checkUpdateStatus).mockResolvedValue({
      root,
      installKind: "package",
      packageManager: "npm",
      deps: {
        manager: "npm",
        status: "ok",
        lockfilePath: null,
        markerPath: null,
      },
    });
  };

  const expectUpdateCallChannel = (channel: string) => {
    const call = vi.mocked(runGatewayUpdate).mock.calls[0]?.[0];
    expect(call?.channel).toBe(channel);
    return call;
  };

  const setupNonInteractiveDowngrade = async () => {
    const tempDir = await createCaseDir("marv-update");
    setTty(false);
    readPackageVersion.mockResolvedValue("2.0.0");

    mockPackageInstallStatus(tempDir);
    vi.mocked(resolveNpmChannelTag).mockResolvedValue({
      tag: "latest",
      version: "0.0.1",
    });
    vi.mocked(runGatewayUpdate).mockResolvedValue({
      status: "ok",
      mode: "npm",
      steps: [],
      durationMs: 100,
    });
    vi.mocked(defaultRuntime.error).mockClear();
    vi.mocked(defaultRuntime.exit).mockClear();

    return tempDir;
  };

  beforeEach(() => {
    confirm.mockReset();
    select.mockReset();
    vi.mocked(runGatewayUpdate).mockReset();
    vi.mocked(runGatewayRollback).mockReset();
    vi.mocked(resolveMarvPackageRoot).mockReset();
    vi.mocked(readConfigFileSnapshot).mockReset();
    vi.mocked(writeConfigFile).mockReset();
    vi.mocked(checkUpdateStatus).mockReset();
    vi.mocked(fetchNpmTagVersion).mockReset();
    vi.mocked(resolveNpmChannelTag).mockReset();
    vi.mocked(runCommandWithTimeout).mockReset();
    vi.mocked(runDaemonRestart).mockReset();
    vi.mocked(doctorCommand).mockReset();
    vi.mocked(defaultRuntime.log).mockReset();
    vi.mocked(defaultRuntime.error).mockReset();
    vi.mocked(defaultRuntime.exit).mockReset();
    readPackageName.mockReset();
    readPackageVersion.mockReset();
    resolveGlobalManager.mockReset();
    serviceLoaded.mockReset();
    prepareRestartScript.mockReset();
    runRestartScript.mockReset();
    vi.mocked(resolveMarvPackageRoot).mockResolvedValue(process.cwd());
    vi.mocked(readConfigFileSnapshot).mockResolvedValue(baseSnapshot);
    vi.mocked(fetchNpmTagVersion).mockResolvedValue({
      tag: "latest",
      version: "9999.0.0",
    });
    vi.mocked(resolveNpmChannelTag).mockResolvedValue({
      tag: "latest",
      version: "9999.0.0",
    });
    vi.mocked(checkUpdateStatus).mockResolvedValue({
      root: "/test/path",
      installKind: "git",
      packageManager: "pnpm",
      git: {
        root: "/test/path",
        sha: "abcdef1234567890",
        tag: "v1.2.3",
        branch: "main",
        upstream: "origin/main",
        dirty: false,
        ahead: 0,
        behind: 0,
        fetchOk: true,
      },
      deps: {
        manager: "pnpm",
        status: "ok",
        lockfilePath: "/test/path/pnpm-lock.yaml",
        markerPath: "/test/path/node_modules",
      },
      registry: {
        latestVersion: "1.2.3",
      },
    });
    vi.mocked(runCommandWithTimeout).mockResolvedValue({
      stdout: "",
      stderr: "",
      code: 0,
      signal: null,
      killed: false,
      termination: "exit",
    });
    readPackageName.mockResolvedValue("marv");
    readPackageVersion.mockResolvedValue("1.0.0");
    resolveGlobalManager.mockResolvedValue("npm");
    serviceLoaded.mockResolvedValue(false);
    prepareRestartScript.mockResolvedValue("/tmp/marv-restart-test.sh");
    runRestartScript.mockResolvedValue(undefined);
    setTty(false);
    setStdoutTty(false);
  });

  it("exports updateCommand and registerUpdateCli", async () => {
    expect(typeof updateCommand).toBe("function");
    expect(typeof updateRollbackCommand).toBe("function");
    expect(typeof registerUpdateCli).toBe("function");
    expect(typeof updateWizardCommand).toBe("function");
  }, 20_000);

  it("updateCommand runs update and outputs result", async () => {
    const mockResult: UpdateRunResult = {
      status: "ok",
      mode: "git",
      root: "/test/path",
      before: { sha: "abc123", version: "1.0.0" },
      after: { sha: "def456", version: "1.0.1" },
      steps: [
        {
          name: "git fetch",
          command: "git fetch",
          cwd: "/test/path",
          durationMs: 100,
          exitCode: 0,
        },
      ],
      durationMs: 500,
    };

    vi.mocked(runGatewayUpdate).mockResolvedValue(mockResult);

    await updateCommand({ json: false });

    expect(runGatewayUpdate).toHaveBeenCalled();
    expect(defaultRuntime.log).toHaveBeenCalled();
  });

  it("updateRollbackCommand runs rollback and outputs result", async () => {
    const mockResult: UpdateRunResult = {
      status: "ok",
      mode: "git",
      root: "/test/path",
      before: { sha: "bad999", version: "1.0.2" },
      after: { sha: "good123", version: "1.0.1" },
      deploy: {
        statePath: "/tmp/deploy-state.json",
        lastKnownGoodSha: "good123",
        targetSha: "good123",
        rolledBackToSha: "good123",
      },
      steps: [
        {
          name: "git reset --hard",
          command: "git reset --hard good123",
          cwd: "/test/path",
          durationMs: 100,
          exitCode: 0,
        },
      ],
      durationMs: 500,
    };

    vi.mocked(runGatewayRollback).mockResolvedValue(mockResult);

    await updateRollbackCommand({ json: false });

    expect(runGatewayRollback).toHaveBeenCalled();
    expect(defaultRuntime.log).toHaveBeenCalled();
  });

  it("updateStatusCommand prints table output", async () => {
    await updateStatusCommand({ json: false });

    const logs = vi.mocked(defaultRuntime.log).mock.calls.map((call) => call[0]);
    expect(logs.join("\n")).toContain("Marv update status");
  });

  it("updateStatusCommand emits JSON", async () => {
    await updateStatusCommand({ json: true });

    const last = vi.mocked(defaultRuntime.log).mock.calls.at(-1)?.[0];
    expect(typeof last).toBe("string");
    const parsed = JSON.parse(String(last));
    expect(parsed.channel.value).toBe("stable");
  });

  it("defaults to dev channel for git installs when unset", async () => {
    vi.mocked(runGatewayUpdate).mockResolvedValue({
      status: "ok",
      mode: "git",
      steps: [],
      durationMs: 100,
    });

    await updateCommand({});

    expectUpdateCallChannel("dev");
  });

  it("defaults to stable channel for package installs when unset", async () => {
    const tempDir = await createCaseDir("marv-update");

    mockPackageInstallStatus(tempDir);
    readPackageName.mockResolvedValue("agentmarv");
    readPackageVersion.mockResolvedValue("1.0.0");

    await updateCommand({ yes: true });

    // Package installs with no stored channel use npm path (stable)
    const installCall = vi
      .mocked(runCommandWithTimeout)
      .mock.calls.find((c) => c[0].some((arg) => arg.includes("agentmarv@latest")));
    expect(installCall).toBeTruthy();
  });

  it("uses stored beta channel when configured", async () => {
    vi.mocked(readConfigFileSnapshot).mockResolvedValue({
      ...baseSnapshot,
      config: { update: { channel: "beta" } } as MarvConfig,
    });
    vi.mocked(runGatewayUpdate).mockResolvedValue({
      status: "ok",
      mode: "git",
      steps: [],
      durationMs: 100,
    });

    await updateCommand({});

    expectUpdateCallChannel("beta");
  });

  it("falls back to latest when beta tag is older than release", async () => {
    const tempDir = await createCaseDir("marv-update");

    mockPackageInstallStatus(tempDir);
    vi.mocked(readConfigFileSnapshot).mockResolvedValue({
      ...baseSnapshot,
      config: { update: { channel: "beta" } } as MarvConfig,
    });
    vi.mocked(resolveNpmChannelTag).mockResolvedValue({
      tag: "latest",
      version: "1.2.3-1",
    });
    readPackageName.mockResolvedValue("agentmarv");
    readPackageVersion.mockResolvedValue("1.0.0");

    await updateCommand({});

    // Package install with stored beta channel uses npm path;
    // tag falls back to "latest" when beta is older.
    const installCall = vi
      .mocked(runCommandWithTimeout)
      .mock.calls.find((c) => c[0].some((arg) => arg.includes("agentmarv@latest")));
    expect(installCall).toBeTruthy();
  });

  it("honors --tag override", async () => {
    const tempDir = await createCaseDir("marv-update");

    vi.mocked(resolveMarvPackageRoot).mockResolvedValue(tempDir);
    vi.mocked(runGatewayUpdate).mockResolvedValue({
      status: "ok",
      mode: "npm",
      steps: [],
      durationMs: 100,
    });

    await updateCommand({ tag: "next" });

    const call = vi.mocked(runGatewayUpdate).mock.calls[0]?.[0];
    expect(call?.tag).toBe("next");
  });

  it("updateCommand outputs JSON when --json is set", async () => {
    const mockResult: UpdateRunResult = {
      status: "ok",
      mode: "git",
      steps: [],
      durationMs: 100,
    };

    vi.mocked(runGatewayUpdate).mockResolvedValue(mockResult);
    vi.mocked(defaultRuntime.log).mockClear();

    await updateCommand({ json: true });

    const logCalls = vi.mocked(defaultRuntime.log).mock.calls;
    const jsonOutput = logCalls.find((call) => {
      try {
        JSON.parse(call[0] as string);
        return true;
      } catch {
        return false;
      }
    });
    expect(jsonOutput).toBeDefined();
  });

  it("updateCommand exits with error on failure", async () => {
    const mockResult: UpdateRunResult = {
      status: "error",
      mode: "git",
      reason: "rebase-failed",
      steps: [],
      durationMs: 100,
    };

    vi.mocked(runGatewayUpdate).mockResolvedValue(mockResult);
    vi.mocked(defaultRuntime.exit).mockClear();

    await updateCommand({});

    expect(defaultRuntime.exit).toHaveBeenCalledWith(1);
  });

  it("updateRollbackCommand exits with error on failure", async () => {
    const mockResult: UpdateRunResult = {
      status: "error",
      mode: "git",
      reason: "rollback-failed",
      steps: [],
      durationMs: 100,
    };

    vi.mocked(runGatewayRollback).mockResolvedValue(mockResult);
    vi.mocked(defaultRuntime.exit).mockClear();

    await updateRollbackCommand({});

    expect(defaultRuntime.exit).toHaveBeenCalledWith(1);
  });

  it("updateCommand restarts daemon by default", async () => {
    const mockResult: UpdateRunResult = {
      status: "ok",
      mode: "git",
      steps: [],
      durationMs: 100,
    };

    vi.mocked(runGatewayUpdate).mockResolvedValue(mockResult);
    vi.mocked(runDaemonRestart).mockResolvedValue(true);

    await updateCommand({});

    expect(runDaemonRestart).toHaveBeenCalled();
  });

  it("updateRollbackCommand restarts daemon by default", async () => {
    const mockResult: UpdateRunResult = {
      status: "ok",
      mode: "git",
      steps: [],
      durationMs: 100,
    };

    vi.mocked(runGatewayRollback).mockResolvedValue(mockResult);
    vi.mocked(runDaemonRestart).mockResolvedValue(true);

    await updateRollbackCommand({});

    expect(runDaemonRestart).toHaveBeenCalled();
  });

  it("updateCommand continues after doctor sub-step and clears update flag", async () => {
    const mockResult: UpdateRunResult = {
      status: "ok",
      mode: "git",
      steps: [],
      durationMs: 100,
    };

    const envSnapshot = captureEnv(["MARV_UPDATE_IN_PROGRESS"]);
    const randomSpy = vi.spyOn(Math, "random").mockReturnValue(0);
    try {
      delete process.env.MARV_UPDATE_IN_PROGRESS;
      vi.mocked(runGatewayUpdate).mockResolvedValue(mockResult);
      vi.mocked(runDaemonRestart).mockResolvedValue(true);
      vi.mocked(doctorCommand).mockResolvedValue(undefined);
      vi.mocked(defaultRuntime.log).mockClear();

      await updateCommand({});

      expect(doctorCommand).toHaveBeenCalledWith(
        defaultRuntime,
        expect.objectContaining({ nonInteractive: true }),
      );
      expect(process.env.MARV_UPDATE_IN_PROGRESS).toBeUndefined();

      const logLines = vi.mocked(defaultRuntime.log).mock.calls.map((call) => String(call[0]));
      expect(
        logLines.some((line) => line.includes("Leveled up! New skills unlocked. You're welcome.")),
      ).toBe(true);
    } finally {
      randomSpy.mockRestore();
      envSnapshot.restore();
    }
  });

  it("updateCommand skips restart when --no-restart is set", async () => {
    const mockResult: UpdateRunResult = {
      status: "ok",
      mode: "git",
      steps: [],
      durationMs: 100,
    };

    vi.mocked(runGatewayUpdate).mockResolvedValue(mockResult);

    await updateCommand({ restart: false });

    expect(runDaemonRestart).not.toHaveBeenCalled();
  });

  it("updateRollbackCommand skips restart when --no-restart is set", async () => {
    const mockResult: UpdateRunResult = {
      status: "ok",
      mode: "git",
      steps: [],
      durationMs: 100,
    };

    vi.mocked(runGatewayRollback).mockResolvedValue(mockResult);

    await updateRollbackCommand({ restart: false });

    expect(runDaemonRestart).not.toHaveBeenCalled();
  });

  it("updateCommand skips success message when restart does not run", async () => {
    const mockResult: UpdateRunResult = {
      status: "ok",
      mode: "git",
      steps: [],
      durationMs: 100,
    };

    vi.mocked(runGatewayUpdate).mockResolvedValue(mockResult);
    vi.mocked(runDaemonRestart).mockResolvedValue(false);
    vi.mocked(defaultRuntime.log).mockClear();

    await updateCommand({ restart: true });

    const logLines = vi.mocked(defaultRuntime.log).mock.calls.map((call) => String(call[0]));
    expect(logLines.some((line) => line.includes("Daemon restarted successfully."))).toBe(false);
  });

  it("updateCommand validates timeout option", async () => {
    vi.mocked(defaultRuntime.error).mockClear();
    vi.mocked(defaultRuntime.exit).mockClear();

    await updateCommand({ timeout: "invalid" });

    expect(defaultRuntime.error).toHaveBeenCalledWith(expect.stringContaining("timeout"));
    expect(defaultRuntime.exit).toHaveBeenCalledWith(1);
  });

  it("updateRollbackCommand validates timeout option", async () => {
    vi.mocked(defaultRuntime.error).mockClear();
    vi.mocked(defaultRuntime.exit).mockClear();

    await updateRollbackCommand({ timeout: "invalid" });

    expect(defaultRuntime.error).toHaveBeenCalledWith(expect.stringContaining("timeout"));
    expect(defaultRuntime.exit).toHaveBeenCalledWith(1);
  });

  it("updateStatusCommand validates timeout option", async () => {
    vi.mocked(defaultRuntime.error).mockClear();
    vi.mocked(defaultRuntime.exit).mockClear();

    await updateStatusCommand({ timeout: "invalid" });

    expect(defaultRuntime.error).toHaveBeenCalledWith(expect.stringContaining("timeout"));
    expect(defaultRuntime.exit).toHaveBeenCalledWith(1);
  });

  it("updateRollbackCommand reports when no last known good deployment exists", async () => {
    const mockResult: UpdateRunResult = {
      status: "skipped",
      mode: "git",
      reason: "no-last-known-good",
      steps: [],
      durationMs: 100,
    };

    vi.mocked(runGatewayRollback).mockResolvedValue(mockResult);

    await updateRollbackCommand({});

    const logLines = vi.mocked(defaultRuntime.log).mock.calls.map((call) => String(call[0]));
    expect(logLines.some((line) => line.includes("last-known-good deployment"))).toBe(true);
    expect(defaultRuntime.exit).toHaveBeenCalledWith(0);
  });

  it("persists update channel when --channel is set", async () => {
    const mockResult: UpdateRunResult = {
      status: "ok",
      mode: "git",
      steps: [],
      durationMs: 100,
    };

    vi.mocked(runGatewayUpdate).mockResolvedValue(mockResult);

    await updateCommand({ channel: "beta" });

    expect(writeConfigFile).toHaveBeenCalled();
    const call = vi.mocked(writeConfigFile).mock.calls[0]?.[0] as {
      update?: { channel?: string };
    };
    expect(call?.update?.channel).toBe("beta");
  });

  it("requires confirmation on downgrade when non-interactive", async () => {
    await setupNonInteractiveDowngrade();

    // Explicit stable channel keeps the npm update path
    await updateCommand({ channel: "stable" });

    expect(defaultRuntime.error).toHaveBeenCalledWith(
      expect.stringContaining("Downgrade confirmation required."),
    );
    expect(defaultRuntime.exit).toHaveBeenCalledWith(1);
  });

  it("allows downgrade with --yes in non-interactive mode", async () => {
    await setupNonInteractiveDowngrade();
    readPackageName.mockResolvedValue("agentmarv");

    // Explicit stable channel keeps the npm update path
    await updateCommand({ yes: true, channel: "stable" });

    expect(defaultRuntime.error).not.toHaveBeenCalledWith(
      expect.stringContaining("Downgrade confirmation required."),
    );
  });

  it("updateWizardCommand requires a TTY", async () => {
    setTty(false);
    vi.mocked(defaultRuntime.error).mockClear();
    vi.mocked(defaultRuntime.exit).mockClear();

    await updateWizardCommand({});

    expect(defaultRuntime.error).toHaveBeenCalledWith(
      expect.stringContaining("Update wizard requires a TTY"),
    );
    expect(defaultRuntime.exit).toHaveBeenCalledWith(1);
  });

  it("updateWizardCommand validates timeout option", async () => {
    setTty(true);
    vi.mocked(defaultRuntime.error).mockClear();
    vi.mocked(defaultRuntime.exit).mockClear();

    await updateWizardCommand({ timeout: "invalid" });

    expect(defaultRuntime.error).toHaveBeenCalledWith(expect.stringContaining("timeout"));
    expect(defaultRuntime.exit).toHaveBeenCalledWith(1);
  });

  it("updateWizardCommand offers dev checkout and forwards selections", async () => {
    const tempDir = await createCaseDir("marv-update-wizard");
    const envSnapshot = captureEnv(["MARV_GIT_DIR"]);
    try {
      setTty(true);
      process.env.MARV_GIT_DIR = tempDir;

      vi.mocked(checkUpdateStatus).mockResolvedValue({
        root: "/test/path",
        installKind: "package",
        packageManager: "npm",
        deps: {
          manager: "npm",
          status: "ok",
          lockfilePath: null,
          markerPath: null,
        },
      });
      select.mockResolvedValue("dev");
      confirm.mockResolvedValueOnce(true).mockResolvedValueOnce(false);
      vi.mocked(runGatewayUpdate).mockResolvedValue({
        status: "ok",
        mode: "git",
        steps: [],
        durationMs: 100,
      });

      await updateWizardCommand({});

      const call = vi.mocked(runGatewayUpdate).mock.calls[0]?.[0];
      expect(call?.channel).toBe("dev");
    } finally {
      envSnapshot.restore();
    }
  });
});
