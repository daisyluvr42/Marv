import { beforeEach, describe, expect, it, vi } from "vitest";

const loadAndMaybeMigrateDoctorConfigMock = vi.hoisted(() => vi.fn());
const readConfigFileSnapshotMock = vi.hoisted(() => vi.fn());

vi.mock("../../commands/doctor-config-flow.js", () => ({
  loadAndMaybeMigrateDoctorConfig: loadAndMaybeMigrateDoctorConfigMock,
}));

vi.mock("../../core/config/config.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../core/config/config.js")>();
  return {
    ...actual,
    readConfigFileSnapshot: readConfigFileSnapshotMock,
  };
});

function makeSnapshot() {
  return {
    exists: false,
    valid: true,
    issues: [],
    legacyIssues: [],
    path: "/tmp/marv.json",
  };
}

function makeRuntime() {
  return {
    error: vi.fn(),
    exit: vi.fn(),
  };
}

describe("ensureConfigReady", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    readConfigFileSnapshotMock.mockResolvedValue(makeSnapshot());
  });

  it("skips doctor flow for read-only fast path commands", async () => {
    vi.resetModules();
    const { ensureConfigReady } = await import("./config-guard.js");
    await ensureConfigReady({ runtime: makeRuntime() as never, commandPath: ["status"] });
    await ensureConfigReady({
      runtime: makeRuntime() as never,
      commandPath: ["config", "validate"],
    });
    await ensureConfigReady({
      runtime: makeRuntime() as never,
      commandPath: ["gateway", "status"],
    });
    await ensureConfigReady({
      runtime: makeRuntime() as never,
      commandPath: ["system", "heartbeat", "last"],
    });
    expect(loadAndMaybeMigrateDoctorConfigMock).not.toHaveBeenCalled();
  });

  it("runs doctor flow for commands that may mutate state", async () => {
    vi.resetModules();
    const { ensureConfigReady } = await import("./config-guard.js");
    await ensureConfigReady({ runtime: makeRuntime() as never, commandPath: ["message"] });
    expect(loadAndMaybeMigrateDoctorConfigMock).toHaveBeenCalledTimes(1);
  });

  it("allows invalid config for declaratively whitelisted gateway probes", async () => {
    vi.resetModules();
    readConfigFileSnapshotMock.mockResolvedValue({
      ...makeSnapshot(),
      exists: true,
      valid: false,
      issues: [{ path: "gateway.port", message: "bad" }],
    });
    const runtime = makeRuntime();
    const { ensureConfigReady } = await import("./config-guard.js");
    await ensureConfigReady({ runtime: runtime as never, commandPath: ["gateway", "probe"] });
    expect(runtime.exit).not.toHaveBeenCalled();
  });

  it("allows invalid config for config inspection and validation commands", async () => {
    vi.resetModules();
    readConfigFileSnapshotMock.mockResolvedValue({
      ...makeSnapshot(),
      exists: true,
      valid: false,
      issues: [{ path: "gateway.port", message: "bad" }],
    });
    const runtime = makeRuntime();
    const { ensureConfigReady } = await import("./config-guard.js");
    await ensureConfigReady({ runtime: runtime as never, commandPath: ["config", "get"] });
    await ensureConfigReady({ runtime: runtime as never, commandPath: ["config", "validate"] });
    expect(runtime.exit).not.toHaveBeenCalled();
  });

  it("still blocks invalid config for commands that require a valid config", async () => {
    vi.resetModules();
    readConfigFileSnapshotMock.mockResolvedValue({
      ...makeSnapshot(),
      exists: true,
      valid: false,
      issues: [{ path: "gateway.port", message: "bad" }],
    });
    const runtime = makeRuntime();
    const { ensureConfigReady } = await import("./config-guard.js");
    await ensureConfigReady({ runtime: runtime as never, commandPath: ["message", "send"] });
    expect(runtime.exit).toHaveBeenCalledWith(1);
  });
});
