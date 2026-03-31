import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const inspectPortUsage = vi.fn();
const sleep = vi.fn(async (ms: number) => {
  fakeNow += ms;
});

let fakeNow = 0;

vi.mock("../../infra/ports.js", () => ({
  inspectPortUsage: (port: number) => inspectPortUsage(port),
  formatPortDiagnostics: (diagnostics: {
    port: number;
    status: "busy" | "free" | "unknown";
    listeners: Array<{ pid?: number; commandLine?: string; command?: string }>;
  }) => {
    if (diagnostics.status !== "busy") {
      return [`Port ${diagnostics.port} is free.`];
    }
    return [
      `Port ${diagnostics.port} is already in use.`,
      ...diagnostics.listeners.map(
        (listener) =>
          `- pid ${listener.pid ?? "?"}: ${listener.commandLine ?? listener.command ?? "unknown"}`,
      ),
    ];
  },
}));

vi.mock("../../utils.js", () => ({
  sleep: (ms: number) => sleep(ms),
}));

describe("gateway restart verification", () => {
  beforeEach(() => {
    fakeNow = 0;
    inspectPortUsage.mockReset();
    sleep.mockClear();
    vi.spyOn(Date, "now").mockImplementation(() => fakeNow);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("accepts a restarted gateway when a new pid owns the configured port", async () => {
    const { __testing } = await import("./lifecycle.js");
    const service = {
      readRuntime: vi.fn().mockResolvedValue({
        status: "running",
        pid: 202,
      }),
    } as const;

    inspectPortUsage.mockResolvedValue({
      port: 4242,
      status: "busy",
      listeners: [
        {
          pid: 202,
          commandLine: "/usr/bin/node /repo/dist/marv.mjs gateway --port 4242",
        },
      ],
      hints: [],
    });

    await expect(
      __testing.verifyGatewayRestart(service as never, {
        port: 4242,
        command: {
          programArguments: ["/usr/bin/node", "/repo/dist/marv.mjs", "gateway", "--port", "4242"],
        },
        runtime: {
          status: "running",
          pid: 101,
        },
        previousPids: new Set([101]),
      }),
    ).resolves.toBeUndefined();
  });

  it("fails when the old pid keeps serving the gateway port after restart", async () => {
    const { __testing } = await import("./lifecycle.js");
    const service = {
      readRuntime: vi.fn().mockResolvedValue({
        status: "running",
        pid: 101,
      }),
    } as const;

    inspectPortUsage.mockResolvedValue({
      port: 4242,
      status: "busy",
      listeners: [
        {
          pid: 101,
          commandLine: "/usr/bin/node /repo/dist/marv.mjs gateway --port 4242",
        },
      ],
      hints: [],
    });

    await expect(
      __testing.verifyGatewayRestart(service as never, {
        port: 4242,
        command: {
          programArguments: ["/usr/bin/node", "/repo/dist/marv.mjs", "gateway", "--port", "4242"],
        },
        runtime: {
          status: "running",
          pid: 101,
        },
        previousPids: new Set([101]),
      }),
    ).rejects.toThrow("old gateway pid 101 is still active after restart");
  });
});
