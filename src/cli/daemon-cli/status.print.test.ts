import { beforeEach, describe, expect, it, vi } from "vitest";

const runtimeLogs: string[] = [];
const runtimeErrors: string[] = [];

vi.mock("../../runtime.js", () => ({
  defaultRuntime: {
    log: (message: string) => runtimeLogs.push(message),
    error: (message: string) => runtimeErrors.push(message),
    exit: (code: number) => {
      throw new Error(`__exit__:${code}`);
    },
  },
}));

vi.mock("../../logging.js", () => ({
  getResolvedLoggerSettings: () => ({ file: "/tmp/marv.log" }),
}));

vi.mock("../../terminal/theme.js", () => ({
  colorize: (_rich: boolean, fn: (value: string) => string, value: string) => fn(value),
  isRich: () => false,
  theme: {
    muted: (value: string) => value,
    accent: (value: string) => value,
    info: (value: string) => value,
    success: (value: string) => value,
    warn: (value: string) => value,
    error: (value: string) => value,
  },
}));

describe("printDaemonStatus", () => {
  beforeEach(() => {
    runtimeLogs.length = 0;
    runtimeErrors.length = 0;
  });

  it("treats a reachable foreground instance as unmanaged instead of broken", async () => {
    const { printDaemonStatus } = await import("./status.print.js");

    printDaemonStatus(
      {
        service: {
          label: "LaunchAgent",
          loaded: false,
          loadedText: "loaded",
          notLoadedText: "not loaded",
          command: {
            programArguments: ["/usr/bin/node", "/tmp/agentmarv/dist/index.js", "gateway"],
            sourcePath: "/Users/test/Library/LaunchAgents/ai.marv.gateway.plist",
          },
          runtime: {
            status: "unknown",
            detail: "Could not find service",
            missingUnit: true,
          },
        },
        rpc: {
          ok: true,
          url: "ws://127.0.0.1:18789",
        },
        extraServices: [],
      },
      { json: false },
    );

    expect(runtimeLogs.join("\n")).toContain("foreground instance reachable");
    expect(runtimeLogs.join("\n")).toContain("Managed service not installed.");
    expect(runtimeLogs.join("\n")).toContain("Last known command:");
    expect(runtimeErrors.join("\n")).not.toContain("Service unit not found.");
  });
});
