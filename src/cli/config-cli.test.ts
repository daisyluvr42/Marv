import { Command } from "commander";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ConfigFileSnapshot, MarvConfig } from "../core/config/types.js";

/**
 * Test for issue #6070:
 * `marv config set/unset` must update snapshot.resolved (user config after $include/${ENV},
 * but before runtime defaults), so runtime defaults don't leak into the written config.
 */

const mockReadConfigFileSnapshot = vi.fn<() => Promise<ConfigFileSnapshot>>();
const mockWriteConfigFile = vi.fn<(cfg: MarvConfig) => Promise<void>>(async () => {});

vi.mock("../core/config/config.js", () => ({
  readConfigFileSnapshot: () => mockReadConfigFileSnapshot(),
  writeConfigFile: (cfg: MarvConfig) => mockWriteConfigFile(cfg),
}));

const mockLog = vi.fn();
const mockError = vi.fn();
const mockExit = vi.fn((code: number) => {
  const errorMessages = mockError.mock.calls.map((c) => c.join(" ")).join("; ");
  throw new Error(`__exit__:${code} - ${errorMessages}`);
});

vi.mock("../runtime.js", () => ({
  defaultRuntime: {
    log: (...args: unknown[]) => mockLog(...args),
    error: (...args: unknown[]) => mockError(...args),
    exit: (code: number) => mockExit(code),
  },
}));

function buildSnapshot(params: { resolved: MarvConfig; config: MarvConfig }): ConfigFileSnapshot {
  return {
    path: "/tmp/marv.json",
    exists: true,
    raw: JSON.stringify(params.resolved),
    parsed: params.resolved,
    resolved: params.resolved,
    valid: true,
    config: params.config,
    issues: [],
    warnings: [],
    legacyIssues: [],
  };
}

function setSnapshot(resolved: MarvConfig, config: MarvConfig) {
  mockReadConfigFileSnapshot.mockResolvedValueOnce(buildSnapshot({ resolved, config }));
}

function setInvalidSnapshot(issues: Array<{ path: string; message: string }>) {
  mockReadConfigFileSnapshot.mockResolvedValueOnce({
    path: "/tmp/marv.json",
    exists: true,
    raw: "{",
    parsed: {},
    resolved: {},
    valid: false,
    config: {},
    issues,
    warnings: [],
    legacyIssues: [],
  } as ConfigFileSnapshot);
}

async function runConfigCommand(args: string[]) {
  const { registerConfigCli } = await import("./config-cli.js");
  const program = new Command();
  program.exitOverride();
  registerConfigCli(program);
  await program.parseAsync(args, { from: "user" });
}

describe("config cli", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("config set - issue #6070", () => {
    it("preserves existing config keys when setting a new value", async () => {
      const resolved: MarvConfig = {
        agents: {
          list: [{ id: "main" }, { id: "oracle", workspace: "~/oracle-workspace" }],
        },
        gateway: { port: 18789 },
        tools: { allow: ["group:fs"] },
        logging: { level: "debug" },
      };
      const runtimeMerged: MarvConfig = {
        ...resolved,
        agents: {
          ...resolved.agents,
          defaults: {
            model: "gpt-5.2",
          } as never,
        } as never,
      };
      setSnapshot(resolved, runtimeMerged);

      await runConfigCommand(["config", "set", "gateway.auth.mode", "token"]);

      expect(mockWriteConfigFile).toHaveBeenCalledTimes(1);
      const written = mockWriteConfigFile.mock.calls[0]?.[0];
      expect(written.gateway?.auth).toEqual({ mode: "token" });
      expect(written.gateway?.port).toBe(18789);
      expect(written.agents).toEqual(resolved.agents);
      expect(written.tools).toEqual(resolved.tools);
      expect(written.logging).toEqual(resolved.logging);
      expect(written.agents).not.toHaveProperty("defaults");
    });

    it("does not inject runtime defaults into the written config", async () => {
      const resolved: MarvConfig = {
        gateway: { port: 18789 },
      };
      const runtimeMerged = {
        ...resolved,
        agents: {
          defaults: {
            model: "gpt-5.2",
            contextWindow: 128_000,
            maxTokens: 16_000,
          },
        } as never,
        messages: { ackReaction: "✅" } as never,
        sessions: { persistence: { enabled: true } } as never,
      } as unknown as MarvConfig;
      setSnapshot(resolved, runtimeMerged);

      await runConfigCommand(["config", "set", "gateway.auth.mode", "token"]);

      expect(mockWriteConfigFile).toHaveBeenCalledTimes(1);
      const written = mockWriteConfigFile.mock.calls[0]?.[0];
      expect(written).not.toHaveProperty("agents.defaults.model");
      expect(written).not.toHaveProperty("agents.defaults.contextWindow");
      expect(written).not.toHaveProperty("agents.defaults.maxTokens");
      expect(written).not.toHaveProperty("messages.ackReaction");
      expect(written).not.toHaveProperty("sessions.persistence");
      expect(written.gateway?.port).toBe(18789);
      expect(written.gateway?.auth).toEqual({ mode: "token" });
    });
  });

  describe("config unset - issue #6070", () => {
    it("preserves existing config keys when unsetting a value", async () => {
      const resolved: MarvConfig = {
        agents: { list: [{ id: "main" }] },
        gateway: { port: 18789 },
        tools: {
          profile: "coding",
          alsoAllow: ["agents_list"],
        },
        logging: { level: "debug" },
      };
      const runtimeMerged: MarvConfig = {
        ...resolved,
        agents: {
          ...resolved.agents,
          defaults: {
            model: "gpt-5.2",
          },
        } as never,
      };
      setSnapshot(resolved, runtimeMerged);

      await runConfigCommand(["config", "unset", "tools.alsoAllow"]);

      expect(mockWriteConfigFile).toHaveBeenCalledTimes(1);
      const written = mockWriteConfigFile.mock.calls[0]?.[0];
      expect(written.tools).not.toHaveProperty("alsoAllow");
      expect(written.agents).not.toHaveProperty("defaults");
      expect(written.agents?.list).toEqual(resolved.agents?.list);
      expect(written.gateway).toEqual(resolved.gateway);
      expect(written.tools?.profile).toBe("coding");
      expect(written.logging).toEqual(resolved.logging);
    });
  });

  describe("config validate", () => {
    it("prints success output for a valid config", async () => {
      const resolved: MarvConfig = { gateway: { port: 18789 } };
      setSnapshot(resolved, resolved);

      await runConfigCommand(["config", "validate"]);

      expect(mockLog).toHaveBeenCalledWith(expect.stringContaining("Config valid:"));
    });

    it("prints JSON issues and exits non-zero for an invalid config", async () => {
      setInvalidSnapshot([{ path: "gateway.port", message: "Expected number, received string" }]);

      await expect(runConfigCommand(["config", "validate", "--json"])).rejects.toThrow(
        "__exit__:1",
      );

      const output = String(mockLog.mock.calls.at(-1)?.[0] ?? "");
      expect(output).toContain('"valid": false');
      expect(output).toContain('"gateway.port"');
    });
  });

  describe("config get with invalid config", () => {
    it("still shows stored values for inspection when the config is invalid", async () => {
      mockReadConfigFileSnapshot.mockResolvedValueOnce({
        path: "/tmp/marv.json",
        exists: true,
        raw: JSON.stringify({ gateway: { port: "oops" } }),
        parsed: { gateway: { port: "oops" } },
        resolved: { gateway: { port: "oops" } },
        valid: false,
        config: {},
        issues: [{ path: "gateway.port", message: "Expected number, received string" }],
        warnings: [],
        legacyIssues: [],
      } as ConfigFileSnapshot);

      await runConfigCommand(["config", "get", "gateway.port"]);

      expect(mockError).toHaveBeenCalledWith(
        expect.stringContaining("Showing the stored value anyway"),
      );
      expect(mockLog).toHaveBeenCalledWith("oops");
    });
  });
});
