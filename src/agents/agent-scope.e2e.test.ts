import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MarvConfig } from "../core/config/config.js";
import { resolveAgentConfig, resolveAgentDir, resolveAgentWorkspaceDir } from "./agent-scope.js";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("resolveAgentConfig", () => {
  it("should return undefined when no agents config exists", () => {
    const cfg: MarvConfig = {};
    const result = resolveAgentConfig(cfg, "main");
    expect(result).toBeUndefined();
  });

  it("should return undefined when agent id does not exist", () => {
    const cfg: MarvConfig = {
      agents: {
        defaults: { workspace: "~/marv" },
      },
    };
    const result = resolveAgentConfig(cfg, "nonexistent");
    expect(result).toBeUndefined();
  });

  it("should return basic agent config", () => {
    const cfg: MarvConfig = {
      agents: {
        defaults: {
          name: "Main Agent",
          workspace: "~/marv",
          agentDir: "~/.marv/agents/main",
          model: { primary: "anthropic/claude-opus-4" },
        },
      },
    };
    const result = resolveAgentConfig(cfg, "main");
    expect(result).toEqual({
      name: "Main Agent",
      workspace: "~/marv",
      agentDir: "~/.marv/agents/main",
      model: { primary: "anthropic/claude-opus-4" },
      identity: undefined,
      groupChat: undefined,
      subagents: undefined,
      sandbox: undefined,
      tools: undefined,
    });
  });

  it("returns per-agent model pool", () => {
    const cfg: MarvConfig = {
      agents: {
        defaults: {
          modelPool: "coding",
        },
      },
    };

    expect(resolveAgentConfig(cfg, "main")?.modelPool).toBe("coding");
  });

  it("should return agent-specific sandbox config", () => {
    const cfg: MarvConfig = {
      agents: {
        defaults: {
          workspace: "~/marv-work",
          sandbox: {
            mode: "all",
            scope: "agent",
            perSession: false,
            workspaceAccess: "ro",
            workspaceRoot: "~/sandboxes",
          },
        },
      },
    };
    const result = resolveAgentConfig(cfg, "main");
    expect(result?.sandbox).toEqual({
      mode: "all",
      scope: "agent",
      perSession: false,
      workspaceAccess: "ro",
      workspaceRoot: "~/sandboxes",
    });
  });

  it("should return agent-specific tools config", () => {
    const cfg: MarvConfig = {
      agents: {
        defaults: {
          workspace: "~/marv-restricted",
          tools: {
            allow: ["read"],
            deny: ["exec", "write", "edit"],
            elevated: {
              enabled: false,
              allowFrom: { whatsapp: ["+15555550123"] },
            },
          },
        },
      },
    };
    const result = resolveAgentConfig(cfg, "main");
    expect(result?.tools).toEqual({
      allow: ["read"],
      deny: ["exec", "write", "edit"],
      elevated: {
        enabled: false,
        allowFrom: { whatsapp: ["+15555550123"] },
      },
    });
  });

  it("should return both sandbox and tools config", () => {
    const cfg: MarvConfig = {
      agents: {
        defaults: {
          workspace: "~/marv-family",
          sandbox: {
            mode: "all",
            scope: "agent",
          },
          tools: {
            allow: ["read"],
            deny: ["exec"],
          },
        },
      },
    };
    const result = resolveAgentConfig(cfg, "main");
    expect(result?.sandbox?.mode).toBe("all");
    expect(result?.tools?.allow).toEqual(["read"]);
  });

  it("should normalize agent id", () => {
    const cfg: MarvConfig = {
      agents: {
        defaults: { workspace: "~/marv" },
      },
    };
    // Should normalize to "main" (default)
    const result = resolveAgentConfig(cfg, "");
    expect(result).toBeDefined();
    expect(result?.workspace).toBe("~/marv");
  });

  it("uses MARV_HOME for default agent workspace", () => {
    const home = path.join(path.sep, "srv", "marv-home");
    vi.stubEnv("MARV_HOME", home);

    const workspace = resolveAgentWorkspaceDir({} as MarvConfig, "main");
    expect(workspace).toBe(path.join(path.resolve(home), ".marv", "workspace"));
  });

  it("uses MARV_HOME for default agentDir", () => {
    const home = path.join(path.sep, "srv", "marv-home");
    vi.stubEnv("MARV_HOME", home);
    // Clear state dir so it falls back to MARV_HOME
    vi.stubEnv("MARV_STATE_DIR", "");

    const agentDir = resolveAgentDir({} as MarvConfig, "main");
    expect(agentDir).toBe(path.join(path.resolve(home), ".marv", "agents", "main", "agent"));
  });
});
