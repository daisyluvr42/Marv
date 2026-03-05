import { afterEach, describe, expect, it, vi } from "vitest";
import { createRequestMissingToolsTool } from "./request-missing-tools-tool.js";

const { discoverMock, installDiscoveredSkillMock, callGatewayToolMock } = vi.hoisted(() => ({
  discoverMock: vi.fn(),
  installDiscoveredSkillMock: vi.fn(),
  callGatewayToolMock: vi.fn(),
}));

vi.mock("./tool-discovery.js", () => ({
  ToolDiscoveryService: class {
    discover = discoverMock;
  },
}));

vi.mock("../skills-install.js", () => ({
  installDiscoveredSkill: installDiscoveredSkillMock,
}));

vi.mock("./gateway.js", () => ({
  callGatewayTool: callGatewayToolMock,
  readGatewayCallOptions: vi.fn().mockReturnValue({}),
}));

describe("request_missing_tools tool", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("discovers and installs skills after approval", async () => {
    discoverMock.mockReturnValue([
      {
        skillId: "github-repos",
        source: "managed",
        metadata: {},
        confidenceScore: 0.92,
      },
    ]);
    callGatewayToolMock.mockResolvedValue({ decision: "allow-once" });
    installDiscoveredSkillMock.mockResolvedValue({
      ok: true,
      message: "Installed",
      stdout: "",
      stderr: "",
      code: 0,
    });

    const tool = createRequestMissingToolsTool({
      workspaceDir: "/tmp/workspace",
      config: { autonomy: { autoInstallSkills: true } },
      agentSessionKey: "agent:main:main",
    });

    const result = await tool.execute("call1", {
      description: "search github repos",
      suggestedTools: ["gh", "github"],
    });
    const details = result.details as {
      discovered: Array<{ skillId: string }>;
      installed: Array<{ skillId: string; ok: boolean }>;
    };

    expect(details.discovered[0]?.skillId).toBe("github-repos");
    expect(details.installed[0]?.skillId).toBe("github-repos");
    expect(details.installed[0]?.ok).toBe(true);
  });

  it("returns discovery results without install when autoInstall is disabled in request", async () => {
    discoverMock.mockReturnValue([
      {
        skillId: "github-repos",
        source: "managed",
        metadata: {},
        confidenceScore: 0.92,
      },
    ]);

    const tool = createRequestMissingToolsTool({
      workspaceDir: "/tmp/workspace",
      config: { autonomy: { autoInstallSkills: true } },
    });

    const result = await tool.execute("call2", {
      description: "search github repos",
      autoInstall: false,
    });
    const details = result.details as { installed: unknown[]; message: string };

    expect(details.installed).toEqual([]);
    expect(details.message).toContain("Auto-install disabled");
    expect(callGatewayToolMock).not.toHaveBeenCalled();
    expect(installDiscoveredSkillMock).not.toHaveBeenCalled();
  });
});
