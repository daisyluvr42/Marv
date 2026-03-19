import fsSync from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { __testing, createRequestMissingToolsTool } from "./request-missing-tools-tool.js";

const {
  discoverMock,
  inspectDiscoveredSkillSafetyMock,
  installDiscoveredSkillMock,
  callGatewayToolMock,
  readSkillUsageRecordsMock,
} = vi.hoisted(() => ({
  discoverMock: vi.fn(),
  inspectDiscoveredSkillSafetyMock: vi.fn(),
  installDiscoveredSkillMock: vi.fn(),
  callGatewayToolMock: vi.fn(),
  readSkillUsageRecordsMock: vi.fn(),
}));

vi.mock("./tool-discovery.js", () => ({
  ToolDiscoveryService: class {
    discover = discoverMock;
    discoverAsync = vi
      .fn()
      .mockImplementation((...args: unknown[]) => Promise.resolve(discoverMock(...args)));
  },
}));

vi.mock("../skills-install.js", () => ({
  inspectDiscoveredSkillSafety: inspectDiscoveredSkillSafetyMock,
  installDiscoveredSkill: installDiscoveredSkillMock,
}));

vi.mock("../skill-usage-records.js", () => ({
  readSkillUsageRecords: readSkillUsageRecordsMock,
  markInstalledSkillUsageRecord: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("./gateway.js", () => ({
  callGatewayTool: callGatewayToolMock,
  readGatewayCallOptions: vi.fn().mockReturnValue({}),
}));

describe("request_missing_tools tool", () => {
  beforeEach(() => {
    readSkillUsageRecordsMock.mockResolvedValue({});
    __testing.resetSeenCapabilitySearches();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("auto-installs clean skills without approval", async () => {
    discoverMock.mockReturnValue([
      {
        skillId: "github-repos",
        source: "managed",
        metadata: {},
        confidenceScore: 0.92,
      },
    ]);
    inspectDiscoveredSkillSafetyMock.mockResolvedValue({
      level: "clean",
      warnings: [],
      findings: [],
      blocked: false,
    });
    installDiscoveredSkillMock.mockResolvedValue({
      ok: true,
      message: "Installed",
      stdout: "",
      stderr: "",
      code: 0,
      scan: { level: "clean" },
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
      installed: Array<{ skillId: string; ok: boolean; approved: string }>;
    };

    expect(details.discovered[0]?.skillId).toBe("github-repos");
    expect(details.installed[0]?.skillId).toBe("github-repos");
    expect(details.installed[0]?.ok).toBe(true);
    expect(details.installed[0]?.approved).toBe("not-needed");
    expect(callGatewayToolMock).not.toHaveBeenCalled();
  });

  it("forwards context task id as the approval task id when warnings require approval", async () => {
    discoverMock.mockReturnValue([
      {
        skillId: "github-repos",
        source: "managed",
        metadata: {},
        confidenceScore: 0.92,
      },
    ]);
    inspectDiscoveredSkillSafetyMock.mockResolvedValue({
      level: "warn",
      warnings: ["warn"],
      findings: [],
      blocked: false,
    });
    callGatewayToolMock.mockResolvedValue({ decision: "deny" });

    const tool = createRequestMissingToolsTool({
      workspaceDir: "/tmp/workspace",
      config: { autonomy: { autoInstallSkills: true } },
      agentSessionKey: "agent:ops:main",
    });

    await tool.execute("call-task", {
      description: "search github repos",
      contextTaskId: "task-99",
    });

    expect(callGatewayToolMock).toHaveBeenCalledWith(
      "exec.approval.request",
      {},
      expect.objectContaining({
        taskId: "task-99",
        agentId: "ops",
      }),
      { expectFinal: true },
    );
  });

  it("blocks installation when scan is critical", async () => {
    discoverMock.mockReturnValue([
      {
        skillId: "github-repos",
        source: "managed",
        metadata: {},
        confidenceScore: 0.92,
      },
    ]);
    inspectDiscoveredSkillSafetyMock.mockResolvedValue({
      level: "critical",
      warnings: ["danger"],
      findings: [],
      blocked: true,
    });

    const tool = createRequestMissingToolsTool({
      workspaceDir: "/tmp/workspace",
      config: { autonomy: { autoInstallSkills: true } },
    });

    const result = await tool.execute("call2", {
      description: "search github repos",
    });
    const details = result.details as {
      installed: Array<{ ok: boolean; approved: string; scanLevel: string }>;
    };

    expect(details.installed[0]?.ok).toBe(false);
    expect(details.installed[0]?.approved).toBe("blocked");
    expect(details.installed[0]?.scanLevel).toBe("critical");
    expect(callGatewayToolMock).not.toHaveBeenCalled();
    expect(installDiscoveredSkillMock).not.toHaveBeenCalled();
  });

  it("requests approval when autoInstall is disabled in the request", async () => {
    discoverMock.mockReturnValue([
      {
        skillId: "github-repos",
        source: "managed",
        metadata: {},
        confidenceScore: 0.92,
      },
    ]);
    inspectDiscoveredSkillSafetyMock.mockResolvedValue({
      level: "clean",
      warnings: [],
      findings: [],
      blocked: false,
    });
    callGatewayToolMock.mockResolvedValue({ decision: "allow-once" });
    installDiscoveredSkillMock.mockResolvedValue({
      ok: true,
      message: "Installed",
      stdout: "",
      stderr: "",
      code: 0,
      scan: { level: "clean" },
    });

    const tool = createRequestMissingToolsTool({
      workspaceDir: "/tmp/workspace",
      config: { autonomy: { autoInstallSkills: true } },
      agentSessionKey: "agent:main:main",
    });

    const result = await tool.execute("call3", {
      description: "search github repos",
      autoInstall: false,
    });
    const details = result.details as {
      installed: Array<{ ok: boolean; approved: string }>;
    };

    expect(details.installed[0]?.ok).toBe(true);
    expect(details.installed[0]?.approved).toBe("allow-once");
    expect(callGatewayToolMock).toHaveBeenCalledTimes(1);
  });

  it("returns a synthesis hint when discovery finds no matching skill", async () => {
    discoverMock.mockReturnValue([]);

    const tool = createRequestMissingToolsTool({
      workspaceDir: "/tmp/workspace",
      config: { autonomy: { autoInstallSkills: true } },
    });

    const result = await tool.execute("call4", {
      description: "inspect parquet files",
    });
    const details = result.details as {
      discovered: unknown[];
      synthesisHint?: { guidance?: string } | null;
      message: string;
    };

    expect(details.discovered).toEqual([]);
    expect(details.synthesisHint?.guidance).toContain("Create one:");
    expect(details.message).toContain("Consider creating an ad-hoc solution");
  });

  it("suppresses synthesis hint when tool synthesis is disabled", async () => {
    discoverMock.mockReturnValue([]);

    const tool = createRequestMissingToolsTool({
      workspaceDir: "/tmp/workspace",
      config: {
        autonomy: {
          autoInstallSkills: true,
          toolSynthesis: { enabled: false },
        },
      },
    });

    const result = await tool.execute("call5", {
      description: "inspect parquet files",
    });
    const details = result.details as {
      synthesisHint?: unknown;
      message: string;
    };

    expect(details.synthesisHint).toBeNull();
    expect(details.message).toBe("No matching skills found for the requested capability.");
  });

  it("blocks registry installs when a nested dependency declares lifecycle scripts", async () => {
    const root = fsSync.mkdtempSync(path.join(os.tmpdir(), "marv-registry-scan-"));
    fsSync.mkdirSync(path.join(root, "node_modules", "outer", "node_modules", "inner"), {
      recursive: true,
    });
    fsSync.writeFileSync(
      path.join(root, "package.json"),
      JSON.stringify({ name: "root-skill", version: "1.0.0" }, null, 2),
    );
    fsSync.writeFileSync(
      path.join(root, "node_modules", "outer", "package.json"),
      JSON.stringify({ name: "outer", version: "1.0.0" }, null, 2),
    );
    fsSync.writeFileSync(
      path.join(root, "node_modules", "outer", "node_modules", "inner", "package.json"),
      JSON.stringify(
        {
          name: "inner",
          version: "1.0.0",
          scripts: { postinstall: "node build.js" },
        },
        null,
        2,
      ),
    );

    try {
      const result = await __testing.scanRegistryInstallDir(root);

      expect(result.blocked).toBe(true);
      expect(result.lifecycleScripts).toEqual([
        {
          pkg: "inner",
          scripts: ["postinstall"],
        },
      ]);
      expect(result.warnings[0]).toContain("inner");
    } finally {
      fsSync.rmSync(root, { recursive: true, force: true });
    }
  });

  it("scans package.json contents when checking registry installs", async () => {
    const root = fsSync.mkdtempSync(path.join(os.tmpdir(), "marv-registry-scan-"));
    fsSync.writeFileSync(
      path.join(root, "package.json"),
      JSON.stringify(
        {
          name: "bad-skill",
          description: 'This manifest contains eval("hack") for testing',
        },
        null,
        2,
      ),
    );

    try {
      const result = await __testing.scanRegistryInstallDir(root);

      expect(result.blocked).toBe(true);
      expect(result.scan?.findings.some((f) => f.file.endsWith("package.json"))).toBe(true);
      expect(result.scan?.findings.some((f) => f.ruleId === "dynamic-code-execution")).toBe(true);
    } finally {
      fsSync.rmSync(root, { recursive: true, force: true });
    }
  });
});
