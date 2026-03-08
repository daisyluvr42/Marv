import { beforeEach, describe, expect, it, vi } from "vitest";
import type { MarvConfig } from "../../core/config/config.js";

const applyNonInteractiveAuthChoice = vi.fn();
const loadValidConfigOrThrow = vi.fn();
const resolveKnownAgentId = vi.fn();
const updateConfig = vi.fn();
const resolveModelsAuthChoice = vi.fn();

vi.mock("../../agents/agent-scope.js", () => ({
  resolveAgentDir: vi.fn((_cfg, agentId: string) => `/tmp/marv-agents/${agentId}`),
  resolveAgentWorkspaceDir: vi.fn(),
  resolveDefaultAgentId: vi.fn(),
}));

vi.mock("../onboard-non-interactive/local/auth-choice.js", () => ({
  applyNonInteractiveAuthChoice,
}));

vi.mock("./auth-choice.js", () => ({
  resolveModelsAuthChoice,
}));

vi.mock("./shared.js", () => ({
  loadValidConfigOrThrow,
  resolveKnownAgentId,
  updateConfig,
}));

describe("modelsAuthSetCommand", () => {
  const runtime = {
    log: vi.fn(),
    error: vi.fn(),
    exit: vi.fn(),
  };

  const baseConfig: MarvConfig = {
    agents: {
      defaults: {
        model: {
          primary: "anthropic/claude-sonnet-4-6",
        },
      },
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
    loadValidConfigOrThrow.mockResolvedValue(baseConfig);
    resolveKnownAgentId.mockImplementation(({ rawAgentId }: { rawAgentId?: string }) => rawAgentId);
    resolveModelsAuthChoice.mockReturnValue({
      choice: "gemini-api-key",
      options: [],
    });
  });

  it("passes agent scope to non-interactive auth setup", async () => {
    let savedConfig: MarvConfig | undefined;
    updateConfig.mockImplementation(async (updater: (cfg: MarvConfig) => MarvConfig) => {
      savedConfig = updater(baseConfig);
    });
    applyNonInteractiveAuthChoice.mockResolvedValue({
      ...baseConfig,
      auth: {
        profiles: {
          "google:default": {
            provider: "google",
            mode: "api_key",
          },
        },
      },
      agents: {
        defaults: {
          model: {
            primary: "google/gemini-2.5-pro",
          },
        },
      },
    } satisfies MarvConfig);

    const { modelsAuthSetCommand } = await import("./auth.js");
    await modelsAuthSetCommand(
      {
        provider: "google",
        method: "gemini-api-key",
        apiKey: "gem-test-key",
        agent: "pairing",
      },
      runtime as never,
    );

    expect(applyNonInteractiveAuthChoice).toHaveBeenCalledWith(
      expect.objectContaining({
        authChoice: "gemini-api-key",
        agentDir: "/tmp/marv-agents/pairing",
      }),
    );
    expect(savedConfig?.agents?.defaults?.model?.primary).toBe("anthropic/claude-sonnet-4-6");
  });

  it("keeps the provider default when --set-default is enabled", async () => {
    let savedConfig: MarvConfig | undefined;
    updateConfig.mockImplementation(async (updater: (cfg: MarvConfig) => MarvConfig) => {
      savedConfig = updater(baseConfig);
    });
    applyNonInteractiveAuthChoice.mockResolvedValue({
      ...baseConfig,
      agents: {
        defaults: {
          model: {
            primary: "google/gemini-2.5-pro",
          },
        },
      },
    } satisfies MarvConfig);

    const { modelsAuthSetCommand } = await import("./auth.js");
    await modelsAuthSetCommand(
      {
        provider: "google",
        method: "gemini-api-key",
        apiKey: "gem-test-key",
        setDefault: true,
      },
      runtime as never,
    );

    expect(savedConfig?.agents?.defaults?.model?.primary).toBe("google/gemini-2.5-pro");
  });
});
