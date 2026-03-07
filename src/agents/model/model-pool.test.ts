import { describe, expect, it, vi } from "vitest";
import type { MarvConfig } from "../../core/config/config.js";
import { resolveRuntimeModelPlan } from "./model-pool.js";

vi.mock("../agent-scope.js", () => ({
  resolveAgentConfig: vi.fn((cfg: MarvConfig, agentId?: string) => {
    if (agentId === "vision") {
      return { modelPool: "vision" };
    }
    return { modelPool: "default" };
  }),
  resolveDefaultAgentId: vi.fn(() => "main"),
}));

vi.mock("../auth-profiles.js", () => ({
  ensureAuthProfileStore: vi.fn(() => ({ profiles: {} })),
  listProfilesForProvider: vi.fn(() => []),
}));

vi.mock("./model-auth.js", () => ({
  getCustomProviderApiKey: vi.fn((cfg: MarvConfig, provider: string) =>
    provider === "openai" ? "key" : undefined,
  ),
  resolveEnvApiKey: vi.fn(() => undefined),
}));

describe("resolveRuntimeModelPlan", () => {
  it("prefers local low-tier candidates before cloud candidates", () => {
    const cfg = {
      models: {
        catalog: {
          "ollama/qwen2.5-coder": {
            location: "local",
            tier: "low",
            capabilities: ["text", "coding", "tools"],
          },
          "openai/gpt-4o": {
            location: "cloud",
            tier: "standard",
            capabilities: ["text", "vision"],
          },
          "anthropic/claude-sonnet-4-5": {
            location: "cloud",
            tier: "high",
            capabilities: ["text", "coding"],
          },
        },
      },
      agents: {
        defaults: {
          modelPool: "default",
        },
      },
    } as MarvConfig;

    const plan = resolveRuntimeModelPlan({ cfg, agentId: "main" });

    expect(plan.poolName).toBe("default");
    expect(plan.configured.map((entry) => entry.ref)).toEqual([
      "ollama/qwen2.5-coder",
      "openai/gpt-4o",
      "anthropic/claude-sonnet-4-5",
    ]);
    expect(plan.candidates.map((entry) => entry.ref)).toEqual([
      "ollama/qwen2.5-coder",
      "openai/gpt-4o",
    ]);
  });

  it("filters candidates by pool capability requirements", () => {
    const cfg = {
      models: {
        catalog: {
          "openai/gpt-4o": {
            location: "cloud",
            tier: "standard",
            capabilities: ["text", "vision"],
          },
          "openai/gpt-4o-mini": {
            location: "cloud",
            tier: "low",
            capabilities: ["text"],
          },
        },
      },
      agents: {
        defaults: {
          modelPool: "default",
        },
        modelPools: {
          vision: {
            requireCapabilities: ["vision"],
          },
        },
      },
    } as MarvConfig;

    const plan = resolveRuntimeModelPlan({ cfg, agentId: "vision" });

    expect(plan.poolName).toBe("vision");
    expect(plan.candidates.map((entry) => entry.ref)).toEqual(["openai/gpt-4o"]);
  });
});
