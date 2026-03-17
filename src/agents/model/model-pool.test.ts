import { describe, expect, it, vi } from "vitest";
import type { MarvConfig } from "../../core/config/config.js";
import type { RuntimeConfiguredModel } from "./model-pool.js";
import {
  applyThinkingModelPreferences,
  resolveRuntimeModelPlan,
  resolveThinkingModelTier,
} from "./model-pool.js";
import { resolveSelectedModelRefs } from "./model-selections.js";

const readRuntimeModelRegistryMock = vi.fn((): unknown => null);
const listConfiguredProvidersMock = vi.fn(
  (_cfg?: unknown, _agentDir?: string) => new Set<string>(),
);

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
  listProfilesForProvider: vi.fn((_store, provider: string) =>
    provider === "google" ? ["google:default"] : [],
  ),
}));

vi.mock("./model-auth.js", () => ({
  getCustomProviderApiKey: vi.fn((cfg: MarvConfig, provider: string) =>
    provider === "openai" ? "key" : undefined,
  ),
  resolveEnvApiKey: vi.fn(() => undefined),
}));

vi.mock("./model-availability-state.js", () => ({
  getRuntimeModelAvailability: vi.fn(() => undefined),
}));

vi.mock("./runtime-model-registry.js", () => ({
  readRuntimeModelRegistry: () => readRuntimeModelRegistryMock(),
  listConfiguredProviders: (cfg: MarvConfig, agentDir?: string) =>
    listConfiguredProvidersMock(cfg, agentDir),
}));

describe("resolveRuntimeModelPlan", () => {
  it("keeps auth-backed runtime registry models even when selections are narrower", () => {
    readRuntimeModelRegistryMock.mockReturnValue({
      models: [
        {
          ref: "google/gemini-2.0-flash",
          provider: "google",
          model: "gemini-2.0-flash",
          displayName: "Gemini 2.0 Flash",
          source: "official_api",
          status: "active",
          capabilities: ["text"],
          tier: "low",
          location: "cloud",
        },
        {
          ref: "google/gemini-2.5-flash",
          provider: "google",
          model: "gemini-2.5-flash",
          displayName: "Gemini 2.5 Flash",
          source: "official_api",
          status: "active",
          capabilities: ["text", "vision"],
          tier: "standard",
          location: "cloud",
        },
      ],
    });
    listConfiguredProvidersMock.mockReturnValue(new Set(["google"]));

    const cfg = {
      auth: {
        profiles: {
          "google:default": {
            provider: "google",
            mode: "api_key",
          },
        },
      },
      models: {
        selections: {
          "google:default": ["google-gemini-cli/gemini-2.0-flash"],
        },
      },
      agents: {
        defaults: {
          modelPool: "default",
        },
      },
    } as MarvConfig;

    const plan = resolveRuntimeModelPlan({ cfg, agentId: "main" });

    expect(plan.configured.map((entry) => entry.ref)).toEqual([
      "google/gemini-2.0-flash",
      "google/gemini-2.5-flash",
    ]);
    expect(plan.candidates.map((entry) => entry.ref)).toEqual([
      "google/gemini-2.0-flash",
      "google/gemini-2.5-flash",
    ]);
  });

  it("prefers local low-tier candidates before cloud candidates", () => {
    readRuntimeModelRegistryMock.mockReturnValue(null);
    listConfiguredProvidersMock.mockReturnValue(new Set());

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
    readRuntimeModelRegistryMock.mockReturnValue(null);
    listConfiguredProvidersMock.mockReturnValue(new Set());

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

  it("builds candidates from auth-backed model selections", () => {
    readRuntimeModelRegistryMock.mockReturnValue(null);
    listConfiguredProvidersMock.mockReturnValue(new Set());

    const cfg = {
      auth: {
        profiles: {
          "google:default": {
            provider: "google",
            mode: "api_key",
          },
        },
      },
      models: {
        selections: {
          "google:default": [
            "google-gemini-cli/gemini-2.0-flash",
            "google-gemini-cli/gemini-2.5-flash",
          ],
        },
        catalog: {
          "google-gemini-cli/gemini-2.0-flash": {
            location: "cloud",
            tier: "low",
            capabilities: ["text"],
          },
          "google-gemini-cli/gemini-2.5-flash": {
            location: "cloud",
            tier: "standard",
            capabilities: ["text", "vision"],
          },
          "openai/gpt-4o": {
            location: "cloud",
            tier: "standard",
            capabilities: ["text", "vision"],
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

    expect(plan.configured.map((entry) => entry.ref)).toEqual([
      "google/gemini-2.0-flash",
      "google/gemini-2.5-flash",
    ]);
    expect(plan.candidates.map((entry) => entry.ref)).toEqual([
      "google/gemini-2.0-flash",
      "google/gemini-2.5-flash",
    ]);
  });

  it("remaps google-gemini-cli model refs to google when source profile is api_key", () => {
    readRuntimeModelRegistryMock.mockReturnValue(null);
    listConfiguredProvidersMock.mockReturnValue(new Set());

    const cfg = {
      auth: {
        profiles: {
          "google:default": {
            provider: "google",
            mode: "api_key",
          },
        },
      },
      models: {
        selections: {
          "google:default": ["google-gemini-cli/gemini-2.5-flash"],
        },
      },
    } as MarvConfig;

    expect(Array.from(resolveSelectedModelRefs({ cfg, defaultProvider: "anthropic" }))).toEqual([
      "google/gemini-2.5-flash",
    ]);
  });

  it("drops runtime-unsupported models from the candidate pool", async () => {
    readRuntimeModelRegistryMock.mockReturnValue(null);
    listConfiguredProvidersMock.mockReturnValue(new Set());

    const availability = await import("./model-availability-state.js");
    vi.mocked(availability.getRuntimeModelAvailability).mockImplementation((ref: string) =>
      ref === "local/qwen-small"
        ? {
            status: "unsupported",
            lastCheckedAt: Date.now(),
            lastError: "context window too small",
          }
        : undefined,
    );

    const cfg = {
      models: {
        catalog: {
          "local/qwen-small": {
            location: "local",
            tier: "low",
            capabilities: ["text", "coding"],
          },
          "openai/gpt-4o": {
            location: "cloud",
            tier: "standard",
            capabilities: ["text", "vision"],
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

    expect(plan.configured.find((entry) => entry.ref === "local/qwen-small")?.available).toBe(
      false,
    );
    expect(
      plan.configured.find((entry) => entry.ref === "local/qwen-small")?.availabilityReason,
    ).toBe("unsupported");
    expect(plan.candidates.map((entry) => entry.ref)).toEqual(["openai/gpt-4o"]);
  });

  it("temporarily cools down rate-limited models until they can be retried", async () => {
    readRuntimeModelRegistryMock.mockReturnValue(null);
    listConfiguredProvidersMock.mockReturnValue(new Set());

    const availability = await import("./model-availability-state.js");
    vi.mocked(availability.getRuntimeModelAvailability).mockImplementation((ref: string) =>
      ref === "openai/gpt-4o"
        ? {
            status: "temporary_unavailable",
            lastCheckedAt: Date.now(),
            retryAfter: Date.now() + 60_000,
            lastError: "429 rate limit reached",
          }
        : undefined,
    );

    const cfg = {
      models: {
        catalog: {
          "openai/gpt-4o": {
            location: "cloud",
            tier: "standard",
            capabilities: ["text", "vision"],
          },
          "google/gemini-2.0-flash": {
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
      },
    } as MarvConfig;

    const plan = resolveRuntimeModelPlan({ cfg, agentId: "main" });

    expect(plan.configured.find((entry) => entry.ref === "openai/gpt-4o")?.available).toBe(false);
    expect(plan.configured.find((entry) => entry.ref === "openai/gpt-4o")?.availabilityReason).toBe(
      "temporary_unavailable",
    );
    expect(plan.candidates.map((entry) => entry.ref)).toEqual(["google/gemini-2.0-flash"]);
  });
});

describe("resolveThinkingModelTier", () => {
  it("maps off/minimal/low to low tier", () => {
    expect(resolveThinkingModelTier("off")).toBe("low");
    expect(resolveThinkingModelTier("minimal")).toBe("low");
    expect(resolveThinkingModelTier("low")).toBe("low");
  });

  it("maps medium to medium tier", () => {
    expect(resolveThinkingModelTier("medium")).toBe("medium");
  });

  it("maps high/xhigh to high tier", () => {
    expect(resolveThinkingModelTier("high")).toBe("high");
    expect(resolveThinkingModelTier("xhigh")).toBe("high");
  });
});

describe("applyThinkingModelPreferences", () => {
  const makeCandidate = (ref: string): RuntimeConfiguredModel => ({
    ref,
    provider: ref.split("/")[0],
    model: ref.split("/")[1],
    location: "cloud",
    tier: "standard",
    capabilities: ["text"],
    priority: 0,
    enabled: true,
    available: true,
    aliases: [],
  });

  const candidates = [
    makeCandidate("google/gemini-2.5-flash"),
    makeCandidate("openai/gpt-4o"),
    makeCandidate("anthropic/claude-opus-4-6"),
  ];

  it("returns candidates unchanged when no thinkingModels config", () => {
    const result = applyThinkingModelPreferences({
      candidates,
      thinkingModels: undefined,
      thinkLevel: "high",
    });
    expect(result.map((c) => c.ref)).toEqual(candidates.map((c) => c.ref));
  });

  it("returns candidates unchanged when no thinkLevel", () => {
    const result = applyThinkingModelPreferences({
      candidates,
      thinkingModels: { high: ["anthropic/claude-opus-4-6"] },
      thinkLevel: undefined,
    });
    expect(result.map((c) => c.ref)).toEqual(candidates.map((c) => c.ref));
  });

  it("moves preferred models to front for matching tier", () => {
    const result = applyThinkingModelPreferences({
      candidates,
      thinkingModels: { high: ["anthropic/claude-opus-4-6", "openai/gpt-4o"] },
      thinkLevel: "high",
    });
    expect(result.map((c) => c.ref)).toEqual([
      "anthropic/claude-opus-4-6",
      "openai/gpt-4o",
      "google/gemini-2.5-flash",
    ]);
  });

  it("preserves order when preferred models are already first", () => {
    const result = applyThinkingModelPreferences({
      candidates,
      thinkingModels: { low: ["google/gemini-2.5-flash"] },
      thinkLevel: "low",
    });
    expect(result.map((c) => c.ref)).toEqual([
      "google/gemini-2.5-flash",
      "openai/gpt-4o",
      "anthropic/claude-opus-4-6",
    ]);
  });

  it("ignores preferred models not in candidates", () => {
    const result = applyThinkingModelPreferences({
      candidates,
      thinkingModels: { medium: ["unknown/model", "openai/gpt-4o"] },
      thinkLevel: "medium",
    });
    expect(result.map((c) => c.ref)).toEqual([
      "openai/gpt-4o",
      "google/gemini-2.5-flash",
      "anthropic/claude-opus-4-6",
    ]);
  });

  it("maps xhigh to high tier", () => {
    const result = applyThinkingModelPreferences({
      candidates,
      thinkingModels: { high: ["anthropic/claude-opus-4-6"] },
      thinkLevel: "xhigh",
    });
    expect(result[0].ref).toBe("anthropic/claude-opus-4-6");
  });

  it("returns candidates unchanged when tier has no preferred models", () => {
    const result = applyThinkingModelPreferences({
      candidates,
      thinkingModels: { high: ["anthropic/claude-opus-4-6"] },
      thinkLevel: "low",
    });
    expect(result.map((c) => c.ref)).toEqual(candidates.map((c) => c.ref));
  });

  it("returns empty array for empty candidates", () => {
    const result = applyThinkingModelPreferences({
      candidates: [],
      thinkingModels: { high: ["anthropic/claude-opus-4-6"] },
      thinkLevel: "high",
    });
    expect(result).toEqual([]);
  });
});
