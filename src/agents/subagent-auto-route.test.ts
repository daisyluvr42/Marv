import { describe, expect, it, vi } from "vitest";

// Mock the model pool module to avoid needing real provider configs.
// Real resolveRuntimeModelPlan sorts candidates high → standard → low (descending tier weight).
vi.mock("./model/model-pool.js", () => ({
  resolveRuntimeModelPlan: vi.fn(() => ({
    poolName: "default",
    configured: [],
    candidates: [
      {
        ref: "anthropic/claude-3-5-sonnet",
        provider: "anthropic",
        model: "claude-3-5-sonnet",
        location: "cloud",
        tier: "high",
        capabilities: ["text", "vision", "coding", "tools"],
        priority: 0,
        enabled: true,
        available: true,
        aliases: [],
      },
      {
        ref: "anthropic/claude-3-haiku",
        provider: "anthropic",
        model: "claude-3-haiku",
        location: "cloud",
        tier: "low",
        capabilities: ["text", "tools"],
        priority: 0,
        enabled: true,
        available: true,
        aliases: [],
      },
    ],
  })),
}));

// Mock the auto-routing classifier.
vi.mock("./auto-routing.js", () => ({
  classifyComplexityByRules: vi.fn(({ prompt }: { prompt: string }) => {
    if (prompt.length > 500) {
      return "expert";
    }
    if (prompt.length > 200) {
      return "complex";
    }
    if (prompt.length > 50) {
      return "moderate";
    }
    return "simple";
  }),
}));

describe("resolveSubagentAutoRoute", () => {
  it("matches a preset by keywords", async () => {
    const { resolveSubagentAutoRoute } = await import("./subagent-auto-route.js");
    const result = resolveSubagentAutoRoute({
      task: "review this code for bugs",
      cfg: {
        agents: {
          defaults: {
            subagents: {
              presets: {
                "code-review": {
                  roles: ["reviewer", "linter"],
                  autoTrigger: {
                    keywords: ["review", "lint"],
                  },
                },
              },
            },
          },
        },
      } as never,
    });

    expect(result.matched).toBe(true);
    expect(result.preset).toBe("code-review");
    expect(result.roles).toEqual(["reviewer", "linter"]);
    expect(result.matchReason).toBe("keywords");
  });

  it("matches a preset by complexity threshold", async () => {
    const { resolveSubagentAutoRoute } = await import("./subagent-auto-route.js");
    // Task long enough to classify as "complex" (>200 chars per mock).
    const longTask = "x".repeat(250);
    const result = resolveSubagentAutoRoute({
      task: longTask,
      cfg: {
        agents: {
          defaults: {
            subagents: {
              presets: {
                "deep-analysis": {
                  roles: ["analyst"],
                  autoTrigger: {
                    minComplexity: "complex",
                  },
                },
              },
            },
          },
        },
      } as never,
    });

    expect(result.matched).toBe(true);
    expect(result.preset).toBe("deep-analysis");
    expect(result.matchReason).toBe("complexity");
  });

  it("returns matched=false when no preset triggers", async () => {
    const { resolveSubagentAutoRoute } = await import("./subagent-auto-route.js");
    const result = resolveSubagentAutoRoute({
      task: "hello",
      cfg: {
        agents: {
          defaults: {
            subagents: {
              presets: {
                "hard-tasks": {
                  roles: ["expert"],
                  autoTrigger: {
                    minComplexity: "expert",
                  },
                },
              },
            },
          },
        },
      } as never,
    });

    expect(result.matched).toBe(false);
  });

  it("prefers 'both' match over single", async () => {
    const { resolveSubagentAutoRoute } = await import("./subagent-auto-route.js");
    // Task >200 chars with keyword "deploy"
    const task = "deploy " + "x".repeat(250);
    const result = resolveSubagentAutoRoute({
      task,
      cfg: {
        agents: {
          defaults: {
            subagents: {
              presets: {
                "deploy-complex": {
                  roles: ["deployer"],
                  autoTrigger: {
                    keywords: ["deploy"],
                    minComplexity: "complex",
                  },
                },
                "generic-complex": {
                  roles: ["worker"],
                  autoTrigger: {
                    minComplexity: "complex",
                  },
                },
              },
            },
          },
        },
      } as never,
    });

    expect(result.matched).toBe(true);
    expect(result.preset).toBe("deploy-complex");
    expect(result.matchReason).toBe("both");
  });
});

describe("resolveTaskAwareModel", () => {
  it("picks low-tier model for simple tasks", async () => {
    const { resolveTaskAwareModel } = await import("./subagent-auto-route.js");
    const result = resolveTaskAwareModel({
      task: "say hi",
      complexity: "simple",
      cfg: {} as never,
    });

    // Simple tasks reverse pool order (high → low becomes low → high), so haiku (low-tier) comes first.
    expect(result.model).toBe("anthropic/claude-3-haiku");
    expect(result.thinking).toBe("off");
  });

  it("picks high-tier model for expert tasks", async () => {
    const { resolveTaskAwareModel } = await import("./subagent-auto-route.js");
    const result = resolveTaskAwareModel({
      task: "design a complex distributed system architecture",
      complexity: "expert",
      cfg: {} as never,
    });

    // Expert tasks keep pool default order (high → low), so sonnet (high-tier) comes first.
    expect(result.model).toBe("anthropic/claude-3-5-sonnet");
    expect(result.thinking).toBe("high");
  });

  it("maps complexity to thinking levels", async () => {
    const { resolveTaskAwareModel } = await import("./subagent-auto-route.js");

    expect(
      resolveTaskAwareModel({ task: "x", complexity: "simple", cfg: {} as never }).thinking,
    ).toBe("off");
    expect(
      resolveTaskAwareModel({ task: "x", complexity: "moderate", cfg: {} as never }).thinking,
    ).toBe("low");
    expect(
      resolveTaskAwareModel({ task: "x", complexity: "complex", cfg: {} as never }).thinking,
    ).toBe("medium");
    expect(
      resolveTaskAwareModel({ task: "x", complexity: "expert", cfg: {} as never }).thinking,
    ).toBe("high");
  });

  it("respects preset thinking override", async () => {
    const { resolveTaskAwareModel } = await import("./subagent-auto-route.js");
    const result = resolveTaskAwareModel({
      task: "test",
      complexity: "expert",
      cfg: {} as never,
      presetAutoTrigger: { thinking: "xhigh" },
    });

    expect(result.thinking).toBe("xhigh");
  });
});
