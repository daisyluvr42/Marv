import { describe, expect, it } from "vitest";
import type { OpenClawConfig } from "../config/config.js";
import {
  buildClassifierPrompt,
  classifyComplexityByRules,
  parseClassifierResponse,
  resolveAutoRouting,
} from "./auto-routing.js";

describe("classifyComplexityByRules", () => {
  it("classifies a short greeting as simple", () => {
    expect(classifyComplexityByRules({ prompt: "Hi there!" })).toBe("simple");
  });

  it("classifies a short question as simple", () => {
    expect(classifyComplexityByRules({ prompt: "What is 2+2?" })).toBe("simple");
  });

  it("classifies a medium question with complexity signals as moderate", () => {
    // Length over simpleMaxChars (200) → score 1, plus "analyze" complex pattern → score 1 = 2 → moderate
    const prompt =
      "Can you explain how JavaScript closures work and give me a detailed example of when they are useful in practice? Also analyze the performance implications of using closures in hot loops versus plain functions.";
    expect(classifyComplexityByRules({ prompt })).toBe("moderate");
  });

  it("classifies code with multiple markers as complex", () => {
    const prompt = `Here's my code:
\`\`\`
async function fetchData() {
  const response = await fetch('/api/data');
  if (!response.ok) {
    throw new Error('Failed');
  }
  return response.json();
}
\`\`\`
Can you debug why this fails?`;
    const result = classifyComplexityByRules({ prompt });
    expect(["complex", "expert"]).toContain(result);
  });

  it("classifies a long multi-step request as expert", () => {
    const prompt = `I need you to implement a complete authentication system for my application.

1. First, create the user model with password hashing using bcrypt
2. Then, implement JWT token generation and validation
3. Also, add middleware for route protection
4. Next, implement the login and registration endpoints
5. Finally, add refresh token rotation

Please analyze the existing codebase first to understand the architecture, then design the implementation following best practices. Make sure to consider security implications and optimize for performance.

The codebase uses TypeScript with Express and Prisma ORM. We need to refactor the existing session-based auth to JWT while maintaining backward compatibility during the migration period.`;
    expect(classifyComplexityByRules({ prompt })).toBe("expert");
  });

  it("bumps complexity when images are present", () => {
    // Use a prompt that already scores 1 from a complex pattern so images push it to moderate.
    const prompt = "Can you analyze this screenshot and explain what's happening?";
    const withoutImages = classifyComplexityByRules({ prompt, hasImages: false });
    const withImages = classifyComplexityByRules({ prompt, hasImages: true });
    expect(withoutImages).toBe("simple");
    expect(withImages).toBe("moderate");
  });

  it("respects custom thresholds", () => {
    const prompt = "Hello, how are you doing today? I hope you are well.";
    // Default simpleMaxChars=200, this is under it → simple
    expect(classifyComplexityByRules({ prompt })).toBe("simple");
    // With very low threshold → should bump up
    expect(
      classifyComplexityByRules({
        prompt,
        thresholds: { simpleMaxChars: 10, moderateMaxChars: 30 },
      }),
    ).toBe("moderate");
  });

  it("detects complex patterns from config", () => {
    // Both default pattern groups hit: "implement" (group 1) + "analyze" (group 2) → score 2 → moderate
    const prompt = "Please implement the auth module and analyze the security implications";
    const result = classifyComplexityByRules({ prompt });
    expect(result).toBe("moderate");
  });

  it("handles empty prompt as simple", () => {
    expect(classifyComplexityByRules({ prompt: "" })).toBe("simple");
  });
});

describe("buildClassifierPrompt", () => {
  it("includes the user message", () => {
    const prompt = buildClassifierPrompt("Hello world");
    expect(prompt).toContain("Hello world");
    expect(prompt).toContain("simple");
    expect(prompt).toContain("expert");
  });

  it("truncates very long messages", () => {
    const longMessage = "x".repeat(3000);
    const prompt = buildClassifierPrompt(longMessage);
    expect(prompt.length).toBeLessThan(3000);
    expect(prompt).toContain("...");
  });
});

describe("parseClassifierResponse", () => {
  it("parses exact tier names", () => {
    expect(parseClassifierResponse("simple")).toBe("simple");
    expect(parseClassifierResponse("moderate")).toBe("moderate");
    expect(parseClassifierResponse("complex")).toBe("complex");
    expect(parseClassifierResponse("expert")).toBe("expert");
  });

  it("handles surrounding whitespace", () => {
    expect(parseClassifierResponse("  complex  \n")).toBe("complex");
  });

  it("extracts tier from longer response", () => {
    expect(parseClassifierResponse("I would classify this as expert")).toBe("expert");
  });

  it("defaults to moderate on unrecognized response", () => {
    expect(parseClassifierResponse("unknown")).toBe("moderate");
  });
});

describe("resolveAutoRouting", () => {
  const baseConfig: OpenClawConfig = {
    agents: {
      defaults: {
        autoRouting: {
          enabled: true,
          rules: [
            { complexity: "simple", model: "anthropic/claude-haiku-4-5", thinking: "off" },
            { complexity: "moderate", model: "anthropic/claude-sonnet-4-6", thinking: "off" },
            { complexity: "complex", model: "anthropic/claude-sonnet-4-6", thinking: "low" },
            { complexity: "expert", model: "anthropic/claude-opus-4-6", thinking: "medium" },
          ],
        },
      },
    },
  };

  it("returns routed=false when disabled", async () => {
    const config: OpenClawConfig = {
      agents: { defaults: { autoRouting: { enabled: false } } },
    };
    const result = await resolveAutoRouting({
      prompt: "Hi",
      config,
      defaultProvider: "anthropic",
      defaultModel: "claude-opus-4-6",
    });
    expect(result.routed).toBe(false);
  });

  it("returns routed=false when config is missing", async () => {
    const result = await resolveAutoRouting({
      prompt: "Hi",
      config: undefined,
      defaultProvider: "anthropic",
      defaultModel: "claude-opus-4-6",
    });
    expect(result.routed).toBe(false);
  });

  it("routes a simple message to the fast model", async () => {
    const result = await resolveAutoRouting({
      prompt: "Hi!",
      config: baseConfig,
      defaultProvider: "anthropic",
      defaultModel: "claude-opus-4-6",
    });
    expect(result.routed).toBe(true);
    expect(result.complexity).toBe("simple");
    expect(result.model).toBe("claude-haiku-4-5");
    expect(result.provider).toBe("anthropic");
    expect(result.thinking).toBe("off");
  });

  it("routes an expert message to the powerful model", async () => {
    const longPrompt = `Please implement a complete microservices architecture with the following:
1. Design the service boundaries and communication patterns
2. Implement the API gateway with rate limiting and authentication
3. Also create the event bus for async communication between services
4. Then implement each service with proper error handling, circuit breakers, and retry logic
5. Finally, optimize the database queries and add comprehensive monitoring

Step 1: analyze the current monolith
Step 2: refactor into bounded contexts

${"x".repeat(1500)}`;
    const result = await resolveAutoRouting({
      prompt: longPrompt,
      config: baseConfig,
      defaultProvider: "anthropic",
      defaultModel: "claude-opus-4-6",
    });
    expect(result.routed).toBe(true);
    expect(result.complexity).toBe("expert");
    expect(result.model).toBe("claude-opus-4-6");
    expect(result.thinking).toBe("medium");
  });

  it("returns routed=false when no rule matches the tier", async () => {
    const config: OpenClawConfig = {
      agents: {
        defaults: {
          autoRouting: {
            enabled: true,
            // Only has an "expert" rule — a simple message won't match it.
            rules: [{ complexity: "expert", model: "anthropic/claude-opus-4-6" }],
          },
        },
      },
    };
    const result = await resolveAutoRouting({
      prompt: "Hi there!",
      config,
      defaultProvider: "anthropic",
      defaultModel: "claude-opus-4-6",
    });
    // Simple prompt but only "expert" rule exists → no match
    expect(result.routed).toBe(false);
    expect(result.complexity).toBe("simple");
  });

  it("uses per-agent config when available", async () => {
    const config: OpenClawConfig = {
      agents: {
        defaults: {
          autoRouting: {
            enabled: true,
            rules: [{ complexity: "simple", model: "anthropic/claude-haiku-4-5" }],
          },
        },
        list: [
          {
            id: "coding-agent",
            autoRouting: {
              enabled: true,
              rules: [{ complexity: "simple", model: "openai/gpt-4o-mini" }],
            },
          },
        ],
      },
    };
    const result = await resolveAutoRouting({
      prompt: "Hi",
      config,
      agentId: "coding-agent",
      defaultProvider: "anthropic",
      defaultModel: "claude-opus-4-6",
    });
    expect(result.routed).toBe(true);
    expect(result.provider).toBe("openai");
    expect(result.model).toBe("gpt-4o-mini");
  });

  it("uses LLM classifier when configured and classifyFn provided", async () => {
    const config: OpenClawConfig = {
      agents: {
        defaults: {
          autoRouting: {
            enabled: true,
            classifier: "llm",
            classifierModel: "anthropic/claude-haiku-4-5",
            rules: [
              { complexity: "simple", model: "anthropic/claude-haiku-4-5" },
              { complexity: "expert", model: "anthropic/claude-opus-4-6" },
            ],
          },
        },
      },
    };
    const result = await resolveAutoRouting({
      prompt: "Hi",
      config,
      defaultProvider: "anthropic",
      defaultModel: "claude-opus-4-6",
      classifyFn: async () => "expert",
    });
    expect(result.routed).toBe(true);
    expect(result.complexity).toBe("expert");
    expect(result.model).toBe("claude-opus-4-6");
  });

  it("falls back to rules when LLM classifier throws", async () => {
    const config: OpenClawConfig = {
      agents: {
        defaults: {
          autoRouting: {
            enabled: true,
            classifier: "llm",
            rules: [{ complexity: "simple", model: "anthropic/claude-haiku-4-5" }],
          },
        },
      },
    };
    const result = await resolveAutoRouting({
      prompt: "Hi",
      config,
      defaultProvider: "anthropic",
      defaultModel: "claude-opus-4-6",
      classifyFn: async () => {
        throw new Error("model unavailable");
      },
    });
    expect(result.routed).toBe(true);
    expect(result.complexity).toBe("simple");
  });

  it("falls back to rules when classifyFn is not provided for llm classifier", async () => {
    const config: OpenClawConfig = {
      agents: {
        defaults: {
          autoRouting: {
            enabled: true,
            classifier: "llm",
            rules: [{ complexity: "simple", model: "anthropic/claude-haiku-4-5" }],
          },
        },
      },
    };
    const result = await resolveAutoRouting({
      prompt: "Hi",
      config,
      defaultProvider: "anthropic",
      defaultModel: "claude-opus-4-6",
    });
    // Without classifyFn, falls to rules classifier
    expect(result.routed).toBe(true);
    expect(result.complexity).toBe("simple");
  });
});
