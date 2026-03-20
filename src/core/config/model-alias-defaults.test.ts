import { describe, expect, it } from "vitest";
import { DEFAULT_CONTEXT_TOKENS } from "../../agents/defaults.js";
import { applyModelDefaults } from "./defaults.js";
import type { MarvConfig } from "./types.js";

describe("applyModelDefaults", () => {
  it("adds default aliases when models are present", () => {
    const cfg = {
      models: {
        metadata: {
          "anthropic/claude-opus-4-6": {},
          "openai/gpt-5.4": {},
        },
      },
    } satisfies MarvConfig;
    const next = applyModelDefaults(cfg);

    expect(next.models?.metadata?.["anthropic/claude-opus-4-6"]?.alias).toBe("opus");
    expect(next.models?.metadata?.["openai/gpt-5.4"]?.alias).toBe("gpt");
  });

  it("does not override existing aliases", () => {
    const cfg = {
      models: {
        metadata: {
          "anthropic/claude-opus-4-5": { alias: "Opus" },
        },
      },
    } satisfies MarvConfig;

    const next = applyModelDefaults(cfg);

    expect(next.models?.metadata?.["anthropic/claude-opus-4-5"]?.alias).toBe("Opus");
  });

  it("respects explicit empty alias disables", () => {
    const cfg = {
      models: {
        metadata: {
          "google/gemini-3.1-pro-preview": { alias: "" },
          "google/gemini-3-flash-preview": {},
        },
      },
    } satisfies MarvConfig;

    const next = applyModelDefaults(cfg);

    expect(next.models?.metadata?.["google/gemini-3.1-pro-preview"]?.alias).toBe("");
    expect(next.models?.metadata?.["google/gemini-3-flash-preview"]?.alias).toBe("gemini-flash");
  });

  it("fills missing model provider defaults", () => {
    const cfg = {
      models: {
        providers: {
          myproxy: {
            baseUrl: "https://proxy.example/v1",
            apiKey: "sk-test",
            api: "openai-completions",
            models: [
              {
                id: "gpt-5.2",
                name: "GPT-5.2",
                reasoning: false,
                input: ["text"],
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                contextWindow: 200_000,
                maxTokens: 8192,
              },
            ],
          },
        },
      },
    } satisfies MarvConfig;

    const next = applyModelDefaults(cfg);
    const model = next.models?.providers?.myproxy?.models?.[0];

    expect(model?.reasoning).toBe(false);
    expect(model?.input).toEqual(["text"]);
    expect(model?.cost).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
    expect(model?.contextWindow).toBe(DEFAULT_CONTEXT_TOKENS);
    expect(model?.maxTokens).toBe(8192);
  });

  it("clamps maxTokens to contextWindow", () => {
    const cfg = {
      models: {
        providers: {
          myproxy: {
            baseUrl: "https://proxy.example/v1",
            apiKey: "sk-test",
            api: "openai-completions",
            models: [
              {
                id: "gpt-5.2",
                name: "GPT-5.2",
                reasoning: false,
                input: ["text"],
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                contextWindow: 32768,
                maxTokens: 40960,
              },
            ],
          },
        },
      },
    } satisfies MarvConfig;

    const next = applyModelDefaults(cfg);
    const model = next.models?.providers?.myproxy?.models?.[0];

    expect(model?.contextWindow).toBe(32768);
    expect(model?.maxTokens).toBe(32768);
  });
});
