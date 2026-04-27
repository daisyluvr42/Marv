import { describe, expect, it } from "vitest";
import type { MarvConfig } from "../../core/config/config.js";
import { resolveOrderedModelRoutePlan } from "./model-route.js";

describe("resolveOrderedModelRoutePlan", () => {
  it("uses configured primary and fallbacks in exact order", () => {
    const cfg = {
      agents: {
        defaults: {
          model: {
            primary: "lmstudio/qwen-local",
            fallbacks: ["google/gemini-3.1-pro-preview", "anthropic/claude-sonnet-4-5"],
          },
        },
      },
    } as MarvConfig;

    const route = resolveOrderedModelRoutePlan({ cfg });

    expect(route.hasConfiguredRoute).toBe(true);
    expect(route.entries.map((entry) => entry.ref)).toEqual([
      "lmstudio/qwen-local",
      "google/gemini-3.1-pro-preview",
      "anthropic/claude-sonnet-4-5",
    ]);
  });

  it("puts a user-assigned current model before the configured route", () => {
    const cfg = {
      agents: {
        defaults: {
          model: {
            primary: "google/gemini-3.1-pro-preview",
            fallbacks: ["anthropic/claude-sonnet-4-5"],
          },
        },
      },
    } as MarvConfig;

    const route = resolveOrderedModelRoutePlan({
      cfg,
      primary: { provider: "lmstudio", model: "qwen-local" },
    });

    expect(route.entries.map((entry) => entry.ref)).toEqual([
      "lmstudio/qwen-local",
      "google/gemini-3.1-pro-preview",
      "anthropic/claude-sonnet-4-5",
    ]);
  });

  it("does not synthesize a configured route when no model list exists", () => {
    const route = resolveOrderedModelRoutePlan({
      cfg: {} as MarvConfig,
      primary: { provider: "openai", model: "gpt-4.1" },
    });

    expect(route.hasConfiguredRoute).toBe(false);
    expect(route.entries.map((entry) => entry.ref)).toEqual(["openai/gpt-4.1"]);
  });
});
