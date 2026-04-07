import { describe, expect, it, vi } from "vitest";
import type { MarvConfig } from "../core/config/config.js";
import { syncProviderSelectionsAfterAuth } from "./auth-choice.default-model.js";

vi.mock("../agents/model/model-availability-state.js", () => ({
  clearProviderFailureStates: vi.fn(),
}));

vi.mock("../agents/model/model-catalog.js", () => ({
  loadModelCatalog: vi.fn(async () => [
    { provider: "google", id: "gemini-2.5-pro" },
    { provider: "google", id: "gemini-2.5-flash" },
    { provider: "openai", id: "gpt-4o-mini" },
  ]),
}));

describe("syncProviderSelectionsAfterAuth", () => {
  it("replaces provider-family selections while preserving unrelated providers", async () => {
    const cfg = {
      auth: {
        profiles: {
          "google:default": { provider: "google", mode: "api_key" },
          "google:work": { provider: "google", mode: "api_key" },
          "openai:default": { provider: "openai", mode: "api_key" },
        },
      },
      models: {
        selections: {
          "google:default": ["google/gemini-2.0-flash"],
          "google:work": ["google/gemini-1.5-pro"],
          "openai:default": ["openai/gpt-4o-mini"],
        },
      },
    } as MarvConfig;

    const next = await syncProviderSelectionsAfterAuth(cfg, "google/gemini-2.5-pro");

    expect(next.models?.selections).toEqual({
      google: ["google/gemini-2.5-pro", "google/gemini-2.5-flash"],
      "openai:default": ["openai/gpt-4o-mini"],
    });
  });
});
