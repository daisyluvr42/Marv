import { describe, expect, it } from "vitest";
import { resolveModelsAuthChoice, resolveModelsAuthChoiceOptions } from "./auth-choice.js";

describe("resolveModelsAuthChoiceOptions", () => {
  it("returns all Google auth methods for provider group", () => {
    expect(resolveModelsAuthChoiceOptions("google").map((option) => option.value)).toEqual([
      "gemini-api-key",
      "google-antigravity",
      "google-gemini-cli",
    ]);
  });

  it("resolves direct auth choice values", () => {
    expect(
      resolveModelsAuthChoiceOptions("google-gemini-cli").map((option) => option.value),
    ).toEqual(["google-gemini-cli"]);
  });
});

describe("resolveModelsAuthChoice", () => {
  it("returns null when provider has multiple methods and no method was specified", () => {
    const result = resolveModelsAuthChoice({ provider: "google" });
    expect(result.choice).toBeNull();
    expect(result.options.map((option) => option.value)).toEqual([
      "gemini-api-key",
      "google-antigravity",
      "google-gemini-cli",
    ]);
  });

  it("resolves a method inside a provider group", () => {
    const result = resolveModelsAuthChoice({
      provider: "google",
      method: "gemini-api-key",
    });
    expect(result.choice).toBe("gemini-api-key");
  });

  it("accepts legacy auth choice aliases as methods", () => {
    const result = resolveModelsAuthChoice({
      provider: "anthropic",
      method: "setup-token",
    });
    expect(result.choice).toBe("token");
  });

  it("resolves a provider with a single matching method", () => {
    const result = resolveModelsAuthChoice({ provider: "xai" });
    expect(result.choice).toBe("xai-api-key");
  });
});
