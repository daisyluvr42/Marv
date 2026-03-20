import { describe, expect, it } from "vitest";
import type { MarvConfig } from "../../core/config/config.js";
import { getBaselineModels } from "./model-baselines.js";
import { loadModelCatalog } from "./model-catalog.js";
import {
  installModelCatalogTestHooks,
  mockCatalogImportFailThenRecover,
} from "./model-catalog.test-harness.js";

describe("loadModelCatalog", () => {
  installModelCatalogTestHooks();

  it("retries after loader failure without poisoning the cache", async () => {
    const getCallCount = mockCatalogImportFailThenRecover();

    const cfg = {} as MarvConfig;
    // First call: models.json loader throws, but baselines are still returned.
    const first = await loadModelCatalog({ config: cfg });
    expect(first.length).toBeGreaterThan(0);
    expect(first).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ provider: "anthropic", id: "claude-opus-4-6" }),
      ]),
    );
    expect(getCallCount()).toBe(1);

    // Second call (with useCache: false): loader recovers, adds models.json models.
    const second = await loadModelCatalog({ config: cfg, useCache: false });
    expect(second).toEqual(
      expect.arrayContaining([expect.objectContaining({ id: "gpt-4.1", provider: "openai" })]),
    );
    expect(getCallCount()).toBe(2);
  });

  it("includes baseline models even without models.json", async () => {
    const cfg = {} as MarvConfig;
    const result = await loadModelCatalog({ config: cfg });
    const baselines = getBaselineModels();

    // All baseline models should be present.
    for (const baseline of baselines) {
      expect(result).toContainEqual(
        expect.objectContaining({ provider: baseline.provider, id: baseline.id }),
      );
    }
  });

  it("includes openai-codex baseline models", async () => {
    const cfg = {} as MarvConfig;
    const result = await loadModelCatalog({ config: cfg });

    expect(result).toContainEqual(
      expect.objectContaining({
        provider: "openai-codex",
        id: "gpt-5.3-codex-spark",
      }),
    );
    expect(result).toContainEqual(
      expect.objectContaining({
        provider: "openai-codex",
        id: "gpt-5.4-codex",
      }),
    );
  });
});
