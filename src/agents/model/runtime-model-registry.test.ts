import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

let stateDir = "";

const fetchWithPrivateNetworkAccessMock = vi.fn();
const loadModelCatalogMock = vi.fn(async () => []);

vi.mock("../../core/config/paths.js", () => ({
  resolveStateDir: () => stateDir,
}));

vi.mock("../../infra/net/private-network-fetch.js", () => ({
  fetchWithPrivateNetworkAccess: (...args: Parameters<typeof fetchWithPrivateNetworkAccessMock>) =>
    fetchWithPrivateNetworkAccessMock(...args),
}));

vi.mock("../auth-profiles.js", () => ({
  ensureAuthProfileStore: vi.fn(() => ({ profiles: {} })),
  listProfilesForProvider: vi.fn(() => []),
}));

vi.mock("./model-auth.js", () => ({
  resolveApiKeyForProvider: vi.fn(),
  getCustomProviderApiKey: vi.fn(() => undefined),
  resolveEnvApiKey: vi.fn(() => undefined),
}));

vi.mock("./model-catalog.js", () => ({
  loadModelCatalog: (...args: Parameters<typeof loadModelCatalogMock>) =>
    loadModelCatalogMock(...args),
}));

import type { MarvConfig } from "../../core/config/config.js";
import { refreshRuntimeModelRegistry } from "./runtime-model-registry.js";

describe("runtime-model-registry", () => {
  beforeEach(() => {
    stateDir = fs.mkdtempSync(path.join(os.tmpdir(), "marv-runtime-model-registry-"));
    fetchWithPrivateNetworkAccessMock.mockReset();
    loadModelCatalogMock.mockReset();
    loadModelCatalogMock.mockResolvedValue([]);
  });

  afterEach(() => {
    if (stateDir) {
      fs.rmSync(stateDir, { recursive: true, force: true });
      stateDir = "";
    }
  });

  it("defaults local OpenAI-compatible models to a usable context window when metadata is missing", async () => {
    fetchWithPrivateNetworkAccessMock.mockResolvedValue({
      response: new Response(JSON.stringify({ data: [{ id: "foo-local-model" }] }), {
        status: 200,
      }),
      release: async () => {},
    });

    const cfg = {
      models: {
        providers: {
          vllm: {
            baseUrl: "http://127.0.0.1:8000/v1",
            models: [],
          },
        },
      },
    } as MarvConfig;

    const registry = await refreshRuntimeModelRegistry({ cfg, force: true });

    expect(fetchWithPrivateNetworkAccessMock).toHaveBeenCalledWith(
      expect.objectContaining({
        url: "http://127.0.0.1:8000/v1/models",
        auditContext: "runtime-model-registry.vllm",
      }),
    );
    expect(registry.models).toContainEqual(
      expect.objectContaining({
        ref: "vllm/foo-local-model",
        provider: "vllm",
        model: "foo-local-model",
        location: "local",
        contextWindow: 128_000,
      }),
    );
  });
});
