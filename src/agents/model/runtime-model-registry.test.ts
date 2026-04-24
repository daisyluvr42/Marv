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
import {
  __resetRuntimeModelRegistryRefreshLoopForTest,
  refreshRuntimeModelRegistry,
  startRuntimeModelRegistryRefreshLoop,
} from "./runtime-model-registry.js";

describe("runtime-model-registry", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    stateDir = fs.mkdtempSync(path.join(os.tmpdir(), "marv-runtime-model-registry-"));
    fetchWithPrivateNetworkAccessMock.mockReset();
    loadModelCatalogMock.mockReset();
    loadModelCatalogMock.mockResolvedValue([]);
  });

  afterEach(() => {
    __resetRuntimeModelRegistryRefreshLoopForTest();
    vi.useRealTimers();
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

  it("uses the latest config after the refresh loop is reconfigured", async () => {
    vi.setSystemTime(new Date("2026-04-12T00:00:00Z"));
    fetchWithPrivateNetworkAccessMock.mockResolvedValue({
      response: new Response(JSON.stringify({ data: [{ id: "foo-local-model" }] }), {
        status: 200,
      }),
      release: async () => {},
    });

    const firstConfig = {
      models: {
        providers: {
          vllm: {
            baseUrl: "http://127.0.0.1:8000/v1",
            models: [],
          },
        },
      },
    } as MarvConfig;
    const secondConfig = {
      models: {
        providers: {
          vllm: {
            baseUrl: "http://127.0.0.1:9000/v1",
            models: [],
          },
        },
      },
    } as MarvConfig;

    startRuntimeModelRegistryRefreshLoop({ cfg: firstConfig });
    await vi.waitFor(() => expect(fetchWithPrivateNetworkAccessMock).toHaveBeenCalledTimes(1));

    startRuntimeModelRegistryRefreshLoop({ cfg: secondConfig });
    await vi.advanceTimersByTimeAsync(8 * 24 * 60 * 60 * 1000);

    const lastCall = fetchWithPrivateNetworkAccessMock.mock.calls.at(-1)?.[0] as
      | { url?: string }
      | undefined;
    expect(lastCall?.url).toBe("http://127.0.0.1:9000/v1/models");
  });
});
