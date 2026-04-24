import { beforeEach, describe, expect, it, vi } from "vitest";

const setGatewaySigusr1RestartPolicyMock = vi.fn();
const resetDirectoryCacheMock = vi.fn();
const setCommandLaneConcurrencyMock = vi.fn();
const clearAllRuntimeModelAvailabilityMock = vi.fn(() => 2);
const ensureMarvModelsJsonMock = vi.fn(async () => ({ agentDir: "/tmp/agent", wrote: true }));
const refreshRuntimeModelRegistryMock = vi.fn(async () => ({}));
const startRuntimeModelRegistryRefreshLoopMock = vi.fn();
const invalidateGatewayModelCatalogCacheMock = vi.fn();

vi.mock("../../hooks/gmail-watcher-lifecycle.js", () => ({
  startGmailWatcherWithLogs: vi.fn(),
}));

vi.mock("../../hooks/gmail-watcher.js", () => ({
  stopGmailWatcher: vi.fn(),
}));

vi.mock("../../infra/env.js", () => ({
  isTruthyEnvValue: vi.fn(() => false),
}));

vi.mock("../../infra/outbound/target-resolver.js", () => ({
  resetDirectoryCache: (...args: Parameters<typeof resetDirectoryCacheMock>) =>
    resetDirectoryCacheMock(...args),
}));

vi.mock("../../infra/restart.js", () => ({
  deferGatewayRestartUntilIdle: vi.fn(),
  emitGatewayRestart: vi.fn(() => true),
  setGatewaySigusr1RestartPolicy: (
    ...args: Parameters<typeof setGatewaySigusr1RestartPolicyMock>
  ) => setGatewaySigusr1RestartPolicyMock(...args),
}));

vi.mock("../../process/command-queue.js", () => ({
  getTotalQueueSize: vi.fn(() => 0),
  setCommandLaneConcurrency: (...args: Parameters<typeof setCommandLaneConcurrencyMock>) =>
    setCommandLaneConcurrencyMock(...args),
}));

vi.mock("../config/agent-limits.js", () => ({
  resolveAgentMaxConcurrent: vi.fn(() => 1),
  resolveSubagentMaxConcurrent: vi.fn(() => 1),
}));

vi.mock("./hooks.js", () => ({
  resolveHooksConfig: vi.fn(() => ({})),
}));

vi.mock("./server-browser.js", () => ({
  startBrowserControlServerIfEnabled: vi.fn(async () => null),
}));

vi.mock("./server-cron.js", () => ({
  buildGatewayCronService: vi.fn(() => ({
    cron: {
      stop: vi.fn(),
      start: vi.fn(async () => {}),
    },
  })),
  startProactiveTaskRunnerIfEnabled: vi.fn(() => null),
}));

vi.mock("../../agents/model/model-availability-state.js", () => ({
  clearAllRuntimeModelAvailability: (
    ...args: Parameters<typeof clearAllRuntimeModelAvailabilityMock>
  ) => clearAllRuntimeModelAvailabilityMock(...args),
}));

vi.mock("../../agents/model/models-config.js", () => ({
  ensureMarvModelsJson: (...args: Parameters<typeof ensureMarvModelsJsonMock>) =>
    ensureMarvModelsJsonMock(...args),
}));

vi.mock("../../agents/model/runtime-model-registry.js", () => ({
  refreshRuntimeModelRegistry: (...args: Parameters<typeof refreshRuntimeModelRegistryMock>) =>
    refreshRuntimeModelRegistryMock(...args),
  startRuntimeModelRegistryRefreshLoop: (
    ...args: Parameters<typeof startRuntimeModelRegistryRefreshLoopMock>
  ) => startRuntimeModelRegistryRefreshLoopMock(...args),
}));

vi.mock("./server-model-catalog.js", () => ({
  invalidateGatewayModelCatalogCache: (
    ...args: Parameters<typeof invalidateGatewayModelCatalogCacheMock>
  ) => invalidateGatewayModelCatalogCacheMock(...args),
}));

import { createGatewayReloadHandlers } from "./server-reload-handlers.js";

describe("createGatewayReloadHandlers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("rebuilds provider runtime state when model provider config changes", async () => {
    const state = {
      hooksConfig: {},
      heartbeatRunner: { updateConfig: vi.fn() },
      cronState: {
        cron: {
          stop: vi.fn(),
          start: vi.fn(async () => {}),
        },
      },
      proactiveRunner: null,
      browserControl: null,
    };
    const setState = vi.fn();
    const logReload = {
      info: vi.fn(),
      warn: vi.fn(),
    };
    const handlers = createGatewayReloadHandlers({
      deps: {} as never,
      broadcast: vi.fn(),
      getState: () => state as never,
      setState,
      startChannel: vi.fn(async () => {}),
      stopChannel: vi.fn(async () => {}),
      logHooks: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
      logBrowser: { error: vi.fn() },
      logChannels: { info: vi.fn(), error: vi.fn() },
      logCron: { error: vi.fn() },
      logReload,
    });

    const nextConfig = {
      models: {
        providers: {
          "local-qwen": {
            baseUrl: "http://10.0.0.1:11434/v1",
            timeoutMs: 120_000,
            models: [{ id: "qwen3.5:122b-a10b" }],
          },
        },
      },
    };

    await handlers.applyHotReload(
      {
        changedPaths: ["models.providers.local-qwen.baseUrl"],
        restartGateway: false,
        restartChannels: new Set(),
        restartReasons: [],
        hotReasons: ["models.providers.local-qwen.baseUrl"],
        reloadHooks: false,
        reloadModelAvailability: true,
        restartGmailWatcher: false,
        restartBrowserControl: false,
        restartCron: false,
        restartHeartbeat: false,
        noopPaths: [],
      },
      nextConfig as never,
    );

    expect(invalidateGatewayModelCatalogCacheMock).toHaveBeenCalledTimes(1);
    expect(clearAllRuntimeModelAvailabilityMock).toHaveBeenCalledTimes(1);
    expect(ensureMarvModelsJsonMock).toHaveBeenCalledWith(nextConfig);
    expect(refreshRuntimeModelRegistryMock).toHaveBeenCalledWith({
      cfg: nextConfig,
      force: true,
    });
    expect(startRuntimeModelRegistryRefreshLoopMock).toHaveBeenCalledWith({
      cfg: nextConfig,
      log: expect.objectContaining({
        warn: expect.any(Function),
      }),
    });
    expect(setState).toHaveBeenCalledTimes(1);
  });
});
