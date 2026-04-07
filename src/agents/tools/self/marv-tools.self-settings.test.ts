import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

const loadSessionStoreMock = vi.fn();
const updateSessionStoreMock = vi.fn();
const writeConfigFileMock = vi.fn(async (_cfg: unknown, _options?: unknown) => {});
const refreshRuntimeModelRegistryMock = vi.fn(async (_params?: unknown) => ({
  models: [{ ref: "google/gemini-2.5-flash" }],
  lastSuccessfulRefreshAt: 123,
}));

vi.mock("../../../core/config/sessions.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../core/config/sessions.js")>();
  return {
    ...actual,
    loadSessionStore: (storePath: string) => loadSessionStoreMock(storePath),
    updateSessionStore: async (
      storePath: string,
      mutator: (store: Record<string, unknown>) => Promise<unknown> | void,
    ) => {
      const store = loadSessionStoreMock(storePath) as Record<string, unknown>;
      const result = await mutator(store);
      updateSessionStoreMock(storePath, store);
      return result;
    },
    resolveStorePath: () => "/tmp/main/sessions.json",
  };
});

vi.mock("../../../core/config/config.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../core/config/config.js")>();
  return {
    ...actual,
    loadConfig: () => ({
      session: { mainKey: "main", scope: "per-sender" },
      tools: {
        elevated: {
          enabled: true,
          allowFrom: {
            whatsapp: ["+1000"],
          },
        },
      },
      agents: {
        defaults: {
          model: { primary: "anthropic/claude-opus-4-5" },
          models: {},
        },
      },
      memory: {
        soul: {
          deepConsolidation: {
            enabled: false,
            schedule: "20 4 * * 0",
          },
        },
      },
    }),
    writeConfigFile: (cfg: unknown, options?: unknown) => writeConfigFileMock(cfg, options),
  };
});

vi.mock("../../model/model-catalog.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../model/model-catalog.js")>();
  return {
    ...actual,
    loadModelCatalog: async () => [
      {
        provider: "anthropic",
        id: "claude-opus-4-5",
        name: "Opus",
        contextWindow: 200000,
      },
      {
        provider: "anthropic",
        id: "claude-sonnet-4-5",
        name: "Sonnet",
        contextWindow: 200000,
      },
    ],
  };
});

vi.mock("../../auth-profiles.js", () => ({
  ensureAuthProfileStore: () => ({
    profiles: {
      "anthropic:work": {
        provider: "anthropic",
      },
    },
  }),
}));

vi.mock("../../../core/gateway/server-methods/sessions.js", () => ({
  sessionsHandlers: {
    "sessions.reset": async ({ respond }: { respond: (ok: boolean, payload?: unknown) => void }) =>
      respond(true, {
        ok: true,
        key: "main",
        entry: { sessionId: "reset-session", updatedAt: 100 },
      }),
  },
}));

vi.mock("../../model/runtime-model-registry.js", () => ({
  refreshRuntimeModelRegistry: (params?: unknown) => refreshRuntimeModelRegistryMock(params),
  readRuntimeModelRegistry: () => null,
  listConfiguredProviders: () => new Set<string>(),
  resolveRuntimeRegistryPathForDisplay: () => "/tmp/main/runtime/model-registry.json",
}));

import "../../test-helpers/fast-core-tools.js";
import {
  createSelfSettingsTool,
  createMemorySettingsTool,
  createHeartbeatSettingsTool,
} from "./self-settings-tool.js";

function getTextContent(result?: { content?: Array<{ type: string; text?: string }> }) {
  const textBlock = result?.content?.find((block) => block.type === "text");
  return textBlock?.text ?? "";
}

function resetSessionStore(store: Record<string, unknown>) {
  loadSessionStoreMock.mockReset();
  updateSessionStoreMock.mockReset();
  writeConfigFileMock.mockReset();
  refreshRuntimeModelRegistryMock.mockClear();
  loadSessionStoreMock.mockReturnValue(store);
}

const tempDirs: string[] = [];

afterEach(async () => {
  await Promise.all(
    tempDirs.splice(0).map(async (dir) => {
      await fs.rm(dir, { recursive: true, force: true }).catch(() => undefined);
    }),
  );
});

describe("self_settings tool", () => {
  it("refreshes the runtime model registry without mutating the session", async () => {
    resetSessionStore({
      main: {
        sessionId: "s1",
        updatedAt: 10,
      },
    });

    const tool = createSelfSettingsTool({ agentSessionKey: "main" });
    const result = await tool.execute("call0", { modelRegistryAction: "refresh" });
    const text = getTextContent(result);
    const details = result.details as {
      ok?: boolean;
      settings?: {
        modelRegistryPath?: string;
        modelRegistryRefreshedAt?: number;
      };
    };

    expect(details.ok).toBe(true);
    expect(text).toContain("model registry refreshed (1 models)");
    expect(refreshRuntimeModelRegistryMock).toHaveBeenCalledTimes(1);
    expect(updateSessionStoreMock).not.toHaveBeenCalled();
    expect(details.settings?.modelRegistryPath).toBe("/tmp/main/runtime/model-registry.json");
    expect(details.settings?.modelRegistryRefreshedAt).toBe(123);
  });

  it("applies current-session model, auth profile, and queue settings", async () => {
    resetSessionStore({
      main: {
        sessionId: "s1",
        updatedAt: 10,
      },
    });

    const tool = createSelfSettingsTool({ agentSessionKey: "main" });
    const result = await tool.execute("call1", {
      model: "anthropic/claude-sonnet-4-5",
      authProfile: "anthropic:work",
      queueMode: "collect",
      queueCap: 5,
    });
    const details = result.details as {
      ok?: boolean;
      settings?: {
        model?: string;
        authProfileOverride?: string;
        queueMode?: string;
        queueCap?: number;
      };
    };
    expect(details.ok).toBe(true);
    expect(details.settings?.model).toBe("claude-sonnet-4-5");
    expect(details.settings?.authProfileOverride).toBe("anthropic:work");
    expect(details.settings?.queueMode).toBe("collect");
    expect(details.settings?.queueCap).toBe(5);

    const [, savedStore] = updateSessionStoreMock.mock.calls.at(-1) as [
      string,
      Record<string, unknown>,
    ];
    const saved = Object.values(savedStore).at(0) as Record<string, unknown> | undefined;
    expect(saved).toBeDefined();
    if (!saved) {
      return;
    }
    expect(saved.providerOverride).toBe("anthropic");
    expect(saved.modelOverride).toBe("claude-sonnet-4-5");
    expect(saved.authProfileOverride).toBe("anthropic:work");
    expect(saved.queueMode).toBe("collect");
    expect(saved.queueCap).toBe(5);
  });

  it("still allows session-level changes for non-direct requests", async () => {
    resetSessionStore({
      main: {
        sessionId: "s1",
        updatedAt: 10,
      },
    });

    const tool = createSelfSettingsTool({
      agentSessionKey: "main",
      directUserInstruction: false,
    });
    const result = await tool.execute("call2", { model: "anthropic/claude-sonnet-4-5" });
    const details = result.details as { ok?: boolean };
    expect(details.ok).toBe(true);
    expect(updateSessionStoreMock).toHaveBeenCalled();
  });

  it("returns a generic denial for non-direct system-level requests", async () => {
    resetSessionStore({
      main: {
        sessionId: "s1",
        updatedAt: 10,
      },
    });

    const tool = createHeartbeatSettingsTool({
      agentSessionKey: "main",
      directUserInstruction: false,
    });
    const result = await tool.execute("call2b", { heartbeatEvery: "30m" });
    const details = result.details as { ok?: boolean; denied?: boolean };
    expect(details.ok).toBe(false);
    expect(details.denied).toBe(true);
    expect(updateSessionStoreMock).not.toHaveBeenCalled();
    expect(writeConfigFileMock).not.toHaveBeenCalled();
  });

  it("returns a generic denial when elevated changes are not allowed", async () => {
    resetSessionStore({
      main: {
        sessionId: "s1",
        updatedAt: 10,
      },
    });

    const tool = createSelfSettingsTool({
      agentSessionKey: "main",
      agentChannel: "whatsapp",
      senderE164: "+1999",
    });
    const result = await tool.execute("call3", { elevatedLevel: "ask" });
    const details = result.details as { ok?: boolean; denied?: boolean };
    expect(details.ok).toBe(false);
    expect(details.denied).toBe(true);
    expect(updateSessionStoreMock).not.toHaveBeenCalled();
  });

  it("updates shared deep-memory settings through a restricted config write", async () => {
    resetSessionStore({
      main: {
        sessionId: "s1",
        updatedAt: 10,
      },
    });

    const tool = createMemorySettingsTool({ agentSessionKey: "main" });
    const result = await tool.execute("call4", {
      deepMemoryEnabled: true,
      deepMemorySchedule: "15 5 * * 0",
      deepMemoryModelProvider: "ollama",
      deepMemoryModelApi: "ollama",
      deepMemoryModel: "qwen2.5:3b",
      deepMemoryMaxItems: 250,
    });
    const text = getTextContent(result);
    const details = result.details as {
      ok?: boolean;
      sharedConfig?: {
        deepMemoryEnabled?: boolean;
        deepMemorySchedule?: string;
        deepMemoryModelProvider?: string;
        deepMemoryModelApi?: string;
        deepMemoryModel?: string;
        deepMemoryMaxItems?: number;
      };
    };

    expect(details.ok).toBe(true);
    expect(text).toContain("Updated shared deep-memory settings");
    expect(writeConfigFileMock).toHaveBeenCalledTimes(1);
    expect(details.sharedConfig).toEqual(
      expect.objectContaining({
        deepMemoryEnabled: true,
        deepMemorySchedule: "15 5 * * 0",
        deepMemoryModelProvider: "ollama",
        deepMemoryModelApi: "ollama",
        deepMemoryModel: "qwen2.5:3b",
        deepMemoryMaxItems: 250,
      }),
    );
  });

  it("rejects invalid deep-memory schedules", async () => {
    resetSessionStore({
      main: {
        sessionId: "s1",
        updatedAt: 10,
      },
    });

    const tool = createMemorySettingsTool({ agentSessionKey: "main" });
    const result = await tool.execute("call5", {
      deepMemorySchedule: "not a cron",
    });
    const details = result.details as { ok?: boolean; invalid?: boolean };

    expect(details.ok).toBe(false);
    expect(details.invalid).toBe(true);
    expect(writeConfigFileMock).not.toHaveBeenCalled();
  });

  it("updates shared memory-search settings through a restricted config write", async () => {
    resetSessionStore({
      main: {
        sessionId: "s1",
        updatedAt: 10,
      },
    });

    const tool = createMemorySettingsTool({
      agentSessionKey: "main",
      config: {
        session: { mainKey: "main", scope: "per-sender" },
        agents: {
          defaults: {
            memorySearch: {
              provider: "openai",
              model: "text-embedding-3-small",
            },
          },
        },
      } as never,
    });
    const result = await tool.execute("call6", {
      memorySearchProvider: "openai",
      memorySearchModel: "Qwen3-Embedding-0.6B",
      memorySearchDimensions: 512,
      memorySearchRemoteBaseUrl: "http://localhost:8080/v1",
      memorySearchRemoteApiKey: "local-key",
      memorySearchRerankerEnabled: true,
      memorySearchRerankerApiUrl: "http://localhost:8081/v1/rerank",
      memorySearchRerankerModel: "Qwen3-Reranker-0.6B",
      memorySearchRerankerApiKey: "rerank-key",
      memorySearchRerankerMaxCandidates: 24,
    });
    const text = getTextContent(result);
    const details = result.details as {
      ok?: boolean;
      sharedConfig?: {
        memorySearchProvider?: string;
        memorySearchModel?: string;
        memorySearchDimensions?: number;
        memorySearchRemoteBaseUrl?: string;
        memorySearchRemoteApiKey?: string;
        memorySearchRerankerEnabled?: boolean;
        memorySearchRerankerApiUrl?: string;
        memorySearchRerankerModel?: string;
        memorySearchRerankerApiKey?: string;
        memorySearchRerankerMaxCandidates?: number;
      };
    };

    expect(details.ok).toBe(true);
    expect(text).toContain("Updated shared memory-search settings");
    expect(text).not.toContain("local-key");
    expect(text).not.toContain("rerank-key");
    expect(writeConfigFileMock).toHaveBeenCalledTimes(1);
    expect(details.sharedConfig).toEqual(
      expect.objectContaining({
        memorySearchProvider: "openai",
        memorySearchModel: "Qwen3-Embedding-0.6B",
        memorySearchDimensions: 512,
        memorySearchRemoteBaseUrl: "http://localhost:8080/v1",
        memorySearchRemoteApiKey: "[redacted]",
        memorySearchRerankerEnabled: true,
        memorySearchRerankerApiUrl: "http://localhost:8081/v1/rerank",
        memorySearchRerankerModel: "Qwen3-Reranker-0.6B",
        memorySearchRerankerApiKey: "[redacted]",
        memorySearchRerankerMaxCandidates: 24,
      }),
    );
  });

  it("rejects incomplete memory-search reranker requests", async () => {
    resetSessionStore({
      main: {
        sessionId: "s1",
        updatedAt: 10,
      },
    });

    const tool = createMemorySettingsTool({
      agentSessionKey: "main",
      config: {
        session: { mainKey: "main", scope: "per-sender" },
        agents: {
          defaults: {
            memorySearch: {
              provider: "openai",
              model: "text-embedding-3-small",
            },
          },
        },
      } as never,
    });
    const result = await tool.execute("call7", {
      memorySearchRerankerEnabled: true,
    });
    const details = result.details as { ok?: boolean; invalid?: boolean };

    expect(details.ok).toBe(false);
    expect(details.invalid).toBe(true);
    expect(writeConfigFileMock).not.toHaveBeenCalled();
  });

  it("updates shared external CLI fallback settings through config", async () => {
    resetSessionStore({
      main: {
        sessionId: "s1",
        updatedAt: 10,
      },
    });

    const tool = createMemorySettingsTool({ agentSessionKey: "main" });
    const result = await tool.execute("call8", {
      externalCliEnabled: true,
      externalCliAvailableBrands: "codex, claude",
      externalCliDefault: "codex",
    });
    const text = getTextContent(result);
    const details = result.details as {
      ok?: boolean;
      sharedConfig?: {
        externalCliEnabled?: boolean;
        externalCliDefault?: string;
        externalCliAvailable?: string[];
      };
    };

    expect(details.ok).toBe(true);
    expect(text).toContain("Updated shared external-CLI settings");
    expect(writeConfigFileMock).toHaveBeenCalledTimes(1);
    expect(details.sharedConfig).toEqual(
      expect.objectContaining({
        externalCliEnabled: true,
        externalCliDefault: "codex",
        externalCliAvailable: ["codex", "claude"],
      }),
    );
  });

  it("updates shared heartbeat settings through config", async () => {
    resetSessionStore({
      main: {
        sessionId: "s1",
        updatedAt: 10,
      },
    });

    const tool = createHeartbeatSettingsTool({ agentSessionKey: "main" });
    const result = await tool.execute("call9", {
      heartbeatEvery: "45m",
      heartbeatModel: "ollama/qwen2.5:3b",
      heartbeatTarget: "telegram",
      heartbeatActiveHoursStart: "09:00",
      heartbeatActiveHoursEnd: "22:00",
      heartbeatIncludeReasoning: true,
    });
    const text = getTextContent(result);
    const details = result.details as {
      ok?: boolean;
      sharedConfig?: {
        heartbeatEvery?: string;
        heartbeatModel?: string;
        heartbeatTarget?: string;
        heartbeatActiveHoursStart?: string;
        heartbeatActiveHoursEnd?: string;
        heartbeatIncludeReasoning?: boolean;
      };
    };

    expect(details.ok).toBe(true);
    expect(text).toContain("Updated shared heartbeat settings");
    expect(writeConfigFileMock).toHaveBeenCalledTimes(1);
    expect(details.sharedConfig).toEqual(
      expect.objectContaining({
        heartbeatEvery: "45m",
        heartbeatModel: "ollama/qwen2.5:3b",
        heartbeatTarget: "telegram",
        heartbeatActiveHoursStart: "09:00",
        heartbeatActiveHoursEnd: "22:00",
        heartbeatIncludeReasoning: true,
      }),
    );
  });

  it("rewrites HEARTBEAT.md through the self settings tool", async () => {
    const workspaceDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-self-settings-heartbeat-"));
    tempDirs.push(workspaceDir);
    resetSessionStore({
      main: {
        sessionId: "s1",
        updatedAt: 10,
      },
    });

    const tool = createHeartbeatSettingsTool({
      agentSessionKey: "main",
      config: {
        session: { mainKey: "main", scope: "per-sender" },
        agents: {
          defaults: {
            workspace: workspaceDir,
            model: { primary: "anthropic/claude-opus-4-5" },
            models: {},
          },
        },
      } as never,
    });

    const result = await tool.execute("call10", {
      heartbeatFileAction: "replace",
      heartbeatFileContent: "# HEARTBEAT.md\n\n- Check inbox\n",
    });
    const text = getTextContent(result);
    const details = result.details as {
      ok?: boolean;
      files?: {
        heartbeat?: {
          action?: string;
          path?: string;
          size?: number;
        };
      };
    };

    expect(details.ok).toBe(true);
    expect(text).toContain("Updated HEARTBEAT.md: replace.");
    expect(writeConfigFileMock).not.toHaveBeenCalled();
    expect(details.files?.heartbeat?.action).toBe("replace");

    const saved = await fs.readFile(path.join(workspaceDir, "HEARTBEAT.md"), "utf-8");
    expect(saved).toContain("- Check inbox");
  });

  it("appends to HEARTBEAT.md through the self settings tool", async () => {
    const workspaceDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-self-settings-heartbeat-"));
    tempDirs.push(workspaceDir);
    await fs.writeFile(path.join(workspaceDir, "HEARTBEAT.md"), "# HEARTBEAT.md\n", "utf-8");
    resetSessionStore({
      main: {
        sessionId: "s1",
        updatedAt: 10,
      },
    });

    const tool = createHeartbeatSettingsTool({
      agentSessionKey: "main",
      config: {
        session: { mainKey: "main", scope: "per-sender" },
        agents: {
          defaults: {
            workspace: workspaceDir,
            model: { primary: "anthropic/claude-opus-4-5" },
            models: {},
          },
        },
      } as never,
    });

    const result = await tool.execute("call11", {
      heartbeatFileAction: "append",
      heartbeatFileContent: "- Check blockers\n",
    });
    const details = result.details as { ok?: boolean };
    expect(details.ok).toBe(true);

    const saved = await fs.readFile(path.join(workspaceDir, "HEARTBEAT.md"), "utf-8");
    expect(saved).toContain("# HEARTBEAT.md\n- Check blockers\n");
  });
});
