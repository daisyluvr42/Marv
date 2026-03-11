import { describe, expect, it, vi } from "vitest";

const loadSessionStoreMock = vi.fn();
const updateSessionStoreMock = vi.fn();
const writeConfigFileMock = vi.fn(async (_cfg: unknown, _options?: unknown) => {});
const refreshRuntimeModelRegistryMock = vi.fn(async (_params?: unknown) => ({
  models: [{ ref: "google/gemini-2.5-flash" }],
  lastSuccessfulRefreshAt: 123,
}));

vi.mock("../../core/config/sessions.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../core/config/sessions.js")>();
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

vi.mock("../../core/config/config.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../core/config/config.js")>();
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

vi.mock("../model/model-catalog.js", () => ({
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
}));

vi.mock("../auth-profiles.js", () => ({
  ensureAuthProfileStore: () => ({
    profiles: {
      "anthropic:work": {
        provider: "anthropic",
      },
    },
  }),
}));

vi.mock("../../core/gateway/server-methods/sessions.js", () => ({
  sessionsHandlers: {
    "sessions.reset": async ({ respond }: { respond: (ok: boolean, payload?: unknown) => void }) =>
      respond(true, {
        ok: true,
        key: "main",
        entry: { sessionId: "reset-session", updatedAt: 100 },
      }),
  },
}));

vi.mock("../model/runtime-model-registry.js", () => ({
  refreshRuntimeModelRegistry: (params?: unknown) => refreshRuntimeModelRegistryMock(params),
  resolveRuntimeRegistryPathForDisplay: () => "/tmp/main/runtime/model-registry.json",
}));

import "../test-helpers/fast-core-tools.js";
import { createSelfSettingsTool } from "./self-settings-tool.js";

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

  it("returns a generic denial for non-direct requests", async () => {
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
    const details = result.details as { ok?: boolean; denied?: boolean };
    expect(details.ok).toBe(false);
    expect(details.denied).toBe(true);
    expect(updateSessionStoreMock).not.toHaveBeenCalled();
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

    const tool = createSelfSettingsTool({ agentSessionKey: "main" });
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

    const tool = createSelfSettingsTool({ agentSessionKey: "main" });
    const result = await tool.execute("call5", {
      deepMemorySchedule: "not a cron",
    });
    const details = result.details as { ok?: boolean; invalid?: boolean };

    expect(details.ok).toBe(false);
    expect(details.invalid).toBe(true);
    expect(writeConfigFileMock).not.toHaveBeenCalled();
  });
});
