import { describe, expect, it, vi } from "vitest";

const loadSessionStoreMock = vi.fn();
const updateSessionStoreMock = vi.fn();

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
    }),
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

import "../test-helpers/fast-core-tools.js";
import { createSelfSettingsTool } from "./self-settings-tool.js";

function resetSessionStore(store: Record<string, unknown>) {
  loadSessionStoreMock.mockReset();
  updateSessionStoreMock.mockReset();
  loadSessionStoreMock.mockReturnValue(store);
}

describe("self_settings tool", () => {
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
});
