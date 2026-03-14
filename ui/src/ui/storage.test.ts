import { beforeEach, describe, expect, it, vi } from "vitest";

type UiSettings = {
  gatewayUrl: string;
  token: string;
  sessionKey: string;
  lastActiveSessionKey: string;
  theme: "light" | "dark" | "system";
  chatFocusMode: boolean;
  chatShowThinking: boolean;
  splitRatio: number;
  navCollapsed: boolean;
  operationsSection: "sessions" | "instances" | "usage" | "cron" | "logs" | "debug";
  agentsSection: "agents" | "skills" | "nodes";
  workspaceSection: "projects" | "memory" | "documents" | "calendar";
  settingsSection: "config";
  locale?: string;
};

function installStorageMock() {
  const store = new Map<string, string>();
  const localStorage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value);
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
    clear: () => {
      store.clear();
    },
  };
  vi.stubGlobal("localStorage", localStorage);
  vi.stubGlobal("window", { localStorage });
  vi.stubGlobal("location", { protocol: "http:", host: "127.0.0.1:18789" });
}

function createSettings(overrides: Partial<UiSettings> = {}): UiSettings {
  return {
    gatewayUrl: "ws://127.0.0.1:18789",
    token: "bootstrap-token",
    sessionKey: "main",
    lastActiveSessionKey: "main",
    theme: "system",
    chatFocusMode: false,
    chatShowThinking: true,
    splitRatio: 0.6,
    navCollapsed: false,
    operationsSection: "sessions",
    agentsSection: "agents",
    workspaceSection: "projects",
    settingsSection: "config",
    ...overrides,
  };
}

describe("storage", () => {
  beforeEach(() => {
    vi.resetModules();
    installStorageMock();
  });

  it("does not persist shared gateway tokens to localStorage", async () => {
    const { saveSettings } = await import("./storage.js");
    saveSettings(createSettings());

    const raw = localStorage.getItem("marv.control.settings.v2");
    expect(raw).not.toBeNull();
    expect(raw).not.toContain("bootstrap-token");
  });

  it("ignores legacy stored shared tokens when loading settings", async () => {
    const { loadSettings } = await import("./storage.js");
    localStorage.setItem(
      "marv.control.settings.v2",
      JSON.stringify({
        gatewayUrl: "ws://127.0.0.1:18789",
        token: "legacy-token",
        sessionKey: "main",
        lastActiveSessionKey: "main",
      }),
    );

    expect(loadSettings().token).toBe("");
  });
});
