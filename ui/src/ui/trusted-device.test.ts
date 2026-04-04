import { describe, expect, it, vi } from "vitest";

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
}

describe("trusted device lifecycle", () => {
  it("forgets the current device and clears stored credentials", async () => {
    vi.resetModules();
    installStorageMock();
    localStorage.setItem(
      "marv.control.settings.v2",
      JSON.stringify({
        gatewayUrl: "ws://127.0.0.1:4242",
        sessionKey: "main",
        lastActiveSessionKey: "main",
      }),
    );
    localStorage.setItem(
      "marv.device.auth.v1",
      JSON.stringify({
        version: 1,
        deviceId: "device-1",
        tokens: {
          operator: {
            token: "device-token",
            role: "operator",
            scopes: ["operator.admin"],
            updatedAtMs: 1,
          },
        },
      }),
    );
    localStorage.setItem(
      "marv-device-identity-v1",
      JSON.stringify({
        version: 1,
        deviceId: "device-1",
        publicKey: "pub",
        privateKey: "priv",
        createdAtMs: 1,
      }),
    );

    const { forgetTrustedDevice } = await import("./trusted-device.js");
    const host: import("./trusted-device.js").TrustedDeviceHost = {
      settings: {
        gatewayUrl: "ws://127.0.0.1:4242",
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
      },
      password: "secret",
      hello: { auth: { deviceToken: "device-token" } },
      lastError: "boom",
      connected: true,
      client: { stop: vi.fn() },
      applySettings(next: {
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
        workspaceSection: "projects" | "workbench" | "memory" | "documents" | "calendar";
        settingsSection: "config";
      }) {
        this.settings = next;
      },
    };

    forgetTrustedDevice(host);

    expect(localStorage.getItem("marv.device.auth.v1")).toBeNull();
    expect(localStorage.getItem("marv-device-identity-v1")).toBeNull();
    expect(host.settings.token).toBe("");
    expect(host.password).toBe("");
    expect(host.connected).toBe(false);
    expect(host.lastError).toBeNull();
    expect(host.client!.stop).toHaveBeenCalledTimes(1);
  });
});
