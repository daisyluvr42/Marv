import { beforeEach, describe, expect, it, vi } from "vitest";

const storage = new Map<string, string>();
const localStorage = {
  getItem: (key: string) => storage.get(key) ?? null,
  setItem: (key: string, value: string) => {
    storage.set(key, value);
  },
  removeItem: (key: string) => {
    storage.delete(key);
  },
  clear: () => {
    storage.clear();
  },
};

Object.defineProperty(globalThis, "localStorage", {
  value: localStorage,
  configurable: true,
});
Object.defineProperty(globalThis, "window", {
  value: {
    setTimeout,
    clearTimeout,
    localStorage,
  },
  configurable: true,
});
Object.defineProperty(globalThis, "location", {
  value: { protocol: "http:", host: "127.0.0.1:18789" },
  configurable: true,
});

const { connectGateway } = await import("./app-gateway.js");

type GatewayClientMock = {
  start: ReturnType<typeof vi.fn>;
  stop: ReturnType<typeof vi.fn>;
  request: ReturnType<typeof vi.fn>;
  emitHello: (hello: unknown) => void;
  emitClose: (code: number, reason?: string) => void;
  emitGap: (expected: number, received: number) => void;
  emitEvent: (evt: { event: string; payload?: unknown; seq?: number }) => void;
};

const gatewayClientInstances: GatewayClientMock[] = [];

vi.mock("./controllers/agents.js", () => ({
  loadAgents: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("./controllers/assistant-identity.js", () => ({
  loadAssistantIdentity: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("./controllers/devices.js", () => ({
  loadDevices: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("./controllers/nodes.js", () => ({
  loadNodes: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("./gateway.js", () => {
  class GatewayBrowserClient {
    readonly start = vi.fn();
    readonly stop = vi.fn();
    readonly request = vi.fn().mockResolvedValue({});

    constructor(
      private opts: {
        onHello?: (hello: unknown) => void;
        onClose?: (info: { code: number; reason: string }) => void;
        onGap?: (info: { expected: number; received: number }) => void;
        onEvent?: (evt: { event: string; payload?: unknown; seq?: number }) => void;
      },
    ) {
      gatewayClientInstances.push({
        start: this.start,
        stop: this.stop,
        request: this.request,
        emitHello: (hello) => {
          this.opts.onHello?.(hello);
        },
        emitClose: (code, reason) => {
          this.opts.onClose?.({ code, reason: reason ?? "" });
        },
        emitGap: (expected, received) => {
          this.opts.onGap?.({ expected, received });
        },
        emitEvent: (evt) => {
          this.opts.onEvent?.(evt);
        },
      });
    }
  }

  return { GatewayBrowserClient };
});

function createHost() {
  return {
    settings: {
      gatewayUrl: "ws://127.0.0.1:18789",
      token: "",
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
    password: "",
    client: null,
    connected: false,
    hello: null,
    lastError: null,
    eventLogBuffer: [],
    eventLog: [],
    tab: "overview",
    operationsSection: "sessions",
    presenceEntries: [],
    presenceError: null,
    presenceStatus: null,
    agentsLoading: false,
    agentsList: null,
    agentsError: null,
    debugHealth: null,
    assistantName: "Marv",
    assistantAvatar: null,
    assistantAgentId: null,
    sessionKey: "main",
    chatRunId: null,
    chatStream: null,
    chatStreamStartedAt: null,
    chatToolMessages: [],
    toolStreamById: new Map(),
    toolStreamOrder: [],
    toolStreamSyncTimer: null,
    compactionStatus: null,
    compactionClearTimer: null,
    refreshSessionsAfterChat: new Set<string>(),
    execApprovalQueue: [],
    execApprovalError: null,
  } as unknown as Parameters<typeof connectGateway>[0];
}

describe("connectGateway", () => {
  beforeEach(() => {
    gatewayClientInstances.length = 0;
    localStorage.clear();
  });

  it("ignores stale client onGap callbacks after reconnect", () => {
    const host = createHost();

    connectGateway(host);
    const firstClient = gatewayClientInstances[0];
    expect(firstClient).toBeDefined();

    connectGateway(host);
    const secondClient = gatewayClientInstances[1];
    expect(secondClient).toBeDefined();

    firstClient.emitGap(10, 13);
    expect(host.lastError).toBeNull();

    secondClient.emitGap(20, 24);
    expect(host.lastError).toBe(
      "event gap detected (expected seq 20, got 24); refresh recommended",
    );
  });

  it("ignores stale client onEvent callbacks after reconnect", () => {
    const host = createHost();

    connectGateway(host);
    const firstClient = gatewayClientInstances[0];
    expect(firstClient).toBeDefined();

    connectGateway(host);
    const secondClient = gatewayClientInstances[1];
    expect(secondClient).toBeDefined();

    firstClient.emitEvent({ event: "presence", payload: { presence: [{ host: "stale" }] } });
    expect(host.eventLogBuffer).toHaveLength(0);

    secondClient.emitEvent({ event: "presence", payload: { presence: [{ host: "active" }] } });
    expect(host.eventLogBuffer).toHaveLength(1);
    expect(host.eventLogBuffer[0]?.event).toBe("presence");
  });

  it("ignores stale client onClose callbacks after reconnect", () => {
    const host = createHost();

    connectGateway(host);
    const firstClient = gatewayClientInstances[0];
    expect(firstClient).toBeDefined();

    connectGateway(host);
    const secondClient = gatewayClientInstances[1];
    expect(secondClient).toBeDefined();

    firstClient.emitClose(1005);
    expect(host.lastError).toBeNull();

    secondClient.emitClose(1005);
    expect(host.lastError).toBe("disconnected (1005): no reason");
  });

  it("clears the runtime shared token once device auth is established", () => {
    const host = createHost();
    host.settings.token = "bootstrap-token";

    connectGateway(host);
    const client = gatewayClientInstances[0];
    expect(client).toBeDefined();

    client.emitHello({
      type: "hello-ok",
      protocol: 3,
      auth: {
        deviceToken: "device-token",
        role: "operator",
        scopes: ["operator.admin"],
      },
    });

    expect(host.settings.token).toBe("");
  });
});
