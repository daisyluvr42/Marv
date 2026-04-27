import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const loadConfig = vi.fn();
const resolveGatewayPort = vi.fn();
const pickPrimaryTailnetIPv4 = vi.fn();
const pickPrimaryLanIPv4 = vi.fn();
const gatewayRequest = vi.fn();

const originalEnvToken = process.env.MARV_GATEWAY_TOKEN;
const originalEnvPassword = process.env.MARV_GATEWAY_PASSWORD;

vi.mock("../core/config/config.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../core/config/config.js")>();
  return {
    ...actual,
    loadConfig,
    resolveGatewayPort,
  };
});

vi.mock("../infra/tailnet.js", () => ({
  pickPrimaryTailnetIPv4,
}));

vi.mock("../core/gateway/net.js", () => ({
  pickPrimaryLanIPv4,
}));

vi.mock("../core/gateway/client.js", () => ({
  GatewayClient: class {
    start() {}
    stop() {}
    async request(method: string, params?: unknown) {
      return await gatewayRequest(method, params);
    }
  },
}));

const { GatewayChatClient, resolveGatewayConnection } = await import("./gateway-chat.js");

describe("GatewayChatClient", () => {
  beforeEach(() => {
    loadConfig.mockReset();
    resolveGatewayPort.mockReset();
    pickPrimaryTailnetIPv4.mockReset();
    pickPrimaryLanIPv4.mockReset();
    gatewayRequest.mockReset();
    loadConfig.mockReturnValue({ gateway: { mode: "local" } });
    resolveGatewayPort.mockReturnValue(4242);
    pickPrimaryTailnetIPv4.mockReturnValue(undefined);
    pickPrimaryLanIPv4.mockReturnValue(undefined);
    gatewayRequest.mockResolvedValue({ ok: true });
  });

  it("resolves exec approvals with the gateway schema", async () => {
    const client = new GatewayChatClient({});

    await client.resolveExecApproval({
      id: "approval-1",
      decision: "allow-once",
    });

    expect(gatewayRequest).toHaveBeenCalledWith("exec.approval.resolve", {
      id: "approval-1",
      decision: "allow-once",
    });
  });
});

describe("resolveGatewayConnection", () => {
  beforeEach(() => {
    loadConfig.mockReset();
    resolveGatewayPort.mockReset();
    pickPrimaryTailnetIPv4.mockReset();
    pickPrimaryLanIPv4.mockReset();
    resolveGatewayPort.mockReturnValue(4242);
    pickPrimaryTailnetIPv4.mockReturnValue(undefined);
    pickPrimaryLanIPv4.mockReturnValue(undefined);
    delete process.env.MARV_GATEWAY_TOKEN;
    delete process.env.MARV_GATEWAY_PASSWORD;
  });

  afterEach(() => {
    if (originalEnvToken === undefined) {
      delete process.env.MARV_GATEWAY_TOKEN;
    } else {
      process.env.MARV_GATEWAY_TOKEN = originalEnvToken;
    }

    if (originalEnvPassword === undefined) {
      delete process.env.MARV_GATEWAY_PASSWORD;
    } else {
      process.env.MARV_GATEWAY_PASSWORD = originalEnvPassword;
    }
  });

  it("throws when url override is missing explicit credentials", () => {
    loadConfig.mockReturnValue({ gateway: { mode: "local" } });

    expect(() => resolveGatewayConnection({ url: "wss://override.example/ws" })).toThrow(
      "explicit credentials",
    );
  });

  it("uses explicit token when url override is set", () => {
    loadConfig.mockReturnValue({ gateway: { mode: "local" } });

    const result = resolveGatewayConnection({
      url: "wss://override.example/ws",
      token: "explicit-token",
    });

    expect(result).toEqual({
      url: "wss://override.example/ws",
      token: "explicit-token",
      password: undefined,
    });
  });

  it("uses explicit password when url override is set", () => {
    loadConfig.mockReturnValue({ gateway: { mode: "local" } });

    const result = resolveGatewayConnection({
      url: "wss://override.example/ws",
      password: "explicit-password",
    });

    expect(result).toEqual({
      url: "wss://override.example/ws",
      token: undefined,
      password: "explicit-password",
    });
  });

  it("uses tailnet host when local bind is tailnet", () => {
    loadConfig.mockReturnValue({ gateway: { mode: "local", bind: "tailnet" } });
    resolveGatewayPort.mockReturnValue(4253);
    pickPrimaryTailnetIPv4.mockReturnValue("100.64.0.1");

    const result = resolveGatewayConnection({});

    expect(result.url).toBe("ws://100.64.0.1:4253");
  });

  it("uses lan host when local bind is lan", () => {
    loadConfig.mockReturnValue({ gateway: { mode: "local", bind: "lan" } });
    resolveGatewayPort.mockReturnValue(4253);
    pickPrimaryLanIPv4.mockReturnValue("192.168.1.42");

    const result = resolveGatewayConnection({});

    expect(result.url).toBe("ws://192.168.1.42:4253");
  });
});
