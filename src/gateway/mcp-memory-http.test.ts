import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

const TEST_GATEWAY_TOKEN = "test-gateway-token-1234567890";

const createMemorySearchToolMock = vi.hoisted(() => vi.fn());
const createMemoryGetToolMock = vi.hoisted(() => vi.fn());
const createMemoryWriteToolMock = vi.hoisted(() => vi.fn());
const memorySearchExecuteMock = vi.hoisted(() => vi.fn());
const memoryGetExecuteMock = vi.hoisted(() => vi.fn());
const memoryWriteExecuteMock = vi.hoisted(() => vi.fn());

let cfg: Record<string, unknown> = {};

vi.mock("../config/config.js", () => ({
  loadConfig: () => cfg,
}));

vi.mock("../config/sessions.js", () => ({
  resolveMainSessionKey: () => "agent:main:main",
}));

vi.mock("./http-auth-helpers.js", () => ({
  authorizeGatewayBearerRequestOrReply: async () => true,
}));

vi.mock("../logger.js", () => ({
  logWarn: () => {},
}));

vi.mock("../agents/tools/memory-tool.js", () => {
  const makeTool = (name: string, execute: ReturnType<typeof vi.fn>) => ({
    name,
    description: `${name} description`,
    parameters: { type: "object", properties: {} },
    execute,
  });
  return {
    createMemorySearchTool: createMemorySearchToolMock.mockImplementation(() =>
      makeTool("memory_search", memorySearchExecuteMock),
    ),
    createMemoryGetTool: createMemoryGetToolMock.mockImplementation(() =>
      makeTool("memory_get", memoryGetExecuteMock),
    ),
    createMemoryWriteTool: createMemoryWriteToolMock.mockImplementation(() =>
      makeTool("memory_write", memoryWriteExecuteMock),
    ),
  };
});

const { handleMcpMemoryHttpRequest } = await import("./mcp-memory-http.js");

let sharedPort = 0;
let sharedServer: ReturnType<typeof createServer> | undefined;

beforeAll(async () => {
  sharedServer = createServer((req, res) => {
    void (async () => {
      const handled = await handleMcpMemoryHttpRequest(req, res, {
        auth: { mode: "token", token: TEST_GATEWAY_TOKEN, allowTailscale: false },
      });
      if (handled) {
        return;
      }
      res.statusCode = 404;
      res.end("not found");
    })().catch((err) => {
      res.statusCode = 500;
      res.end(String(err));
    });
  });

  await new Promise<void>((resolve, reject) => {
    sharedServer?.once("error", reject);
    sharedServer?.listen(0, "127.0.0.1", () => {
      const address = sharedServer?.address() as AddressInfo | null;
      sharedPort = address?.port ?? 0;
      resolve();
    });
  });
});

afterAll(async () => {
  const server = sharedServer;
  if (!server) {
    return;
  }
  await new Promise<void>((resolve) => server.close(() => resolve()));
  sharedServer = undefined;
});

beforeEach(() => {
  cfg = {};
  createMemorySearchToolMock.mockClear();
  createMemoryGetToolMock.mockClear();
  createMemoryWriteToolMock.mockClear();
  memorySearchExecuteMock.mockReset();
  memoryGetExecuteMock.mockReset();
  memoryWriteExecuteMock.mockReset();
  memorySearchExecuteMock.mockResolvedValue({
    details: { ok: true, source: "search" },
  });
  memoryGetExecuteMock.mockResolvedValue({
    details: { ok: true, source: "get" },
  });
  memoryWriteExecuteMock.mockResolvedValue({
    details: { ok: true, source: "write" },
  });
});

async function postMcp(payload: unknown): Promise<Response> {
  return await fetch(`http://127.0.0.1:${sharedPort}/mcp`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${TEST_GATEWAY_TOKEN}` },
    body: JSON.stringify(payload),
  });
}

describe("POST /mcp", () => {
  it("responds to initialize with Marv-mem server info", async () => {
    const res = await postMcp({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {},
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.jsonrpc).toBe("2.0");
    expect(body.id).toBe(1);
    expect(body.result.serverInfo.name).toBe("Marv-mem");
    expect(body.result.capabilities).toHaveProperty("tools");
  });

  it("lists memory tools", async () => {
    const res = await postMcp({
      jsonrpc: "2.0",
      id: "list",
      method: "tools/list",
      params: {},
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    const names = (body.result.tools as Array<{ name: string }>).map((entry) => entry.name);
    expect(names).toEqual(["memory_search", "memory_get", "memory_write"]);
  });

  it("calls memory_search and routes sessionKey to tool context", async () => {
    memorySearchExecuteMock.mockResolvedValueOnce({
      details: { ok: true, payload: "search-ok" },
    });
    const res = await postMcp({
      jsonrpc: "2.0",
      id: 22,
      method: "tools/call",
      params: {
        name: "memory_search",
        arguments: {
          query: "deploy notes",
          sessionKey: "agent:ops:main",
        },
      },
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(createMemorySearchToolMock).toHaveBeenCalledTimes(1);
    expect(createMemorySearchToolMock.mock.calls[0]?.[0]).toMatchObject({
      agentSessionKey: "agent:ops:main",
    });
    expect(memorySearchExecuteMock).toHaveBeenCalledTimes(1);
    expect(memorySearchExecuteMock.mock.calls[0]?.[1]).toEqual({
      query: "deploy notes",
    });
    expect(body.result.isError).toBe(false);
    expect(body.result.structuredContent).toMatchObject({ payload: "search-ok" });
  });

  it("falls back to main session key when tools/call omits sessionKey", async () => {
    const res = await postMcp({
      jsonrpc: "2.0",
      id: 23,
      method: "tools/call",
      params: {
        name: "memory_get",
        arguments: {
          path: "soul-memory/mem_x",
        },
      },
    });
    expect(res.status).toBe(200);
    expect(createMemoryGetToolMock).toHaveBeenCalledTimes(1);
    expect(createMemoryGetToolMock.mock.calls[0]?.[0]).toMatchObject({
      agentSessionKey: "agent:main:main",
    });
  });

  it("returns method-not-found for unsupported tools", async () => {
    const res = await postMcp({
      jsonrpc: "2.0",
      id: "bad-tool",
      method: "tools/call",
      params: {
        name: "not_exists",
        arguments: {},
      },
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.error.code).toBe(-32601);
  });

  it("accepts initialized notifications without response body", async () => {
    const res = await postMcp({
      jsonrpc: "2.0",
      method: "notifications/initialized",
      params: {},
    });
    expect(res.status).toBe(202);
    expect(await res.text()).toBe("");
  });
});
