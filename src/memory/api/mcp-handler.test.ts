import { beforeEach, describe, expect, it, vi } from "vitest";
import { ToolInputError } from "../../agents/tools/common.js";

const createMemorySearchToolMock = vi.hoisted(() => vi.fn());
const createMemoryGetToolMock = vi.hoisted(() => vi.fn());
const createMemoryWriteToolMock = vi.hoisted(() => vi.fn());
const memorySearchExecuteMock = vi.hoisted(() => vi.fn());
const memoryGetExecuteMock = vi.hoisted(() => vi.fn());
const memoryWriteExecuteMock = vi.hoisted(() => vi.fn());

let cfg: Record<string, unknown> = {};

vi.mock("../../core/config/config.js", () => ({
  loadConfig: () => cfg,
}));

vi.mock("../../core/config/sessions.js", () => ({
  resolveMainSessionKey: () => "agent:main:main",
}));

vi.mock("../../agents/tools/memory-tool.js", () => {
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

const { handleRpcPayload, handleRpcRequest } = await import("./mcp-handler.js");

describe("mcp-handler", () => {
  beforeEach(() => {
    cfg = {};
    createMemorySearchToolMock.mockClear();
    createMemoryGetToolMock.mockClear();
    createMemoryWriteToolMock.mockClear();
    memorySearchExecuteMock.mockReset();
    memoryGetExecuteMock.mockReset();
    memoryWriteExecuteMock.mockReset();
    memorySearchExecuteMock.mockResolvedValue({ details: { ok: true, source: "search" } });
    memoryGetExecuteMock.mockResolvedValue({ details: { ok: true, source: "get" } });
    memoryWriteExecuteMock.mockResolvedValue({ details: { ok: true, source: "write" } });
  });

  it("responds to initialize", async () => {
    const response = await handleRpcRequest({
      payload: {
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: {},
      },
    });

    expect(response?.id).toBe(1);
    const responseResult = response?.result as { serverInfo?: { name?: string } } | undefined;
    expect(responseResult?.serverInfo?.name).toBe("Marv-mem");
  });

  it("lists memory tools", async () => {
    const response = await handleRpcRequest({
      payload: {
        jsonrpc: "2.0",
        id: "list",
        method: "tools/list",
      },
    });

    const result = response?.result as { tools?: Array<{ name: string }> } | undefined;
    expect(result?.tools?.map((entry) => entry.name)).toEqual([
      "memory_search",
      "memory_get",
      "memory_write",
    ]);
  });

  it("calls tool with explicit sessionKey from args", async () => {
    const response = await handleRpcRequest({
      payload: {
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
      },
    });

    expect(createMemorySearchToolMock).toHaveBeenCalledTimes(1);
    expect(createMemorySearchToolMock.mock.calls[0]?.[0]).toMatchObject({
      agentSessionKey: "agent:ops:main",
    });
    expect(memorySearchExecuteMock.mock.calls[0]?.[1]).toEqual({ query: "deploy notes" });
    const result = response?.result as { isError?: boolean } | undefined;
    expect(result?.isError).toBe(false);
  });

  it("falls back to sessionKey in _meta", async () => {
    await handleRpcRequest({
      payload: {
        jsonrpc: "2.0",
        id: 23,
        method: "tools/call",
        params: {
          name: "memory_get",
          _meta: {
            sessionKey: "agent:meta:session",
          },
          arguments: {
            path: "soul-memory/mem_x",
          },
        },
      },
    });

    expect(createMemoryGetToolMock).toHaveBeenCalledTimes(1);
    expect(createMemoryGetToolMock.mock.calls[0]?.[0]).toMatchObject({
      agentSessionKey: "agent:meta:session",
    });
  });

  it("maps ToolInputError to invalid params", async () => {
    memoryWriteExecuteMock.mockRejectedValueOnce(new ToolInputError("content required"));

    const response = await handleRpcRequest({
      payload: {
        jsonrpc: "2.0",
        id: "bad",
        method: "tools/call",
        params: {
          name: "memory_write",
          arguments: {
            content: "",
          },
        },
      },
    });

    const error = response?.error as { code?: number } | undefined;
    expect(error?.code).toBe(-32602);
  });

  it("returns null for notifications", async () => {
    const response = await handleRpcRequest({
      payload: {
        jsonrpc: "2.0",
        method: "notifications/initialized",
        params: {},
      },
    });
    expect(response).toBeNull();
  });

  it("handles batch payloads and drops notification-only entries", async () => {
    const response = await handleRpcPayload([
      {
        jsonrpc: "2.0",
        method: "notifications/initialized",
      },
      {
        jsonrpc: "2.0",
        id: 1,
        method: "ping",
      },
    ]);

    expect(Array.isArray(response)).toBe(true);
    const entries = response as Array<{ id: number }>;
    expect(entries).toHaveLength(1);
    expect(entries[0]?.id).toBe(1);
  });

  it("returns invalid request for empty batch", async () => {
    const response = await handleRpcPayload([]);
    const error = (response as { error?: { code?: number } } | null)?.error;
    expect(error?.code).toBe(-32600);
  });
});
