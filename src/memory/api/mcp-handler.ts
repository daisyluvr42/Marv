import { randomUUID } from "node:crypto";
import { ToolInputError } from "../../agents/tools/common.js";
import {
  createMemoryGetTool,
  createMemorySearchTool,
  createMemoryWriteTool,
} from "../../agents/tools/memory-tool.js";
import { loadConfig } from "../../core/config/config.js";
import { resolveMainSessionKey } from "../../core/config/sessions.js";
import type {
  JsonRpcId,
  McpJsonRpcRequest,
  McpJsonRpcResponse,
  McpToolListEntry,
} from "./mcp-types.js";

export const MCP_PROTOCOL_VERSION = "2024-11-05";
export const MCP_SERVER_NAME = "Marv-mem";
export const MCP_SERVER_VERSION = "0.1.0";

export const JSON_RPC_VERSION = "2.0" as const;
export const INVALID_REQUEST_CODE = -32600;
export const METHOD_NOT_FOUND_CODE = -32601;
export const INVALID_PARAMS_CODE = -32602;
export const INTERNAL_ERROR_CODE = -32603;
export const SERVER_ERROR_CODE = -32000;

const SEARCH_TOOL_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    query: { type: "string" },
    maxResults: { type: "number" },
    minScore: { type: "number" },
    sessionKey: { type: "string" },
  },
  required: ["query"],
} satisfies Record<string, unknown>;

const GET_TOOL_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    path: { type: "string" },
    from: { type: "number" },
    lines: { type: "number" },
    sessionKey: { type: "string" },
  },
  required: ["path"],
} satisfies Record<string, unknown>;

const WRITE_TOOL_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    content: { type: "string" },
    kind: { type: "string" },
    scopeType: { type: "string" },
    scopeId: { type: "string" },
    confidence: { type: "number" },
    source: { type: "string" },
    sessionKey: { type: "string" },
  },
  required: ["content"],
} satisfies Record<string, unknown>;

function asRecord(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return null;
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function normalizeRpcId(raw: unknown): JsonRpcId | undefined {
  if (raw === null) {
    return null;
  }
  if (typeof raw === "string" || typeof raw === "number") {
    return raw;
  }
  return undefined;
}

export function rpcResult(id: JsonRpcId, result: unknown): McpJsonRpcResponse {
  return {
    jsonrpc: JSON_RPC_VERSION,
    id,
    result,
  };
}

export function rpcError(
  id: JsonRpcId,
  code: number,
  message: string,
  data?: unknown,
): McpJsonRpcResponse {
  return {
    jsonrpc: JSON_RPC_VERSION,
    id,
    error: {
      code,
      message,
      data,
    },
  };
}

export function buildToolList(): McpToolListEntry[] {
  return [
    {
      name: "memory_search",
      description:
        "Search structured Marv memory (soul-memory) with multi-signal ranking and optional legacy fallback.",
      inputSchema: SEARCH_TOOL_SCHEMA,
    },
    {
      name: "memory_get",
      description: "Fetch a memory snippet by path returned from memory_search.",
      inputSchema: GET_TOOL_SCHEMA,
    },
    {
      name: "memory_write",
      description: "Write durable structured memory into soul-memory storage.",
      inputSchema: WRITE_TOOL_SCHEMA,
    },
  ];
}

export function resolveToolContext(params: {
  cfg: ReturnType<typeof loadConfig>;
  paramsRecord: Record<string, unknown>;
  argsRecord: Record<string, unknown>;
}): { sessionKey: string; toolArgs: Record<string, unknown> } {
  const sessionKeyFromParams = asString(params.paramsRecord.sessionKey);
  const sessionKeyFromArgs = asString(params.argsRecord.sessionKey);
  const meta = asRecord(params.paramsRecord._meta);
  const sessionKeyFromMeta = asString(meta?.sessionKey);
  const sessionKeyRaw =
    sessionKeyFromParams ??
    sessionKeyFromArgs ??
    sessionKeyFromMeta ??
    resolveMainSessionKey(params.cfg);

  const toolArgs = { ...params.argsRecord };
  delete toolArgs.sessionKey;
  return { sessionKey: sessionKeyRaw, toolArgs };
}

function buildStructuredToolResult(result: unknown): {
  content: Array<{ type: "text"; text: string }>;
  structuredContent: Record<string, unknown>;
  isError: false;
} {
  const maybeRecord = asRecord(result);
  const details = maybeRecord?.details ?? result;
  const detailsRecord = asRecord(details);
  return {
    content: [
      {
        type: "text",
        text: typeof details === "string" ? details : JSON.stringify(details, null, 2),
      },
    ],
    structuredContent: detailsRecord ?? { value: details },
    isError: false,
  };
}

type MemoryToolName = "memory_search" | "memory_get" | "memory_write";

type ExecutableTool = {
  execute?: (callId: string, args: Record<string, unknown>) => unknown;
};

export async function callMemoryTool(params: {
  name: MemoryToolName;
  cfg: ReturnType<typeof loadConfig>;
  sessionKey: string;
  toolArgs: Record<string, unknown>;
}): Promise<ReturnType<typeof buildStructuredToolResult>> {
  const common = {
    config: params.cfg,
    agentSessionKey: params.sessionKey,
  };
  const tool =
    params.name === "memory_search"
      ? createMemorySearchTool(common)
      : params.name === "memory_get"
        ? createMemoryGetTool(common)
        : createMemoryWriteTool(common);

  const executable = tool as ExecutableTool;
  const raw = await executable.execute?.(`mcp-${randomUUID()}`, params.toolArgs);
  return buildStructuredToolResult(raw);
}

export async function handleRpcRequest(params: {
  payload: unknown;
}): Promise<McpJsonRpcResponse | null> {
  const request = asRecord(params.payload) as McpJsonRpcRequest | null;
  if (!request) {
    return rpcError(null, INVALID_REQUEST_CODE, "Invalid Request");
  }

  const id = normalizeRpcId(request.id);
  const hasId = Object.prototype.hasOwnProperty.call(request, "id");
  const isNotification = !hasId;

  const method = asString(request.method);
  if (!method) {
    if (isNotification) {
      return null;
    }
    return rpcError(id ?? null, INVALID_REQUEST_CODE, "Invalid Request");
  }

  if (method === "notifications/initialized") {
    if (isNotification) {
      return null;
    }
    return rpcResult(id ?? null, {});
  }

  if (isNotification) {
    return null;
  }

  if (id === undefined) {
    return rpcError(null, INVALID_REQUEST_CODE, "Invalid Request");
  }

  if (method === "ping") {
    return rpcResult(id, {});
  }

  if (method === "initialize") {
    return rpcResult(id, {
      protocolVersion: MCP_PROTOCOL_VERSION,
      capabilities: {
        tools: {},
      },
      serverInfo: {
        name: MCP_SERVER_NAME,
        version: MCP_SERVER_VERSION,
      },
      instructions:
        "Marv-mem exposes memory_search, memory_get, and memory_write over MCP JSON-RPC.",
    });
  }

  if (method === "tools/list") {
    return rpcResult(id, {
      tools: buildToolList(),
    });
  }

  if (method === "tools/call") {
    const cfg = loadConfig();
    const paramsRecord = asRecord(request.params);
    if (!paramsRecord) {
      return rpcError(id, INVALID_PARAMS_CODE, "tools/call requires params");
    }
    const name = asString(paramsRecord.name) ?? asString(paramsRecord.tool);
    if (!name) {
      return rpcError(id, INVALID_PARAMS_CODE, "tools/call requires params.name");
    }
    if (name !== "memory_search" && name !== "memory_get" && name !== "memory_write") {
      return rpcError(id, METHOD_NOT_FOUND_CODE, `Unknown MCP tool: ${name}`);
    }
    const argsRecord = asRecord(paramsRecord.arguments) ?? {};
    const { sessionKey, toolArgs } = resolveToolContext({
      cfg,
      paramsRecord,
      argsRecord,
    });
    try {
      const result = await callMemoryTool({
        name,
        cfg,
        sessionKey,
        toolArgs,
      });
      return rpcResult(id, result);
    } catch (err) {
      if (err instanceof ToolInputError) {
        return rpcError(id, INVALID_PARAMS_CODE, err.message);
      }
      const message = err instanceof Error ? err.message : String(err);
      return rpcError(id, SERVER_ERROR_CODE, message);
    }
  }

  return rpcError(id, METHOD_NOT_FOUND_CODE, `Method not found: ${method}`);
}

export async function handleRpcPayload(
  payload: unknown,
): Promise<McpJsonRpcResponse | McpJsonRpcResponse[] | null> {
  if (!Array.isArray(payload)) {
    return await handleRpcRequest({ payload });
  }

  if (payload.length === 0) {
    return rpcError(null, INVALID_REQUEST_CODE, "Invalid Request");
  }

  const responses: McpJsonRpcResponse[] = [];
  for (const entry of payload) {
    const response = await handleRpcRequest({ payload: entry });
    if (response) {
      responses.push(response);
    }
  }
  if (responses.length === 0) {
    return null;
  }
  return responses;
}
