import { randomUUID } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";
import { ToolInputError } from "../../agents/tools/common.js";
import {
  createMemoryGetTool,
  createMemorySearchTool,
  createMemoryWriteTool,
} from "../../agents/tools/memory-tool.js";
import { logWarn } from "../../logger.js";
import { loadConfig } from "../config/config.js";
import { resolveMainSessionKey } from "../config/sessions.js";
import type { AuthRateLimiter } from "./auth-rate-limit.js";
import type { ResolvedGatewayAuth } from "./auth.js";
import { sendJson } from "./http-common.js";
import { handleGatewayPostJsonEndpoint } from "./http-endpoint-helpers.js";

const DEFAULT_BODY_BYTES = 2 * 1024 * 1024;
const MCP_PATH = "/mcp";
const MCP_PROTOCOL_VERSION = "2024-11-05";
const MCP_SERVER_NAME = "Marv-mem";
const MCP_SERVER_VERSION = "0.1.0";

const JSON_RPC_VERSION = "2.0";
const INVALID_REQUEST_CODE = -32600;
const METHOD_NOT_FOUND_CODE = -32601;
const INVALID_PARAMS_CODE = -32602;
const INTERNAL_ERROR_CODE = -32603;
const SERVER_ERROR_CODE = -32000;

type JsonRpcId = string | number | null;

type JsonRpcRequest = {
  jsonrpc?: unknown;
  id?: unknown;
  method?: unknown;
  params?: unknown;
};

type JsonRpcError = {
  code: number;
  message: string;
  data?: unknown;
};

type JsonRpcResponse = {
  jsonrpc: "2.0";
  id: JsonRpcId;
  result?: unknown;
  error?: JsonRpcError;
};

type ToolListEntry = {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
};

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

function rpcResult(id: JsonRpcId, result: unknown): JsonRpcResponse {
  return {
    jsonrpc: JSON_RPC_VERSION,
    id,
    result,
  };
}

function rpcError(id: JsonRpcId, code: number, message: string, data?: unknown): JsonRpcResponse {
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

function buildToolList(): ToolListEntry[] {
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

function resolveToolContext(params: {
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

async function callMemoryTool(params: {
  name: string;
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
        : params.name === "memory_write"
          ? createMemoryWriteTool(common)
          : null;
  if (!tool) {
    throw new Error(`Tool unavailable: ${params.name}`);
  }
  // oxlint-disable-next-line typescript/no-explicit-any
  const raw = await (tool as any).execute?.(`mcp-${randomUUID()}`, params.toolArgs);
  return buildStructuredToolResult(raw);
}

async function handleRpcRequest(params: {
  payload: unknown;
  req: IncomingMessage;
}): Promise<JsonRpcResponse | null> {
  const request = asRecord(params.payload) as JsonRpcRequest | null;
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

export async function handleMcpMemoryHttpRequest(
  req: IncomingMessage,
  res: ServerResponse,
  opts: {
    auth: ResolvedGatewayAuth;
    maxBodyBytes?: number;
    trustedProxies?: string[];
    rateLimiter?: AuthRateLimiter;
  },
): Promise<boolean> {
  const handled = await handleGatewayPostJsonEndpoint(req, res, {
    pathname: MCP_PATH,
    auth: opts.auth,
    trustedProxies: opts.trustedProxies,
    rateLimiter: opts.rateLimiter,
    maxBodyBytes: opts.maxBodyBytes ?? DEFAULT_BODY_BYTES,
  });
  if (handled === false) {
    return false;
  }
  if (!handled) {
    return true;
  }

  try {
    const payload = handled.body;
    if (Array.isArray(payload)) {
      if (payload.length === 0) {
        sendJson(res, 200, rpcError(null, INVALID_REQUEST_CODE, "Invalid Request"));
        return true;
      }
      const responses: JsonRpcResponse[] = [];
      for (const entry of payload) {
        const response = await handleRpcRequest({ payload: entry, req });
        if (response) {
          responses.push(response);
        }
      }
      if (responses.length === 0) {
        res.statusCode = 202;
        res.end();
        return true;
      }
      sendJson(res, 200, responses);
      return true;
    }

    const response = await handleRpcRequest({ payload, req });
    if (!response) {
      res.statusCode = 202;
      res.end();
      return true;
    }
    sendJson(res, 200, response);
    return true;
  } catch (err) {
    logWarn(`mcp-memory: request failed: ${String(err)}`);
    sendJson(res, 500, rpcError(null, INTERNAL_ERROR_CODE, "Internal error"));
    return true;
  }
}
