export type JsonRpcId = string | number | null;

export type McpJsonRpcRequest = {
  jsonrpc?: unknown;
  id?: unknown;
  method?: unknown;
  params?: unknown;
};

export type McpJsonRpcError = {
  code: number;
  message: string;
  data?: unknown;
};

export type McpJsonRpcResponse = {
  jsonrpc: "2.0";
  id: JsonRpcId;
  result?: unknown;
  error?: McpJsonRpcError;
};

export type McpToolListEntry = {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
};

export type McpToolCallParams = {
  name?: string;
  tool?: string;
  arguments?: Record<string, unknown>;
  sessionKey?: string;
  _meta?: Record<string, unknown>;
};

export type McpMemorySearchArgs = {
  query: string;
  maxResults?: number;
  minScore?: number;
  sessionKey?: string;
};

export type McpMemoryGetArgs = {
  path: string;
  from?: number;
  lines?: number;
  sessionKey?: string;
};

export type McpMemoryWriteArgs = {
  content: string;
  kind?: string;
  scopeType?: string;
  scopeId?: string;
  confidence?: number;
  source?: string;
  sessionKey?: string;
};
