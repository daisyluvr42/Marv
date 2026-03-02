import type { IncomingMessage, ServerResponse } from "node:http";
import { logWarn } from "../../logger.js";
import { handleRpcPayload, INTERNAL_ERROR_CODE, rpcError } from "../../memory/api/mcp-handler.js";
import type { AuthRateLimiter } from "./auth-rate-limit.js";
import type { ResolvedGatewayAuth } from "./auth.js";
import { sendJson } from "./http-common.js";
import { handleGatewayPostJsonEndpoint } from "./http-endpoint-helpers.js";

const DEFAULT_BODY_BYTES = 2 * 1024 * 1024;
const MCP_PATH = "/mcp";

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
    const rpcResponse = await handleRpcPayload(handled.body);
    if (!rpcResponse) {
      res.statusCode = 202;
      res.end();
      return true;
    }
    sendJson(res, 200, rpcResponse);
    return true;
  } catch (err) {
    logWarn(`mcp-memory: request failed: ${String(err)}`);
    sendJson(res, 500, rpcError(null, INTERNAL_ERROR_CODE, "Internal error"));
    return true;
  }
}
