import type { IncomingMessage, ServerResponse } from "node:http";
import type { createSubsystemLogger } from "../../../logging/subsystem.js";
import type { PluginRegistry } from "../../../plugins/registry.js";

type SubsystemLogger = ReturnType<typeof createSubsystemLogger>;

export type PluginHttpRequestHandler = (
  req: IncomingMessage,
  res: ServerResponse,
) => Promise<boolean>;

export function createGatewayPluginRequestHandler(params: {
  registry: PluginRegistry;
  log: SubsystemLogger;
}): PluginHttpRequestHandler {
  const { registry, log } = params;
  return async (req, res) => {
    const routes = registry.httpRoutes ?? [];
    if (routes.length === 0) {
      return false;
    }

    const url = new URL(req.url ?? "/", "http://localhost");
    const route = routes.find((entry) => entry.path === url.pathname);
    if (!route) {
      return false;
    }

    try {
      await route.handler(req, res);
      return true;
    } catch (err) {
      log.warn(`plugin http route failed (${route.pluginId ?? "unknown"}): ${String(err)}`);
      if (!res.headersSent) {
        res.statusCode = 500;
        res.setHeader("Content-Type", "text/plain; charset=utf-8");
        res.end("Internal Server Error");
      }
      return true;
    }
  };
}
