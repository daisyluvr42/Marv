import chalk from "chalk";
import { DEFAULT_MODEL, DEFAULT_PROVIDER } from "../../agents/defaults.js";
import { resolveConfiguredModelRef } from "../../agents/model/model-resolve.js";
import { VERSION } from "../../infra/version.js";
import { getResolvedLoggerSettings } from "../../logging.js";
import type { loadConfig } from "../config/config.js";

export function logGatewayStartup(params: {
  cfg: ReturnType<typeof loadConfig>;
  bindHost: string;
  bindHosts?: string[];
  port: number;
  tlsEnabled?: boolean;
  log: { info: (msg: string, meta?: Record<string, unknown>) => void };
  isNixMode: boolean;
}) {
  const { provider: agentProvider, model: agentModel } = resolveConfiguredModelRef({
    cfg: params.cfg,
    defaultProvider: DEFAULT_PROVIDER,
    defaultModel: DEFAULT_MODEL,
  });
  const modelRef = `${agentProvider}/${agentModel}`;
  params.log.info(`agent model: ${modelRef}`, {
    consoleMessage: `agent model: ${chalk.whiteBright(modelRef)}`,
  });
  const execPath = process.execPath;
  const entrypoint = process.argv[1] || execPath;
  params.log.info(`build: version ${VERSION} · exec ${execPath} · entry ${entrypoint}`);
  const scheme = params.tlsEnabled ? "wss" : "ws";
  const formatHost = (host: string) => (host.includes(":") ? `[${host}]` : host);
  const hosts =
    params.bindHosts && params.bindHosts.length > 0 ? params.bindHosts : [params.bindHost];
  const primaryHost = hosts[0] ?? params.bindHost;
  params.log.info(
    `listening on ${scheme}://${formatHost(primaryHost)}:${params.port} (PID ${process.pid})`,
  );
  for (const host of hosts.slice(1)) {
    params.log.info(`listening on ${scheme}://${formatHost(host)}:${params.port}`);
  }
  params.log.info(`log file: ${getResolvedLoggerSettings().file}`);
  if (params.isNixMode) {
    params.log.info("gateway: running in Nix mode (config managed externally)");
  }
}
