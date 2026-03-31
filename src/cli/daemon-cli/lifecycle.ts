import path from "node:path";
import { readConfigFileSnapshot, resolveGatewayPort } from "../../core/config/config.js";
import type { GatewayService } from "../../infra/daemon/service.js";
import { resolveGatewayService } from "../../infra/daemon/service.js";
import { formatPortDiagnostics, inspectPortUsage, type PortListener } from "../../infra/ports.js";
import { sleep } from "../../utils.js";
import {
  runServiceRestart,
  runServiceStart,
  runServiceStop,
  runServiceUninstall,
} from "./lifecycle-core.js";
import { renderGatewayServiceStartHints } from "./shared.js";
import type { DaemonLifecycleOptions } from "./types.js";

const GATEWAY_RESTART_VERIFY_TIMEOUT_MS = 10_000;
const GATEWAY_RESTART_VERIFY_POLL_MS = 250;

type GatewayCommandSnapshot = Awaited<ReturnType<GatewayService["readCommand"]>>;
type GatewayRuntimeSnapshot = Awaited<ReturnType<GatewayService["readRuntime"]>>;

type GatewayRestartSnapshot = {
  port: number;
  command: GatewayCommandSnapshot;
  runtime: GatewayRuntimeSnapshot | undefined;
  previousPids: Set<number>;
};

function parsePositiveInt(value: string | undefined): number | null {
  if (!value?.trim()) {
    return null;
  }
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null;
  }
  return parsed;
}

async function resolveGatewayRestartPort(service: GatewayService): Promise<number> {
  const snapshot = await readConfigFileSnapshot().catch(() => null);
  if (snapshot?.valid) {
    return resolveGatewayPort(snapshot.config, process.env);
  }

  const command = await service.readCommand(process.env).catch(() => null);
  const args = command?.programArguments ?? [];
  const portFlagIndex = args.indexOf("--port");
  const argPort = portFlagIndex >= 0 ? parsePositiveInt(args[portFlagIndex + 1]) : null;
  if (argPort) {
    return argPort;
  }

  const envPort =
    parsePositiveInt(command?.environment?.MARV_GATEWAY_PORT) ??
    parsePositiveInt(command?.environment?.CLAWDBOT_GATEWAY_PORT) ??
    parsePositiveInt(process.env.MARV_GATEWAY_PORT) ??
    parsePositiveInt(process.env.CLAWDBOT_GATEWAY_PORT);
  return envPort ?? resolveGatewayPort(undefined, process.env);
}

function collectPreviousPids(
  runtime: GatewayRuntimeSnapshot | undefined,
  listeners: PortListener[],
): Set<number> {
  const pids = new Set<number>();
  if (typeof runtime?.pid === "number" && runtime.pid > 0) {
    pids.add(runtime.pid);
  }
  for (const listener of listeners) {
    if (typeof listener.pid === "number" && listener.pid > 0) {
      pids.add(listener.pid);
    }
  }
  return pids;
}

function normalizeMatchText(value: string): string {
  return value.replaceAll("\\", "/").toLowerCase();
}

function collectGatewayIdentityTokens(command: GatewayCommandSnapshot): string[] {
  const args = command?.programArguments ?? [];
  if (args.length === 0) {
    return [];
  }

  const normalized = args.map((value) => normalizeMatchText(value)).filter(Boolean);
  const execBase = normalized[0] ? path.posix.basename(normalized[0]) : "";
  const entrypoint = normalized
    .slice(1)
    .find((value) => value.includes("/") || /\.(?:[cm]?js|ts)$/i.test(value));
  const entrypointBase = entrypoint ? path.posix.basename(entrypoint) : "";
  const tokens = new Set<string>();

  if (entrypoint) {
    tokens.add(entrypoint);
  }
  if (entrypointBase && entrypointBase !== execBase) {
    tokens.add(entrypointBase);
  }
  if (!entrypoint && execBase && execBase !== "node" && execBase !== "bun") {
    tokens.add(execBase);
  }

  return [...tokens];
}

function listenerLooksLikeExpectedGateway(
  listener: PortListener,
  command: GatewayCommandSnapshot,
): boolean {
  const haystack = normalizeMatchText(`${listener.commandLine ?? ""} ${listener.command ?? ""}`);
  if (!haystack) {
    return false;
  }
  if (!haystack.includes("gateway")) {
    return false;
  }

  const identityTokens = collectGatewayIdentityTokens(command);
  if (identityTokens.length === 0) {
    return haystack.includes("marv") || haystack.includes("agentmarv");
  }

  return identityTokens.some((token) => haystack.includes(token));
}

function formatGatewayRestartFailure(params: {
  port: number;
  detail: string;
  runtime?: GatewayRuntimeSnapshot;
  listeners?: PortListener[];
  diagnostics?: string[];
}): string {
  const lines = [`Gateway restart verification failed: ${params.detail}`];
  if (typeof params.runtime?.pid === "number" && params.runtime.pid > 0) {
    lines.push(`Service runtime pid: ${params.runtime.pid}`);
  }
  if (params.listeners && params.listeners.length > 0) {
    lines.push(
      ...formatPortDiagnostics({
        port: params.port,
        status: "busy",
        listeners: params.listeners,
        hints: [],
        detail: undefined,
      }),
    );
  } else if (params.diagnostics && params.diagnostics.length > 0) {
    lines.push(...params.diagnostics);
  }
  return lines.join("\n");
}

async function captureGatewayRestartSnapshot(
  service: GatewayService,
): Promise<GatewayRestartSnapshot> {
  const port = await resolveGatewayRestartPort(service);
  const [command, runtime, portUsage] = await Promise.all([
    service.readCommand(process.env).catch(() => null),
    service.readRuntime(process.env).catch(() => undefined),
    inspectPortUsage(port).catch(() => ({
      port,
      status: "unknown" as const,
      listeners: [],
      hints: [],
    })),
  ]);

  return {
    port,
    command,
    runtime,
    previousPids: collectPreviousPids(runtime, portUsage.listeners),
  };
}

async function verifyGatewayRestart(
  service: GatewayService,
  snapshot: GatewayRestartSnapshot,
): Promise<{ warnings?: string[] } | void> {
  const deadline = Date.now() + GATEWAY_RESTART_VERIFY_TIMEOUT_MS;
  let lastDetail = `gateway is not listening on port ${snapshot.port}`;
  let lastRuntime: GatewayRuntimeSnapshot | undefined;
  let lastListeners: PortListener[] = [];
  let lastDiagnostics: string[] = [];

  while (Date.now() <= deadline) {
    const [runtime, portUsage] = await Promise.all([
      service.readRuntime(process.env).catch(() => undefined),
      inspectPortUsage(snapshot.port).catch(() => ({
        port: snapshot.port,
        status: "unknown" as const,
        listeners: [],
        hints: [],
      })),
    ]);

    lastRuntime = runtime;
    lastListeners = portUsage.listeners;
    lastDiagnostics =
      portUsage.status === "busy"
        ? formatPortDiagnostics(portUsage)
        : [`Port ${snapshot.port} is free.`];

    const activePids = collectPreviousPids(runtime, portUsage.listeners);
    const stalePid = [...snapshot.previousPids].find((pid) => activePids.has(pid));
    if (typeof stalePid === "number") {
      lastDetail = `old gateway pid ${stalePid} is still active after restart`;
    } else if (portUsage.status !== "busy") {
      lastDetail = `gateway is not listening on port ${snapshot.port}`;
    } else if (portUsage.listeners.length === 0) {
      if (
        typeof runtime?.pid === "number" &&
        runtime.pid > 0 &&
        !snapshot.previousPids.has(runtime.pid)
      ) {
        return {
          warnings: [
            `Gateway restarted on port ${snapshot.port}, but the serving process could not be identified from port inspection.`,
          ],
        };
      }
      lastDetail = `port ${snapshot.port} is busy, but the listener identity is unavailable`;
    } else {
      const runtimePid = runtime?.pid;
      const matchingListener =
        (typeof runtimePid === "number" && runtimePid > 0
          ? portUsage.listeners.find((listener) => listener.pid === runtimePid)
          : undefined) ??
        portUsage.listeners.find((listener) =>
          listenerLooksLikeExpectedGateway(listener, snapshot.command),
        );
      if (matchingListener) {
        return;
      }
      lastDetail = `port ${snapshot.port} is held by a process that does not match the configured gateway command`;
    }

    if (Date.now() >= deadline) {
      break;
    }
    await sleep(GATEWAY_RESTART_VERIFY_POLL_MS);
  }

  throw new Error(
    formatGatewayRestartFailure({
      port: snapshot.port,
      detail: lastDetail,
      runtime: lastRuntime,
      listeners: lastListeners,
      diagnostics: lastDiagnostics,
    }),
  );
}

export async function runDaemonUninstall(opts: DaemonLifecycleOptions = {}) {
  return await runServiceUninstall({
    serviceNoun: "Gateway",
    service: resolveGatewayService(),
    opts,
    stopBeforeUninstall: true,
    assertNotLoadedAfterUninstall: true,
  });
}

export async function runDaemonStart(opts: DaemonLifecycleOptions = {}) {
  return await runServiceStart({
    serviceNoun: "Gateway",
    service: resolveGatewayService(),
    renderStartHints: renderGatewayServiceStartHints,
    opts,
  });
}

export async function runDaemonStop(opts: DaemonLifecycleOptions = {}) {
  return await runServiceStop({
    serviceNoun: "Gateway",
    service: resolveGatewayService(),
    opts,
  });
}

/**
 * Restart the gateway service service.
 * @returns `true` if restart succeeded, `false` if the service was not loaded.
 * Throws/exits on check or restart failures.
 */
export async function runDaemonRestart(opts: DaemonLifecycleOptions = {}): Promise<boolean> {
  const service = resolveGatewayService();
  const restartSnapshot = await captureGatewayRestartSnapshot(service);
  return await runServiceRestart({
    serviceNoun: "Gateway",
    service,
    renderStartHints: renderGatewayServiceStartHints,
    opts,
    checkTokenDrift: true,
    postRestartCheck: async () => await verifyGatewayRestart(service, restartSnapshot),
  });
}

export const __testing = {
  captureGatewayRestartSnapshot,
  verifyGatewayRestart,
  listenerLooksLikeExpectedGateway,
};
