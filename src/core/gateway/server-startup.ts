import { DEFAULT_MODEL, DEFAULT_PROVIDER } from "../../agents/defaults.js";
import { loadModelCatalog } from "../../agents/model/model-catalog.js";
import {
  getModelRefStatus,
  resolveConfiguredModelRef,
  resolveHooksGmailModel,
} from "../../agents/model/model-resolve.js";
import {
  ensureRuntimeModelRegistry,
  startRuntimeModelRegistryRefreshLoop,
} from "../../agents/model/runtime-model-registry.js";
import { resolveAgentSessionDirs } from "../../agents/session-dirs.js";
import { cleanStaleLockFiles } from "../../agents/session-write-lock.js";
import type { CliDeps } from "../../cli/deps.js";
import { startGmailWatcherWithLogs } from "../../hooks/gmail-watcher-lifecycle.js";
import { isTruthyEnvValue } from "../../infra/env.js";
import type { loadMarvPlugins } from "../../plugins/loader.js";
import { type PluginServicesHandle, startPluginServices } from "../../plugins/services.js";
import type { loadConfig } from "../config/config.js";
import { resolveStateDir } from "../config/paths.js";
import { startBrowserControlServerIfEnabled } from "./server-browser.js";
import {
  scheduleRestartSentinelWake,
  shouldWakeFromRestartSentinel,
} from "./server-restart-sentinel.js";
import { startGatewayMemoryBackend } from "./server-startup-memory.js";

const SESSION_LOCK_STALE_MS = 30 * 60 * 1000;

export async function startGatewaySidecars(params: {
  cfg: ReturnType<typeof loadConfig>;
  pluginRegistry: ReturnType<typeof loadMarvPlugins>;
  defaultWorkspaceDir: string;
  deps: CliDeps;
  startChannels: () => Promise<void>;
  log: { warn: (msg: string) => void };
  logHooks: {
    info: (msg: string) => void;
    warn: (msg: string) => void;
    error: (msg: string) => void;
  };
  logChannels: { info: (msg: string) => void; error: (msg: string) => void };
  logBrowser: { error: (msg: string) => void };
}) {
  try {
    const stateDir = resolveStateDir(process.env);
    const sessionDirs = await resolveAgentSessionDirs(stateDir);
    for (const sessionsDir of sessionDirs) {
      await cleanStaleLockFiles({
        sessionsDir,
        staleMs: SESSION_LOCK_STALE_MS,
        removeStale: true,
        log: { warn: (message) => params.log.warn(message) },
      });
    }
  } catch (err) {
    params.log.warn(`session lock cleanup failed on startup: ${String(err)}`);
  }

  try {
    await ensureRuntimeModelRegistry({ cfg: params.cfg });
    startRuntimeModelRegistryRefreshLoop({
      cfg: params.cfg,
      log: { warn: (message) => params.log.warn(message) },
    });
  } catch (err) {
    params.log.warn(`model registry startup refresh failed: ${String(err)}`);
  }

  // Start Marv browser control server (unless disabled via config).
  let browserControl: Awaited<ReturnType<typeof startBrowserControlServerIfEnabled>> = null;
  try {
    browserControl = await startBrowserControlServerIfEnabled();
  } catch (err) {
    params.logBrowser.error(`server failed to start: ${String(err)}`);
  }

  // Start Gmail watcher if configured (hooks.gmail.account).
  await startGmailWatcherWithLogs({
    cfg: params.cfg,
    log: params.logHooks,
  });

  // Validate hooks.gmail.model if configured.
  if (params.cfg.hooks?.gmail?.model) {
    const hooksModelRef = resolveHooksGmailModel({
      cfg: params.cfg,
      defaultProvider: DEFAULT_PROVIDER,
    });
    if (hooksModelRef) {
      const { provider: defaultProvider, model: defaultModel } = resolveConfiguredModelRef({
        cfg: params.cfg,
        defaultProvider: DEFAULT_PROVIDER,
        defaultModel: DEFAULT_MODEL,
      });
      const catalog = await loadModelCatalog({ config: params.cfg });
      const status = getModelRefStatus({
        cfg: params.cfg,
        catalog,
        ref: hooksModelRef,
        defaultProvider,
        defaultModel,
      });
      if (!status.allowed) {
        params.logHooks.warn(
          `hooks.gmail.model "${status.key}" not available through the current model selections/pool (will use primary instead)`,
        );
      }
      if (!status.inCatalog) {
        params.logHooks.warn(
          `hooks.gmail.model "${status.key}" not in the model catalog (may fail at runtime)`,
        );
      }
    }
  }

  // Launch configured channels so gateway replies via the surface the message came from.
  // Tests can opt out via MARV_SKIP_CHANNELS/MARV_SKIP_PROVIDERS (legacy MARV_* also supported).
  const skipChannels =
    isTruthyEnvValue(process.env.MARV_SKIP_CHANNELS) ||
    isTruthyEnvValue(process.env.MARV_SKIP_PROVIDERS) ||
    isTruthyEnvValue(process.env.MARV_SKIP_CHANNELS) ||
    isTruthyEnvValue(process.env.MARV_SKIP_PROVIDERS);
  if (!skipChannels) {
    try {
      await params.startChannels();
    } catch (err) {
      params.logChannels.error(`channel startup failed: ${String(err)}`);
    }
  } else {
    params.logChannels.info(
      "skipping channel start (MARV_SKIP_CHANNELS=1 or MARV_SKIP_PROVIDERS=1; legacy MARV_* also supported)",
    );
  }

  let pluginServices: PluginServicesHandle | null = null;
  try {
    pluginServices = await startPluginServices({
      registry: params.pluginRegistry,
      config: params.cfg,
      workspaceDir: params.defaultWorkspaceDir,
    });
  } catch (err) {
    params.log.warn(`plugin services failed to start: ${String(err)}`);
  }

  void startGatewayMemoryBackend({ cfg: params.cfg, log: params.log }).catch((err) => {
    params.log.warn(`qmd memory startup initialization failed: ${String(err)}`);
  });

  if (shouldWakeFromRestartSentinel()) {
    setTimeout(() => {
      void scheduleRestartSentinelWake({ deps: params.deps });
    }, 750);
  }

  return { browserControl, pluginServices };
}
